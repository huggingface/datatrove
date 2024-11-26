use std::io::Cursor;
use std::collections::HashMap;
use anyhow::{Context, Result};
use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::task;
use std::sync::{Arc, Mutex};
use tokio_retry::Retry;
use tokio_retry::strategy::{ExponentialBackoff, jitter};
use std::time::Duration;

async fn with_retry<F, Fut, T>(f: F) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let retry_strategy = ExponentialBackoff::from_millis(1000)
        .max_delay(Duration::from_secs(30))
        .map(jitter)
        .take(3);

    Retry::spawn(retry_strategy, || async {
        f().await
    }).await
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input S3 folder path (e.g., s3://bucket/input/)
    #[arg(long)]
    input_folder: String,

    /// Output S3 folder path (e.g., s3://bucket/output/)
    #[arg(long)]
    output_folder: String,

    /// Total number of files to process
    #[arg(long)]
    total_files: usize,
}

#[derive(Debug)]
struct S3Path {
    bucket: String,
    prefix: String,
}

impl S3Path {
    fn from_path(path: &str) -> Result<Self> {
        let parts: Vec<&str> = path.trim_start_matches("s3://").split('/').collect();
        if parts.len() < 2 {
            anyhow::bail!("Invalid S3 path: {}", path);
        }
        Ok(S3Path {
            bucket: parts[0].to_string(),
            prefix: parts[1..].join("/"),
        })
    }

    fn with_key(&self, key: &str) -> String {
        format!("{}/{}", self.prefix.trim_end_matches('/'), key)
    }
}

#[derive(Debug)]
struct UnionFind {
    union_set: HashMap<(u32, u32), (u32, u32)>,
    set_size: HashMap<(u32, u32), usize>,
}

impl UnionFind {
    fn new() -> Self {
        UnionFind {
            union_set: HashMap::new(),
            set_size: HashMap::new(),
        }
    }

    fn find_parent(&mut self, x: (u32, u32)) -> (u32, u32) {
        if !self.union_set.contains_key(&x) || self.union_set.get(&x) == Some(&x) {
            self.union_set.insert(x, x);
            return x;
        }

        // Get parent and recurse
        let parent = *self.union_set.get(&x).unwrap();
        let root = self.find_parent(parent);
        self.union_set.insert(x, root);
        root
    }

    fn union(&mut self, v_a: (u32, u32), v_b: (u32, u32)) {
        let mut root_a = self.find_parent(v_a);
        let mut root_b = self.find_parent(v_b);

        if root_a == root_b {
            return;
        }

        let size_a = *self.set_size.get(&root_a).unwrap_or(&1);
        let size_b = *self.set_size.get(&root_b).unwrap_or(&1);

        if size_a < size_b {
            std::mem::swap(&mut root_a, &mut root_b);
        }

        self.union_set.insert(root_b, root_a);
        let new_size = size_a + size_b;
        self.set_size.insert(root_a, new_size);
        self.set_size.remove(&root_b);
    }
}

struct S3StreamWriter {
    client: Client,
    bucket: String,
    key: String,
    upload_id: String,
    buffer: Vec<u8>,
    part_number: i32,
    completed_parts: Vec<CompletedPart>,
    buffer_threshold: usize,
}

impl S3StreamWriter {
    async fn new(
        client: &Client,
        bucket: &str,
        key: &str,
        buffer_threshold: usize,
    ) -> Result<Self> {
        let create_multipart_upload_output = with_retry(|| async {
            client
                .create_multipart_upload()
                .bucket(bucket)
                .key(key)
                .send()
                .await
                .context("Failed to create multipart upload")
        }).await?;

        Ok(Self {
            client: client.clone(),
            bucket: bucket.to_string(),
            key: key.to_string(),
            upload_id: create_multipart_upload_output.upload_id().unwrap().to_string(),
            buffer: Vec::new(),
            part_number: 1,
            completed_parts: Vec::new(),
            buffer_threshold,
        })
    }

    async fn write(&mut self, data: &[u8]) -> Result<()> {
        self.buffer.extend_from_slice(data);

        if self.buffer.len() >= self.buffer_threshold {
            self.flush().await?;
        }

        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let buffer_clone = self.buffer.clone();
        let upload_part_output = with_retry(|| async {
            let part_body = ByteStream::from(buffer_clone.clone());
            self.client
                .upload_part()
                .bucket(&self.bucket)
                .key(&self.key)
                .upload_id(&self.upload_id)
                .part_number(self.part_number)
                .body(part_body)
                .send()
                .await
                .context("Failed to upload part")
        }).await?;

        let completed_part = CompletedPart::builder()
            .e_tag(upload_part_output.e_tag().unwrap_or_default())
            .part_number(self.part_number)
            .build();

        self.completed_parts.push(completed_part);
        self.part_number += 1;
        self.buffer.clear();

        Ok(())
    }

    async fn finalize(mut self) -> Result<()> {
        self.flush().await?;

        let completed_multipart_upload = CompletedMultipartUpload::builder()
            .set_parts(Some(self.completed_parts.clone()))
            .build();

        with_retry(|| async {
            self.client
                .complete_multipart_upload()
                .bucket(&self.bucket)
                .key(&self.key)
                .upload_id(&self.upload_id)
                .multipart_upload(completed_multipart_upload.clone())
                .send()
                .await
                .context("Failed to complete multipart upload")
        }).await?;

        Ok(())
    }
}

async fn list_s3_files(client: &Client, s3_path: &S3Path, total_files: usize) -> Result<Vec<String>> {
    let resp = with_retry(|| async {
        client
            .list_objects_v2()
            .bucket(&s3_path.bucket)
            .prefix(&s3_path.prefix)
            .send()
            .await
            .context("Failed to list S3 objects")
    }).await?;

    let mut files: Vec<String> = resp
        .contents()
        .iter()
        .filter_map(|obj| obj.key()
            .map(|key| format!("s3://{}/{}", s3_path.bucket, key)))
        .collect();

    // Sort files lexicographically
    files.sort();

    if files.len() != total_files {
        anyhow::bail!(
            "Expected {} files, found {} in s3://{}/{}",
            total_files,
            files.len(),
            s3_path.bucket,
            s3_path.prefix
        );
    }

    Ok(files)
}

async fn download_and_parse_file(client: &Client, file_path: &str) -> Result<Vec<(u32, u32, u32, u32)>> {
    let s3_path = S3Path::from_path(file_path)?;

    let resp = with_retry(|| async {
        client
            .get_object()
            .bucket(&s3_path.bucket)
            .key(&s3_path.prefix)
            .send()
            .await
            .context("Failed to download S3 object")
    }).await?;

    let body = resp.body.collect().await?.into_bytes();
    let mut reader = Cursor::new(body);
    let mut tuples = Vec::new();

    while let (Ok(f1), Ok(d1), Ok(f2), Ok(d2)) = (
        reader.read_u32::<LittleEndian>(),
        reader.read_u32::<LittleEndian>(),
        reader.read_u32::<LittleEndian>(),
        reader.read_u32::<LittleEndian>(),
    ) {
        tuples.push((f1, d1, f2, d2));
    }

    Ok(tuples)
}

async fn process_post_union(
    client: &Client,
    output_path: &S3Path,
    union_find: &mut UnionFind,
) -> Result<(usize, usize)> {
    let mut to_remove = 0;
    let mut clusters = 0;
    const BUFFER_THRESHOLD: usize = 5 * 1024 * 1024; // 5MB

    let mut writers: HashMap<String, S3StreamWriter> = HashMap::new();

    // Create sorted list of nodes
    let mut nodes: Vec<(u32, u32)> = union_find.union_set.keys().cloned().collect();
    nodes.sort();

    println!("Writing output files...");
    let pb = ProgressBar::new(nodes.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    for node in nodes {
        let (file, doc) = node;
        let p = union_find.find_parent(node);
        let size = *union_find.set_size.get(&p).unwrap_or(&1);

        // Write sizes
        let sizes_key = format!("{:06}.sizes", file);
        if !writers.contains_key(&sizes_key) {
            writers.insert(
                sizes_key.clone(),
                S3StreamWriter::new(
                    client,
                    &output_path.bucket,
                    &output_path.with_key(&sizes_key),
                    BUFFER_THRESHOLD,
                )
                .await?,
            );
        }

        let mut buffer = Vec::new();
        buffer.write_u32::<LittleEndian>(doc)?;
        buffer.write_u32::<LittleEndian>(size as u32)?;

        if let Some(writer) = writers.get_mut(&sizes_key) {
            writer.write(&buffer).await?;
        }

        // Handle removal markers
        if node != p {
            let remove_key = format!("{:06}.remove", file);
            if !writers.contains_key(&remove_key) {
                writers.insert(
                    remove_key.clone(),
                    S3StreamWriter::new(
                        client,
                        &output_path.bucket,
                        &output_path.with_key(&remove_key),
                        BUFFER_THRESHOLD,
                    )
                    .await?,
                );
            }

            let mut remove_buffer = Vec::new();
            remove_buffer.write_u32::<LittleEndian>(doc)?;

            if let Some(writer) = writers.get_mut(&remove_key) {
                writer.write(&remove_buffer).await?;
            }

            to_remove += 1;
        }

        if node == p {
            clusters += 1;
        }

        pb.inc(1);
    }

    pb.finish_with_message("Output writing complete");

    // Finalize all writers
    let num_writers = writers.len();
    println!("Finalizing {} output files...", num_writers);
    let pb = ProgressBar::new(num_writers as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    for (_, writer) in writers {
        writer.finalize().await?;
        pb.inc(1);
    }
    pb.finish_with_message("All files finalized");

    Ok((to_remove, clusters))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = Client::new(&config);

    let input_path = S3Path::from_path(&args.input_folder)?;
    let output_path = S3Path::from_path(&args.output_folder)?;

    let files = list_s3_files(&client, &input_path, args.total_files).await?;

    let union_find = Arc::new(Mutex::new(UnionFind::new()));

    println!("Processing {} input files...", files.len());
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    let mut handles = Vec::new();

    for file_path in files {
        let client = client.clone();
        let union_find = Arc::clone(&union_find);
        let pb = pb.clone();

        let tuples = download_and_parse_file(&client, &file_path).await?;
        let mut uf = union_find.lock().unwrap();
        for (f1, d1, f2, d2) in tuples {
            uf.union((f1, d1), (f2, d2));
        }
        pb.inc(1);
//
//
//         let handle = task::spawn(async move {
//             let tuples = download_and_parse_file(&client, &file_path).await?;
//             let mut uf = union_find.lock().unwrap();
//             for (f1, d1, f2, d2) in tuples {
//                 uf.union((f1, d1), (f2, d2));
//             }
//             pb.inc(1);
//             Ok::<(), anyhow::Error>(())
//         });

//         handles.push(handle);
    }

//     for handle in handles {
//         handle.await??;
//     }
    pb.finish_with_message("File processing complete");

    let mut union_find = match Arc::try_unwrap(union_find) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(_) => panic!("Failed to unwrap Arc, some references still exist"),
    };

    let (to_remove, clusters) = process_post_union(&client, &output_path, &mut union_find).await?;

    println!("Processing complete:");
    println!("  Total clusters: {}", clusters);
    println!("  Documents to remove: {}", to_remove);

    Ok(())
}