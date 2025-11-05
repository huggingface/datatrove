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
use tokio::time::{Duration, sleep};
use tokio::sync::Semaphore;

const SENTINEL: u32 = u32::MAX;

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

async fn with_retry<F, Fut, T>(f: F) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let retry_strategy = ExponentialBackoff::from_millis(1000)
        .max_delay(Duration::from_secs(300))
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

    /// Total number of concurrent downloads
    #[arg(long, default_value = "0")]
    downloads: usize,
}

#[derive(Debug, Clone)]
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
struct UnionFindData {
    union_set: HashMap<(u32, u32), (u32, u32)>,
    set_size: HashMap<(u32, u32), usize>,
}

#[derive(Debug)]
struct UnionFind {
    data: Arc<Mutex<UnionFindData>>,
}

impl UnionFind {
    fn new() -> Self {
        UnionFind {
            data: Arc::new(Mutex::new(UnionFindData {
                union_set: HashMap::new(),
                set_size: HashMap::new(),
            })),
        }
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

async fn process_single_file(
    client: &Client,
    output_path: &S3Path,
    file_number: u32,
    union_find: &Arc<UnionFindData>,
    pb: &ProgressBar,
) -> Result<(usize, usize)> {
    let mut to_remove = 0;
    let mut clusters = 0;
    const BUFFER_THRESHOLD: usize = 5 * 1024 * 1024;

    let nodes_data = {
        let mut docs = union_find.union_set.keys()
            .filter(|(f, _)| *f == file_number)
            .map(|(_, d)| *d)
            .collect::<Vec<_>>();
        docs.sort_unstable();

        docs.into_iter().map(|doc| {
            let node = (file_number, doc);
            let mut current = node;
            while let Some(&parent) = union_find.union_set.get(&current) {
                if parent == current {
                    break;
                }
                current = parent;
            }
            let root = current;
            let size = *union_find.set_size.get(&root).unwrap_or(&1);
            (doc, root, size)
        }).collect::<Vec<_>>()
    };

    let mut sizes_writer = S3StreamWriter::new(
        client,
        &output_path.bucket,
        &output_path.with_key(&format!("{:06}.sizes", file_number)),
        BUFFER_THRESHOLD,
    ).await?;

    let mut remove_writer = S3StreamWriter::new(
        client,
        &output_path.bucket,
        &output_path.with_key(&format!("{:06}.remove", file_number)),
        BUFFER_THRESHOLD,
    ).await?;

    for (doc, root, size) in nodes_data {
        let node = (file_number, doc);

        // Write sizes
        let mut buffer = Vec::new();
        buffer.write_u32::<LittleEndian>(doc)?;
        buffer.write_u32::<LittleEndian>(size as u32)?;
        sizes_writer.write(&buffer).await?;

        // Handle removal markers
        if node != root {
            let mut remove_buffer = Vec::new();
            remove_buffer.write_u32::<LittleEndian>(doc)?;
            remove_writer.write(&remove_buffer).await?;
            to_remove += 1;
        }

        if node == root {
            clusters += 1;
        }

        pb.inc(1);
    }

    sizes_writer.finalize().await?;
    remove_writer.finalize().await?;

    Ok((to_remove, clusters))
}

async fn process_post_union(
    client: &Client,
    output_path: &S3Path,
    union_find: UnionFind,  // Changed from &UnionFind to take ownership
) -> Result<(usize, usize)> {
    let data = union_find.data.lock().unwrap();
    let mut files: Vec<_> = data.union_set.keys()
        .map(|(f, _)| *f)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let total_nodes = data.union_set.len();
    drop(data);

    // Convert to immutable Arc
    let union_find_data = Arc::new(Arc::try_unwrap(union_find.data)
        .expect("All threads should be finished")
        .into_inner()
        .unwrap());

    files.sort_unstable();

    println!("Processing {} files in parallel...", files.len());
    let pb = Arc::new(ProgressBar::new(total_nodes as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let pb_clone = Arc::clone(&pb);
    tokio::spawn(async move {
        while !pb_clone.is_finished() {
            let elapsed = pb_clone.elapsed(); // Time elapsed since progress bar creation
            let eta = pb_clone.eta();         // Estimated time remaining
            eprintln!(
                "Progress: {}/{} | Elapsed: {} | Remaining: {}",
                pb_clone.position(),
                pb_clone.length().unwrap_or(0),
                format_duration(elapsed),
                format_duration(eta)
            );
            sleep(Duration::from_secs(5)).await;
        }
    });


    let semaphore = Arc::new(Semaphore::new(100));


    let mut handles = Vec::new();
    for file_number in files {
        let client = client.clone();
        let output_path = output_path.clone();
        let union_find_data = Arc::clone(&union_find_data);
        let pb = pb.clone();
        let semaphore = Arc::clone(&semaphore);

        let handle = task::spawn(async move {
            let _permit = semaphore.acquire().await?;
            process_single_file(
                &client,
                &output_path,
                file_number,
                &union_find_data,
                &pb,
            ).await
        });
        handles.push(handle);
    }

    let mut total_to_remove = 0;
    let mut total_clusters = 0;

    for handle in handles {
        let (to_remove, clusters) = handle.await??;
        total_to_remove += to_remove;
        total_clusters += clusters;
    }

    pb.finish_with_message("Output writing complete");

    Ok((total_to_remove, total_clusters))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = Client::new(&config);

    let input_path = S3Path::from_path(&args.input_folder)?;
    let output_path = S3Path::from_path(&args.output_folder)?;

    let files = list_s3_files(&client, &input_path, args.total_files).await?;

    let union_find = UnionFind::new();
    let semaphore = Arc::new(if args.downloads == 0 {
        Semaphore::new(args.total_files)  // Effectively unlimited
    } else {
        Semaphore::new(args.downloads)
    });

    println!("Processing {} input files...", files.len());
    let pb = Arc::new(ProgressBar::new(files.len() as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let pb_clone = Arc::clone(&pb);
    tokio::spawn(async move {
        while !pb_clone.is_finished() {
            let elapsed = pb_clone.elapsed(); // Time elapsed since progress bar creation
            let eta = pb_clone.eta();         // Estimated time remaining
            eprintln!(
                "Progress: {}/{} | Elapsed: {} | Remaining: {}",
                pb_clone.position(),
                pb_clone.length().unwrap_or(0),
                format_duration(elapsed),
                format_duration(eta)
            );
            sleep(Duration::from_secs(5)).await;
        }
    });
    let mut handles = Vec::new();

    for file_path in files {
        let client = client.clone();
        let union_find = Arc::clone(&union_find.data);
        let pb = pb.clone();
        let semaphore = Arc::clone(&semaphore);

        let handle = task::spawn(async move {
            let _permit = semaphore.acquire().await?;
            let tuples = download_and_parse_file(&client, &file_path).await?;

            let mut data = union_find.lock().unwrap();
            for (f1, d1, f2, d2) in tuples {
                let v_a = (f1, d1);
                let v_b = (f2, d2);

                let root_a = {
                    let mut current = v_a;
                    let mut path = Vec::new();
                    while let Some(&parent) = data.union_set.get(&current) {
                        if parent == current {
                            break;
                        }
                        path.push(current);
                        current = parent;
                    }
                    if !data.union_set.contains_key(&current) {
                        data.union_set.insert(current, current);
                    }
                    for node in path {
                        data.union_set.insert(node, current);
                    }
                    current
                };

                let root_b = {
                    let mut current = v_b;
                    let mut path = Vec::new();
                    while let Some(&parent) = data.union_set.get(&current) {
                        if parent == current {
                            break;
                        }
                        path.push(current);
                        current = parent;
                    }
                    if !data.union_set.contains_key(&current) {
                        data.union_set.insert(current, current);
                    }
                    for node in path {
                        data.union_set.insert(node, current);
                    }
                    current
                };

                if root_a != root_b {
                    let size_a = *data.set_size.get(&root_a).unwrap_or(&1);
                    let size_b = *data.set_size.get(&root_b).unwrap_or(&1);

                    let (big_root, small_root) = if root_a == (SENTINEL, SENTINEL) || (size_a >= size_b && root_b != (SENTINEL, SENTINEL)) {
                        (root_a, root_b)
                    } else {
                        (root_b, root_a)
                    };

                    data.union_set.insert(small_root, big_root);
                    data.set_size.insert(big_root, size_a + size_b);
                    data.set_size.remove(&small_root);
                }
            }
            drop(data);
            pb.inc(1);
            Ok::<(), anyhow::Error>(())
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.await??;
    }
    pb.finish_with_message("File processing complete");

    let (to_remove, clusters) = process_post_union(&client, &output_path, union_find).await?;

    println!("Processing complete:");
    println!("  Total clusters: {}", clusters);
    println!("  Documents to remove: {}", to_remove);

    Ok(())
}
