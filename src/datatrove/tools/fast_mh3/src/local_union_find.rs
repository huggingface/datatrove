use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use anyhow::{Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use tokio::task;
use std::sync::{Arc, Mutex};
// use tokio::time::{Duration, sleep};
use tokio::sync::Semaphore;

const SENTINEL: u32 = u32::MAX;

// fn format_duration(duration: Duration) -> String {
//     let secs = duration.as_secs();
//     let hours = secs / 3600;
//     let minutes = (secs % 3600) / 60;
//     let seconds = secs % 60;
//     format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
// }

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input folder path
    #[arg(long)]
    input_folder: String,

    /// Output folder path
    #[arg(long)]
    output_folder: String,

    /// Total number of files to process
    #[arg(long)]
    total_files: usize,

    /// Total number of concurrent operations
    #[arg(long, default_value = "0")]
    concurrent_ops: usize,
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

async fn list_files(input_folder: &str, total_files: usize) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(input_folder)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect();

    files.sort();

    if files.len() != total_files {
        anyhow::bail!(
            "Expected {} files, found {} in {}",
            total_files,
            files.len(),
            input_folder
        );
    }

    Ok(files)
}

fn read_and_parse_file(file_path: &Path) -> Result<Vec<(u32, u32, u32, u32)>> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
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
    output_folder: &Path,
    file_number: u32,
    union_find: &Arc<UnionFindData>,
    pb: &ProgressBar,
) -> Result<(usize, usize)> {
    let mut to_remove = 0;
    let mut clusters = 0;

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

    let sizes_path = output_folder.join(format!("{:06}.sizes", file_number));
    let remove_path = output_folder.join(format!("{:06}.remove", file_number));

    let mut sizes_writer = BufWriter::new(File::create(sizes_path)?);
    let mut remove_writer = BufWriter::new(File::create(remove_path)?);

    for (doc, root, size) in nodes_data {
        let node = (file_number, doc);

        // Write sizes
        sizes_writer.write_u32::<LittleEndian>(doc)?;
        sizes_writer.write_u32::<LittleEndian>(size as u32)?;

        // Handle removal markers
        if node != root {
            remove_writer.write_u32::<LittleEndian>(doc)?;
            to_remove += 1;
        }

        if node == root {
            clusters += 1;
        }

        pb.inc(1);
    }

    sizes_writer.flush()?;
    remove_writer.flush()?;

    Ok((to_remove, clusters))
}

async fn process_post_union(
    output_folder: &Path,
    union_find: UnionFind,
) -> Result<(usize, usize)> {
    let data = union_find.data.lock().unwrap();
    let mut files: Vec<_> = data.union_set.keys()
        .map(|(f, _)| *f)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let total_nodes = data.union_set.len();
    drop(data);

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

//     let pb_clone = Arc::clone(&pb);
//     tokio::spawn(async move {
//         while !pb_clone.is_finished() {
//             let elapsed = pb_clone.elapsed();
//             let eta = pb_clone.eta();
// //             eprintln!(
//                 "Progress: {}/{} | Elapsed: {} | Remaining: {}",
//                 pb_clone.position(),
//                 pb_clone.length().unwrap_or(0),
//                 format_duration(elapsed),
//                 format_duration(eta)
//             );
//             sleep(Duration::from_secs(5)).await;
//         }
//     });

    let semaphore = Arc::new(Semaphore::new(100));
    let mut handles = Vec::new();

    for file_number in files {
        let output_folder = output_folder.to_path_buf();
        let union_find_data = Arc::clone(&union_find_data);
        let pb = pb.clone();
        let semaphore = Arc::clone(&semaphore);

        let handle = task::spawn(async move {
            let _permit = semaphore.acquire().await?;
            process_single_file(
                &output_folder,
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

    // Create output directory if it doesn't exist
    fs::create_dir_all(&args.output_folder)?;

    let files = list_files(&args.input_folder, args.total_files).await?;

    let union_find = UnionFind::new();
    let semaphore = Arc::new(if args.concurrent_ops == 0 {
        Semaphore::new(args.total_files)  // Effectively unlimited
    } else {
        Semaphore::new(args.concurrent_ops)
    });

    println!("Processing {} input files...", files.len());
    let pb = Arc::new(ProgressBar::new(files.len() as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

//     let pb_clone = Arc::clone(&pb);
//     tokio::spawn(async move {
//         while !pb_clone.is_finished() {
//             let elapsed = pb_clone.elapsed();
//             let eta = pb_clone.eta();
//             eprintln!(
//                 "Progress: {}/{} | Elapsed: {} | Remaining: {}",
//                 pb_clone.position(),
//                 pb_clone.length().unwrap_or(0),
//                 format_duration(elapsed),
//                 format_duration(eta)
//             );
//             sleep(Duration::from_secs(5)).await;
//         }
//     });

    let mut handles = Vec::new();

    for file_path in files {
        let union_find = Arc::clone(&union_find.data);
        let pb = pb.clone();
        let semaphore = Arc::clone(&semaphore);
        let file_path = file_path.clone();

        let handle = task::spawn(async move {
            let _permit = semaphore.acquire().await?;
            let tuples = read_and_parse_file(&file_path)?;

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

    let (to_remove, clusters) = process_post_union(Path::new(&args.output_folder), union_find).await?;

    println!("Processing complete:");
    println!("  Total clusters: {}", clusters);
    println!("  Documents to remove: {}", to_remove);

    Ok(())
}
