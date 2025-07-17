use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use flate2::read::GzDecoder;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: linecount <path/to/file.fastq.gz>");
        std::process::exit(1);
    }
    let path = &args[1];
    let file = File::open(path).expect("Failed to open file");
    let gz = GzDecoder::new(file);
    let reader = BufReader::new(gz);
    let mut count = 0u64;
    for line in reader.lines() {
        match line {
            Ok(_) => count += 1,
            Err(e) => {
                eprintln!("Error at line {}: {}", count, e);
                break;
            }
        }
        if count % 10_000_000 == 0 && count > 0 {
            println!("Read {} lines so far...", count);
        }
    }
    println!("Total lines: {}", count);
} 