// CLI entry point for genome-scale sequence alignment

use clap::Parser;
mod smith_waterman;

#[derive(Parser)]
#[command(name = "rustseq_mini")]
#[command(about = "High-performance sequence alignment for genome-scale data")]
struct Args {
    /// first sequence or file path
    #[arg(short, long)]
    seq1: String,
    
    /// second sequence or file path  
    #[arg(short, long)]
    seq2: String,
    
    /// treat inputs as file paths instead of direct sequences
    #[arg(short, long, default_value = "false")]
    files: bool,
    
    /// chunk size in MB for large sequences (default: 1MB, chosen just because, no math reason)
    #[arg(short, long, default_value = "1")]
    chunk_size: usize,
}

fn main() {
    let args = Args::parse();
    
    if args.files {
        // process genome files
        match smith_waterman::align_from_files(&args.seq1, &args.seq2) {
            Ok(score) => println!("Total alignment score: {}", score),
            Err(e) => {
                eprintln!("Error reading files: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // process direct sequence input
        let score = smith_waterman::align(&args.seq1, &args.seq2);
        println!("Alignment score: {}", score);
    }
}