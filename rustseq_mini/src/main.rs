// CLI entry point

use clap::Parser;
mod smith_waterman;

#[derive(Parser)]
struct Args {
    seq1: String,
    seq2: String,
}

fn main() {
    let args = Args::parse();
    let score = smith_waterman::align(&args.seq1, &args.seq2);
    println!("Score: {}", score);
}