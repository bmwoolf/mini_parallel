use clap::Parser;
mod kmer_counter;

#[derive(Parser)]
struct Args {
    input: String,
    k: usize,
}

fn main() {
    let args = Args::parse();
    kmer_counter::count_kmers(&args.input, args.k);
}
