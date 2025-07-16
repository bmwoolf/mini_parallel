// CLI entry point for genome-scale sequence alignment

use clap::Parser;
mod smith_waterman;
mod gpu;

#[derive(Parser)]
#[command(name = "rustseq_mini")]
#[command(about = "High-performance sequence alignment for genome-scale data")]
struct Args {
    /// first sequence or file path
    #[arg(short = '1', long)]
    seq1: String,
    
    /// second sequence or file path  
    #[arg(short = '2', long)]
    seq2: String,
    
    /// treat inputs as file paths instead of direct sequences
    #[arg(short, long, default_value = "false")]
    files: bool,
    
    /// chunk size in MB for large sequences (default: 1MB, chosen just because, no math reason)
    #[arg(short, long, default_value = "1")]
    chunk_size: usize,
    
    /// use GPU acceleration if available
    #[arg(short, long, default_value = "false")]
    gpu: bool,
    
    /// number of files to process (for multi-file mode)
    #[arg(short, long)]
    num_files: Option<usize>,
    
    /// test mode: read WGS files from USB drive
    #[arg(short, long, default_value = "false")]
    test_wgs: bool,
}

fn main() {
    let args = Args::parse();
    
    // Test WGS files from USB drive
    if args.test_wgs {
        println!("Testing WGS file reading from USB drive...");
        let wgs_path = "/media/bradley/64GB/";
        
        // Test reading first few files
        let test_files = [
            "102324-WGS-C3115440_S19_L001_R1_001.fastq.gz",
            "102324-WGS-C3115440_S19_L001_R2_001.fastq.gz",
        ];
        
        for file in &test_files {
            let full_path = format!("{}{}", wgs_path, file);
            println!("Testing: {}", full_path);
            
            match gpu::aligner::load_sequence_from_file(&full_path) {
                Ok(seq) => {
                    println!("✅ Successfully loaded {} bases from {}", seq.len(), file);
                    println!("   First 100 bases: {}", &seq[..seq.len().min(100)]);
                },
                Err(e) => {
                    println!("❌ Error loading {}: {}", file, e);
                }
            }
        }
        return;
    }
    
    // GPU only
    if !args.gpu || !gpu::is_gpu_available() {
        eprintln!("error: gpu acceleration is required and no compatible gpu was found");
        std::process::exit(1);
    }
    println!("GPU acceleration enabled");
    let devices = gpu::get_gpu_devices();
    for device in &devices {
        println!("  Found GPU: {} ({} GB)", device.name, device.memory_gb);
    }
    
    if args.files {
        match gpu::aligner::gpu_align_pair(&args.seq1, &args.seq2, &devices[0]) {
            Ok(result) => {
                println!("GPU Alignment Result:");
                println!("  Score: {}", result.score);
                println!("  Processing time: {:.2} ms", result.processing_time_ms);
                println!("  GPU device: {}", result.gpu_device);
            },
            Err(e) => {
                eprintln!("GPU alignment error: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        match gpu::aligner::gpu_align(&args.seq1, &args.seq2, &devices[0]) {
            Ok(score) => println!("GPU Alignment score: {}", score),
            Err(e) => {
                eprintln!("GPU alignment error: {}", e);
                std::process::exit(1);
            }
        }
    }
}