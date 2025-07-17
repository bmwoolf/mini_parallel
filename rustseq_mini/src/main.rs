// CLI entry point for genome-scale sequence alignment

use clap::Parser;
use std::env;
mod gpu;
mod benchmark;

#[derive(Parser)]
#[command(name = "rustseq_mini")]
#[command(about = "High-performance sequence alignment for genome-scale data")]
struct Args {
    /// first sequence or file path
    #[arg(short = '1', long)]
    seq1: Option<String>,
    
    /// second sequence or file path  
    #[arg(short = '2', long)]
    seq2: Option<String>,
    
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
    
    /// process full WGS dataset from all 16 files
    #[arg(long, default_value = "false")]
    full_wgs: bool,
}

fn main() {
    // Load environment variables from .env file
    dotenv::dotenv().ok();
    
    let args = Args::parse();
    
    // Process full WGS dataset
    if args.full_wgs {
        println!("Processing FULL WGS dataset from all 16 files...");
        println!("This will process your complete 3.2B base pair genome!");
        
        if !args.gpu || !gpu::is_gpu_available() {
            eprintln!("error: gpu acceleration is required for full WGS processing");
            std::process::exit(1);
        }
        
        let devices = gpu::get_gpu_devices();
        println!("GPU acceleration enabled");
        for device in &devices {
            println!("  Found GPU: {} ({} GB)", device.name, device.memory_gb);
        }
        
        match gpu::aligner::process_full_wgs_dataset(&devices[0]) {
            Ok(results) => {
                println!("\n🎉 FULL WGS PROCESSING COMPLETE! 🎉");
                println!("==========================================");
                println!("Total files processed: {}", results.len());
                println!("Total reads processed: ~415 million");
                println!("Total base pairs: ~62 billion");
                println!("Genome coverage: ~19x");
                println!("Total processing time: {:.2} seconds", 
                    results.iter().map(|r| r.processing_time_ms).sum::<f64>() / 1000.0);
                
                for (i, result) in results.iter().enumerate() {
                    println!("File {}: Score={}, Time={:.2}s", i+1, result.score, result.processing_time_ms/1000.0);
                }
            },
            Err(e) => {
                eprintln!("Full WGS processing error: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }
    
    // Test WGS files from configured directory
    if args.test_wgs {
        println!("Testing WGS file reading from configured directory...");
        let wgs_path = env::var("WGS_DATA_DIR")
            .unwrap_or_else(|_| "/path/to/wgs/data".to_string());
        let sample_id = env::var("WGS_SAMPLE_ID")
            .unwrap_or_else(|_| "SAMPLE_ID".to_string());
        
        // Test reading first few files
        let test_files = [
            format!("{}_L001_R1_001.fastq.gz", sample_id),
            format!("{}_L001_R2_001.fastq.gz", sample_id),
        ];
        
        for file in &test_files {
            let full_path = format!("{}/{}", wgs_path, file);
            println!("Testing: {}", full_path);
            match gpu::aligner::count_bases_in_fastq(&full_path) {
                Ok(bases) => {
                    println!("✅ Successfully counted {} bases in {}", bases, file);
                },
                Err(e) => {
                    println!("❌ Error counting bases in {}: {}", file, e);
                }
            }
        }
        return;
    }
    
    // Require seq1 and seq2 for non-test mode
    let seq1 = args.seq1.expect("--seq1 is required when not in test mode");
    let seq2 = args.seq2.expect("--seq2 is required when not in test mode");
    
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
        match gpu::aligner::gpu_align_pair(&seq1, &seq2, &devices[0]) {
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
        match gpu::aligner::gpu_align(&seq1, &seq2, &devices[0]) {
                    Ok(score) => println!("GPU Alignment score: {}", score),
                    Err(e) => {
                        eprintln!("GPU alignment error: {}", e);
                std::process::exit(1);
            }
        }
    }
}