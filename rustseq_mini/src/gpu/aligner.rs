// GPU-accelerated sequence aligner for 16 files
// Handles the main alignment logic and orchestrates GPU operations

use crate::gpu::{GpuAlignmentResult, GpuDevice, GPU_CHUNK_SIZE};
use std::time::Instant;

// Main GPU alignment function for 16 files
pub fn gpu_align_16_files(file_paths: [String; 16]) -> Vec<GpuAlignmentResult> {
    let mut results = Vec::new();
    
    // Check GPU availability
    if !super::is_gpu_available() {
        eprintln!("Warning: GPU not available, falling back to CPU");
        return results;
    }
    
    // Get GPU devices
    let devices = super::get_gpu_devices();
    if devices.is_empty() {
        eprintln!("Error: No GPU devices found");
        return results;
    }
    
    println!("Found {} GPU device(s)", devices.len());
    for device in &devices {
        println!("  - {} ({} GB)", device.name, device.memory_gb);
    }
    
    // process all pairs in one loop (16 choose 2 = 120 pairs)
    // TODO: replace with vecorization instead of loops
    for i in 0..16 {
        for j in (i + 1)..16 {
            let start_time = Instant::now();
            
            // match checks if gpu_align_pair succeeded
            match gpu_align_pair(&file_paths[i], &file_paths[j], &devices[0]) {
                Ok(mut result) => {
                    result.processing_time_ms = start_time.elapsed().as_millis() as f64;
                    results.push(result);
                },
                Err(e) => {
                    eprintln!("Error aligning {} vs {}: {}", file_paths[i], file_paths[j], e);
                }
            }
        }
    }
    
    results
}

// GPU alignment for a single pair of files
pub fn gpu_align_pair(file1: &str, file2: &str, device: &GpuDevice) -> Result<GpuAlignmentResult, String> {
    // Load sequences from files
    let seq1 = load_sequence_from_file(file1)?;
    let seq2 = load_sequence_from_file(file2)?;
    
    // Perform GPU alignment
    let score = gpu_align(&seq1, &seq2, device)?;
    
    Ok(GpuAlignmentResult {
        score,
        processing_time_ms: 0.0, // Will be set by caller
        memory_used_mb: 0.0, // Will be calculated
        gpu_device: device.name.clone(),
    })
}

// Main GPU alignment function for two sequences
pub fn gpu_align(seq1: &str, seq2: &str, device: &GpuDevice) -> Result<i32, String> {
    let bytes1 = seq1.as_bytes();
    let bytes2 = seq2.as_bytes();
    let len = bytes1.len().min(bytes2.len());
    
    if len == 0 {
        return Ok(0);
    }
    
    // Calculate optimal GPU grid configuration
    let threads_per_block = device.max_threads_per_block.min(256);
    let blocks_needed = ((len + threads_per_block as usize - 1) / threads_per_block as usize) as u32;
    let blocks = blocks_needed.min(super::GPU_MAX_BLOCKS);
    
    println!("GPU Grid: {} blocks × {} threads = {} total threads", 
             blocks, threads_per_block, blocks * threads_per_block);
    
    // Allocate GPU memory
    let gpu_memory = crate::gpu::memory::allocate_gpu_memory(len)?;
    
    // Copy data to GPU
    crate::gpu::memory::copy_to_gpu(&gpu_memory, bytes1, bytes2)?;
    
    // Launch GPU kernel
    let score = crate::gpu::kernel::launch_alignment_kernel(&gpu_memory, len, blocks, threads_per_block)?;
    
    // Clean up GPU memory
    crate::gpu::memory::free_gpu_memory(gpu_memory)?;
    
    Ok(score)
}

// Load sequence from FASTA file
fn load_sequence_from_file(filepath: &str) -> Result<String, String> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    
    let file = File::open(filepath)
        .map_err(|e| format!("Failed to open file {}: {}", filepath, e))?;
    
    let reader = BufReader::new(file);
    let mut sequence = String::new();
    
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
        
        // Skip header lines (start with '>')
        if line.starts_with('>') {
            continue;
        }
        
        // Append sequence data
        sequence.push_str(&line);
    }
    
    if sequence.is_empty() {
        return Err(format!("No sequence data found in {}", filepath));
    }
    
    Ok(sequence)
}

// GPU alignment with reference (all vs one)
pub fn gpu_align_against_reference(reference_file: &str, query_files: [String; 15]) -> Vec<GpuAlignmentResult> {
    let mut results = Vec::new();
    
    // Load reference sequence once
    let reference_seq = match load_sequence_from_file(reference_file) {
        Ok(seq) => seq,
        Err(e) => {
            eprintln!("Error loading reference file: {}", e);
            return results;
        }
    };
    
    // Get GPU device
    let devices = super::get_gpu_devices();
    if devices.is_empty() {
        eprintln!("Error: No GPU devices found");
        return results;
    }
    
    // Align each query against reference
    for query_file in query_files {
        let start_time = Instant::now();
        
        match load_sequence_from_file(&query_file) {
            Ok(query_seq) => {
                match gpu_align(&reference_seq, &query_seq, &devices[0]) {
                    Ok(score) => {
                        let mut result = GpuAlignmentResult {
                            score,
                            processing_time_ms: start_time.elapsed().as_millis() as f64,
                            memory_used_mb: 0.0,
                            gpu_device: devices[0].name.clone(),
                        };
                        results.push(result);
                    },
                    Err(e) => {
                        eprintln!("Error aligning {} vs reference: {}", query_file, e);
                    }
                }
            },
            Err(e) => {
                eprintln!("Error loading query file {}: {}", query_file, e);
            }
        }
    }
    
    results
} 