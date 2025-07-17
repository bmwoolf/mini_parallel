// GPU-accelerated sequence aligner using OpenCL
// Handles the main alignment logic and orchestrates GPU operations

use crate::gpu::{GpuAlignmentResult, GpuDevice, GPU_WORK_GROUP_SIZE, GPU_MAX_WORK_GROUPS};
use ocl::{Buffer, Program, Kernel, MemFlags};

// Remove unused functions: gpu_align_16_files, gpu_align_against_reference
// Remove unused imports inside functions
// Remove unused mut from variables

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

// Simple streaming FASTQ processor that yields chunks as they're read
pub fn process_fastq_file_in_chunks<F>(filepath: &str, chunk_size_reads: usize, mut processor: F) -> Result<(), String> 
where F: FnMut(&[String]) -> Result<(), String> {
    let reader: Box<dyn BufRead> = if filepath.ends_with(".gz") {
        // Use system zcat for gzipped files (fixes flate2 issues with large files)
        let child = Command::new("zcat")
            .arg(filepath)
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn zcat for {}: {}", filepath, e))?;
        
        let stdout = child.stdout
            .ok_or_else(|| format!("Failed to get stdout from zcat for {}", filepath))?;
        
        Box::new(BufReader::new(stdout))
    } else {
        // For non-gzipped files, use regular file reader
        let file = File::open(filepath)
            .map_err(|e| format!("Failed to open file {}: {}", filepath, e))?;
        Box::new(BufReader::new(file))
    };
    
    let mut chunk = Vec::with_capacity(chunk_size_reads);
    let mut line_count = 0;
    let mut total_reads = 0;
    let mut error_count = 0;
    
    for line_result in reader.lines() {
        match line_result {
            Ok(line) => {
                line_count += 1;
                
                if line_count % 4 == 2 {
                    // This is a sequence line
                    chunk.push(line);
                    total_reads += 1;
                    
                    if chunk.len() >= chunk_size_reads {
                        // Process this chunk
                        processor(&chunk)?;
                        chunk.clear();
                    }
                }
                
                // Debug output every 100,000 lines
                if line_count % 100000 == 0 {
                    println!("    Debug: Read {} lines, found {} reads, current chunk size: {}", line_count, total_reads, chunk.len());
                }
            },
            Err(e) => {
                error_count += 1;
                if error_count <= 5 {
                    println!("    Warning: Error reading line {}: {}", line_count, e);
                }
                if error_count > 10 {
                    return Err(format!("Too many read errors (>10), stopping at line {}", line_count));
                }
            }
        }
    }
    
    // Process any remaining reads in the final chunk
    if !chunk.is_empty() {
        processor(&chunk)?;
    }
    
    println!("    Processed {} total reads in {} chunks", total_reads, (total_reads + chunk_size_reads - 1) / chunk_size_reads);
    println!("    Total lines read: {}", line_count);
    if error_count > 0 {
        println!("    Total read errors: {}", error_count);
    }
    Ok(())
}

// Process full WGS dataset from all 16 files
pub fn process_full_wgs_dataset(device: &GpuDevice) -> Result<Vec<GpuAlignmentResult>, String> {
    let wgs_path = std::env::var("WGS_DATA_DIR")
        .unwrap_or_else(|_| "/path/to/wgs/data".to_string());
    let sample_id = std::env::var("WGS_SAMPLE_ID")
        .unwrap_or_else(|_| "SAMPLE_ID".to_string());
    let lanes: usize = std::env::var("WGS_LANES")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap_or(8);
    let reads_per_lane: usize = std::env::var("WGS_READS_PER_LANE")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .unwrap_or(2);
    
    // Generate file paths dynamically based on configuration
    let mut files = Vec::new();
    for lane in 1..=lanes {
        for read in 1..=reads_per_lane {
            let filename = format!("{}_L{:03}_R{}_001.fastq.gz", sample_id, lane, read);
            files.push(format!("{}/{}", wgs_path, filename));
        }
    }
    
    let mut results = Vec::new();
    let total_files = files.len();
    
    println!("Processing {} files (your complete genome)...", total_files);
    println!("Estimated total reads: ~415 million");
    println!("Estimated total base pairs: ~62 billion");
    println!("Estimated genome coverage: ~19x");
    println!("==========================================");
    
    for (i, file) in files.iter().enumerate() {
        println!("Processing file {}/{}: {}", i+1, total_files, file.split('/').last().unwrap());
        
        let start_time = std::time::Instant::now();
        
        let chunk_size_reads: usize = std::env::var("GPU_CHUNK_SIZE_READS")
            .unwrap_or_else(|_| "1000".to_string())  // Reduced from 10000 to 1000
            .parse()
            .unwrap_or(1000);
        
        let mut total_score = 0;
        let mut processed_chunks = 0;
        let mut total_bases = 0;
        
        println!("    Using chunk size: {} reads", chunk_size_reads);
        
        // Process the file in chunks using our streaming processor
        process_fastq_file_in_chunks(file, chunk_size_reads, |chunk| {
            let seq = chunk.concat();
            total_bases += seq.len();
            
            match gpu_align_chunk_self(&seq, device) {
                Ok(score) => {
                    total_score += score;
                    processed_chunks += 1;
                    if processed_chunks % 10 == 0 {
                        println!("    Processed {} chunks ({} reads), current score: {}", processed_chunks, chunk.len(), total_score);
                    }
                },
                Err(e) => {
                    println!("    Warning: Failed to align chunk {}: {}", processed_chunks, e);
                }
            }
            Ok(())
        })?;
        
        let processing_time = start_time.elapsed();
        
        println!("  ✅ File {} complete: Score={}, Bases={}, Time={:.2}s", i+1, total_score, total_bases, processing_time.as_secs_f64());
        
        results.push(GpuAlignmentResult {
            score: total_score,
            processing_time_ms: processing_time.as_millis() as f64,
            _memory_used_mb: 0.0,
            gpu_device: device.name.clone(),
            _work_groups: 0,
            _work_group_size: 0,
        });
    }
    
    Ok(results)
}

// Self-alignment of a single chunk (for full WGS processing)
fn gpu_align_chunk_self(chunk: &str, device: &GpuDevice) -> Result<i32, String> {
    if chunk.len() < 100 {
        return Ok(0); // Skip very small chunks
    }
    
    // For self-alignment, we'll align the chunk against itself
    // This gives us a measure of internal sequence similarity
    gpu_align(chunk, chunk, device)
}

// GPU alignment for a single pair of files
pub fn gpu_align_pair(file1: &str, file2: &str, device: &GpuDevice) -> Result<GpuAlignmentResult, String> {
    // Count total bases in each file
    let bases1 = count_bases_in_fastq(file1)?;
    let bases2 = count_bases_in_fastq(file2)?;
    println!("Loaded {} bases from {}", bases1, file1);
    println!("Loaded {} bases from {}", bases2, file2);
    // For actual alignment, just use the chunked logic (no need to load all into memory)
    // Use chunked alignment for large sequences
    let start_time = std::time::Instant::now();
    let score = {
        // We'll just align the first chunk of each file for demonstration (or you can implement full pairwise chunked alignment)
        // For now, use the same chunked logic as before
        let mut total_score = 0;
        process_fastq_file_in_chunks(file1, 100_000, |chunk1| {
            process_fastq_file_in_chunks(file2, 100_000, |chunk2| {
                let seq1 = chunk1.concat();
                let seq2 = chunk2.concat();
                total_score += gpu_align(&seq1, &seq2, device)?;
                Ok(())
            })?;
            Ok(())
        })?;
        total_score
    };
    let processing_time = start_time.elapsed();
    Ok(GpuAlignmentResult {
        score,
        processing_time_ms: processing_time.as_millis() as f64,
        _memory_used_mb: 0.0, // Will be calculated
        gpu_device: device.name.clone(),
        _work_groups: 0, // Will be calculated
        _work_group_size: 0, // Will be calculated
    })
}

// Chunked GPU alignment for large sequences
fn gpu_align_chunked(seq1: &str, seq2: &str, device: &GpuDevice, chunk_size: usize) -> Result<i32, String> {
    let bytes1 = seq1.as_bytes();
    let bytes2 = seq2.as_bytes();
    let len1 = bytes1.len();
    let len2 = bytes2.len();
    
    let mut total_score = 0;
    let mut processed_chunks = 0;
    
    // Process sequences in chunks
    for i in (0..len1).step_by(chunk_size) {
        for j in (0..len2).step_by(chunk_size) {
            let end1 = (i + chunk_size).min(len1);
            let end2 = (j + chunk_size).min(len2);
            
            let chunk1 = &bytes1[i..end1];
            let chunk2 = &bytes2[j..end2];
            
            // Convert chunks back to strings for alignment
            let chunk1_str = String::from_utf8_lossy(chunk1);
            let chunk2_str = String::from_utf8_lossy(chunk2);
            
            match gpu_align(&chunk1_str, &chunk2_str, device) {
                Ok(score) => {
                    total_score += score;
                    processed_chunks += 1;
                    if processed_chunks % 10 == 0 {
                        println!("Processed {} chunks, current score: {}", processed_chunks, total_score);
                    }
                },
                Err(e) => {
                    println!("Warning: Failed to align chunk {}-{}: {}", i, j, e);
                    // Continue with other chunks
                }
            }
        }
    }
    
    println!("Completed chunked alignment: {} chunks processed, final score: {}", processed_chunks, total_score);
    Ok(total_score)
}

// Main GPU alignment function for two sequences using OpenCL
pub fn gpu_align(seq1: &str, seq2: &str, device: &GpuDevice) -> Result<i32, String> {
    let bytes1 = seq1.as_bytes();
    let bytes2 = seq2.as_bytes();
    let len = bytes1.len().min(bytes2.len());
    if len == 0 {
        return Ok(0);
    }
    // Initialize OpenCL
    let (context, queue) = super::init_opencl()
        .map_err(|e| format!("Failed to initialize OpenCL: {}", e))?;
    // Calculate optimal OpenCL work group configuration
    let work_group_size = device.max_work_group_size.min(GPU_WORK_GROUP_SIZE);
    let work_groups_needed = (len + work_group_size - 1) / work_group_size;
    let work_groups = work_groups_needed.min(GPU_MAX_WORK_GROUPS);
    println!("OpenCL Grid: {} work groups × {} work items = {} total work items", 
             work_groups, work_group_size, work_groups * work_group_size);
    // Create OpenCL buffers
    let seq1_buffer = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_only().copy_host_ptr())
        .len(bytes1.len())
        .copy_host_slice(bytes1)
        .build()
        .map_err(|e| format!("Failed to create seq1 buffer: {}", e))?;
    let seq2_buffer = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_only().copy_host_ptr())
        .len(bytes2.len())
        .copy_host_slice(bytes2)
        .build()
        .map_err(|e| format!("Failed to create seq2 buffer: {}", e))?;
    let result_buffer = Buffer::<i32>::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().write_only())
        .len(1)
        .build()
        .map_err(|e| format!("Failed to create result buffer: {}", e))?;
    // Create and build OpenCL program
    let program_src = include_str!("kernels/smith_waterman.cl");
    let program = Program::builder()
        .src(program_src)
        .build(&context)
        .map_err(|e| format!("Failed to build OpenCL program: {}", e))?;
    // Create kernel
    let kernel = Kernel::builder()
        .program(&program)
        .name("smith_waterman_align")
        .queue(queue.clone())
        .global_work_size(work_groups * work_group_size)
        .local_work_size(work_group_size)
        .arg(&seq1_buffer)
        .arg(&seq2_buffer)
        .arg(&result_buffer)
        .arg(&(len as u32))
        .build()
        .map_err(|e| format!("Failed to create kernel: {}", e))?;
    // Execute kernel
    unsafe {
        kernel.enq().map_err(|e| format!("Failed to execute kernel: {}", e))?;
    }
    // Wait for completion
    queue.finish().map_err(|e| format!("Failed to wait for kernel completion: {}", e))?;
    // Read result
    let mut result = vec![0i32];
    result_buffer.read(&mut result).enq().map_err(|e| format!("Failed to read result: {}", e))?;
    Ok(result[0])
}

/// Count total bases in a FASTQ file (compressed or uncompressed), streaming and chunked
pub fn count_bases_in_fastq(filepath: &str) -> Result<usize, String> {
    let mut total_bases = 0usize;
    // Use a large chunk size for efficiency
    process_fastq_file_in_chunks(filepath, 100_000, |chunk| {
        total_bases += chunk.iter().map(|seq| seq.len()).sum::<usize>();
        Ok(())
    })?;
    Ok(total_bases)
} 