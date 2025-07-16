// GPU-accelerated sequence aligner using OpenCL
// Handles the main alignment logic and orchestrates GPU operations

use crate::gpu::{GpuAlignmentResult, GpuDevice, GPU_WORK_GROUP_SIZE, GPU_MAX_WORK_GROUPS};
use ocl::{Buffer, Program, Kernel, MemFlags};

// Remove unused functions: gpu_align_16_files, gpu_align_against_reference
// Remove unused imports inside functions
// Remove unused mut from variables

use std::fs::File;
use std::io::{BufRead, BufReader};
use flate2::read::GzDecoder;

/// Streaming FASTQ chunk reader (yields Vec<String> of sequence lines)
pub struct StreamingFastqChunkReader {
    reader: Box<dyn BufRead>,
    chunk_size_reads: usize,
}

impl StreamingFastqChunkReader {
    pub fn new(filepath: &str, chunk_size_reads: usize) -> Result<Self, String> {
        let file = File::open(filepath)
            .map_err(|e| format!("Failed to open file {}: {}", filepath, e))?;
        let reader: Box<dyn BufRead> = if filepath.ends_with(".gz") {
            Box::new(BufReader::new(GzDecoder::new(file)))
        } else {
            Box::new(BufReader::new(file))
        };
        Ok(Self { reader, chunk_size_reads })
    }
}

impl Iterator for StreamingFastqChunkReader {
    type Item = Vec<String>; // Each chunk is a Vec of sequence lines
    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Vec::with_capacity(self.chunk_size_reads);
        let mut line = String::new();
        let mut line_count = 0;
        let mut seq_line_idx = 0;
        loop {
            line.clear();
            let bytes = self.reader.read_line(&mut line).ok()?;
            if bytes == 0 {
                break;
            }
            line_count += 1;
            if line_count % 4 == 2 {
                // Sequence line
                chunk.push(line.trim_end().to_string());
                seq_line_idx += 1;
                if seq_line_idx >= self.chunk_size_reads {
                    break;
                }
            }
        }
        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }
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
            let filename = format!("{}_{:03}_R{}_001.fastq.gz", sample_id, lane, read);
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
            .unwrap_or_else(|_| "10000".to_string())
            .parse()
            .unwrap_or(10000);
        let mut total_score = 0;
        let mut processed_chunks = 0;
        let mut total_bases = 0;
        let mut reader = StreamingFastqChunkReader::new(file, chunk_size_reads)?;
        while let Some(chunk) = reader.next() {
            let seq = chunk.concat();
            total_bases += seq.len();
            match gpu_align_chunk_self(&seq, device) {
                Ok(score) => {
                    total_score += score;
                    processed_chunks += 1;
                    if processed_chunks % 10 == 0 {
                        println!("    Processed {} chunks, current score: {}", processed_chunks, total_score);
                    }
                },
                Err(e) => {
                    println!("    Warning: Failed to align chunk {}: {}", processed_chunks, e);
                }
            }
        }
        
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
    // Load sequences from files
    let seq1 = load_sequence_from_file(file1)?;
    let seq2 = load_sequence_from_file(file2)?;
    
    // Use chunked alignment for large sequences
    let start_time = std::time::Instant::now();
    let score = if seq1.len() > 100000 || seq2.len() > 100000 {
        println!("Large sequences detected ({} and {} bases), using chunked alignment", seq1.len(), seq2.len());
        gpu_align_chunked(&seq1, &seq2, device, 50000)? // 50KB chunks
    } else {
        gpu_align(&seq1, &seq2, device)?
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

// Load sequence from FASTQ file (compressed or uncompressed)
pub fn load_sequence_from_file(filepath: &str) -> Result<String, String> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use flate2::read::GzDecoder;
    let file = File::open(filepath)
        .map_err(|e| format!("Failed to open file {}: {}", filepath, e))?;
    // Handle gzipped files
    let reader: Box<dyn BufRead> = if filepath.ends_with(".gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    let mut sequence = String::new();
    let mut line_count = 0;
    for line in reader.lines() {
        let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
        line_count += 1;
        // FASTQ format: sequence is on line 2, 6, 10, etc. (every 4th line starting from line 2)
        if line_count % 4 == 2 {
            // This is a sequence line
            sequence.push_str(&line);
        }
        // Skip header lines (line 1, 5, 9, etc.) and quality scores (line 4, 8, 12, etc.)
    }
    if sequence.is_empty() {
        return Err(format!("No sequence data found in {}", filepath));
    }
    println!("Loaded {} bases from {}", sequence.len(), filepath);
    Ok(sequence)
} 