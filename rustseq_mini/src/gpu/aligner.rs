// GPU-accelerated sequence aligner using OpenCL
// Handles the main alignment logic and orchestrates GPU operations

use crate::gpu::{GpuAlignmentResult, GpuDevice, GPU_WORK_GROUP_SIZE, GPU_MAX_WORK_GROUPS};
use ocl::{Buffer, Program, Kernel, MemFlags};

// Remove unused functions: gpu_align_16_files, gpu_align_against_reference
// Remove unused imports inside functions
// Remove unused mut from variables

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
        _memory_used_mb: 0.0, // Will be calculated
        gpu_device: device.name.clone(),
        _work_groups: 0, // Will be calculated
        _work_group_size: 0, // Will be calculated
    })
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