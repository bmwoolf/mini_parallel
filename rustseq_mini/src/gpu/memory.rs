// GPU memory management for sequence alignment using CUDA
// Optimized for RTX 4070

use std::ptr;
use cuda::memory::{DeviceBuffer, UnifiedBuffer};
use cuda::prelude::*;

// GPU memory handle for CUDA
pub struct GpuMemory {
    pub seq1_buffer: Option<DeviceBuffer<u8>>,
    pub seq2_buffer: Option<DeviceBuffer<u8>>,
    pub result_buffer: Option<UnifiedBuffer<i32>>,
    pub seq1_size: usize,
    pub seq2_size: usize,
    pub allocated: bool,
}

impl GpuMemory {
    pub fn new() -> Self {
        GpuMemory {
            seq1_buffer: None,
            seq2_buffer: None,
            result_buffer: None,
            seq1_size: 0,
            seq2_size: 0,
            allocated: false,
        }
    }
}

// Initialize CUDA context
pub fn init_cuda() -> Result<(), String> {
    match cuda::init() {
        Ok(_) => {
            println!("CUDA initialized successfully");
            Ok(())
        },
        Err(e) => Err(format!("Failed to initialize CUDA: {}", e))
    }
}

// Allocate GPU memory for sequence alignment using CUDA
pub fn allocate_gpu_memory(sequence_length: usize) -> Result<GpuMemory, String> {
    let mut gpu_mem = GpuMemory::new();
    
    // Calculate memory requirements
    let seq1_size = sequence_length;
    let seq2_size = sequence_length;
    
    println!("Allocating CUDA memory for RTX 4070:");
    println!("  Sequence 1: {} bytes", seq1_size);
    println!("  Sequence 2: {} bytes", seq2_size);
    println!("  Result: {} bytes", std::mem::size_of::<i32>());
    println!("  Total: {} bytes ({:.2} MB)", 
             seq1_size + seq2_size + std::mem::size_of::<i32>(),
             (seq1_size + seq2_size + std::mem::size_of::<i32>()) as f64 / (1024.0 * 1024.0));
    
    // Initialize CUDA if not already done
    init_cuda()?;
    
    // Allocate device memory for sequences
    match DeviceBuffer::new(seq1_size) {
        Ok(seq1_buffer) => {
            gpu_mem.seq1_buffer = Some(seq1_buffer);
        },
        Err(e) => return Err(format!("Failed to allocate seq1 buffer: {}", e))
    }
    
    match DeviceBuffer::new(seq2_size) {
        Ok(seq2_buffer) => {
            gpu_mem.seq2_buffer = Some(seq2_buffer);
        },
        Err(e) => return Err(format!("Failed to allocate seq2 buffer: {}", e))
    }
    
    // Allocate unified memory for result (accessible from both CPU and GPU)
    match UnifiedBuffer::new(1) {
        Ok(result_buffer) => {
            gpu_mem.result_buffer = Some(result_buffer);
        },
        Err(e) => return Err(format!("Failed to allocate result buffer: {}", e))
    }
    
    gpu_mem.seq1_size = seq1_size;
    gpu_mem.seq2_size = seq2_size;
    gpu_mem.allocated = true;
    
    println!("CUDA memory allocated successfully");
    Ok(gpu_mem)
}

// Copy sequence data to GPU memory using CUDA
pub fn copy_to_gpu(gpu_mem: &GpuMemory, seq1: &[u8], seq2: &[u8]) -> Result<(), String> {
    if !gpu_mem.allocated {
        return Err("GPU memory not allocated".to_string());
    }
    
    let len = seq1.len().min(seq2.len());
    
    println!("Copying {} bytes to CUDA device memory", len * 2);
    
    // Copy sequence 1 to device
    if let Some(ref seq1_buffer) = gpu_mem.seq1_buffer {
        match seq1_buffer.copy_from(seq1) {
            Ok(_) => println!("Sequence 1 copied to GPU"),
            Err(e) => return Err(format!("Failed to copy seq1 to GPU: {}", e))
        }
    }
    
    // Copy sequence 2 to device
    if let Some(ref seq2_buffer) = gpu_mem.seq2_buffer {
        match seq2_buffer.copy_from(seq2) {
            Ok(_) => println!("Sequence 2 copied to GPU"),
            Err(e) => return Err(format!("Failed to copy seq2 to GPU: {}", e))
        }
    }
    
    // Initialize result buffer to 0
    if let Some(ref result_buffer) = gpu_mem.result_buffer {
        let zero: i32 = 0;
        match result_buffer.copy_from(&[zero]) {
            Ok(_) => println!("Result buffer initialized"),
            Err(e) => return Err(format!("Failed to initialize result buffer: {}", e))
        }
    }
    
    println!("Data copied to CUDA device successfully");
    Ok(())
}

// Copy result from GPU memory to CPU using CUDA
pub fn copy_from_gpu(gpu_mem: &GpuMemory) -> Result<i32, String> {
    if !gpu_mem.allocated {
        return Err("GPU memory not allocated".to_string());
    }
    
    let mut result: i32 = 0;
    
    if let Some(ref result_buffer) = gpu_mem.result_buffer {
        match result_buffer.copy_to(&mut [result]) {
            Ok(_) => println!("Result copied from GPU"),
            Err(e) => return Err(format!("Failed to copy result from GPU: {}", e))
        }
    }
    
    Ok(result)
}

// Free GPU memory using CUDA
pub fn free_gpu_memory(mut gpu_mem: GpuMemory) -> Result<(), String> {
    if !gpu_mem.allocated {
        return Ok(());
    }
    
    // Drop CUDA buffers (they will be freed automatically)
    gpu_mem.seq1_buffer = None;
    gpu_mem.seq2_buffer = None;
    gpu_mem.result_buffer = None;
    
    gpu_mem.allocated = false;
    gpu_mem.seq1_size = 0;
    gpu_mem.seq2_size = 0;
    
    println!("CUDA memory freed successfully");
    Ok(())
}

// Get GPU memory usage statistics for RTX 4070
pub fn get_gpu_memory_stats() -> Result<(f64, f64), String> {
    // RTX 4070 has 12GB VRAM
    let total_vram_gb = 12.0;
    let total_vram_mb = total_vram_gb * 1024.0;
    
    // This would query actual GPU memory usage via CUDA
    // For now, return estimated values
    let used_mb = 0.0; // Would be queried from CUDA
    let total_mb = total_vram_mb;
    
    Ok((used_mb, total_mb))
}

// Check if GPU has enough memory for given sequence length
pub fn check_gpu_memory_availability(sequence_length: usize) -> Result<bool, String> {
    let required_mb = (sequence_length * 2 + std::mem::size_of::<i32>()) as f64 / (1024.0 * 1024.0);
    
    match get_gpu_memory_stats() {
        Ok((used_mb, total_mb)) => {
            let available_mb = total_mb - used_mb;
            let has_enough = available_mb >= required_mb;
            
            println!("RTX 4070 Memory Check:");
            println!("  Required: {:.2} MB", required_mb);
            println!("  Available: {:.2} MB", available_mb);
            println!("  Total VRAM: {:.2} MB", total_mb);
            println!("  Sufficient: {}", has_enough);
            
            Ok(has_enough)
        },
        Err(e) => Err(format!("Failed to check GPU memory: {}", e))
    }
}

// Get optimal CUDA grid configuration for RTX 4070
pub fn get_optimal_cuda_config(sequence_length: usize) -> (u32, u32) {
    // RTX 4070 specs:
    // - 5888 CUDA cores
    // - 46 SMs (Streaming Multiprocessors)
    // - 256 threads per block optimal
    // - Max 1024 threads per block
    
    let threads_per_block = 256; // Optimal for RTX 4070
    let blocks_needed = ((sequence_length + threads_per_block as usize - 1) / threads_per_block as usize) as u32;
    let max_blocks = 65535; // CUDA limit
    let blocks = blocks_needed.min(max_blocks);
    
    println!("RTX 4070 CUDA Configuration:");
    println!("  Threads per block: {}", threads_per_block);
    println!("  Blocks needed: {}", blocks_needed);
    println!("  Blocks allocated: {}", blocks);
    println!("  Total threads: {}", blocks * threads_per_block);
    
    (blocks, threads_per_block)
} 