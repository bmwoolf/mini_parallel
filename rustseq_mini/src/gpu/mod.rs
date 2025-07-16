// GPU module for parallel sequence alignment using CUDA
// Optimized for RTX 4070 (Ada Lovelace architecture)

pub mod aligner;
pub mod memory;
pub mod kernel;

// Re-export main GPU alignment function
pub use aligner::gpu_align;

// GPU configuration constants for RTX 4070
pub const GPU_CHUNK_SIZE: usize = 1024 * 1024; // 1MB chunks for GPU
pub const GPU_THREADS_PER_BLOCK: u32 = 256; // Optimal for RTX 4070
pub const GPU_MAX_BLOCKS: u32 = 65535; // Maximum CUDA blocks
pub const RTX_4070_VRAM_GB: f32 = 12.0; // RTX 4070 has 12GB VRAM

// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub name: String,
    pub memory_gb: f32,
    pub compute_capability: (u32, u32),
    pub max_threads_per_block: u32,
    pub cuda_cores: u32,
    pub sm_count: u32,
}

// GPU alignment result
#[derive(Debug, Clone)]
pub struct GpuAlignmentResult {
    pub score: i32,
    pub processing_time_ms: f64,
    pub memory_used_mb: f64,
    pub gpu_device: String,
    pub cuda_blocks: u32,
    pub cuda_threads: u32,
}

// Check if GPU is available using CUDA
pub fn is_gpu_available() -> bool {
    match cuda::init() {
        Ok(_) => {
            // Check if we can get device count
            match cuda::device::count() {
                Ok(count) => count > 0,
                Err(_) => false
            }
        },
        Err(_) => false
    }
}

// Get available GPU devices using CUDA
pub fn get_gpu_devices() -> Vec<GpuDevice> {
    let mut devices = Vec::new();
    
    match cuda::init() {
        Ok(_) => {
            match cuda::device::count() {
                Ok(count) => {
                    for i in 0..count {
                        if let Ok(device) = cuda::device::get(i) {
                            if let Ok(name) = device.name() {
                                if let Ok(compute_cap) = device.compute_capability() {
                                    let device_info = GpuDevice {
                                        name: name.clone(),
                                        memory_gb: if name.contains("RTX 4070") { RTX_4070_VRAM_GB } else { 8.0 },
                                        compute_capability: compute_cap,
                                        max_threads_per_block: 1024, // CUDA standard
                                        cuda_cores: if name.contains("RTX 4070") { 5888 } else { 0 },
                                        sm_count: if name.contains("RTX 4070") { 46 } else { 0 },
                                    };
                                    devices.push(device_info);
                                }
                            }
                        }
                    }
                },
                Err(e) => {
                    eprintln!("Failed to get CUDA device count: {}", e);
                }
            }
        },
        Err(e) => {
            eprintln!("Failed to initialize CUDA: {}", e);
        }
    }
    
    devices
}

// Get RTX 4070 specific device
pub fn get_rtx_4070_device() -> Option<GpuDevice> {
    let devices = get_gpu_devices();
    devices.into_iter().find(|d| d.name.contains("RTX 4070"))
}

// Check if RTX 4070 is available
pub fn is_rtx_4070_available() -> bool {
    get_rtx_4070_device().is_some()
}

// Get optimal configuration for RTX 4070
pub fn get_rtx_4070_config() -> (u32, u32) {
    // RTX 4070 optimal configuration:
    // - 256 threads per block (optimal for Ada Lovelace)
    // - Up to 65535 blocks
    // - 46 SMs available
    (GPU_THREADS_PER_BLOCK, GPU_MAX_BLOCKS)
}

// Print RTX 4070 specifications
pub fn print_rtx_4070_specs() {
    println!("RTX 4070 Specifications:");
    println!("  Architecture: Ada Lovelace");
    println!("  CUDA Cores: 5,888");
    println!("  SMs (Streaming Multiprocessors): 46");
    println!("  VRAM: 12 GB GDDR6X");
    println!("  Memory Bandwidth: 504 GB/s");
    println!("  Compute Capability: 8.9");
    println!("  Max Threads per Block: 1,024");
    println!("  Max Blocks per Grid: 65,535");
    println!("  Optimal Threads per Block: 256");
} 