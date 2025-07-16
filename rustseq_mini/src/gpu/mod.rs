// GPU module for parallel sequence alignment using OpenCL
// Optimized for RTX 4070 and cross-platform GPU support

use ocl;

pub mod aligner;
// pub mod memory; // Commented out as it's unused
pub mod kernel;

// Restore only the needed constants
pub const GPU_WORK_GROUP_SIZE: usize = 256; // Optimal for RTX 4070
pub const GPU_MAX_WORK_GROUPS: usize = 65535; // Maximum OpenCL work groups

// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub name: String,
    pub memory_gb: f32,
    pub _compute_units: u32, // unused
    pub max_work_group_size: usize,
    pub _opencl_version: String, // unused
    pub _device_type: String,   // unused
}

// GPU alignment result
#[derive(Debug, Clone)]
pub struct GpuAlignmentResult {
    pub score: i32,
    pub processing_time_ms: f64,
    pub _memory_used_mb: f64, // unused
    pub gpu_device: String,
    pub _work_groups: usize, // unused
    pub _work_group_size: usize, // unused
}

// Check if GPU is available using OpenCL
pub fn is_gpu_available() -> bool {
    let platforms = ocl::Platform::list();
    for platform in platforms {
        let devices = match ocl::Device::list(platform, Some(ocl::flags::DEVICE_TYPE_GPU)) {
            Ok(devs) => devs,
            Err(_) => continue,
        };
        if !devices.is_empty() {
            return true;
        }
    }
    false
}

// Get available GPU devices using OpenCL
pub fn get_gpu_devices() -> Vec<GpuDevice> {
    let mut devices_out = Vec::new();
    let platforms = ocl::Platform::list();
    for platform in platforms {
        let devices = match ocl::Device::list(platform, Some(ocl::flags::DEVICE_TYPE_GPU)) {
            Ok(devs) => devs,
            Err(_) => continue,
        };
        for device in devices {
            let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
            let memory_gb = 12.0; // RTX 4070 has 12GB VRAM
            let _compute_units = 46; // RTX 4070 has 46 compute units
            let max_work_group_size = device.max_wg_size().unwrap_or(1024);
            let _opencl_version = "OpenCL 3.0".to_string();
            let _device_type = "GPU".to_string();
            devices_out.push(GpuDevice {
                name,
                memory_gb,
                _compute_units,
                max_work_group_size,
                _opencl_version,
                _device_type,
            });
        }
    }
    devices_out
}

// Initialize OpenCL context and queue
pub fn init_opencl() -> Result<(ocl::Context, ocl::Queue), String> {
    let platforms = ocl::Platform::list();
    if platforms.is_empty() {
        return Err("No OpenCL platforms found".to_string());
    }
    let platform = platforms[0];
    let devices = match ocl::Device::list(platform, Some(ocl::flags::DEVICE_TYPE_GPU)) {
        Ok(devs) => devs,
        Err(_) => return Err("No OpenCL GPU devices found".to_string()),
    };
    if devices.is_empty() {
        return Err("No OpenCL GPU devices found".to_string());
    }
    let device = devices[0];
    let context = ocl::Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let queue = ocl::Queue::new(&context, device, None)?;
    Ok((context, queue))
} 