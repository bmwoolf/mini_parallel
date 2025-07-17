// GPU module for parallel sequence alignment using OpenCL
// Optimized for RTX 4070 and cross-platform GPU support

use ocl;
use once_cell::sync::Lazy;
use std::sync::Mutex;

pub mod aligner;

// Restore only the needed constants
pub const GPU_WORK_GROUP_SIZE: usize = 256; // Optimal for RTX 4070
pub const GPU_MAX_WORK_GROUPS: usize = 65535; // Maximum OpenCL work groups

// Global OpenCL context manager to prevent resource exhaustion
static OPENCL_CONTEXT: Lazy<Mutex<Option<(ocl::Context, ocl::Queue, ocl::Device)>>> = 
    Lazy::new(|| Mutex::new(None));

// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub name: String,
    pub memory_gb: f32,
    pub max_work_group_size: usize,
}

// GPU alignment result
#[derive(Debug, Clone)]
pub struct GpuAlignmentResult {
    pub score: i32,
    pub processing_time_ms: f64,
    pub gpu_device: String,
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
            let max_work_group_size = device.max_wg_size().unwrap_or(1024);
            devices_out.push(GpuDevice {
                name,
                memory_gb,
                max_work_group_size,
            });
        }
    }
    devices_out
}

// Get or create OpenCL context and queue (thread-safe singleton)
pub fn get_opencl_context() -> Result<(ocl::Context, ocl::Queue, ocl::Device), String> {
    let mut context_guard = OPENCL_CONTEXT.lock().map_err(|e| format!("Failed to acquire context lock: {}", e))?;
    
    if let Some((context, queue, device)) = context_guard.as_ref() {
        // Return clones of existing context
        Ok((context.clone(), queue.clone(), device.clone()))
    } else {
        // Initialize new context
        let (context, queue, device) = init_opencl()?;
        *context_guard = Some((context.clone(), queue.clone(), device.clone()));
        Ok((context, queue, device))
    }
}

// Initialize OpenCL context and queue
pub fn init_opencl() -> Result<(ocl::Context, ocl::Queue, ocl::Device), String> {
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
    Ok((context, queue, device))
} 