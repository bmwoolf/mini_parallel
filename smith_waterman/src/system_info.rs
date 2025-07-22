use std::process::Command;

/// System information structure containing hardware capabilities
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub gpu_name: String,
    pub gpu_memory_gb: f64,
    pub cpu_cores: usize,
    pub total_ram_gb: f64,
    pub opencl_platform: String,
    pub opencl_device: String,
}

impl SystemInfo {
    /// Get system information by dynamically detecting hardware
    pub fn new() -> Result<Self, String> {
        let gpu_info = Self::detect_gpu_info()?;
        let cpu_cores = num_cpus::get();
        let total_ram = Self::detect_total_ram()?;
        
        Ok(SystemInfo {
            gpu_name: gpu_info.name,
            gpu_memory_gb: gpu_info.memory_gb,
            cpu_cores,
            total_ram_gb: total_ram,
            opencl_platform: gpu_info.platform,
            opencl_device: gpu_info.device,
        })
    }
    
    /// Detect GPU information using multiple methods
    fn detect_gpu_info() -> Result<GpuInfo, String> {
        // Try NVIDIA-SMI first (most reliable for NVIDIA GPUs)
        if let Ok(gpu_info) = Self::detect_nvidia_gpu() {
            return Ok(gpu_info);
        }
        
        // Try OpenCL detection as fallback
        if let Ok(gpu_info) = Self::detect_opencl_gpu() {
            return Ok(gpu_info);
        }
        
        // Fallback to basic detection
        Ok(Self::detect_basic_gpu())
    }
    
    /// Detect NVIDIA GPU using nvidia-smi
    fn detect_nvidia_gpu() -> Result<GpuInfo, String> {
        let output = Command::new("nvidia-smi")
            .args(&["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
            .output()
            .map_err(|e| format!("Failed to run nvidia-smi: {}", e))?;
        
        if !output.status.success() {
            return Err("nvidia-smi command failed".to_string());
        }
        
        let output_str = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = output_str.lines().collect();
        
        if lines.is_empty() {
            return Err("No GPU information found in nvidia-smi output".to_string());
        }
        
        let first_line = lines[0].trim();
        let parts: Vec<&str> = first_line.split(',').collect();
        
        if parts.len() < 2 {
            return Err("Invalid nvidia-smi output format".to_string());
        }
        
        let name = parts[0].trim().to_string();
        let memory_mb: f64 = parts[1].trim().parse()
            .map_err(|e| format!("Failed to parse GPU memory: {}", e))?;
        let memory_gb = memory_mb / 1024.0;
        
        Ok(GpuInfo {
            name,
            memory_gb,
            platform: "NVIDIA".to_string(),
            device: "GPU".to_string(),
        })
    }
    
    /// Detect GPU using OpenCL
    fn detect_opencl_gpu() -> Result<GpuInfo, String> {
        use ocl;
        
        let platforms = ocl::Platform::list();
        if platforms.is_empty() {
            return Err("No OpenCL platforms found".to_string());
        }
        
        for platform in &platforms {
            let platform_name = platform.name().unwrap_or_else(|_| "Unknown".to_string());
            let devices = match ocl::Device::list(*platform, Some(ocl::flags::DEVICE_TYPE_GPU)) {
                Ok(devs) => devs,
                Err(_) => continue,
            };
            
            for device in devices {
                let name = device.name().unwrap_or_else(|_| "Unknown GPU".to_string());
                // Use a conservative estimate for OpenCL devices
                let memory_gb = 8.0; // Conservative fallback
                
                return Ok(GpuInfo {
                    name,
                    memory_gb,
                    platform: platform_name,
                    device: "OpenCL GPU".to_string(),
                });
            }
        }
        
        Err("No OpenCL GPU devices found".to_string())
    }
    
    /// Basic GPU detection fallback
    fn detect_basic_gpu() -> GpuInfo {
        // Try to detect common GPU patterns from system
        let gpu_name = Self::detect_gpu_name_from_system()
            .unwrap_or_else(|_| "Unknown GPU".to_string());
        
        // Estimate memory based on GPU name patterns
        let memory_gb = Self::estimate_gpu_memory(&gpu_name);
        
        GpuInfo {
            name: gpu_name,
            memory_gb,
            platform: "Unknown".to_string(),
            device: "GPU".to_string(),
        }
    }
    
    /// Try to detect GPU name from system files
    fn detect_gpu_name_from_system() -> Result<String, String> {
        // Try /proc/driver/nvidia/gpus/ (NVIDIA)
        if let Ok(entries) = std::fs::read_dir("/proc/driver/nvidia/gpus/") {
            for entry in entries {
                if let Ok(entry) = entry {
                    if let Ok(name) = std::fs::read_to_string(entry.path().join("name")) {
                        return Ok(name.trim().to_string());
                    }
                }
            }
        }
        
        // Try lspci as fallback
        let output = Command::new("lspci")
            .args(&["-d", "10de:", "-m"]) // NVIDIA vendor ID
            .output();
        
        if let Ok(output) = output {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                for line in output_str.lines() {
                    if line.contains("VGA") || line.contains("3D") {
                        let parts: Vec<&str> = line.split(':').collect();
                        if parts.len() >= 3 {
                            return Ok(parts[2].trim().to_string());
                        }
                    }
                }
            }
        }
        
        Err("Could not detect GPU name".to_string())
    }
    
    /// Estimate GPU memory based on name patterns
    fn estimate_gpu_memory(gpu_name: &str) -> f64 {
        let name_lower = gpu_name.to_lowercase();
        
        // NVIDIA RTX series
        if name_lower.contains("rtx 4090") { return 24.0; }
        if name_lower.contains("rtx 4080") { return 16.0; }
        if name_lower.contains("rtx 4070") { return 12.0; }
        if name_lower.contains("rtx 4060") { return 8.0; }
        
        // NVIDIA GTX series
        if name_lower.contains("gtx 1080 ti") { return 11.0; }
        if name_lower.contains("gtx 1080") { return 8.0; }
        if name_lower.contains("gtx 1070") { return 8.0; }
        if name_lower.contains("gtx 1060") { return 6.0; }
        
        // AMD RX series
        if name_lower.contains("rx 7900 xtx") { return 24.0; }
        if name_lower.contains("rx 7900 xt") { return 20.0; }
        if name_lower.contains("rx 7800 xt") { return 16.0; }
        if name_lower.contains("rx 7700 xt") { return 12.0; }
        
        // Default fallback
        8.0
    }
    
    /// Detect total system RAM
    fn detect_total_ram() -> Result<f64, String> {
        // Try /proc/meminfo first
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<f64>() {
                            return Ok(kb / (1024.0 * 1024.0)); // Convert KB to GB
                        }
                    }
                }
            }
        }
        
        // Try free command as fallback
        let output = Command::new("free")
            .args(&["-g"])
            .output()
            .map_err(|e| format!("Failed to run free command: {}", e))?;
        
        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = output_str.lines().collect();
            if lines.len() >= 2 {
                let parts: Vec<&str> = lines[1].split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(gb) = parts[1].parse::<f64>() {
                        return Ok(gb);
                    }
                }
            }
        }
        
        // Default fallback
        Ok(16.0)
    }
    
    /// Get available GPU memory for computation (80% of total)
    pub fn available_gpu_memory_gb(&self) -> f64 {
        self.gpu_memory_gb * 0.8
    }
    
    /// Get available GPU memory in bytes
    pub fn available_gpu_memory_bytes(&self) -> usize {
        (self.available_gpu_memory_gb() * 1024.0 * 1024.0 * 1024.0) as usize
    }
    
    /// Print system information
    pub fn print_info(&self) {
        println!("=== System Information ===");
        println!("GPU: {} ({} GB VRAM)", self.gpu_name, self.gpu_memory_gb);
        println!("Available GPU Memory: {:.1} GB", self.available_gpu_memory_gb());
        println!("CPU Cores: {}", self.cpu_cores);
        println!("Total RAM: {:.1} GB", self.total_ram_gb);
        println!("OpenCL Platform: {}", self.opencl_platform);
        println!("OpenCL Device: {}", self.opencl_device);
        println!("==========================");
    }
}

/// GPU information structure
#[derive(Debug, Clone)]
struct GpuInfo {
    name: String,
    memory_gb: f64,
    platform: String,
    device: String,
}

/// Global system information singleton
use once_cell::sync::Lazy;

static SYSTEM_INFO: Lazy<Result<SystemInfo, String>> = Lazy::new(|| SystemInfo::new());

/// Get system information (thread-safe singleton)
pub fn get_system_info() -> Result<&'static SystemInfo, String> {
    SYSTEM_INFO.as_ref().map_err(|e| e.clone())
} 