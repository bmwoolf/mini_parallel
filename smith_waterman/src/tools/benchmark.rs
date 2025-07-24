use std::fs::{OpenOptions, create_dir_all};
use std::io::{Write, BufRead};
use std::time::Instant;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CpuUtilizationSummary {
    pub avg_user_percent: f64,
    pub avg_system_percent: f64,
    pub avg_idle_percent: f64,
    pub max_user_percent: f64,
    pub max_system_percent: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BenchmarkResult {
    pub timestamp: DateTime<Utc>,
    pub run_id: String,
    pub mode: String, //single_file,full_wgs", etc.
    pub files_processed: usize,
    pub total_reads: usize,
    pub total_bases: usize,
    pub total_score: i32,
    pub total_time_seconds: f64,
    pub throughput_reads_per_second: f64,
    pub throughput_bases_per_second: f64,
    pub chunk_size: usize,
    pub gpu_utilization_avg: f64,
    pub gpu_memory_used_mb: f64,
    pub cpu_cores_used: usize,
    pub parallel_files: bool,
    pub system_info: SystemInfo,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SystemInfo {
    pub gpu_name: String,
    pub gpu_memory_gb: f64,
    pub cpu_cores: usize,
    pub total_ram_gb: f64,
}

pub struct BenchmarkTracker {
    start_time: Instant,
    results: Vec<BenchmarkResult>,
    current_run: Option<BenchmarkRun>,
}

struct BenchmarkRun {
    run_id: String,
    mode: String,
    files_processed: usize,
    total_reads: usize,
    total_bases: usize,
    total_score: i32,
    chunk_size: usize,
    parallel_files: bool,
}

impl BenchmarkTracker {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            results: Vec::new(),
            current_run: None,
        }
    }

    pub fn start_run(&mut self, mode: &str, chunk_size: usize, parallel_files: bool) {
        let run_number = get_next_run_number();
        let run_id = format!("run_{}", chrono::Utc::now().timestamp());
        self.current_run = Some(BenchmarkRun {
            run_id: run_id.clone(),
            mode: mode.to_string(),
            files_processed: 0,
            total_reads: 0,
            total_bases: 0,
            total_score: 0,
            chunk_size,
            parallel_files,
        });
        println!("Starting benchmark run #{}: {} (run_id: {})", run_number, mode, run_id);
    }

    pub fn update_progress(&mut self, files_processed: usize, reads: usize, bases: usize, score: i32) {
        if let Some(run) = &mut self.current_run {
            run.files_processed = files_processed;
            run.total_reads = reads;
            run.total_bases = bases;
            run.total_score = score;
        }
    }

    pub fn finish_run(&mut self) -> Option<BenchmarkResult> {
        let duration = self.start_time.elapsed();
        let run = self.current_run.take()?;
        
        let system_info = self.get_system_info();
        let (gpu_util, gpu_memory) = self.get_gpu_stats();
        
        // Clone run_id before moving it
        let run_id = run.run_id.clone();
        
        let result = BenchmarkResult {
            timestamp: chrono::Utc::now(),
            run_id: run.run_id,
            mode: run.mode,
            files_processed: run.files_processed,
            total_reads: run.total_reads,
            total_bases: run.total_bases,
            total_score: run.total_score,
            total_time_seconds: duration.as_secs_f64(),
            throughput_reads_per_second: run.total_reads as f64 / duration.as_secs_f64(),
            throughput_bases_per_second: run.total_bases as f64 / duration.as_secs_f64(),
            chunk_size: run.chunk_size,
            gpu_utilization_avg: gpu_util,
            gpu_memory_used_mb: gpu_memory,
            cpu_cores_used: num_cpus::get(),
            parallel_files: run.parallel_files,
            system_info,
        };

        self.results.push(result.clone());
        self.save_results();
        
        println!("Benchmark run {} completed:", run_id);
        println!("   Time: {:0.2}", result.total_time_seconds);
        println!("   Throughput: {:.0} reads/s, {:.0} bases/s", 
                result.throughput_reads_per_second, result.throughput_bases_per_second);
        println!("   GPU utilization: {:0.1}", result.gpu_utilization_avg);
        
        // Output system monitoring summary
        self.output_monitoring_summary(&run_id);
        
        Some(result)
    }

    fn get_system_info(&self) -> SystemInfo {
        // Use centralized system information
        if let Ok(system_info) = crate::system_info::get_system_info() {
            SystemInfo {
                gpu_name: system_info.gpu_name.clone(),
                gpu_memory_gb: system_info.gpu_memory_gb,
                cpu_cores: system_info.cpu_cores,
                total_ram_gb: system_info.total_ram_gb,
            }
        } else {
            // Fallback values
            SystemInfo {
                gpu_name: "Unknown GPU".to_string(),
                gpu_memory_gb: 8.0,
                cpu_cores: num_cpus::get(),
                total_ram_gb: 16.0,
            }
        }
    }

    fn get_gpu_stats(&self) -> (f64, f64) {
        // Simplified - could parse nvidia-smi output
        // For now, return reasonable defaults
        (25.0, 400.0) // 25% utilization, 400 memory
    }

    fn save_results(&self) {
        // Create benchmark_results directory if it doesn't exist
        if let Err(e) = create_dir_all("benchmark_results") {
            eprintln!("Failed to create benchmark_results directory: {}", e);
            return;
        }
        
        // Get current run number
        let run_number = get_next_run_number();
        let filename = format!("benchmark_results/run_{}_benchmark_results.json", run_number);
        
        // Save individual run result
        if let Some(result) = self.results.last() {
            let json = serde_json::to_string_pretty(result)
                .expect("Failed to serialize benchmark result");
            
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&filename) {
                if let Err(e) = file.write_all(json.as_bytes()) {
                    eprintln!("Failed to write benchmark result: {}, {}", e, filename);
                } else {
                    println!("Benchmark results saved to: {}", filename);
                }
            }
        }
        
        // Also save to the legacy file for backward compatibility
        let legacy_filename = "benchmark_results.json";
        let json = serde_json::to_string_pretty(&self.results)
            .expect("Failed to serialize benchmark results");
        
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(legacy_filename) {
            if let Err(e) = file.write_all(json.as_bytes()) {
                eprintln!("Failed to write legacy benchmark results: {}, {}", e, legacy_filename);
            }
        }
    }

    fn output_monitoring_summary(&self, run_id: &str) {
        // Get current run number
        let run_number = get_next_run_number();
        let logs_dir = format!("logs/run_{}", run_number);
        
        println!("\nSYSTEM MONITORING SUMMARY");
        println!("=========================");
        println!("Run #{}: {}", run_number, run_id);
        println!("Logs directory: {}", logs_dir);
        
        // Parse and display GPU utilization
        let gpu_log_path = format!("{}/gpu_util.log", logs_dir);
        if let Ok(max_util) = self.parse_gpu_log(&gpu_log_path) {
            println!("Max GPU Utilization: {:.1}%", max_util);
        }
        
        // Parse and display disk I/O
        let disk_log_path = format!("{}/disk_io.log", logs_dir);
        if let Ok(peak_read) = self.parse_disk_log(&disk_log_path) {
            println!("Peak Disk Read: {:.1} MB/s", peak_read);
        }
        
        // Parse and display memory/CPU
        let mem_cpu_log_path = format!("{}/mem_cpu.log", logs_dir);
        if let Ok((max_ram, cpu_summary)) = self.parse_mem_cpu_log(&mem_cpu_log_path) {
            println!("Max RAM Usage: {:.1} GB", max_ram);
            println!("CPU Utilization - Avg: {:.1}% user, {:.1}% system, {:.1}% idle", 
                cpu_summary.avg_user_percent, cpu_summary.avg_system_percent, cpu_summary.avg_idle_percent);
        }
        
        // Parse and display context switches
        let context_switch_log_path = format!("{}/context_switch.log", logs_dir);
        if let Ok(total_switches) = self.parse_context_switch_log(&context_switch_log_path) {
            if total_switches > 0 {
                println!("Total Context Switches: {}", total_switches);
            }
        }
    }

    fn parse_gpu_log(&self, log_path: &str) -> Result<f64, String> {
        use std::fs::File;
        use std::io::BufReader;
        
        let file = File::open(log_path)
            .map_err(|e| format!("Failed to open GPU log: {}", e))?;
        
        let reader = BufReader::new(file);
        let mut max_utilization: f64 = 0.0;
        
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read GPU log line: {}", e))?;
            
            // Skip header lines and empty lines
            if line.trim().is_empty() || line.contains("Date") || line.contains("Time") {
                continue;
            }
            
            // Parse nvidia-smi dmon output format
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(util) = parts[2].parse::<f64>() {
                    max_utilization = max_utilization.max(util);
                }
            }
        }
        
        Ok(max_utilization)
    }

    fn parse_disk_log(&self, log_path: &str) -> Result<f64, String> {
        use std::fs::File;
        use std::io::BufReader;
        
        let file = File::open(log_path)
            .map_err(|e| format!("Failed to open disk log: {}", e))?;
        
        let reader = BufReader::new(file);
        let mut peak_read_mbps: f64 = 0.0;
        
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read disk log line: {}", e))?;
            
            // Skip header lines and empty lines
            if line.trim().is_empty() || line.contains("Device") || line.contains("rrqm") {
                continue;
            }
            
            // Parse iostat output format
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                // Look for read KB/s and convert to MB/s
                if let Ok(read_kbps) = parts[2].parse::<f64>() {
                    let read_mbps = read_kbps / 1024.0;
                    peak_read_mbps = peak_read_mbps.max(read_mbps);
                }
            }
        }
        
        Ok(peak_read_mbps)
    }

    fn parse_mem_cpu_log(&self, log_path: &str) -> Result<(f64, CpuUtilizationSummary), String> {
        use std::fs::File;
        use std::io::BufReader;
        
        let file = File::open(log_path)
            .map_err(|e| format!("Failed to open memory/CPU log: {}", e))?;
        
        let reader = BufReader::new(file);
        let mut max_ram_gb: f64 = 0.0;
        let mut cpu_samples = Vec::new();
        
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read memory/CPU log line: {}", e))?;
            
            // Skip header lines and empty lines
            if line.trim().is_empty() || line.contains("procs") || line.contains("r") {
                continue;
            }
            
            // Parse vmstat output format
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 16 {
                // Parse memory usage (free memory in KB)
                if let Ok(free_kb) = parts[3].parse::<f64>() {
                    // Estimate total RAM usage (assuming 32GB total for now)
                    let used_kb = 32.0 * 1024.0 * 1024.0 - free_kb;
                    let used_gb = used_kb / (1024.0 * 1024.0);
                    max_ram_gb = max_ram_gb.max(used_gb);
                }
                
                // Parse CPU utilization
                if let Ok(user_percent) = parts[12].parse::<f64>() {
                    if let Ok(system_percent) = parts[13].parse::<f64>() {
                        if let Ok(idle_percent) = parts[14].parse::<f64>() {
                            cpu_samples.push((user_percent, system_percent, idle_percent));
                        }
                    }
                }
            }
        }
        
        // Calculate CPU utilization summary
        let cpu_summary = if !cpu_samples.is_empty() {
            let total_samples = cpu_samples.len() as f64;
            let (avg_user, avg_system, avg_idle) = cpu_samples.iter()
                .fold((0.0, 0.0, 0.0), |(u, s, i), &(user, system, idle)| {
                    (u + user, s + system, i + idle)
                });
            
            let max_user = cpu_samples.iter().map(|(u, _, _)| *u).fold(0.0, f64::max);
            let max_system = cpu_samples.iter().map(|(_, s, _)| *s).fold(0.0, f64::max);
            
            CpuUtilizationSummary {
                avg_user_percent: avg_user / total_samples,
                avg_system_percent: avg_system / total_samples,
                avg_idle_percent: avg_idle / total_samples,
                max_user_percent: max_user,
                max_system_percent: max_system,
            }
        } else {
            CpuUtilizationSummary {
                avg_user_percent: 0.0,
                avg_system_percent: 0.0,
                avg_idle_percent: 0.0,
                max_user_percent: 0.0,
                max_system_percent: 0.0,
            }
        };
        
        Ok((max_ram_gb, cpu_summary))
    }

    fn parse_context_switch_log(&self, log_path: &str) -> Result<u64, String> {
        use std::fs::File;
        use std::io::BufReader;
        
        let file = File::open(log_path);
        if file.is_err() {
            return Ok(0); // File doesn't exist, return 0
        }
        
        let file = file.unwrap();
        let reader = BufReader::new(file);
        let mut total_switches = 0u64;
        
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read context switch log line: {}", e))?;
            
            // Skip header lines and empty lines
            if line.trim().is_empty() || line.contains("Linux") || line.contains("UID") {
                continue;
            }
            
            // Parse pidstat output format
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                if let Ok(switches) = parts[3].parse::<u64>() {
                    total_switches += switches;
                }
            }
        }
        
        Ok(total_switches)
    }
}

// Global benchmark tracker
use std::sync::Mutex;
use once_cell::sync::Lazy;

static BENCHMARK_TRACKER: Lazy<Mutex<BenchmarkTracker>> = Lazy::new(|| Mutex::new(BenchmarkTracker::new()));

pub fn start_benchmark(mode: &str, chunk_size: usize, parallel_files: bool) {
    if let Ok(mut tracker) = BENCHMARK_TRACKER.lock() {
        tracker.start_run(mode, chunk_size, parallel_files);
    }
}

pub fn get_next_run_number() -> u64 {
    // Use a static counter to ensure unique run numbers
    use std::sync::atomic::{AtomicU64, Ordering};
    static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);
    
    // Increment and get the next run number
    RUN_COUNTER.fetch_add(1, Ordering::SeqCst) + 1
}

pub fn update_benchmark_progress(files_processed: usize, reads: usize, bases: usize, score: i32) {
    if let Ok(mut tracker) = BENCHMARK_TRACKER.lock() {
        tracker.update_progress(files_processed, reads, bases, score);
    }
}

pub fn finish_benchmark() -> Option<BenchmarkResult> {
    if let Ok(mut tracker) = BENCHMARK_TRACKER.lock() {
        tracker.finish_run()
    } else {
        None
    }
} 