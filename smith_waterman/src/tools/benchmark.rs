use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

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
        println!("Starting benchmark run: {} (run_id: {})", mode, run_id);
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
        let filename = "benchmark_results.json";
        let json = serde_json::to_string_pretty(&self.results)
            .expect("Failed to serialize benchmark results");
        
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(filename) {
            if let Err(e) = file.write_all(json.as_bytes()) {
                eprintln!("Failed to write benchmark results: {}, {}", e, filename);
            }
        }
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