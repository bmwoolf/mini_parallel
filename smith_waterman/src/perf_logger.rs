use std::process::{Child, Command, Stdio};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

static MONITORS_RUNNING: AtomicBool = AtomicBool::new(false);
static RUN_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub struct SystemMonitors {
    gpu_monitor: Option<Child>,
    disk_monitor: Option<Child>,
    mem_cpu_monitor: Option<Child>,
    context_switch_monitor: Option<Child>,
    perf_record: Option<Child>,
    run_number: u64,
    run_id: String,
    logs_dir: String,
}

impl SystemMonitors {
    pub fn new() -> Self {
        let run_number = RUN_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let run_id = format!("run_{}", timestamp);
        let logs_dir = format!("logs/run_{}", run_number);
        
        Self {
            gpu_monitor: None,
            disk_monitor: None,
            mem_cpu_monitor: None,
            context_switch_monitor: None,
            perf_record: None,
            run_number,
            run_id,
            logs_dir,
        }
    }

    pub fn start(&mut self) -> Result<(), String> {
        if MONITORS_RUNNING.load(Ordering::SeqCst) {
            return Err("System monitors already running".to_string());
        }

        // Create logs directory
        fs::create_dir_all(&self.logs_dir)
            .map_err(|e| format!("Failed to create logs directory: {}", e))?;

        println!("Starting system monitors for run #{}: {}", self.run_number, self.run_id);
        println!("Logs directory: {}", self.logs_dir);

        // Start GPU utilization monitor
        self.start_gpu_monitor()?;

        // Start disk I/O monitor
        self.start_disk_monitor()?;

        // Start memory/CPU monitor
        self.start_mem_cpu_monitor()?;

        // Start context switch monitor (optional)
        self.start_context_switch_monitor()?;

        // Start perf record (optional)
        self.start_perf_record()?;

        MONITORS_RUNNING.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn start_gpu_monitor(&mut self) -> Result<(), String> {
        let log_file = format!("{}/gpu_util.log", self.logs_dir);
        
        let child = Command::new("nvidia-smi")
            .args(&["dmon", "-s", "u", "-o", "DT", "-f", &log_file])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start GPU monitor: {}", e))?;

        self.gpu_monitor = Some(child);
        println!("  GPU monitor started: {}", log_file);
        Ok(())
    }

    fn start_disk_monitor(&mut self) -> Result<(), String> {
        let log_file = format!("{}/disk_io.log", self.logs_dir);
        
        let output_file = fs::File::create(&log_file)
            .map_err(|e| format!("Failed to create disk I/O log file: {}", e))?;

        let child = Command::new("iostat")
            .args(&["-dx", "1"])
            .stdout(Stdio::from(output_file))
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start disk I/O monitor: {}", e))?;

        self.disk_monitor = Some(child);
        println!("  Disk I/O monitor started: {}", log_file);
        Ok(())
    }

    fn start_mem_cpu_monitor(&mut self) -> Result<(), String> {
        let log_file = format!("{}/mem_cpu.log", self.logs_dir);
        
        let output_file = fs::File::create(&log_file)
            .map_err(|e| format!("Failed to create memory/CPU log file: {}", e))?;

        let child = Command::new("vmstat")
            .args(&["1"])
            .stdout(Stdio::from(output_file))
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start memory/CPU monitor: {}", e))?;

        self.mem_cpu_monitor = Some(child);
        println!("  Memory/CPU monitor started: {}", log_file);
        Ok(())
    }

    fn start_context_switch_monitor(&mut self) -> Result<(), String> {
        let log_file = format!("{}/context_switch.log", self.logs_dir);
        
        let output_file = fs::File::create(&log_file)
            .map_err(|e| format!("Failed to create context switch log file: {}", e))?;

        let child = Command::new("pidstat")
            .args(&["-w", "1"])
            .stdout(Stdio::from(output_file))
            .stderr(Stdio::null())
            .spawn();

        match child {
            Ok(child) => {
                self.context_switch_monitor = Some(child);
                println!("  Context switch monitor started: {}", log_file);
            }
            Err(_) => {
                println!("  Context switch monitor not available (pidstat not found)");
            }
        }
        Ok(())
    }

    fn start_perf_record(&mut self) -> Result<(), String> {
        let perf_data_path = format!("{}/perf.data", self.logs_dir);
        
        let child = Command::new("perf")
            .args(&["record", "-g", "-o", &perf_data_path, "-p", &std::process::id().to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();

        match child {
            Ok(child) => {
                self.perf_record = Some(child);
                println!("  Perf record started: {}", perf_data_path);
            }
            Err(_) => {
                println!("  Perf record not available (perf not found or insufficient permissions)");
            }
        }
        Ok(())
    }

    pub fn stop(&mut self) -> Result<(), String> {
        if !MONITORS_RUNNING.load(Ordering::SeqCst) {
            return Ok(());
        }

        println!("Stopping system monitors...");

        // Stop all monitors
        if let Some(child) = self.gpu_monitor.take() {
            self.stop_monitor(&mut Some(child), "GPU");
        }
        if let Some(child) = self.disk_monitor.take() {
            self.stop_monitor(&mut Some(child), "Disk I/O");
        }
        if let Some(child) = self.mem_cpu_monitor.take() {
            self.stop_monitor(&mut Some(child), "Memory/CPU");
        }
        if let Some(child) = self.context_switch_monitor.take() {
            self.stop_monitor(&mut Some(child), "Context Switch");
        }
        if let Some(child) = self.perf_record.take() {
            self.stop_monitor(&mut Some(child), "Perf Record");
        }

        MONITORS_RUNNING.store(false, Ordering::SeqCst);
        println!("System monitors stopped. Logs saved to: {}", self.logs_dir);
        Ok(())
    }

    fn stop_monitor(&mut self, monitor: &mut Option<Child>, name: &str) {
        if let Some(mut child) = monitor.take() {
            match child.kill() {
                Ok(_) => println!("  {} monitor stopped", name),
                Err(e) => println!("  Warning: Failed to stop {} monitor: {}", name, e),
            }
        }
    }

    pub fn get_run_number(&self) -> u64 {
        self.run_number
    }

    pub fn get_run_id(&self) -> &str {
        &self.run_id
    }

    pub fn get_logs_dir(&self) -> &str {
        &self.logs_dir
    }
}

impl Drop for SystemMonitors {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

// Global system monitors instance
use std::sync::Mutex;
use once_cell::sync::Lazy;

static SYSTEM_MONITORS: Lazy<Mutex<Option<SystemMonitors>>> = Lazy::new(|| Mutex::new(None));

pub fn start_system_monitors() -> Result<(), String> {
    let mut monitors = SystemMonitors::new();
    monitors.start()?;
    
    if let Ok(mut global_monitors) = SYSTEM_MONITORS.lock() {
        *global_monitors = Some(monitors);
    }
    Ok(())
}

pub fn stop_system_monitors() -> Result<(), String> {
    if let Ok(mut global_monitors) = SYSTEM_MONITORS.lock() {
        if let Some(mut monitors) = global_monitors.take() {
            monitors.stop()?;
        }
    }
    Ok(())
}

pub fn get_current_run_number() -> Option<u64> {
    if let Ok(global_monitors) = SYSTEM_MONITORS.lock() {
        if let Some(monitors) = global_monitors.as_ref() {
            return Some(monitors.get_run_number());
        }
    }
    None
}

pub fn get_current_run_id() -> Option<String> {
    if let Ok(global_monitors) = SYSTEM_MONITORS.lock() {
        if let Some(monitors) = global_monitors.as_ref() {
            return Some(monitors.get_run_id().to_string());
        }
    }
    None
}

// Signal handler setup
pub fn setup_signal_handlers() {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    
    ctrlc::set_handler(move || {
        println!("\nReceived interrupt signal, stopping monitors...");
        r.store(false, Ordering::SeqCst);
        let _ = stop_system_monitors();
        std::process::exit(0);
    }).expect("Error setting Ctrl-C handler");
} 