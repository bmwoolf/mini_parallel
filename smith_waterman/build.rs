use std::env;

fn main() {
    // Check for OpenCL development libraries
    let opencl_paths = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
        "/opt/rocm/lib",
    ];
    
    for path in &opencl_paths {
        if std::path::Path::new(path).exists() {
            println!("cargo:rustc-link-search=native={}", path);
        }
    }
    
    // Link OpenCL library
    println!("cargo:rustc-link-lib=dylib=OpenCL");
    
    // Re-run if OpenCL kernel source changes
    println!("cargo:rerun-if-changed=src/smith_waterman.cl");
} 