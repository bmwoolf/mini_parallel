use std::env;
use std::path::PathBuf;

fn main() {
    // Check if CUDA is available
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-env=CUDA_PATH={}", cuda_path);
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        // Try common CUDA installation paths
        let cuda_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8",
        ];
        
        for path in &cuda_paths {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-env=CUDA_PATH={}", path);
                println!("cargo:rustc-link-search=native={}/lib64", path);
                break;
            }
        }
    }
    
    // Link CUDA libraries
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=nvrtc");
    
    // Compile CUDA kernel if nvcc is available
    if let Ok(nvcc_output) = std::process::Command::new("nvcc")
        .arg("--version")
        .output() 
    {
        if nvcc_output.status.success() {
            println!("cargo:warning=NVCC found, compiling CUDA kernels");
            compile_cuda_kernels();
        } else {
            println!("cargo:warning=NVCC not found, using pre-compiled kernels");
        }
    } else {
        println!("cargo:warning=NVCC not found, using pre-compiled kernels");
    }
    
    // Re-run if CUDA kernel source changes
    println!("cargo:rerun-if-changed=src/gpu/kernels/smith_waterman.cu");
}

fn compile_cuda_kernels() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let kernel_src = "src/gpu/kernels/smith_waterman.cu";
    let kernel_dst = format!("{}/smith_waterman.ptx", out_dir);
    
    if std::path::Path::new(kernel_src).exists() {
        let status = std::process::Command::new("nvcc")
            .args(&[
                "--ptx",
                "-arch=sm_89", // RTX 4070 uses Ada Lovelace architecture (SM 8.9)
                "-O3",
                "-o", &kernel_dst,
                kernel_src,
            ])
            .status();
            
        match status {
            Ok(exit_status) => {
                if exit_status.success() {
                    println!("cargo:warning=CUDA kernel compiled successfully");
                    println!("cargo:rustc-env=CUDA_PTX_PATH={}", kernel_dst);
                } else {
                    println!("cargo:warning=Failed to compile CUDA kernel");
                }
            }
            Err(e) => {
                println!("cargo:warning=Failed to run nvcc: {}", e);
            }
        }
    }
} 