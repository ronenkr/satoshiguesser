use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=cuda/toy_search.cu");

    if env::var_os("CARGO_FEATURE_CUDA_TOY").is_none() {
        return;
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let lib_path = out_dir.join("libtoy_cuda.a");
    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_61".to_owned());

    let status = Command::new("nvcc")
        .arg("-O3")
        .arg(format!("-arch={cuda_arch}"))
        .arg("--lib")
        .arg("cuda/toy_search.cu")
        .arg("-o")
        .arg(&lib_path)
        .status()
        .expect("failed to run nvcc; install the NVIDIA CUDA toolkit or build without --features cuda-toy");

    if !status.success() {
        panic!("nvcc failed while compiling cuda/toy_search.cu");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=toy_cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");

    if let Some(cuda_home) = env::var_os("CUDA_HOME").or_else(|| env::var_os("CUDA_PATH")) {
        let lib64 = PathBuf::from(cuda_home).join("lib64");
        println!("cargo:rustc-link-search=native={}", lib64.display());
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
}
