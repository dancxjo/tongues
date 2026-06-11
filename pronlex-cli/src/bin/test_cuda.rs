use burn_cuda::{CudaDevice, Cuda};
use burn::tensor::Tensor;
use std::panic;

fn is_cuda_available() -> bool {
    // Suppress panic messages during the check
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {
        // Do nothing to suppress print
    }));
    
    let result = panic::catch_unwind(|| {
        let device = CudaDevice::default();
        type B = Cuda<f32, i32>;
        // Perform a small computation to ensure CUDA is functional
        let tensor = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let _sum = tensor.sum();
    });
    
    // Restore the default panic hook
    panic::set_hook(default_hook);
    
    result.is_ok()
}

fn main() {
    println!("Checking CUDA availability...");
    if is_cuda_available() {
        println!("CUDA is available and working!");
    } else {
        println!("CUDA is NOT available, falling back to CPU.");
    }
}
