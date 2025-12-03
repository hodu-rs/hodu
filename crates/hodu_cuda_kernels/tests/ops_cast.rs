use hodu_cuda_kernels::{kernel::Kernels, kernels::*};

fn device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn kernels() -> Kernels {
    Kernels::new()
}

fn approx_f64(v: Vec<f64>, digits: i32) -> Vec<f64> {
    let b = 10f64.powi(digits);
    v.iter().map(|t| f64::round(t * b) / b).collect()
}

fn run_cast<T, O>(input: &[T], kernel: hodu_cuda_kernels::kernels::Kernel) -> Vec<O>
where
    T: cudarc::driver::DeviceRepr + Clone,
    O: cudarc::driver::DeviceRepr + Clone,
{
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let mut output: cudarc::driver::CudaSlice<O> = unsafe { stream.alloc(input.len()).unwrap() };

    let num_els = input.len();
    let num_dims = 1;
    let shape = vec![input.len()];
    let strides = vec![1];

    // Build metadata: [num_els, num_dims, shape..., strides..., offset]
    let mut metadata = Vec::with_capacity(2 + num_dims * 2 + 1);
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&strides);
    metadata.push(0); // offset

    call_ops_cast(kernel, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; input.len()];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[test]
fn cast_f32_to_i32() {
    let input = vec![1.5f32, -2.7, 3.2, -4.9];
    let results: Vec<i32> = run_cast(&input, Kernel("hodu_cuda_cast_f32_to_i32"));
    assert_eq!(results, vec![1i32, -2, 3, -4]);
}

#[test]
fn cast_i32_to_f32() {
    let input = vec![1i32, -2, 3, -4];
    let results: Vec<f32> = run_cast(&input, Kernel("hodu_cuda_cast_i32_to_f32"));
    assert_eq!(results, vec![1.0f32, -2.0, 3.0, -4.0]);
}

#[test]
fn cast_f32_to_u32() {
    let input = vec![1.5f32, 2.7, 3.2, 4.9];
    let results: Vec<u32> = run_cast(&input, Kernel("hodu_cuda_cast_f32_to_u32"));
    assert_eq!(results, vec![1u32, 2, 3, 4]);
}

#[test]
fn cast_u32_to_f32() {
    let input = vec![1u32, 2, 3, 4];
    let results: Vec<f32> = run_cast(&input, Kernel("hodu_cuda_cast_u32_to_f32"));
    assert_eq!(results, vec![1.0f32, 2.0, 3.0, 4.0]);
}

#[test]
fn cast_f64_to_f32() {
    let input = vec![1.5f64, 2.7, 3.2, 4.9];
    let results: Vec<f32> = run_cast(&input, Kernel("hodu_cuda_cast_f64_to_f32"));
    assert_eq!(results, vec![1.5f32, 2.7, 3.2, 4.9]);
}

#[test]
fn cast_f32_to_f64() {
    let input = vec![1.5f32, 2.7, 3.2, 4.9];
    let results: Vec<f64> = run_cast(&input, Kernel("hodu_cuda_cast_f32_to_f64"));
    // f32 cannot exactly represent 2.7, 3.2, 4.9, so we need to approximate
    assert_eq!(approx_f64(results, 1), vec![1.5f64, 2.7, 3.2, 4.9]);
}
