use hodu_cuda_kernels::{compat::*, kernel::Kernels, kernels::*};

fn device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn kernels() -> Kernels {
    Kernels::new()
}

fn run_const_set<T>(shape: &[usize], strides: &[usize], offset: usize, const_val: T, kernel: Kernel) -> Vec<T>
where
    T: cudarc::driver::DeviceRepr + Clone,
{
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    // Calculate the actual buffer size needed for strided layout
    let buffer_size = if shape.is_empty() {
        1
    } else {
        offset
            + shape
                .iter()
                .zip(strides.iter())
                .map(|(s, stride)| (s - 1) * stride)
                .sum::<usize>()
            + 1
    };

    let mut output: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(buffer_size).unwrap() };

    // Build metadata: [num_els, num_dims, shape..., strides..., offset]
    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend_from_slice(shape);
    metadata.extend_from_slice(strides);
    metadata.push(offset);

    call_const_set(kernel, &device, &mut output, &metadata, const_val).unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; buffer_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[test]
fn test_const_set_f32() {
    let shape = vec![10];
    let strides = vec![1];
    let result = run_const_set(&shape, &strides, 0, std::f32::consts::PI, const_set::F32);
    assert_eq!(result, vec![std::f32::consts::PI; 10]);
}

#[test]
fn test_const_set_i32() {
    let shape = vec![8];
    let strides = vec![1];
    let result = run_const_set(&shape, &strides, 0, 42i32, const_set::I32);
    assert_eq!(result, vec![42i32; 8]);
}

#[test]
fn test_const_set_2d() {
    let shape = vec![3, 4];
    let strides = vec![4, 1];
    let result = run_const_set(&shape, &strides, 0, 7.0f32, const_set::F32);
    assert_eq!(result, vec![7.0f32; 12]);
}

#[test]
fn test_const_set_strided() {
    // Test with non-contiguous strides
    // shape [2, 3], strides [6, 2]
    // Elements at indices: [0, 2, 4, 6, 8, 10]
    // Buffer size: 0 + (2-1)*6 + (3-1)*2 + 1 = 0 + 6 + 4 + 1 = 11
    let shape = vec![2, 3];
    let strides = vec![6, 2]; // Non-contiguous
    let result = run_const_set(&shape, &strides, 0, 9.0f32, const_set::F32);

    // Check the values at the strided positions
    // Position (0,0) = 0*6 + 0*2 = 0 -> should be 9.0
    // Position (0,1) = 0*6 + 1*2 = 2 -> should be 9.0
    // Position (0,2) = 0*6 + 2*2 = 4 -> should be 9.0
    // Position (1,0) = 1*6 + 0*2 = 6 -> should be 9.0
    // Position (1,1) = 1*6 + 1*2 = 8 -> should be 9.0
    // Position (1,2) = 1*6 + 2*2 = 10 -> should be 9.0
    assert_eq!(result.len(), 11);
    assert_eq!(result[0], 9.0f32);
    assert_eq!(result[2], 9.0f32);
    assert_eq!(result[4], 9.0f32);
    assert_eq!(result[6], 9.0f32);
    assert_eq!(result[8], 9.0f32);
    assert_eq!(result[10], 9.0f32);

    // Other positions should be 0.0 (untouched)
    assert_eq!(result[1], 0.0f32);
    assert_eq!(result[3], 0.0f32);
    assert_eq!(result[5], 0.0f32);
    assert_eq!(result[7], 0.0f32);
    assert_eq!(result[9], 0.0f32);
}

#[test]
fn test_const_set_with_offset() {
    // Test with offset
    let shape = vec![2, 2];
    let strides = vec![2, 1];
    let offset = 3;
    let result = run_const_set(&shape, &strides, offset, 5.0f32, const_set::F32);

    // Buffer size: 3 + (2-1)*2 + (2-1)*1 + 1 = 3 + 2 + 1 + 1 = 7
    // Elements at: [3, 4, 5, 6]
    assert_eq!(result.len(), 7);
    assert_eq!(result[3], 5.0f32);
    assert_eq!(result[4], 5.0f32);
    assert_eq!(result[5], 5.0f32);
    assert_eq!(result[6], 5.0f32);
}
