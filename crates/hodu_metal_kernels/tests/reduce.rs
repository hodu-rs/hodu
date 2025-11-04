use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{argmax, argmin, call_reduce, max, mean, min, sum, Kernel},
    metal::{create_command_buffer, Buffer, Device},
    utils::BufferOffset,
    RESOURCE_OPTIONS,
};
use std::ffi::c_void;

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

fn new_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let options = RESOURCE_OPTIONS;
    let ptr = data.as_ptr() as *const c_void;
    let size = std::mem::size_of_val(data);
    device.new_buffer_with_data(ptr, size, options).unwrap()
}

fn device() -> Device {
    Device::system_default().unwrap()
}

#[allow(clippy::too_many_arguments)]
fn run_reduce<T: Clone>(
    input: &[T],
    input_shape: &[usize],
    reduce_dims: &[usize],
    keep_dim: bool,
    name: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;
    let input_buffer = new_buffer(&device, input);

    // Calculate output shape (will be computed inside call_reduce, but we need it for buffer size)
    let mut output_shape = input_shape.to_vec();
    for &dim in reduce_dims.iter() {
        if keep_dim {
            output_shape[dim] = 1;
        } else {
            output_shape[dim] = 0;
        }
    }
    if !keep_dim {
        output_shape.retain(|&size| size != 0);
        if output_shape.is_empty() {
            output_shape = vec![1];
        }
    }
    let output_size: usize = output_shape.iter().product();

    // Calculate reduce size
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();

    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    // Calculate strides
    let mut strides = vec![1; input_shape.len()];
    for i in (0..input_shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * input_shape[i + 1];
    }

    call_reduce(
        &device,
        &command_buffer,
        &kernels,
        name,
        input_shape,
        BufferOffset::zero_offset(&input_buffer),
        &strides,
        0,
        reduce_dims,
        reduce_size,
        keep_dim,
        &output,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, output_size)
}

#[allow(clippy::too_many_arguments)]
fn run_reduce_output<T: Clone, O: Clone>(
    input: &[T],
    input_shape: &[usize],
    reduce_dims: &[usize],
    keep_dim: bool,
    name: Kernel,
) -> Vec<O> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;
    let input_buffer = new_buffer(&device, input);

    // Calculate output shape
    let mut output_shape = input_shape.to_vec();
    for &dim in reduce_dims.iter() {
        if keep_dim {
            output_shape[dim] = 1;
        } else {
            output_shape[dim] = 0;
        }
    }
    if !keep_dim {
        output_shape.retain(|&size| size != 0);
        if output_shape.is_empty() {
            output_shape = vec![1];
        }
    }
    let output_size: usize = output_shape.iter().product();

    // Calculate reduce size
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();

    let output = device
        .new_buffer(output_size * std::mem::size_of::<O>(), options)
        .unwrap();

    // Calculate strides
    let mut strides = vec![1; input_shape.len()];
    for i in (0..input_shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * input_shape[i + 1];
    }

    call_reduce(
        &device,
        &command_buffer,
        &kernels,
        name,
        input_shape,
        BufferOffset::zero_offset(&input_buffer),
        &strides,
        0,
        reduce_dims,
        reduce_size,
        keep_dim,
        &output,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, output_size)
}

#[test]
fn test_reduce_sum_f32() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3]; // 2x3 matrix
    let reduce_dims = vec![1]; // reduce along columns

    let result: Vec<f32> = run_reduce(&input, &shape, &reduce_dims, false, sum::F32);
    // [[1, 2, 3], [4, 5, 6]] -> sum along dim 1 -> [6, 15]
    assert_eq!(result, vec![6.0, 15.0]);
}

#[test]
fn test_reduce_max_f32() {
    let input: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 8.0, 1.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];

    let result: Vec<f32> = run_reduce(&input, &shape, &reduce_dims, false, max::F32);
    // [[1, 5, 3], [2, 8, 1]] -> max along dim 1 -> [5, 8]
    assert_eq!(result, vec![5.0, 8.0]);
}

#[test]
fn test_reduce_min_f32() {
    let input: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 8.0, 1.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];

    let result: Vec<f32> = run_reduce(&input, &shape, &reduce_dims, false, min::F32);
    // [[1, 5, 3], [2, 8, 1]] -> min along dim 1 -> [1, 1]
    assert_eq!(result, vec![1.0, 1.0]);
}

#[test]
fn test_reduce_mean_f32() {
    let input: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];

    let result: Vec<f32> = run_reduce(&input, &shape, &reduce_dims, false, mean::F32);
    // [[2, 4, 6], [8, 10, 12]] -> mean along dim 1 -> [4, 10]
    assert_eq!(result, vec![4.0, 10.0]);
}

#[test]
fn test_reduce_argmax_f32() {
    let input: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 8.0, 1.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];

    let result: Vec<i32> = run_reduce_output(&input, &shape, &reduce_dims, false, argmax::F32);
    // [[1, 5, 3], [2, 8, 1]] -> argmax along dim 1 -> [1, 1]
    assert_eq!(result, vec![1, 1]);
}

#[test]
fn test_reduce_argmin_f32() {
    let input: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 8.0, 1.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];

    let result: Vec<i32> = run_reduce_output(&input, &shape, &reduce_dims, false, argmin::F32);
    // [[1, 5, 3], [2, 8, 1]] -> argmin along dim 1 -> [0, 2]
    assert_eq!(result, vec![0, 2]);
}

#[test]
fn test_reduce_sum_3d() {
    let input: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let shape = vec![2, 3, 4]; // 2x3x4 tensor
    let reduce_dims = vec![2]; // reduce along last dimension

    let result: Vec<f32> = run_reduce(&input, &shape, &reduce_dims, false, sum::F32);
    // Sum along last dimension (groups of 4)
    // [1,2,3,4] -> 10, [5,6,7,8] -> 26, [9,10,11,12] -> 42,
    // [13,14,15,16] -> 58, [17,18,19,20] -> 74, [21,22,23,24] -> 90
    assert_eq!(result, vec![10.0, 26.0, 42.0, 58.0, 74.0, 90.0]);
}

#[test]
fn test_reduce_sum_f32_keep_dim() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3]; // 2x3 matrix
    let reduce_dims = vec![1]; // reduce along columns

    let result: Vec<f32> = run_reduce(&input, &shape, &reduce_dims, true, sum::F32);
    // [[1, 2, 3], [4, 5, 6]] -> sum along dim 1 with keep_dim -> [[6], [15]]
    // Output shape: [2, 1], flattened: [6, 15]
    assert_eq!(result, vec![6.0, 15.0]);
}

#[test]
fn test_reduce_max_f32_keep_dim() {
    let input: Vec<f32> = vec![1.0, 5.0, 3.0, 2.0, 8.0, 1.0];
    let shape = vec![2, 3];
    let reduce_dims = vec![1];

    let result: Vec<f32> = run_reduce(&input, &shape, &reduce_dims, true, max::F32);
    // [[1, 5, 3], [2, 8, 1]] -> max along dim 1 with keep_dim -> [[5], [8]]
    assert_eq!(result, vec![5.0, 8.0]);
}
