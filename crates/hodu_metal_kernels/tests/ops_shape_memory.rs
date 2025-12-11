use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_ops_flip, flip},
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

#[test]
fn flip_f32_1d_dim0() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(4 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_els = 4usize;
    let num_dims = 1usize;
    let shape = vec![4usize];
    let flip_mask = vec![1usize]; // flip dim 0

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, 4);
    assert_eq!(results, vec![4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn flip_f32_1d_no_flip() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(4 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_els = 4usize;
    let num_dims = 1usize;
    let shape = vec![4usize];
    let flip_mask = vec![0usize]; // no flip

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, 4);
    assert_eq!(results, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn flip_f32_2d_dim0() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Input: [[1, 2, 3], [4, 5, 6]]  (2x3)
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(6 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_els = 6usize;
    let num_dims = 2usize;
    let shape = vec![2usize, 3usize];
    let flip_mask = vec![1usize, 0usize]; // flip dim 0 only

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Output: [[4, 5, 6], [1, 2, 3]]
    let results: Vec<f32> = read_to_vec(&output, 6);
    assert_eq!(results, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
}

#[test]
fn flip_f32_2d_dim1() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Input: [[1, 2, 3], [4, 5, 6]]  (2x3)
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(6 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_els = 6usize;
    let num_dims = 2usize;
    let shape = vec![2usize, 3usize];
    let flip_mask = vec![0usize, 1usize]; // flip dim 1 only

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Output: [[3, 2, 1], [6, 5, 4]]
    let results: Vec<f32> = read_to_vec(&output, 6);
    assert_eq!(results, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
}

#[test]
fn flip_f32_2d_both_dims() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Input: [[1, 2, 3], [4, 5, 6]]  (2x3)
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(6 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_els = 6usize;
    let num_dims = 2usize;
    let shape = vec![2usize, 3usize];
    let flip_mask = vec![1usize, 1usize]; // flip both dims

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Output: [[6, 5, 4], [3, 2, 1]]
    let results: Vec<f32> = read_to_vec(&output, 6);
    assert_eq!(results, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn flip_i32_1d() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input = vec![10i32, 20, 30, 40];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(4 * std::mem::size_of::<i32>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_els = 4usize;
    let num_dims = 1usize;
    let shape = vec![4usize];
    let flip_mask = vec![1usize];

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::I32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<i32> = read_to_vec(&output, 4);
    assert_eq!(results, vec![40, 30, 20, 10]);
}

#[test]
fn flip_f32_3d() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Input: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  (2x2x2)
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(8 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_els = 8usize;
    let num_dims = 3usize;
    let shape = vec![2usize, 2usize, 2usize];
    let flip_mask = vec![0usize, 0usize, 1usize]; // flip dim 2 only

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Output: [[[2, 1], [4, 3]], [[6, 5], [8, 7]]]
    let results: Vec<f32> = read_to_vec(&output, 8);
    assert_eq!(results, vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]);
}

#[test]
fn flip_u8() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input = vec![1u8, 2, 3, 4, 5];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(5 * std::mem::size_of::<u8>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_els = 5usize;
    let num_dims = 1usize;
    let shape = vec![5usize];
    let flip_mask = vec![1usize];

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape);
    metadata.extend(&flip_mask);

    call_ops_flip(
        flip::U8,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<u8> = read_to_vec(&output, 5);
    assert_eq!(results, vec![5, 4, 3, 2, 1]);
}
