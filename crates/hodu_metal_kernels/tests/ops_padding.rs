use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{
        call_ops_pad_circular, call_ops_pad_constant, call_ops_pad_reflect, call_ops_pad_replicate, pad_circular,
        pad_constant, pad_reflect, pad_replicate,
    },
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
fn pad_constant_f32_1d() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input = vec![1.0f32, 2.0, 3.0];
    let pad_value = vec![0.0f32];
    let input_buffer = new_buffer(&device, &input);
    let pad_value_buffer = new_buffer(&device, &pad_value);
    let output = device
        .new_buffer(6 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_dims = 1usize;
    let input_shape = vec![3usize];
    let output_shape = vec![6usize];
    let pad_before = vec![2usize];
    let num_els = 6usize;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_constant(
        pad_constant::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        BufferOffset::zero_offset(&pad_value_buffer),
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, 6);
    assert_eq!(results, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0]);
}

#[test]
fn pad_constant_f32_2d() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let pad_value = vec![0.0f32];
    let input_buffer = new_buffer(&device, &input);
    let pad_value_buffer = new_buffer(&device, &pad_value);
    let output = device
        .new_buffer(16 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_dims = 2usize;
    let input_shape = vec![2usize, 2];
    let output_shape = vec![4usize, 4];
    let pad_before = vec![1usize, 1];
    let num_els = 16usize;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_constant(
        pad_constant::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        BufferOffset::zero_offset(&pad_value_buffer),
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, 16);
    #[rustfmt::skip]
    let expected = vec![
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 2.0, 0.0,
        0.0, 3.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(results, expected);
}

#[test]
fn pad_reflect_f32_1d() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(8 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_dims = 1usize;
    let input_shape = vec![4usize];
    let output_shape = vec![8usize];
    let pad_before = vec![2usize];
    let num_els = 8usize;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_reflect(
        pad_reflect::F32,
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

    let results: Vec<f32> = read_to_vec(&output, 8);
    assert_eq!(results, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
}

#[test]
fn pad_replicate_f32_1d() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input = vec![1.0f32, 2.0, 3.0];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(7 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_dims = 1usize;
    let input_shape = vec![3usize];
    let output_shape = vec![7usize];
    let pad_before = vec![2usize];
    let num_els = 7usize;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_replicate(
        pad_replicate::F32,
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

    let results: Vec<f32> = read_to_vec(&output, 7);
    assert_eq!(results, vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
}

#[test]
fn pad_circular_f32_1d() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input = vec![1.0f32, 2.0, 3.0];
    let input_buffer = new_buffer(&device, &input);
    let output = device
        .new_buffer(7 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    let num_dims = 1usize;
    let input_shape = vec![3usize];
    let output_shape = vec![7usize];
    let pad_before = vec![2usize];
    let num_els = 7usize;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&output_shape);
    metadata.extend(&pad_before);

    call_ops_pad_circular(
        pad_circular::F32,
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

    let results: Vec<f32> = read_to_vec(&output, 7);
    assert_eq!(results, vec![2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]);
}
