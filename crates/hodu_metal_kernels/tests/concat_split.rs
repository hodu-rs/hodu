use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_concat, call_split, concat, split},
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
fn concat_f32_dim0() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Two 2x2 matrices concatenated along dim 0 to form 4x2
    // Input 1: [[1, 2], [3, 4]]
    // Input 2: [[5, 6], [7, 8]]
    // Output: [[1, 2], [3, 4], [5, 6], [7, 8]]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_buffer = new_buffer(&device, &input);

    let output_shape = vec![4, 2];
    let concat_dim = 0;
    let _num_inputs = 2;

    // Input shapes for 2 inputs: [2, 2, 2, 2] (flattened)
    let input_shapes = vec![2, 2, 2, 2];
    // Input strides: [2, 1, 2, 1]
    let input_strides = vec![2, 1, 2, 1];
    // Input offsets into buffer: [0, 4] (second matrix starts at index 4)
    let input_offsets = vec![0, 4];
    // Buffer offsets (in elements): [0, 0] (both from same buffer)
    let input_buffer_offsets = vec![0, 0];

    let output = device
        .new_buffer(8 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    // Build metadata: [num_els, num_dims, output_shape..., concat_dim, num_inputs, input_shapes..., input_strides..., input_offsets..., input_buffer_offsets...]
    let num_els = 8;
    let num_dims = 2;
    let num_inputs = 2;
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&output_shape);
    metadata.push(concat_dim);
    metadata.push(num_inputs);
    metadata.extend(&input_shapes);
    metadata.extend(&input_strides);
    metadata.extend(&input_offsets);
    metadata.extend(&input_buffer_offsets);

    call_concat(
        &device,
        &command_buffer,
        &kernels,
        concat::F32,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, 8);
    assert_eq!(results, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn split_f32_dim0() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Input: 4x2 matrix [[1, 2], [3, 4], [5, 6], [7, 8]]
    // Split along dim 0, taking first 2 rows: [[1, 2], [3, 4]]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_buffer = new_buffer(&device, &input);

    let input_shape = vec![4, 2];
    let strides = vec![2, 1];
    let split_dim = 0;
    let output_size_on_dim = 2; // Take 2 rows
    let split_offset = 0; // Start from beginning

    let output = device
        .new_buffer(4 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    // Build metadata: [num_els, num_dims, input_shape..., strides..., offset, split_dim, output_size_on_dim, split_offset]
    let num_els = 4; // output size
    let num_dims = 2;
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0); // input offset
    metadata.push(split_dim);
    metadata.push(output_size_on_dim);
    metadata.push(split_offset);

    call_split(
        &device,
        &command_buffer,
        &kernels,
        split::F32,
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
fn split_f32_dim0_offset() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Input: 4x2 matrix [[1, 2], [3, 4], [5, 6], [7, 8]]
    // Split along dim 0, taking 2 rows starting from offset 2: [[5, 6], [7, 8]]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_buffer = new_buffer(&device, &input);

    let input_shape = vec![4, 2];
    let strides = vec![2, 1];
    let split_dim = 0;
    let output_size_on_dim = 2;
    let split_offset = 2; // Start from row 2

    let output = device
        .new_buffer(4 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    // Build metadata: [num_els, num_dims, input_shape..., strides..., offset, split_dim, output_size_on_dim, split_offset]
    let num_els = 4; // output size
    let num_dims = 2;
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0); // input offset
    metadata.push(split_dim);
    metadata.push(output_size_on_dim);
    metadata.push(split_offset);

    call_split(
        &device,
        &command_buffer,
        &kernels,
        split::F32,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, 4);
    assert_eq!(results, vec![5.0, 6.0, 7.0, 8.0]);
}
