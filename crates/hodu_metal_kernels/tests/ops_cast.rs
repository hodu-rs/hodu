use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_ops_cast, Kernel},
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

fn run_cast<T: Clone, O: Clone>(input: &[T], kernel: Kernel) -> Vec<O> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    let input_buffer = new_buffer(&device, input);
    let output_size = input.len() * std::mem::size_of::<O>();
    let output = device.new_buffer(output_size, RESOURCE_OPTIONS).unwrap();

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

    call_ops_cast(
        kernel,
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
    read_to_vec(&output, input.len())
}

#[test]
fn cast_f32_to_i32() {
    let input = vec![1.5f32, -2.7, 3.2, -4.9];
    let results: Vec<i32> = run_cast(&input, Kernel("hodu_metal_cast_f32_to_i32"));
    assert_eq!(results, vec![1i32, -2, 3, -4]);
}

#[test]
fn cast_i32_to_f32() {
    let input = vec![1i32, -2, 3, -4];
    let results: Vec<f32> = run_cast(&input, Kernel("hodu_metal_cast_i32_to_f32"));
    assert_eq!(results, vec![1.0f32, -2.0, 3.0, -4.0]);
}
