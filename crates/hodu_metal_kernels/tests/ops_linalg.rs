use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_ops_det, det},
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

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

#[test]
fn det_f32_2x2() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // 2x2 matrix: [[1, 2], [3, 4]]
    // det = 1*4 - 2*3 = -2
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_buffer = new_buffer(&device, &input);

    let output = device.new_buffer(std::mem::size_of::<f32>(), RESOURCE_OPTIONS).unwrap();

    // Metadata: [batch_size, n, ndim, shape..., strides..., offset]
    let metadata = vec![1, 2, 2, 2, 2, 2, 1, 0];

    call_ops_det(
        det::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        1,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, 1);
    assert_eq!(approx(results, 4), vec![-2.0]);
}

#[test]
fn det_f32_3x3() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // 3x3 matrix: [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
    // det = -306
    let input = vec![6.0f32, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0];
    let input_buffer = new_buffer(&device, &input);

    let output = device.new_buffer(std::mem::size_of::<f32>(), RESOURCE_OPTIONS).unwrap();

    let metadata = vec![1, 3, 2, 3, 3, 3, 1, 0];

    call_ops_det(
        det::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        1,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, 1);
    assert_eq!(approx(results, 2), vec![-306.0]);
}

#[test]
fn det_f32_4x4_identity() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // 4x4 identity matrix: det(I) = 1
    let input = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let input_buffer = new_buffer(&device, &input);

    let output = device.new_buffer(std::mem::size_of::<f32>(), RESOURCE_OPTIONS).unwrap();

    let metadata = vec![1, 4, 2, 4, 4, 4, 1, 0];

    call_ops_det(
        det::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        1,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, 1);
    assert_eq!(approx(results, 4), vec![1.0]);
}

#[test]
fn det_f32_batch() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Batch of 2 matrices:
    // Matrix 0: [[1, 2], [3, 4]], det = -2
    // Matrix 1: [[5, 6], [7, 8]], det = -2
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_buffer = new_buffer(&device, &input);

    let output = device
        .new_buffer(2 * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    // Metadata: [batch_size, n, ndim, shape..., strides..., offset]
    // shape = [2, 2, 2], strides = [4, 2, 1], offset = 0
    let metadata = vec![2, 2, 3, 2, 2, 2, 4, 2, 1, 0];

    call_ops_det(
        det::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output,
        2,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, 2);
    assert_eq!(approx(results, 4), vec![-2.0, -2.0]);
}
