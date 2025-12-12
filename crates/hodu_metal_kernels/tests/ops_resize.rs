use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_ops_resize, resize, Kernel},
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

fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// Resize modes
const RESIZE_MODE_NEAREST: usize = 0;
const RESIZE_MODE_LINEAR: usize = 1;
const RESIZE_MODE_CUBIC: usize = 2;

// Coordinate transformation modes
const RESIZE_COORD_HALF_PIXEL: usize = 0;
const RESIZE_COORD_ASYMMETRIC: usize = 1;
const RESIZE_COORD_ALIGN_CORNERS: usize = 2;

// Nearest rounding modes
const RESIZE_NEAREST_FLOOR: usize = 0;

fn build_resize_metadata(
    input_shape: &[usize],
    input_strides: &[usize],
    offset: usize,
    output_shape: &[usize],
    mode: usize,
    coord_transform: usize,
    nearest_mode: usize,
) -> Vec<usize> {
    let output_size: usize = output_shape.iter().product();
    let num_dims = input_shape.len();
    let mut metadata = vec![output_size, num_dims];
    metadata.extend(input_shape);
    metadata.extend(input_strides);
    metadata.push(offset);
    metadata.extend(output_shape);
    metadata.push(mode);
    metadata.push(coord_transform);
    metadata.push(nearest_mode);
    metadata
}

fn run_resize<T: Clone + Default>(
    input: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    mode: usize,
    coord_transform: usize,
    nearest_mode: usize,
    kernel: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;
    let input_buffer = new_buffer(&device, input);

    let output_size: usize = output_shape.iter().product();
    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    let strides = calculate_strides(input_shape);
    let metadata = build_resize_metadata(
        input_shape,
        &strides,
        0,
        output_shape,
        mode,
        coord_transform,
        nearest_mode,
    );

    call_ops_resize(
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
    read_to_vec(&output, output_size)
}

#[test]
fn test_resize_nearest_upsample_2x_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let result = run_resize(
        &input,
        &[1, 1, 2, 2],
        &[1, 1, 4, 4],
        RESIZE_MODE_NEAREST,
        RESIZE_COORD_ASYMMETRIC,
        RESIZE_NEAREST_FLOOR,
        resize::F32,
    );
    let expected = vec![
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_resize_bilinear_upsample_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let result = run_resize(
        &input,
        &[1, 1, 2, 2],
        &[1, 1, 4, 4],
        RESIZE_MODE_LINEAR,
        RESIZE_COORD_ALIGN_CORNERS,
        RESIZE_NEAREST_FLOOR,
        resize::F32,
    );
    // With align_corners, corners should match exactly
    assert!((result[0] - 1.0).abs() < 1e-5); // top-left
    assert!((result[3] - 2.0).abs() < 1e-5); // top-right
    assert!((result[12] - 3.0).abs() < 1e-5); // bottom-left
    assert!((result[15] - 4.0).abs() < 1e-5); // bottom-right
}

#[test]
fn test_resize_nearest_downsample_f32() {
    let input = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let result = run_resize(
        &input,
        &[1, 1, 4, 4],
        &[1, 1, 2, 2],
        RESIZE_MODE_NEAREST,
        RESIZE_COORD_ASYMMETRIC,
        RESIZE_NEAREST_FLOOR,
        resize::F32,
    );
    assert_eq!(result, vec![1.0, 3.0, 9.0, 11.0]);
}

#[test]
fn test_resize_bilinear_half_pixel_f32() {
    let input = vec![0.0f32, 10.0, 10.0, 20.0];
    let result = run_resize(
        &input,
        &[1, 1, 2, 2],
        &[1, 1, 4, 4],
        RESIZE_MODE_LINEAR,
        RESIZE_COORD_HALF_PIXEL,
        RESIZE_NEAREST_FLOOR,
        resize::F32,
    );
    for &val in &result {
        assert!(val >= 0.0 && val <= 20.0);
    }
}

#[test]
fn test_resize_bicubic_f32() {
    let input = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let result = run_resize(
        &input,
        &[1, 1, 4, 4],
        &[1, 1, 8, 8],
        RESIZE_MODE_CUBIC,
        RESIZE_COORD_HALF_PIXEL,
        RESIZE_NEAREST_FLOOR,
        resize::F32,
    );
    for &val in &result {
        assert!(val >= 0.0 && val <= 20.0);
    }
}

#[test]
fn test_resize_identity_f32() {
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let result = run_resize(
        &input,
        &[1, 1, 2, 2],
        &[1, 1, 2, 2],
        RESIZE_MODE_LINEAR,
        RESIZE_COORD_HALF_PIXEL,
        RESIZE_NEAREST_FLOOR,
        resize::F32,
    );
    for (a, b) in result.iter().zip(input.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_resize_batch_channel_f32() {
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let result = run_resize(
        &input,
        &[2, 2, 2, 2],
        &[2, 2, 4, 4],
        RESIZE_MODE_NEAREST,
        RESIZE_COORD_ASYMMETRIC,
        RESIZE_NEAREST_FLOOR,
        resize::F32,
    );
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[16] - 5.0).abs() < 1e-5);
    assert!((result[32] - 9.0).abs() < 1e-5);
    assert!((result[48] - 13.0).abs() < 1e-5);
}
