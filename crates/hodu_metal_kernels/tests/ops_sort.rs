use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_topk, topk, Kernel},
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

// Helper function to build topk metadata
fn build_topk_metadata(
    k: usize,
    last_dim_size: usize,
    outer_size: usize,
    largest: bool,
    sorted: bool,
    offset: usize,
) -> Vec<usize> {
    let output_size = k * outer_size;
    vec![
        output_size,
        k,
        last_dim_size,
        outer_size,
        if largest { 1 } else { 0 },
        if sorted { 1 } else { 0 },
        offset,
    ]
}

fn run_topk<T: Clone + Default>(
    input: &[T],
    k: usize,
    last_dim_size: usize,
    outer_size: usize,
    largest: bool,
    sorted: bool,
    kernel: Kernel,
) -> (Vec<T>, Vec<i32>) {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;
    let input_buffer = new_buffer(&device, input);

    let output_size = k * outer_size;
    let values = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();
    let indices = device
        .new_buffer(output_size * std::mem::size_of::<i32>(), options)
        .unwrap();

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, largest, sorted, 0);

    call_topk(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &values,
        &indices,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    (read_to_vec(&values, output_size), read_to_vec(&indices, output_size))
}

// topk - 1D tensor, largest
#[test]
fn test_topk_1d_largest_f32() {
    let input = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let (values, indices) = run_topk(&input, 3, 8, 1, true, true, topk::F32);

    // Top 3 largest: 9.0 (idx 5), 6.0 (idx 7), 5.0 (idx 4)
    assert_eq!(values, vec![9.0, 6.0, 5.0]);
    assert_eq!(indices, vec![5, 7, 4]);
}

// topk - 1D tensor, smallest
#[test]
fn test_topk_1d_smallest_f32() {
    let input = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let (values, indices) = run_topk(&input, 3, 8, 1, false, true, topk::F32);

    // Top 3 smallest: 1.0 (idx 1 or 3), 1.0, 2.0 (idx 6)
    assert_eq!(values, vec![1.0, 1.0, 2.0]);
}

// topk - 2D tensor (batch)
#[test]
fn test_topk_2d_f32() {
    // Shape [2, 5] - two rows of 5 elements each
    let input = vec![
        5.0f32, 2.0, 8.0, 1.0, 9.0, // row 0
        3.0, 7.0, 4.0, 6.0, 0.0, // row 1
    ];
    let (values, indices) = run_topk(&input, 2, 5, 2, true, true, topk::F32);

    // Row 0: top 2 largest are 9.0 (idx 4), 8.0 (idx 2)
    // Row 1: top 2 largest are 7.0 (idx 1), 6.0 (idx 3)
    assert_eq!(values[0..2], [9.0, 8.0]);
    assert_eq!(indices[0..2], [4, 2]);
    assert_eq!(values[2..4], [7.0, 6.0]);
    assert_eq!(indices[2..4], [1, 3]);
}

// topk - k = 1
#[test]
fn test_topk_k1_f32() {
    let input = vec![3.0f32, 1.0, 4.0, 1.0, 5.0];
    let (values, indices) = run_topk(&input, 1, 5, 1, true, true, topk::F32);

    // Max element: 5.0 at index 4
    assert_eq!(values, vec![5.0]);
    assert_eq!(indices, vec![4]);
}

// topk - integer type i32
#[test]
fn test_topk_i32() {
    let input = vec![3i32, 1, 4, 1, 5, 9, 2, 6];
    let (values, indices) = run_topk(&input, 3, 8, 1, true, true, topk::I32);

    assert_eq!(values, vec![9, 6, 5]);
    assert_eq!(indices, vec![5, 7, 4]);
}

// topk - u32
#[test]
fn test_topk_u32() {
    let input = vec![3u32, 1, 4, 1, 5, 9, 2, 6];
    let (values, indices) = run_topk(&input, 3, 8, 1, true, true, topk::U32);

    assert_eq!(values, vec![9, 6, 5]);
    assert_eq!(indices, vec![5, 7, 4]);
}

// topk - half precision
#[test]
fn test_topk_f16() {
    let input: Vec<half::f16> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let (values, indices) = run_topk(&input, 3, 8, 1, true, true, topk::F16);

    let expected: Vec<half::f16> = vec![9.0, 6.0, 5.0].into_iter().map(half::f16::from_f32).collect();
    assert_eq!(values, expected);
    assert_eq!(indices, vec![5, 7, 4]);
}

// topk - bfloat16
#[test]
fn test_topk_bf16() {
    let input: Vec<half::bf16> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        .into_iter()
        .map(half::bf16::from_f32)
        .collect();
    let (values, indices) = run_topk(&input, 3, 8, 1, true, true, topk::BF16);

    let expected: Vec<half::bf16> = vec![9.0, 6.0, 5.0].into_iter().map(half::bf16::from_f32).collect();
    assert_eq!(values, expected);
    assert_eq!(indices, vec![5, 7, 4]);
}

// topk - negative values
#[test]
fn test_topk_negative_f32() {
    let input = vec![-3.0f32, -1.0, -4.0, -1.0, -5.0];
    let (values, indices) = run_topk(&input, 2, 5, 1, true, true, topk::F32);

    // Top 2 largest: -1.0, -1.0
    assert_eq!(values, vec![-1.0, -1.0]);
}

// topk - u8
#[test]
fn test_topk_u8() {
    let input = vec![3u8, 1, 4, 1, 5, 9, 2, 6];
    let (values, indices) = run_topk(&input, 3, 8, 1, true, true, topk::U8);

    assert_eq!(values, vec![9, 6, 5]);
    assert_eq!(indices, vec![5, 7, 4]);
}

// topk - i8
#[test]
fn test_topk_i8() {
    let input = vec![3i8, 1, 4, 1, 5, 9, 2, 6];
    let (values, indices) = run_topk(&input, 3, 8, 1, true, true, topk::I8);

    assert_eq!(values, vec![9, 6, 5]);
    assert_eq!(indices, vec![5, 7, 4]);
}
