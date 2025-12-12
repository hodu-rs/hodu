use hodu_cuda_kernels::{
    cuda::{CudaContext, CudaSlice},
    kernel::Kernels,
    kernels::{call_topk, topk, Kernel},
    source::Source,
};
use std::sync::Arc;

fn context() -> Arc<CudaContext> {
    CudaContext::new(0).unwrap()
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

fn run_topk_f32(
    input: &[f32],
    k: usize,
    last_dim_size: usize,
    outer_size: usize,
    largest: bool,
    sorted: bool,
) -> (Vec<f32>, Vec<i32>) {
    let ctx = context();
    let kernels = Kernels::new();
    let stream = ctx.default_stream();

    let input_dev: CudaSlice<f32> = stream.memcpy_stod(input).unwrap();
    let output_size = k * outer_size;
    let mut values_dev: CudaSlice<f32> = ctx.alloc_zeros(output_size).unwrap();
    let mut indices_dev: CudaSlice<i32> = ctx.alloc_zeros(output_size).unwrap();

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, largest, sorted, 0);

    call_topk(
        topk::F32,
        &kernels,
        &ctx,
        &input_dev,
        &mut values_dev,
        &mut indices_dev,
        &metadata,
    )
    .unwrap();

    let values: Vec<f32> = stream.memcpy_dtov(&values_dev).unwrap();
    let indices: Vec<i32> = stream.memcpy_dtov(&indices_dev).unwrap();

    (values, indices)
}

fn run_topk_i32(
    input: &[i32],
    k: usize,
    last_dim_size: usize,
    outer_size: usize,
    largest: bool,
    sorted: bool,
) -> (Vec<i32>, Vec<i32>) {
    let ctx = context();
    let kernels = Kernels::new();
    let stream = ctx.default_stream();

    let input_dev: CudaSlice<i32> = stream.memcpy_stod(input).unwrap();
    let output_size = k * outer_size;
    let mut values_dev: CudaSlice<i32> = ctx.alloc_zeros(output_size).unwrap();
    let mut indices_dev: CudaSlice<i32> = ctx.alloc_zeros(output_size).unwrap();

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, largest, sorted, 0);

    call_topk(
        topk::I32,
        &kernels,
        &ctx,
        &input_dev,
        &mut values_dev,
        &mut indices_dev,
        &metadata,
    )
    .unwrap();

    let values: Vec<i32> = stream.memcpy_dtov(&values_dev).unwrap();
    let indices: Vec<i32> = stream.memcpy_dtov(&indices_dev).unwrap();

    (values, indices)
}

// topk - 1D tensor, largest
#[test]
fn test_topk_1d_largest_f32() {
    let input = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let (values, indices) = run_topk_f32(&input, 3, 8, 1, true, true);

    // Top 3 largest: 9.0 (idx 5), 6.0 (idx 7), 5.0 (idx 4)
    assert_eq!(values, vec![9.0, 6.0, 5.0]);
    assert_eq!(indices, vec![5, 7, 4]);
}

// topk - 1D tensor, smallest
#[test]
fn test_topk_1d_smallest_f32() {
    let input = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let (values, _indices) = run_topk_f32(&input, 3, 8, 1, false, true);

    // Top 3 smallest: 1.0, 1.0, 2.0
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
    let (values, indices) = run_topk_f32(&input, 2, 5, 2, true, true);

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
    let (values, indices) = run_topk_f32(&input, 1, 5, 1, true, true);

    // Max element: 5.0 at index 4
    assert_eq!(values, vec![5.0]);
    assert_eq!(indices, vec![4]);
}

// topk - integer type i32
#[test]
fn test_topk_i32() {
    let input = vec![3i32, 1, 4, 1, 5, 9, 2, 6];
    let (values, indices) = run_topk_i32(&input, 3, 8, 1, true, true);

    assert_eq!(values, vec![9, 6, 5]);
    assert_eq!(indices, vec![5, 7, 4]);
}

// topk - negative values
#[test]
fn test_topk_negative_f32() {
    let input = vec![-3.0f32, -1.0, -4.0, -1.0, -5.0];
    let (values, _indices) = run_topk_f32(&input, 2, 5, 1, true, true);

    // Top 2 largest: -1.0, -1.0
    assert_eq!(values, vec![-1.0, -1.0]);
}
