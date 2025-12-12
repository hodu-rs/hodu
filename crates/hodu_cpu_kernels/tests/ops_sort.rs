use hodu_cpu_kernels::*;

// Helper function to build topk metadata
// Layout: [output_size, k, last_dim_size, outer_size, largest, sorted, offset]
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

// topk - 1D tensor, largest
#[test]
fn test_topk_1d_largest_f32() {
    let input = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let k = 3;
    let last_dim_size = 8;
    let outer_size = 1;
    let mut values = vec![0.0f32; k];
    let mut indices = vec![0i32; k];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, true, true, 0);

    call_topk(
        topk::F32,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Top 3 largest: 9.0 (idx 5), 6.0 (idx 7), 5.0 (idx 4)
    assert_eq!(values, vec![9.0, 6.0, 5.0]);
    assert_eq!(indices, vec![5, 7, 4]);
}

// topk - 1D tensor, smallest
#[test]
fn test_topk_1d_smallest_f32() {
    let input = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let k = 3;
    let last_dim_size = 8;
    let outer_size = 1;
    let mut values = vec![0.0f32; k];
    let mut indices = vec![0i32; k];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, false, true, 0);

    call_topk(
        topk::F32,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Top 3 smallest: 1.0 (idx 1), 1.0 (idx 3), 2.0 (idx 6)
    assert_eq!(values, vec![1.0, 1.0, 2.0]);
    // Note: indices might vary for equal values
    assert!(indices.contains(&1) || indices.contains(&3));
}

// topk - 2D tensor (batch)
#[test]
fn test_topk_2d_f32() {
    // Shape [2, 5] - two rows of 5 elements each
    let input = [
        5.0f32, 2.0, 8.0, 1.0, 9.0, // row 0
        3.0, 7.0, 4.0, 6.0, 0.0, // row 1
    ];
    let k = 2;
    let last_dim_size = 5;
    let outer_size = 2;
    let mut values = vec![0.0f32; k * outer_size];
    let mut indices = vec![0i32; k * outer_size];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, true, true, 0);

    call_topk(
        topk::F32,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Row 0: top 2 largest are 9.0 (idx 4), 8.0 (idx 2)
    // Row 1: top 2 largest are 7.0 (idx 1), 6.0 (idx 3)
    assert_eq!(values[0..2], [9.0, 8.0]);
    assert_eq!(indices[0..2], [4, 2]);
    assert_eq!(values[2..4], [7.0, 6.0]);
    assert_eq!(indices[2..4], [1, 3]);
}

// topk - k equals dimension size
#[test]
fn test_topk_k_equals_dim_f32() {
    let input = [3.0f32, 1.0, 4.0, 1.0, 5.0];
    let k = 5;
    let last_dim_size = 5;
    let outer_size = 1;
    let mut values = vec![0.0f32; k];
    let mut indices = vec![0i32; k];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, true, true, 0);

    call_topk(
        topk::F32,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // All elements sorted: 5.0, 4.0, 3.0, 1.0, 1.0
    assert_eq!(values, vec![5.0, 4.0, 3.0, 1.0, 1.0]);
}

// topk - k = 1
#[test]
fn test_topk_k1_f32() {
    let input = [3.0f32, 1.0, 4.0, 1.0, 5.0];
    let k = 1;
    let last_dim_size = 5;
    let outer_size = 1;
    let mut values = vec![0.0f32; k];
    let mut indices = vec![0i32; k];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, true, true, 0);

    call_topk(
        topk::F32,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Max element: 5.0 at index 4
    assert_eq!(values, vec![5.0]);
    assert_eq!(indices, vec![4]);
}

// topk - integer type i32
#[test]
fn test_topk_i32() {
    let input = [3i32, 1, 4, 1, 5, 9, 2, 6];
    let k = 3;
    let last_dim_size = 8;
    let outer_size = 1;
    let mut values = vec![0i32; k];
    let mut indices = vec![0i32; k];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, true, true, 0);

    call_topk(
        topk::I32,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(values, vec![9, 6, 5]);
    assert_eq!(indices, vec![5, 7, 4]);
}

// topk - f64
#[test]
fn test_topk_f64() {
    let input = [3.0f64, 1.0, 4.0, 1.0, 5.0];
    let k = 2;
    let last_dim_size = 5;
    let outer_size = 1;
    let mut values = vec![0.0f64; k];
    let mut indices = vec![0i32; k];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, true, true, 0);

    call_topk(
        topk::F64,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(values, vec![5.0, 4.0]);
    assert_eq!(indices, vec![4, 2]);
}

// topk - with offset
#[test]
fn test_topk_with_offset_f32() {
    let input = [0.0f32, 0.0, 3.0, 1.0, 4.0, 1.0, 5.0]; // offset 2 -> [3, 1, 4, 1, 5]
    let k = 2;
    let last_dim_size = 5;
    let outer_size = 1;
    let mut values = vec![0.0f32; k];
    let mut indices = vec![0i32; k];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, true, true, 2);

    call_topk(
        topk::F32,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // From [3, 1, 4, 1, 5], top 2 largest: 5.0 (idx 4), 4.0 (idx 2)
    assert_eq!(values, vec![5.0, 4.0]);
    assert_eq!(indices, vec![4, 2]);
}

// topk - unsorted
#[test]
fn test_topk_unsorted_f32() {
    let input = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let k = 3;
    let last_dim_size = 8;
    let outer_size = 1;
    let mut values = vec![0.0f32; k];
    let mut indices = vec![0i32; k];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, true, false, 0);

    call_topk(
        topk::F32,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Top 3 largest are 9.0, 6.0, 5.0 but order may not be sorted
    let mut sorted_values = values.clone();
    sorted_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(sorted_values, vec![9.0, 6.0, 5.0]);
}

// topk - negative values
#[test]
fn test_topk_negative_f32() {
    let input = [-3.0f32, -1.0, -4.0, -1.0, -5.0];
    let k = 2;
    let last_dim_size = 5;
    let outer_size = 1;
    let mut values = vec![0.0f32; k];
    let mut indices = vec![0i32; k];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, true, true, 0);

    call_topk(
        topk::F32,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Top 2 largest: -1.0, -1.0 (at indices 1 and 3)
    assert_eq!(values, vec![-1.0, -1.0]);
}

// topk - u8
#[test]
fn test_topk_u8() {
    let input = [3u8, 1, 4, 1, 5, 9, 2, 6];
    let k = 3;
    let last_dim_size = 8;
    let outer_size = 1;
    let mut values = vec![0u8; k];
    let mut indices = vec![0i32; k];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, true, true, 0);

    call_topk(
        topk::U8,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(values, vec![9, 6, 5]);
    assert_eq!(indices, vec![5, 7, 4]);
}

// topk - u32
#[test]
fn test_topk_u32() {
    let input = [3u32, 1, 4, 1, 5, 9, 2, 6];
    let k = 3;
    let last_dim_size = 8;
    let outer_size = 1;
    let mut values = vec![0u32; k];
    let mut indices = vec![0i32; k];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, true, true, 0);

    call_topk(
        topk::U32,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(values, vec![9, 6, 5]);
    assert_eq!(indices, vec![5, 7, 4]);
}

// topk - i64
#[test]
fn test_topk_i64() {
    let input = [3i64, 1, 4, 1, 5, 9, 2, 6];
    let k = 3;
    let last_dim_size = 8;
    let outer_size = 1;
    let mut values = vec![0i64; k];
    let mut indices = vec![0i32; k];

    let metadata = build_topk_metadata(k, last_dim_size, outer_size, true, true, 0);

    call_topk(
        topk::I64,
        input.as_ptr() as *const core::ffi::c_void,
        values.as_mut_ptr() as *mut core::ffi::c_void,
        indices.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    assert_eq!(values, vec![9, 6, 5]);
    assert_eq!(indices, vec![5, 7, 4]);
}
