use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_ops_dot, call_ops_matmul, dot, matmul},
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
fn dot_f32_simple() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Matrix A: 2x3 = [[1, 2, 3], [4, 5, 6]]
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Matrix B: 3x2 = [[7, 8], [9, 10], [11, 12]]
    let rhs = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    let lhs_buffer = new_buffer(&device, &lhs);
    let rhs_buffer = new_buffer(&device, &rhs);

    let m = 2; // rows of A
    let k = 3; // cols of A / rows of B
    let n = 2; // cols of B

    let output = device
        .new_buffer(m * n * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    // Metadata layout (9 elements):
    // [M, K, N, lhs_stride_m, lhs_stride_k, rhs_stride_k, rhs_stride_n, lhs_offset, rhs_offset]
    let metadata = vec![m, k, n, 3, 1, 2, 1, 0, 0];

    call_ops_dot(
        dot::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&lhs_buffer),
        BufferOffset::zero_offset(&rhs_buffer),
        &output,
        m,
        n,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, m * n);
    // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //         = [[58, 64], [139, 154]]
    assert_eq!(approx(results, 4), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_f32_2d() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Matrix A: 2x3 = [[1, 2, 3], [4, 5, 6]]
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // Matrix B: 3x2 = [[7, 8], [9, 10], [11, 12]]
    let rhs = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

    let lhs_buffer = new_buffer(&device, &lhs);
    let rhs_buffer = new_buffer(&device, &rhs);

    let m = 2;
    let k = 3;
    let n = 2;
    let num_els = m * n;

    let output = device
        .new_buffer(num_els * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    // Metadata for matmul: [num_els, lhs_ndim, rhs_ndim, batch_ndim, lhs_shape, rhs_shape,
    //                       lhs_strides, rhs_strides, lhs_offset, rhs_offset, M, K, N]
    let lhs_ndim = 2;
    let rhs_ndim = 2;
    let batch_ndim = 0;
    let metadata = vec![
        num_els, lhs_ndim, rhs_ndim, batch_ndim, m, k, // lhs shape
        k, n, // rhs shape
        k, 1, // lhs strides
        n, 1, // rhs strides
        0, 0, // offsets
        m, k, n, // M, K, N
    ];

    call_ops_matmul(
        matmul::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&lhs_buffer),
        BufferOffset::zero_offset(&rhs_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, num_els);
    // Expected: [[58, 64], [139, 154]]
    assert_eq!(approx(results, 4), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_f32_batched() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Batched matmul: [2, 2, 3] @ [2, 3, 2] = [2, 2, 2]
    // Batch 0: [[1, 2, 3], [4, 5, 6]] @ [[7, 8], [9, 10], [11, 12]]
    // Batch 1: [[13, 14, 15], [16, 17, 18]] @ [[19, 20], [21, 22], [23, 24]]
    let lhs = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 0
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, // Batch 1
    ];
    let rhs = vec![
        7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, // Batch 0
        19.0, 20.0, 21.0, 22.0, 23.0, 24.0, // Batch 1
    ];

    let lhs_buffer = new_buffer(&device, &lhs);
    let rhs_buffer = new_buffer(&device, &rhs);

    let batch = 2;
    let m = 2;
    let k = 3;
    let n = 2;
    let num_els = batch * m * n;

    let output = device
        .new_buffer(num_els * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    // Metadata for batched matmul:
    // [num_els, lhs_ndim, rhs_ndim, batch_ndim, lhs_shape, rhs_shape, batch_shape,
    //  lhs_strides, rhs_strides, lhs_offset, rhs_offset, M, K, N]
    let lhs_ndim = 3;
    let rhs_ndim = 3;
    let batch_ndim = 1;
    let metadata = vec![
        num_els,
        lhs_ndim,
        rhs_ndim,
        batch_ndim,
        batch,
        m,
        k, // lhs shape: [2, 2, 3]
        batch,
        k,
        n,     // rhs shape: [2, 3, 2]
        batch, // batch_shape: [2]
        m * k,
        k,
        1, // lhs strides: [6, 3, 1]
        k * n,
        n,
        1, // rhs strides: [6, 2, 1]
        0,
        0, // offsets
        m,
        k,
        n, // M, K, N
    ];

    call_ops_matmul(
        matmul::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&lhs_buffer),
        BufferOffset::zero_offset(&rhs_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, num_els);
    // Batch 0: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //        = [[58, 64], [139, 154]]
    // Batch 1: [[13*19+14*21+15*23, 13*20+14*22+15*24], [16*19+17*21+18*23, 16*20+17*22+18*24]]
    //        = [[886, 928], [1075, 1126]]
    assert_eq!(
        approx(results, 4),
        vec![58.0, 64.0, 139.0, 154.0, 886.0, 928.0, 1075.0, 1126.0]
    );
}

#[test]
fn matmul_f32_broadcast() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Broadcasting matmul: [2, 2, 3] @ [3, 2] = [2, 2, 2]
    // The rhs is broadcasted across the batch dimension
    let lhs = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 0
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // Batch 1
    ];
    let rhs = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, // No batch dimension
    ];

    let lhs_buffer = new_buffer(&device, &lhs);
    let rhs_buffer = new_buffer(&device, &rhs);

    let batch = 2;
    let m = 2;
    let k = 3;
    let n = 2;
    let num_els = batch * m * n;

    let output = device
        .new_buffer(num_els * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    // lhs: [2, 2, 3], rhs: [3, 2], output: [2, 2, 2]
    let lhs_ndim = 3;
    let rhs_ndim = 2;
    let batch_ndim = 1;
    let metadata = vec![
        num_els,
        lhs_ndim,
        rhs_ndim,
        batch_ndim,
        batch,
        m,
        k, // lhs shape: [2, 2, 3]
        k,
        n,     // rhs shape: [3, 2]
        batch, // batch_shape: [2]
        m * k,
        k,
        1, // lhs strides: [6, 3, 1]
        n,
        1, // rhs strides: [2, 1]
        0,
        0, // offsets
        m,
        k,
        n, // M, K, N
    ];

    call_ops_matmul(
        matmul::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&lhs_buffer),
        BufferOffset::zero_offset(&rhs_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, num_els);
    // Batch 0: [[1, 2, 3], [4, 5, 6]] @ [[1, 2], [3, 4], [5, 6]]
    //        = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
    //        = [[22, 28], [49, 64]]
    // Batch 1: [[7, 8, 9], [10, 11, 12]] @ [[1, 2], [3, 4], [5, 6]] (same rhs broadcasted)
    //        = [[7*1+8*3+9*5, 7*2+8*4+9*6], [10*1+11*3+12*5, 10*2+11*4+12*6]]
    //        = [[76, 100], [103, 136]]
    assert_eq!(
        approx(results, 4),
        vec![22.0, 28.0, 49.0, 64.0, 76.0, 100.0, 103.0, 136.0]
    );
}

#[test]
fn dot_f32_identity() {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();

    // Matrix A: 2x2 = [[1, 2], [3, 4]]
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
    // Identity matrix: 2x2 = [[1, 0], [0, 1]]
    let rhs = vec![1.0f32, 0.0, 0.0, 1.0];

    let lhs_buffer = new_buffer(&device, &lhs);
    let rhs_buffer = new_buffer(&device, &rhs);

    let m = 2;
    let k = 2;
    let n = 2;

    let output = device
        .new_buffer(m * n * std::mem::size_of::<f32>(), RESOURCE_OPTIONS)
        .unwrap();

    let metadata = vec![m, k, n, 2, 1, 2, 1, 0, 0];

    call_ops_dot(
        dot::F32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&lhs_buffer),
        BufferOffset::zero_offset(&rhs_buffer),
        &output,
        m,
        n,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let results: Vec<f32> = read_to_vec(&output, m * n);
    // Should return the same as lhs
    assert_eq!(approx(results, 4), vec![1.0, 2.0, 3.0, 4.0]);
}
