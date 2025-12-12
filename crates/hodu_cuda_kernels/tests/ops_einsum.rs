use hodu_cuda_kernels::{kernel::Kernels, kernels::*};

fn device() -> std::sync::Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn kernels() -> Kernels {
    Kernels::new()
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

// Build einsum metadata
// Header: [num_output_els, num_inputs, num_total_indices, num_contraction_indices, output_ndim, output_shape...]
// Per-input: [input_ndim, input_shape..., input_strides..., input_offset, dim_to_index_map...]
// Index info: [contraction_index_ids..., index_sizes..., output_index_ids...]
fn build_einsum_metadata(
    num_inputs: usize,
    input_shapes: &[&[usize]],
    input_strides: &[&[usize]],
    input_offsets: &[usize],
    output_shape: &[usize],
    all_indices: &[char],
    input_subscripts: &[&[char]],
    output_subscripts: &[char],
    contraction_indices: &[char],
) -> Vec<usize> {
    let num_output_els: usize = output_shape.iter().product::<usize>().max(1);
    let num_total_indices = all_indices.len();
    let num_contraction_indices = contraction_indices.len();
    let output_ndim = output_shape.len();

    let mut metadata = vec![
        num_output_els,
        num_inputs,
        num_total_indices,
        num_contraction_indices,
        output_ndim,
    ];
    metadata.extend(output_shape);

    // Per-input sections
    for i in 0..num_inputs {
        let input_ndim = input_shapes[i].len();
        metadata.push(input_ndim);
        metadata.extend(input_shapes[i]);
        metadata.extend(input_strides[i]);
        metadata.push(input_offsets[i]);

        // dim_to_index_map: for each dim, which index id does it map to
        for &subscript_char in input_subscripts[i] {
            let idx_id = all_indices.iter().position(|&x| x == subscript_char).unwrap();
            metadata.push(idx_id);
        }
    }

    // Contraction index ids
    for c in contraction_indices {
        let id = all_indices.iter().position(|&x| x == *c).unwrap();
        metadata.push(id);
    }

    // Index sizes
    for idx_char in all_indices {
        let mut size = 1usize;
        for (i, subs) in input_subscripts.iter().enumerate() {
            if let Some(pos) = subs.iter().position(|&c| c == *idx_char) {
                size = input_shapes[i][pos];
                break;
            }
        }
        metadata.push(size);
    }

    // Output index ids
    for c in output_subscripts {
        let id = all_indices.iter().position(|&x| x == *c).unwrap();
        metadata.push(id);
    }

    metadata
}

// Test: matrix multiplication "ij,jk->ik"
#[test]
fn test_einsum_matmul_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let input0_dev = stream.memcpy_stod(&a).unwrap();
    let input1_dev = stream.memcpy_stod(&b).unwrap();

    let a_shape = [2usize, 3];
    let b_shape = [3usize, 4];
    let output_shape = [2usize, 4];
    let a_strides = calculate_strides(&a_shape);
    let b_strides = calculate_strides(&b_shape);

    let metadata = build_einsum_metadata(
        2,
        &[&a_shape, &b_shape],
        &[&a_strides, &b_strides],
        &[0, 0],
        &output_shape,
        &['i', 'j', 'k'],
        &[&['i', 'j'], &['j', 'k']],
        &['i', 'k'],
        &['j'],
    );

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(8).unwrap() };

    call_ops_einsum(
        einsum::F32,
        &kernels,
        &device,
        &[&input0_dev, &input1_dev],
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![0.0f32; 8];
    stream.memcpy_dtoh(&output, &mut results).unwrap();

    let expected = vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
    assert_eq!(results, expected);
}

// Test: transpose "ij->ji"
#[test]
fn test_einsum_transpose_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input0_dev = stream.memcpy_stod(&a).unwrap();

    let a_shape = [2usize, 3];
    let output_shape = [3usize, 2];
    let a_strides = calculate_strides(&a_shape);

    let metadata = build_einsum_metadata(
        1,
        &[&a_shape],
        &[&a_strides],
        &[0],
        &output_shape,
        &['i', 'j'],
        &[&['i', 'j']],
        &['j', 'i'],
        &[],
    );

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(6).unwrap() };

    call_ops_einsum(einsum::F32, &kernels, &device, &[&input0_dev], &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 6];
    stream.memcpy_dtoh(&output, &mut results).unwrap();

    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_eq!(results, expected);
}

// Test: trace "ii->" (diagonal sum)
#[test]
fn test_einsum_trace_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let input0_dev = stream.memcpy_stod(&a).unwrap();

    let a_shape = [3usize, 3];
    let output_shape: [usize; 0] = [];
    let a_strides = calculate_strides(&a_shape);

    let metadata = build_einsum_metadata(
        1,
        &[&a_shape],
        &[&a_strides],
        &[0],
        &output_shape,
        &['i'],
        &[&['i', 'i']],
        &[],
        &['i'],
    );

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(1).unwrap() };

    call_ops_einsum(einsum::F32, &kernels, &device, &[&input0_dev], &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; 1];
    stream.memcpy_dtoh(&output, &mut results).unwrap();

    assert_eq!(results, vec![15.0]);
}

// Test: outer product "i,j->ij"
#[test]
fn test_einsum_outer_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0];

    let input0_dev = stream.memcpy_stod(&a).unwrap();
    let input1_dev = stream.memcpy_stod(&b).unwrap();

    let a_shape = [3usize];
    let b_shape = [2usize];
    let output_shape = [3usize, 2];
    let a_strides = calculate_strides(&a_shape);
    let b_strides = calculate_strides(&b_shape);

    let metadata = build_einsum_metadata(
        2,
        &[&a_shape, &b_shape],
        &[&a_strides, &b_strides],
        &[0, 0],
        &output_shape,
        &['i', 'j'],
        &[&['i'], &['j']],
        &['i', 'j'],
        &[],
    );

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(6).unwrap() };

    call_ops_einsum(
        einsum::F32,
        &kernels,
        &device,
        &[&input0_dev, &input1_dev],
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![0.0f32; 6];
    stream.memcpy_dtoh(&output, &mut results).unwrap();

    let expected = vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0];
    assert_eq!(results, expected);
}

// Test: dot product "i,i->"
#[test]
fn test_einsum_dot_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0, 6.0];

    let input0_dev = stream.memcpy_stod(&a).unwrap();
    let input1_dev = stream.memcpy_stod(&b).unwrap();

    let a_shape = [3usize];
    let b_shape = [3usize];
    let output_shape: [usize; 0] = [];
    let a_strides = calculate_strides(&a_shape);
    let b_strides = calculate_strides(&b_shape);

    let metadata = build_einsum_metadata(
        2,
        &[&a_shape, &b_shape],
        &[&a_strides, &b_strides],
        &[0, 0],
        &output_shape,
        &['i'],
        &[&['i'], &['i']],
        &[],
        &['i'],
    );

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(1).unwrap() };

    call_ops_einsum(
        einsum::F32,
        &kernels,
        &device,
        &[&input0_dev, &input1_dev],
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![0.0f32; 1];
    stream.memcpy_dtoh(&output, &mut results).unwrap();

    assert_eq!(results, vec![32.0]);
}

// Test: batch matrix multiplication "bij,bjk->bik"
#[test]
fn test_einsum_batch_matmul_f32() {
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let a: Vec<f32> = vec![
        // batch 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 1
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let b: Vec<f32> = vec![
        // batch 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 1
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    let input0_dev = stream.memcpy_stod(&a).unwrap();
    let input1_dev = stream.memcpy_stod(&b).unwrap();

    let a_shape = [2usize, 2, 3];
    let b_shape = [2usize, 3, 2];
    let output_shape = [2usize, 2, 2];
    let a_strides = calculate_strides(&a_shape);
    let b_strides = calculate_strides(&b_shape);

    let metadata = build_einsum_metadata(
        2,
        &[&a_shape, &b_shape],
        &[&a_strides, &b_strides],
        &[0, 0],
        &output_shape,
        &['b', 'i', 'j', 'k'],
        &[&['b', 'i', 'j'], &['b', 'j', 'k']],
        &['b', 'i', 'k'],
        &['j'],
    );

    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(8).unwrap() };

    call_ops_einsum(
        einsum::F32,
        &kernels,
        &device,
        &[&input0_dev, &input1_dev],
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![0.0f32; 8];
    stream.memcpy_dtoh(&output, &mut results).unwrap();

    let expected = vec![22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0];
    assert_eq!(results, expected);
}
