use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_ops_einsum, einsum, Kernel},
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

fn run_einsum_2inputs<T: Clone + Default>(
    input0: &[T],
    input1: &[T],
    input_shapes: &[&[usize]],
    output_shape: &[usize],
    all_indices: &[char],
    input_subscripts: &[&[char]],
    output_subscripts: &[char],
    contraction_indices: &[char],
    kernel: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;

    let input0_buffer = new_buffer(&device, input0);
    let input1_buffer = new_buffer(&device, input1);

    let output_size: usize = output_shape.iter().product::<usize>().max(1);
    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    let strides: Vec<Vec<usize>> = input_shapes.iter().map(|s| calculate_strides(s)).collect();
    let stride_refs: Vec<&[usize]> = strides.iter().map(|s| s.as_slice()).collect();

    let metadata = build_einsum_metadata(
        2,
        input_shapes,
        &stride_refs,
        &[0, 0],
        output_shape,
        all_indices,
        input_subscripts,
        output_subscripts,
        contraction_indices,
    );

    let inputs = [
        BufferOffset::zero_offset(&input0_buffer),
        BufferOffset::zero_offset(&input1_buffer),
    ];

    call_ops_einsum(kernel, &kernels, &device, &command_buffer, &inputs, &output, &metadata).unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, output_size)
}

fn run_einsum_1input<T: Clone + Default>(
    input0: &[T],
    input_shape: &[usize],
    output_shape: &[usize],
    all_indices: &[char],
    input_subscripts: &[&[char]],
    output_subscripts: &[char],
    contraction_indices: &[char],
    kernel: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;

    let input0_buffer = new_buffer(&device, input0);

    let output_size: usize = output_shape.iter().product::<usize>().max(1);
    let output = device
        .new_buffer(output_size * std::mem::size_of::<T>(), options)
        .unwrap();

    let strides = calculate_strides(input_shape);

    let metadata = build_einsum_metadata(
        1,
        &[input_shape],
        &[&strides],
        &[0],
        output_shape,
        all_indices,
        input_subscripts,
        output_subscripts,
        contraction_indices,
    );

    let inputs = [BufferOffset::zero_offset(&input0_buffer)];

    call_ops_einsum(kernel, &kernels, &device, &command_buffer, &inputs, &output, &metadata).unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    read_to_vec(&output, output_size)
}

// Test: matrix multiplication "ij,jk->ik"
#[test]
fn test_einsum_matmul_f32() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let result = run_einsum_2inputs(
        &a,
        &b,
        &[&[2, 3], &[3, 4]],
        &[2, 4],
        &['i', 'j', 'k'],
        &[&['i', 'j'], &['j', 'k']],
        &['i', 'k'],
        &['j'],
        einsum::F32,
    );

    let expected = vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
    assert_eq!(result, expected);
}

// Test: transpose "ij->ji"
#[test]
fn test_einsum_transpose_f32() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let result = run_einsum_1input(
        &a,
        &[2, 3],
        &[3, 2],
        &['i', 'j'],
        &[&['i', 'j']],
        &['j', 'i'],
        &[],
        einsum::F32,
    );

    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_eq!(result, expected);
}

// Test: trace "ii->" (diagonal sum)
#[test]
fn test_einsum_trace_f32() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    let result = run_einsum_1input(&a, &[3, 3], &[], &['i'], &[&['i', 'i']], &[], &['i'], einsum::F32);

    assert_eq!(result, vec![15.0]);
}

// Test: outer product "i,j->ij"
#[test]
fn test_einsum_outer_f32() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0];

    let result = run_einsum_2inputs(
        &a,
        &b,
        &[&[3], &[2]],
        &[3, 2],
        &['i', 'j'],
        &[&['i'], &['j']],
        &['i', 'j'],
        &[],
        einsum::F32,
    );

    let expected = vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0];
    assert_eq!(result, expected);
}

// Test: dot product "i,i->"
#[test]
fn test_einsum_dot_f32() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0, 6.0];

    let result = run_einsum_2inputs(
        &a,
        &b,
        &[&[3], &[3]],
        &[],
        &['i'],
        &[&['i'], &['i']],
        &[],
        &['i'],
        einsum::F32,
    );

    assert_eq!(result, vec![32.0]);
}

// Test: batch matrix multiplication "bij,bjk->bik"
#[test]
fn test_einsum_batch_matmul_f32() {
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

    let result = run_einsum_2inputs(
        &a,
        &b,
        &[&[2, 2, 3], &[2, 3, 2]],
        &[2, 2, 2],
        &['b', 'i', 'j', 'k'],
        &[&['b', 'i', 'j'], &['b', 'j', 'k']],
        &['b', 'i', 'k'],
        &['j'],
        einsum::F32,
    );

    let expected = vec![22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0];
    assert_eq!(result, expected);
}
