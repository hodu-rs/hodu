use hodu_cpu_kernels::*;

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
    // A: [2, 3], B: [3, 4] -> C: [2, 4]
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut output = vec![0.0f32; 8];

    let a_shape = [2usize, 3];
    let b_shape = [3usize, 4];
    let output_shape = [2usize, 4];
    let a_strides = calculate_strides(&a_shape);
    let b_strides = calculate_strides(&b_shape);

    // "ij,jk->ik"
    let all_indices = ['i', 'j', 'k'];
    let input_subscripts: [&[char]; 2] = [&['i', 'j'], &['j', 'k']];
    let output_subscripts = ['i', 'k'];
    let contraction_indices = ['j'];

    let metadata = build_einsum_metadata(
        2,
        &[&a_shape, &b_shape],
        &[&a_strides, &b_strides],
        &[0, 0],
        &output_shape,
        &all_indices,
        &input_subscripts,
        &output_subscripts,
        &contraction_indices,
    );

    let inputs: [*const core::ffi::c_void; 2] = [
        a.as_ptr() as *const core::ffi::c_void,
        b.as_ptr() as *const core::ffi::c_void,
    ];

    call_ops_einsum(
        einsum::F32,
        &inputs,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Expected: standard matrix multiplication
    // C[0,0] = 1*1 + 2*5 + 3*9 = 1 + 10 + 27 = 38
    // C[0,1] = 1*2 + 2*6 + 3*10 = 2 + 12 + 30 = 44
    // C[0,2] = 1*3 + 2*7 + 3*11 = 3 + 14 + 33 = 50
    // C[0,3] = 1*4 + 2*8 + 3*12 = 4 + 16 + 36 = 56
    // C[1,0] = 4*1 + 5*5 + 6*9 = 4 + 25 + 54 = 83
    // C[1,1] = 4*2 + 5*6 + 6*10 = 8 + 30 + 60 = 98
    // C[1,2] = 4*3 + 5*7 + 6*11 = 12 + 35 + 66 = 113
    // C[1,3] = 4*4 + 5*8 + 6*12 = 16 + 40 + 72 = 128
    let expected = vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
    assert_eq!(output, expected);
}

// Test: transpose "ij->ji"
#[test]
fn test_einsum_transpose_f32() {
    // A: [2, 3] -> B: [3, 2]
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = vec![0.0f32; 6];

    let a_shape = [2usize, 3];
    let output_shape = [3usize, 2];
    let a_strides = calculate_strides(&a_shape);

    // "ij->ji"
    let all_indices = ['i', 'j'];
    let input_subscripts: [&[char]; 1] = [&['i', 'j']];
    let output_subscripts = ['j', 'i'];
    let contraction_indices: [char; 0] = [];

    let metadata = build_einsum_metadata(
        1,
        &[&a_shape],
        &[&a_strides],
        &[0],
        &output_shape,
        &all_indices,
        &input_subscripts,
        &output_subscripts,
        &contraction_indices,
    );

    let inputs: [*const core::ffi::c_void; 1] = [a.as_ptr() as *const core::ffi::c_void];

    call_ops_einsum(
        einsum::F32,
        &inputs,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Input: [[1, 2, 3], [4, 5, 6]]
    // Transpose: [[1, 4], [2, 5], [3, 6]]
    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_eq!(output, expected);
}

// Test: trace "ii->" (diagonal sum)
#[test]
fn test_einsum_trace_f32() {
    // A: [3, 3] -> scalar
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut output = vec![0.0f32; 1];

    let a_shape = [3usize, 3];
    let output_shape: [usize; 0] = [];
    let a_strides = calculate_strides(&a_shape);

    // "ii->" (sum of diagonal)
    let all_indices = ['i'];
    let input_subscripts: [&[char]; 1] = [&['i', 'i']];
    let output_subscripts: [char; 0] = [];
    let contraction_indices = ['i'];

    let metadata = build_einsum_metadata(
        1,
        &[&a_shape],
        &[&a_strides],
        &[0],
        &output_shape,
        &all_indices,
        &input_subscripts,
        &output_subscripts,
        &contraction_indices,
    );

    let inputs: [*const core::ffi::c_void; 1] = [a.as_ptr() as *const core::ffi::c_void];

    call_ops_einsum(
        einsum::F32,
        &inputs,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Trace = 1 + 5 + 9 = 15
    assert_eq!(output, vec![15.0]);
}

// Test: outer product "i,j->ij"
#[test]
fn test_einsum_outer_f32() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0];
    let mut output = vec![0.0f32; 6];

    let a_shape = [3usize];
    let b_shape = [2usize];
    let output_shape = [3usize, 2];
    let a_strides = calculate_strides(&a_shape);
    let b_strides = calculate_strides(&b_shape);

    // "i,j->ij"
    let all_indices = ['i', 'j'];
    let input_subscripts: [&[char]; 2] = [&['i'], &['j']];
    let output_subscripts = ['i', 'j'];
    let contraction_indices: [char; 0] = [];

    let metadata = build_einsum_metadata(
        2,
        &[&a_shape, &b_shape],
        &[&a_strides, &b_strides],
        &[0, 0],
        &output_shape,
        &all_indices,
        &input_subscripts,
        &output_subscripts,
        &contraction_indices,
    );

    let inputs: [*const core::ffi::c_void; 2] = [
        a.as_ptr() as *const core::ffi::c_void,
        b.as_ptr() as *const core::ffi::c_void,
    ];

    call_ops_einsum(
        einsum::F32,
        &inputs,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Outer product: [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]] = [[4, 5], [8, 10], [12, 15]]
    let expected = vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0];
    assert_eq!(output, expected);
}

// Test: dot product "i,i->"
#[test]
fn test_einsum_dot_f32() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0, 6.0];
    let mut output = vec![0.0f32; 1];

    let a_shape = [3usize];
    let b_shape = [3usize];
    let output_shape: [usize; 0] = [];
    let a_strides = calculate_strides(&a_shape);
    let b_strides = calculate_strides(&b_shape);

    // "i,i->" (dot product)
    let all_indices = ['i'];
    let input_subscripts: [&[char]; 2] = [&['i'], &['i']];
    let output_subscripts: [char; 0] = [];
    let contraction_indices = ['i'];

    let metadata = build_einsum_metadata(
        2,
        &[&a_shape, &b_shape],
        &[&a_strides, &b_strides],
        &[0, 0],
        &output_shape,
        &all_indices,
        &input_subscripts,
        &output_subscripts,
        &contraction_indices,
    );

    let inputs: [*const core::ffi::c_void; 2] = [
        a.as_ptr() as *const core::ffi::c_void,
        b.as_ptr() as *const core::ffi::c_void,
    ];

    call_ops_einsum(
        einsum::F32,
        &inputs,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(output, vec![32.0]);
}

// Test: batch matrix multiplication "bij,bjk->bik"
#[test]
fn test_einsum_batch_matmul_f32() {
    // A: [2, 2, 3], B: [2, 3, 2] -> C: [2, 2, 2]
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
    let mut output = vec![0.0f32; 8];

    let a_shape = [2usize, 2, 3];
    let b_shape = [2usize, 3, 2];
    let output_shape = [2usize, 2, 2];
    let a_strides = calculate_strides(&a_shape);
    let b_strides = calculate_strides(&b_shape);

    // "bij,bjk->bik"
    let all_indices = ['b', 'i', 'j', 'k'];
    let input_subscripts: [&[char]; 2] = [&['b', 'i', 'j'], &['b', 'j', 'k']];
    let output_subscripts = ['b', 'i', 'k'];
    let contraction_indices = ['j'];

    let metadata = build_einsum_metadata(
        2,
        &[&a_shape, &b_shape],
        &[&a_strides, &b_strides],
        &[0, 0],
        &output_shape,
        &all_indices,
        &input_subscripts,
        &output_subscripts,
        &contraction_indices,
    );

    let inputs: [*const core::ffi::c_void; 2] = [
        a.as_ptr() as *const core::ffi::c_void,
        b.as_ptr() as *const core::ffi::c_void,
    ];

    call_ops_einsum(
        einsum::F32,
        &inputs,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]]
    // = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
    // = [[1+6+15, 2+8+18], [4+15+30, 8+20+36]]
    // = [[22, 28], [49, 64]]
    // Batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]]
    // = [[7*7+8*9+9*11, 7*8+8*10+9*12], [10*7+11*9+12*11, 10*8+11*10+12*12]]
    // = [[49+72+99, 56+80+108], [70+99+132, 80+110+144]]
    // = [[220, 244], [301, 334]]
    let expected = vec![22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0];
    assert_eq!(output, expected);
}
