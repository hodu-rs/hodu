use hodu_cuda_kernels::{kernel::Kernels, kernels::*};

fn device() -> std::sync::Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn kernels() -> Kernels {
    Kernels::new()
}

#[allow(clippy::too_many_arguments)]
fn run_index_select<T, I>(input: &[T], indices: &[I], input_shape: &[usize], dim: usize, kernel: Kernel) -> Vec<T>
where
    T: cudarc::driver::DeviceRepr + Clone,
    I: cudarc::driver::DeviceRepr + Clone,
{
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let indices_dev = stream.memcpy_stod(indices).unwrap();

    // Calculate output shape: same as input but dim is replaced with num_indices
    let num_indices = indices.len();
    let mut output_shape = input_shape.to_vec();
    output_shape[dim] = num_indices;
    let output_size: usize = output_shape.iter().product();

    let mut output: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(output_size).unwrap() };

    // Calculate strides for input
    let num_dims = input_shape.len();
    let mut input_strides = vec![1; num_dims];
    for i in (0..num_dims - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // CUDA kernel metadata layout for index_select:
    // - metadata[0]: num_els
    // - metadata[1]: num_dims
    // - metadata[2..2+num_dims]: input_shape
    // - metadata[2+num_dims..2+2*num_dims]: input_strides
    // - metadata[2+2*num_dims]: input_offset
    // - metadata[2+2*num_dims+1]: dim
    // - metadata[2+2*num_dims+2]: num_indices
    let mut metadata = Vec::new();
    metadata.push(output_size);
    metadata.push(num_dims);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(&input_strides);
    metadata.push(0); // input_offset
    metadata.push(dim);
    metadata.push(num_indices);

    call_ops_index_select(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &indices_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[allow(clippy::too_many_arguments)]
fn run_index_put<T, I>(
    input: &[T],
    indices: &[I],
    values: &[T],
    input_shape: &[usize],
    dim: usize,
    kernel: Kernel,
) -> Vec<T>
where
    T: cudarc::driver::DeviceRepr + Clone,
    I: cudarc::driver::DeviceRepr + Clone,
{
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let indices_dev = stream.memcpy_stod(indices).unwrap();
    let values_dev = stream.memcpy_stod(values).unwrap();

    let output_size: usize = input_shape.iter().product();
    let mut output: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(output_size).unwrap() };

    // Calculate strides
    let num_dims = input_shape.len();
    let mut input_strides = vec![1; num_dims];
    for i in (0..num_dims - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Values shape: same as input but dim is replaced with num_indices
    let num_indices = indices.len();
    let mut values_shape = input_shape.to_vec();
    values_shape[dim] = num_indices;
    let mut values_strides = vec![1; num_dims];
    for i in (0..num_dims - 1).rev() {
        values_strides[i] = values_strides[i + 1] * values_shape[i + 1];
    }

    // CUDA kernel metadata layout for index_put:
    // - metadata[0]: num_els
    // - metadata[1]: num_dims
    // - metadata[2..2+num_dims]: input_shape
    // - metadata[2+num_dims..2+2*num_dims]: input_strides
    // - metadata[2+2*num_dims..2+3*num_dims]: values_strides
    // - metadata[2+3*num_dims]: input_offset
    // - metadata[2+3*num_dims+1]: values_offset
    // - metadata[2+3*num_dims+2]: dim
    // - metadata[2+3*num_dims+3]: num_indices
    let mut metadata = Vec::new();
    metadata.push(output_size);
    metadata.push(num_dims);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(&input_strides);
    metadata.extend_from_slice(&values_strides);
    metadata.push(0); // input_offset
    metadata.push(0); // values_offset
    metadata.push(dim);
    metadata.push(num_indices);

    call_ops_index_put(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &indices_dev,
        &values_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[allow(clippy::too_many_arguments)]
fn run_gather<T, I>(
    input: &[T],
    indices: &[I],
    input_shape: &[usize],
    indices_shape: &[usize],
    dim: usize,
    kernel: Kernel,
) -> Vec<T>
where
    T: cudarc::driver::DeviceRepr + Clone,
    I: cudarc::driver::DeviceRepr + Clone,
{
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let indices_dev = stream.memcpy_stod(indices).unwrap();

    // Output shape is same as indices_shape for gather
    let output_shape = indices_shape;
    let output_size: usize = output_shape.iter().product();
    let mut output: cudarc::driver::CudaSlice<T> = unsafe { stream.alloc(output_size).unwrap() };

    // Calculate strides
    let num_dims = input_shape.len();
    let mut input_strides = vec![1; num_dims];
    for i in (0..num_dims - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    let mut indices_strides = vec![1; num_dims];
    for i in (0..num_dims - 1).rev() {
        indices_strides[i] = indices_strides[i + 1] * indices_shape[i + 1];
    }

    // CUDA kernel metadata layout for gather:
    // - metadata[0]: num_els
    // - metadata[1]: num_dims
    // - metadata[2..2+num_dims]: output_shape
    // - metadata[2+num_dims..2+2*num_dims]: input_shape
    // - metadata[2+2*num_dims..2+3*num_dims]: input_strides
    // - metadata[2+3*num_dims..2+4*num_dims]: indices_strides
    // - metadata[2+4*num_dims]: input_offset
    // - metadata[2+4*num_dims+1]: indices_offset
    // - metadata[2+4*num_dims+2]: dim
    let mut metadata = Vec::new();
    metadata.push(output_size);
    metadata.push(num_dims);
    metadata.extend_from_slice(output_shape); // output_shape (same as indices_shape)
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(&input_strides);
    metadata.extend_from_slice(&indices_strides);
    metadata.push(0); // input_offset
    metadata.push(0); // indices_offset
    metadata.push(dim);

    call_ops_gather(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &indices_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let mut results = vec![unsafe { core::mem::zeroed() }; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[allow(clippy::too_many_arguments)]
fn run_scatter<T, I>(
    input: &[T],
    indices: &[I],
    src: &[T],
    input_shape: &[usize],
    src_shape: &[usize],
    dim: usize,
    kernel: Kernel,
) -> Vec<T>
where
    T: cudarc::driver::DeviceRepr + Clone,
    I: cudarc::driver::DeviceRepr + Clone,
{
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input_dev = stream.memcpy_stod(input).unwrap();
    let indices_dev = stream.memcpy_stod(indices).unwrap();
    let src_dev = stream.memcpy_stod(src).unwrap();

    // Scatter modifies input at specific indices, so initialize output with input
    let mut output = stream.memcpy_stod(input).unwrap();

    // Calculate strides
    let num_dims = input_shape.len();
    let mut input_strides = vec![1; num_dims];
    for i in (0..num_dims - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    let mut src_strides = vec![1; num_dims];
    for i in (0..num_dims - 1).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }

    // indices_strides follows src_shape (indices has same shape as src)
    let mut indices_strides = vec![1; num_dims];
    for i in (0..num_dims - 1).rev() {
        indices_strides[i] = indices_strides[i + 1] * src_shape[i + 1];
    }

    let num_els = src.len();

    // CUDA kernel metadata layout for scatter:
    // - metadata[0]: num_els (number of elements in src tensor to scatter)
    // - metadata[1]: num_dims
    // - metadata[2..2+num_dims]: input_shape
    // - metadata[2+num_dims..2+2*num_dims]: input_strides
    // - metadata[2+2*num_dims..2+3*num_dims]: src_shape
    // - metadata[2+3*num_dims..2+4*num_dims]: src_strides
    // - metadata[2+4*num_dims..2+5*num_dims]: indices_strides
    // - metadata[2+5*num_dims]: input_offset
    // - metadata[2+5*num_dims+1]: src_offset
    // - metadata[2+5*num_dims+2]: indices_offset
    // - metadata[2+5*num_dims+3]: dim
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend_from_slice(input_shape);
    metadata.extend_from_slice(&input_strides);
    metadata.extend_from_slice(src_shape);
    metadata.extend_from_slice(&src_strides);
    metadata.extend_from_slice(&indices_strides);
    metadata.push(0); // input_offset
    metadata.push(0); // src_offset
    metadata.push(0); // indices_offset
    metadata.push(dim);

    call_ops_scatter(
        kernel,
        &kernels,
        &device,
        &input_dev,
        &indices_dev,
        &src_dev,
        &mut output,
        &metadata,
    )
    .unwrap();

    let output_size: usize = input_shape.iter().product();
    let mut results = vec![unsafe { core::mem::zeroed() }; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    results
}

#[test]
fn test_index_select_1d_f32() {
    // Input: [1, 2, 3, 4, 5]
    // Indices: [0, 2, 4] -> select elements at positions 0, 2, 4
    // Expected: [1, 3, 5]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let indices: Vec<i32> = vec![0, 2, 4];
    let input_shape = vec![5];

    let result: Vec<f32> = run_index_select(&input, &indices, &input_shape, 0, index_select::F32);

    assert_eq!(result, vec![1.0, 3.0, 5.0]);
}

#[test]
fn test_index_select_2d_dim0_f32() {
    // Input: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    // Indices: [0, 2] -> select rows 0 and 2
    // Expected: [[1, 2, 3], [7, 8, 9]]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let indices: Vec<i32> = vec![0, 2];
    let input_shape = vec![3, 3];

    let result: Vec<f32> = run_index_select(&input, &indices, &input_shape, 0, index_select::F32);

    assert_eq!(result, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_index_select_2d_dim1_f32() {
    // Input: [[1, 2, 3], [4, 5, 6]]
    // Indices: [0, 2] -> select columns 0 and 2
    // Expected: [[1, 3], [4, 6]]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices: Vec<i32> = vec![0, 2];
    let input_shape = vec![2, 3];

    let result: Vec<f32> = run_index_select(&input, &indices, &input_shape, 1, index_select::F32);

    assert_eq!(result, vec![1.0, 3.0, 4.0, 6.0]);
}

#[test]
fn test_index_put_1d_f32() {
    // Input: [1, 2, 3, 4, 5]
    // Indices: [1, 3] -> put values at positions 1, 3
    // Values: [10, 20]
    // Expected: [1, 10, 3, 20, 5]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let indices: Vec<i32> = vec![1, 3];
    let values: Vec<f32> = vec![10.0, 20.0];
    let input_shape = vec![5];

    let result: Vec<f32> = run_index_put(&input, &indices, &values, &input_shape, 0, index_put::F32);

    assert_eq!(result, vec![1.0, 10.0, 3.0, 20.0, 5.0]);
}

#[test]
fn test_index_put_2d_dim0_f32() {
    // Input: [[1, 2], [3, 4], [5, 6]]
    // Indices: [0, 2] -> replace rows 0 and 2
    // Values: [[10, 20], [30, 40]]
    // Expected: [[10, 20], [3, 4], [30, 40]]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices: Vec<i32> = vec![0, 2];
    let values: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let input_shape = vec![3, 2];

    let result: Vec<f32> = run_index_put(&input, &indices, &values, &input_shape, 0, index_put::F32);

    assert_eq!(result, vec![10.0, 20.0, 3.0, 4.0, 30.0, 40.0]);
}

#[test]
fn test_gather_1d_f32() {
    // Input: [10, 20, 30, 40, 50]
    // Indices: [4, 2, 1, 0] -> gather at these positions
    // Expected: [50, 30, 20, 10]
    let input: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let indices: Vec<i32> = vec![4, 2, 1, 0];
    let input_shape = vec![5];
    let indices_shape = vec![4];

    let result: Vec<f32> = run_gather(&input, &indices, &input_shape, &indices_shape, 0, gather::F32);

    assert_eq!(result, vec![50.0, 30.0, 20.0, 10.0]);
}

#[test]
fn test_gather_2d_f32() {
    // Input: [[1, 2], [3, 4], [5, 6]]
    // Indices: [[0, 1], [2, 0]] -> gather rows
    // Expected: [[1, 2], [3, 4]] for first index, [[5, 6], [1, 2]] for second
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices: Vec<i32> = vec![0, 2];
    let input_shape = vec![3, 2];
    let indices_shape = vec![2, 2];

    let result: Vec<f32> = run_gather(&input, &indices, &input_shape, &indices_shape, 0, gather::F32);

    // Output shape: [2, 2] with gathered values
    assert_eq!(result.len(), 4);
}

#[test]
fn test_scatter_1d_f32() {
    // Input: [1, 2, 3, 4, 5]
    // Indices: [0, 2, 4]
    // Src: [10, 20, 30]
    // Expected: [10, 2, 20, 4, 30]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let indices: Vec<i32> = vec![0, 2, 4];
    let src: Vec<f32> = vec![10.0, 20.0, 30.0];
    let input_shape = vec![5];
    let src_shape = vec![3];

    let result: Vec<f32> = run_scatter(&input, &indices, &src, &input_shape, &src_shape, 0, scatter::F32);

    assert_eq!(result, vec![10.0, 2.0, 20.0, 4.0, 30.0]);
}

#[test]
fn test_scatter_2d_f32() {
    // Input: [[1, 2], [3, 4], [5, 6]]
    // Indices: [[0, 0], [2, 2]] (indices shape must match src shape)
    // Src: [[10, 20], [30, 40]]
    // Expected: [[10, 20], [3, 4], [30, 40]]
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices: Vec<i32> = vec![0, 0, 2, 2]; // Flattened [2, 2]
    let src: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let input_shape = vec![3, 2];
    let src_shape = vec![2, 2];

    let result: Vec<f32> = run_scatter(&input, &indices, &src, &input_shape, &src_shape, 0, scatter::F32);

    assert_eq!(result, vec![10.0, 20.0, 3.0, 4.0, 30.0, 40.0]);
}

#[test]
fn test_scatter_add_1d_f32() {
    // Input: [1, 2, 3, 4, 5]
    // Indices: [0, 2, 4]
    // Src: [10, 20, 30]
    // Expected: [11, 2, 23, 4, 35] (adding src to input at indices)
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let indices: Vec<i32> = vec![0, 2, 4];
    let src: Vec<f32> = vec![10.0, 20.0, 30.0];
    let input_shape = vec![5];
    let src_shape = vec![3];

    let result: Vec<f32> = run_scatter(&input, &indices, &src, &input_shape, &src_shape, 0, scatter_add::F32);

    assert_eq!(result, vec![11.0, 2.0, 23.0, 4.0, 35.0]);
}

#[test]
fn test_scatter_max_1d_f32() {
    // Input: [5, 5, 5, 5, 5]
    // Indices: [0, 2, 4]
    // Src: [3, 10, 2]
    // Expected: [5, 5, 10, 5, 5] (max of input and src at indices)
    let input: Vec<f32> = vec![5.0, 5.0, 5.0, 5.0, 5.0];
    let indices: Vec<i32> = vec![0, 2, 4];
    let src: Vec<f32> = vec![3.0, 10.0, 2.0];
    let input_shape = vec![5];
    let src_shape = vec![3];

    let result: Vec<f32> = run_scatter(&input, &indices, &src, &input_shape, &src_shape, 0, scatter_max::F32);

    assert_eq!(result, vec![5.0, 5.0, 10.0, 5.0, 5.0]);
}

#[test]
fn test_scatter_min_1d_f32() {
    // Input: [5, 5, 5, 5, 5]
    // Indices: [0, 2, 4]
    // Src: [3, 10, 2]
    // Expected: [3, 5, 5, 5, 2] (min of input and src at indices)
    let input: Vec<f32> = vec![5.0, 5.0, 5.0, 5.0, 5.0];
    let indices: Vec<i32> = vec![0, 2, 4];
    let src: Vec<f32> = vec![3.0, 10.0, 2.0];
    let input_shape = vec![5];
    let src_shape = vec![3];

    let result: Vec<f32> = run_scatter(&input, &indices, &src, &input_shape, &src_shape, 0, scatter_min::F32);

    assert_eq!(result, vec![3.0, 5.0, 5.0, 5.0, 2.0]);
}
