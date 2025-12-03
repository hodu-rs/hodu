use hodu_cuda_kernels::{kernel::Kernels, kernels::*};

fn device() -> Arc<cudarc::driver::CudaContext> {
    cudarc::driver::CudaContext::new(0).unwrap()
}

fn kernels() -> Kernels {
    Kernels::new()
}

fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
    let b = 10f32.powi(digits);
    v.iter().map(|t| f32::round(t * b) / b).collect()
}

#[test]
fn reduce_sum_f32_simple() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    // Simple 1D array: [1, 2, 3, 4, 5]
    let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let input_shape = vec![5];
    let reduce_dims = vec![0];
    let keep_dim = false;

    let output_size = 1;
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    // Calculate strides
    let strides = vec![1];
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();

    // Build metadata
    let num_dims = input_shape.len();
    let output_shape = if keep_dim {
        input_shape.clone()
    } else {
        // Remove dimensions that are being reduced
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_dims.contains(i))
            .map(|(_, &size)| size)
            .collect()
    };
    let output_shape_len = output_shape.len();

    let mut metadata = Vec::new();
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0); // offset
    metadata.push(output_shape_len);
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_ops_reduce(sum::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // Expected: 1 + 2 + 3 + 4 + 5 = 15
    assert_eq!(approx(results, 4), vec![15.0]);
}

#[test]
fn reduce_mean_f32() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input = vec![2.0f32, 4.0, 6.0, 8.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let input_shape = vec![4];
    let reduce_dims = vec![0];
    let keep_dim = false;

    let output_size = 1;
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    let strides = vec![1];
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();

    let num_dims = input_shape.len();
    let output_shape = if keep_dim {
        input_shape.clone()
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_dims.contains(i))
            .map(|(_, &size)| size)
            .collect()
    };
    let output_shape_len = output_shape.len();

    let mut metadata = Vec::new();
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape_len);
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_ops_reduce(mean::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // Expected: (2 + 4 + 6 + 8) / 4 = 5
    assert_eq!(approx(results, 4), vec![5.0]);
}

#[test]
fn reduce_max_f32() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input = vec![3.0f32, 7.0, 2.0, 9.0, 1.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let input_shape = vec![5];
    let reduce_dims = vec![0];
    let keep_dim = false;

    let output_size = 1;
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    let strides = vec![1];
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();

    let num_dims = input_shape.len();
    let output_shape = if keep_dim {
        input_shape.clone()
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_dims.contains(i))
            .map(|(_, &size)| size)
            .collect()
    };
    let output_shape_len = output_shape.len();

    let mut metadata = Vec::new();
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape_len);
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_ops_reduce(max::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(approx(results, 4), vec![9.0]);
}

#[test]
fn reduce_min_f32() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input = vec![3.0f32, 7.0, 2.0, 9.0, 1.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let input_shape = vec![5];
    let reduce_dims = vec![0];
    let keep_dim = false;

    let output_size = 1;
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    let strides = vec![1];
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();

    let num_dims = input_shape.len();
    let output_shape = if keep_dim {
        input_shape.clone()
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_dims.contains(i))
            .map(|(_, &size)| size)
            .collect()
    };
    let output_shape_len = output_shape.len();

    let mut metadata = Vec::new();
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape_len);
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_ops_reduce(min::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    assert_eq!(approx(results, 4), vec![1.0]);
}

#[test]
fn reduce_prod_f32() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input = vec![2.0f32, 3.0, 4.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let input_shape = vec![3];
    let reduce_dims = vec![0];
    let keep_dim = false;

    let output_size = 1;
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    let strides = vec![1];
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();

    let num_dims = input_shape.len();
    let output_shape = if keep_dim {
        input_shape.clone()
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_dims.contains(i))
            .map(|(_, &size)| size)
            .collect()
    };
    let output_shape_len = output_shape.len();

    let mut metadata = Vec::new();
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape_len);
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_ops_reduce(prod::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // Expected: 2 * 3 * 4 = 24
    assert_eq!(approx(results, 4), vec![24.0]);
}

#[test]
fn reduce_norm_f32() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input = vec![3.0f32, 4.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let input_shape = vec![2];
    let reduce_dims = vec![0];
    let keep_dim = false;

    let output_size = 1;
    let mut output: cudarc::driver::CudaSlice<f32> = unsafe { stream.alloc(output_size).unwrap() };

    let strides = vec![1];
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();

    let num_dims = input_shape.len();
    let output_shape = if keep_dim {
        input_shape.clone()
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_dims.contains(i))
            .map(|(_, &size)| size)
            .collect()
    };
    let output_shape_len = output_shape.len();

    let mut metadata = Vec::new();
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape_len);
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_ops_reduce(norm::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0.0f32; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // Expected: sqrt(3^2 + 4^2) = sqrt(25) = 5.0
    assert_eq!(approx(results, 4), vec![5.0]);
}

#[test]
fn reduce_argmax_f32() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input = vec![3.0f32, 7.0, 2.0, 9.0, 1.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let input_shape = vec![5];
    let reduce_dims = vec![0];
    let keep_dim = false;

    let output_size = 1;
    let mut output: cudarc::driver::CudaSlice<i32> = unsafe { stream.alloc(output_size).unwrap() };

    let strides = vec![1];
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();

    let num_dims = input_shape.len();
    let output_shape = if keep_dim {
        input_shape.clone()
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_dims.contains(i))
            .map(|(_, &size)| size)
            .collect()
    };
    let output_shape_len = output_shape.len();

    let mut metadata = Vec::new();
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape_len);
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_ops_reduce(argmax::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0i32; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // Expected: index of max value (9.0) is 3
    assert_eq!(results, vec![3]);
}

#[test]
fn reduce_argmin_f32() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input = vec![3.0f32, 7.0, 2.0, 9.0, 1.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let input_shape = vec![5];
    let reduce_dims = vec![0];
    let keep_dim = false;

    let output_size = 1;
    let mut output: cudarc::driver::CudaSlice<i32> = unsafe { stream.alloc(output_size).unwrap() };

    let strides = vec![1];
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();

    let num_dims = input_shape.len();
    let output_shape = if keep_dim {
        input_shape.clone()
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_dims.contains(i))
            .map(|(_, &size)| size)
            .collect()
    };
    let output_shape_len = output_shape.len();

    let mut metadata = Vec::new();
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape_len);
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_ops_reduce(argmin::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![0i32; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // Expected: index of min value (1.0) is 4
    assert_eq!(results, vec![4]);
}

#[test]
fn reduce_any_f32() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input = vec![0.0f32, 0.0, 1.0, 0.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let input_shape = vec![4];
    let reduce_dims = vec![0];
    let keep_dim = false;

    let output_size = 1;
    let mut output: cudarc::driver::CudaSlice<bool> = unsafe { stream.alloc(output_size).unwrap() };

    let strides = vec![1];
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();

    let num_dims = input_shape.len();
    let output_shape = if keep_dim {
        input_shape.clone()
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_dims.contains(i))
            .map(|(_, &size)| size)
            .collect()
    };
    let output_shape_len = output_shape.len();

    let mut metadata = Vec::new();
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape_len);
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_ops_reduce(any::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![false; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // Expected: true (at least one non-zero value)
    assert_eq!(results, vec![true]);
}

#[test]
fn reduce_all_f32() {
    let kernels = kernels();

    let device = device();
    let stream = device.default_stream();

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_dev = stream.memcpy_stod(&input).unwrap();

    let input_shape = vec![4];
    let reduce_dims = vec![0];
    let keep_dim = false;

    let output_size = 1;
    let mut output: cudarc::driver::CudaSlice<bool> = unsafe { stream.alloc(output_size).unwrap() };

    let strides = vec![1];
    let reduce_size: usize = reduce_dims.iter().map(|&d| input_shape[d]).product();

    let num_dims = input_shape.len();
    let output_shape = if keep_dim {
        input_shape.clone()
    } else {
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_dims.contains(i))
            .map(|(_, &size)| size)
            .collect()
    };
    let output_shape_len = output_shape.len();

    let mut metadata = Vec::new();
    metadata.push(num_dims);
    metadata.extend(&input_shape);
    metadata.extend(&strides);
    metadata.push(0);
    metadata.push(output_shape_len);
    metadata.extend(&output_shape);
    metadata.push(reduce_dims.len());
    metadata.extend(&reduce_dims);
    metadata.push(if keep_dim { 1 } else { 0 });
    metadata.push(reduce_size);

    call_ops_reduce(all::F32, &kernels, &device, &input_dev, &mut output, &metadata).unwrap();

    let mut results = vec![false; output_size];
    stream.memcpy_dtoh(&output, &mut results).unwrap();
    // Expected: true (all values are non-zero)
    assert_eq!(results, vec![true]);
}
