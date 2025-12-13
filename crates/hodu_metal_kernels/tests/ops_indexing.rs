use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_nonzero_count, call_nonzero_fill, call_ops_onehot, nonzero_count, nonzero_fill, onehot, Kernel},
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

fn run_onehot<T: Clone>(
    indices: &[i32],
    num_classes: usize,
    axis: usize,
    output_shape: &[usize],
    kernel: Kernel,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let command_buffer = create_command_buffer(&command_queue).unwrap();
    let options = RESOURCE_OPTIONS;

    let indices_buffer = new_buffer(&device, indices);

    let num_els: usize = output_shape.iter().product();
    let num_input_els = indices.len();
    let num_dims_out = output_shape.len();

    let output = device.new_buffer(num_els * std::mem::size_of::<T>(), options).unwrap();

    // Build metadata
    // - metadata[0]: num_els
    // - metadata[1]: num_input_els
    // - metadata[2]: num_classes
    // - metadata[3]: axis
    // - metadata[4]: num_dims_out
    // - metadata[5..]: output_shape
    let mut metadata = Vec::with_capacity(5 + num_dims_out);
    metadata.push(num_els);
    metadata.push(num_input_els);
    metadata.push(num_classes);
    metadata.push(axis);
    metadata.push(num_dims_out);
    metadata.extend_from_slice(output_shape);

    call_ops_onehot(
        kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&indices_buffer),
        &output,
        &metadata,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();
    read_to_vec(&output, num_els)
}

#[test]
fn test_onehot_f32_1d() {
    // Input indices: [0, 1, 2]
    // num_classes: 4
    // axis: 1 (append at end)
    // Output: [[1,0,0,0], [0,1,0,0], [0,0,1,0]] (3x4)
    let indices = vec![0i32, 1, 2];
    let num_classes = 4;
    let axis = 1;
    let output_shape = vec![3, 4];

    let result: Vec<f32> = run_onehot(&indices, num_classes, axis, &output_shape, onehot::F32);

    assert_eq!(
        result,
        vec![
            1.0, 0.0, 0.0, 0.0, // index 0
            0.0, 1.0, 0.0, 0.0, // index 1
            0.0, 0.0, 1.0, 0.0 // index 2
        ]
    );
}

#[test]
fn test_onehot_f32_axis0() {
    // Input indices: [0, 2]
    // num_classes: 3
    // axis: 0 (prepend at start)
    // Output shape: (3, 2) - num_classes first, then input size
    let indices = vec![0i32, 2];
    let num_classes = 3;
    let axis = 0;
    let output_shape = vec![3, 2];

    let result: Vec<f32> = run_onehot(&indices, num_classes, axis, &output_shape, onehot::F32);

    assert_eq!(
        result,
        vec![
            1.0, 0.0, // class 0: index 0 is 1, index 2 is 0
            0.0, 0.0, // class 1: both are 0
            0.0, 1.0 // class 2: index 0 is 0, index 2 is 1
        ]
    );
}

#[test]
fn test_onehot_i32() {
    // Test with integer output type
    let indices = vec![1i32, 0, 2];
    let num_classes = 3;
    let axis = 1;
    let output_shape = vec![3, 3];

    let result: Vec<i32> = run_onehot(&indices, num_classes, axis, &output_shape, onehot::I32);

    assert_eq!(
        result,
        vec![
            0, 1, 0, // index 1
            1, 0, 0, // index 0
            0, 0, 1 // index 2
        ]
    );
}

#[test]
fn test_onehot_f32_2d_input() {
    // Input indices: [[0, 1], [2, 0]] (2x2)
    // num_classes: 3
    // axis: 2 (append at last dim)
    // Output shape: (2, 2, 3)
    let indices = vec![0i32, 1, 2, 0];
    let num_classes = 3;
    let axis = 2;
    let output_shape = vec![2, 2, 3];

    let result: Vec<f32> = run_onehot(&indices, num_classes, axis, &output_shape, onehot::F32);

    assert_eq!(
        result,
        vec![
            1.0, 0.0, 0.0, // [0,0] = 0
            0.0, 1.0, 0.0, // [0,1] = 1
            0.0, 0.0, 1.0, // [1,0] = 2
            1.0, 0.0, 0.0 // [1,1] = 0
        ]
    );
}

fn run_nonzero<T: Clone>(input: &[T], shape: &[usize], count_kernel: Kernel, fill_kernel: Kernel) -> (usize, Vec<i32>) {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let options = RESOURCE_OPTIONS;

    let input_buffer = new_buffer(&device, input);

    let num_els: usize = shape.iter().product();
    let num_dims = shape.len();

    // Calculate strides
    let mut strides = vec![1usize; num_dims];
    for i in (0..num_dims.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Build metadata
    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend_from_slice(shape);
    metadata.extend_from_slice(&strides);
    metadata.push(0); // offset

    // Count pass
    let count_buffer = device.new_buffer(std::mem::size_of::<u32>(), options).unwrap();
    // Initialize count to 0
    unsafe {
        let ptr = count_buffer.contents() as *mut u32;
        *ptr = 0;
    }

    let command_buffer = create_command_buffer(&command_queue).unwrap();
    call_nonzero_count(
        count_kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &count_buffer,
        &metadata,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let count = unsafe { *(count_buffer.contents() as *const u32) } as usize;

    if count == 0 {
        return (0, vec![]);
    }

    // Fill pass
    let output_buffer = device
        .new_buffer(count * num_dims * std::mem::size_of::<i32>(), options)
        .unwrap();
    let counter_buffer = device.new_buffer(std::mem::size_of::<u32>(), options).unwrap();
    // Initialize counter to 0
    unsafe {
        let ptr = counter_buffer.contents() as *mut u32;
        *ptr = 0;
    }

    let command_buffer = create_command_buffer(&command_queue).unwrap();
    call_nonzero_fill(
        fill_kernel,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &output_buffer,
        &counter_buffer,
        &metadata,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let output = read_to_vec(&output_buffer, count * num_dims);
    (count, output)
}

#[test]
fn test_nonzero_f32_1d() {
    // Input: [0, 1, 0, 2, 0, 3]
    // Non-zero indices: [1, 3, 5]
    let input = vec![0.0f32, 1.0, 0.0, 2.0, 0.0, 3.0];
    let shape = vec![6];

    let (count, indices) = run_nonzero(&input, &shape, nonzero_count::F32, nonzero_fill::F32);

    assert_eq!(count, 3);
    assert_eq!(indices, vec![1i32, 3, 5]);
}

#[test]
fn test_nonzero_f32_2d() {
    // Input: [[0, 1, 0], [2, 0, 3]]
    // Non-zero indices: [[0, 1], [1, 0], [1, 2]]
    let input = vec![0.0f32, 1.0, 0.0, 2.0, 0.0, 3.0];
    let shape = vec![2, 3];

    let (count, indices) = run_nonzero(&input, &shape, nonzero_count::F32, nonzero_fill::F32);

    assert_eq!(count, 3);
    // Output shape is [3, 2]: 3 non-zero elements, 2 dimensions
    assert_eq!(indices, vec![0i32, 1, 1, 0, 1, 2]);
}

#[test]
fn test_nonzero_i32() {
    let input = vec![0i32, 5, 0, 0, 10, 15];
    let shape = vec![6];

    let (count, indices) = run_nonzero(&input, &shape, nonzero_count::I32, nonzero_fill::I32);

    assert_eq!(count, 3);
    assert_eq!(indices, vec![1i32, 4, 5]);
}

#[test]
fn test_nonzero_all_zeros() {
    let input = vec![0.0f32, 0.0, 0.0];
    let shape = vec![3];

    let (count, indices) = run_nonzero(&input, &shape, nonzero_count::F32, nonzero_fill::F32);

    assert_eq!(count, 0);
    assert!(indices.is_empty());
}

#[test]
fn test_nonzero_all_nonzero() {
    let input = vec![1.0f32, 2.0, 3.0];
    let shape = vec![3];

    let (count, indices) = run_nonzero(&input, &shape, nonzero_count::F32, nonzero_fill::F32);

    assert_eq!(count, 3);
    assert_eq!(indices, vec![0i32, 1, 2]);
}
