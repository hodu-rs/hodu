use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{
        call_nonzero_count, call_nonzero_fill, call_ops_onehot, call_unique_bitonic_step, call_unique_build,
        call_unique_count, call_unique_mark, call_unique_prefix_sum, call_unique_sort, nonzero_count, nonzero_fill,
        onehot, unique_bitonic_step, unique_build, unique_count, unique_mark, unique_sort, Kernel,
    },
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

// ============================================================================
// UNIQUE TESTS
// ============================================================================

fn run_unique_i32(input: &[i32]) -> (usize, Vec<i32>, Vec<i32>, Vec<i32>) {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue().unwrap();
    let options = RESOURCE_OPTIONS;

    let num_els = input.len();
    let padded_size = num_els.next_power_of_two();
    let input_buffer = new_buffer(&device, input);
    let metadata = vec![num_els, 0, padded_size]; // num_els, offset, padded_size

    // Step 1: Copy input and initialize indices (with padding)
    let sorted_values = device
        .new_buffer(padded_size * std::mem::size_of::<i32>(), options)
        .unwrap();
    let sorted_indices = device
        .new_buffer(padded_size * std::mem::size_of::<i32>(), options)
        .unwrap();

    let command_buffer = create_command_buffer(&command_queue).unwrap();
    call_unique_sort(
        unique_sort::I32,
        &kernels,
        &device,
        &command_buffer,
        BufferOffset::zero_offset(&input_buffer),
        &sorted_values,
        &sorted_indices,
        &metadata,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Step 2: Bitonic sort
    let mut k = 2;
    while k <= padded_size {
        let mut j = k / 2;
        while j >= 1 {
            let bitonic_metadata = vec![num_els, 0, padded_size, k, j];
            let command_buffer = create_command_buffer(&command_queue).unwrap();
            call_unique_bitonic_step(
                unique_bitonic_step::I32,
                &kernels,
                &device,
                &command_buffer,
                &sorted_values,
                &sorted_indices,
                &bitonic_metadata,
            )
            .unwrap();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            j /= 2;
        }
        k *= 2;
    }

    // Step 3: Count unique elements
    let count_buffer = device.new_buffer(std::mem::size_of::<u32>(), options).unwrap();
    unsafe {
        let ptr = count_buffer.contents() as *mut u32;
        *ptr = 0;
    }

    let command_buffer = create_command_buffer(&command_queue).unwrap();
    call_unique_count(
        unique_count::I32,
        &kernels,
        &device,
        &command_buffer,
        &sorted_values,
        &count_buffer,
        &metadata,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let unique_count = unsafe { *(count_buffer.contents() as *const u32) } as usize;

    if unique_count == 0 {
        return (0, vec![], vec![], vec![]);
    }

    // Step 4: Mark unique boundaries
    let marks = device
        .new_buffer(num_els * std::mem::size_of::<u32>(), options)
        .unwrap();

    let command_buffer = create_command_buffer(&command_queue).unwrap();
    call_unique_mark(
        unique_mark::I32,
        &kernels,
        &device,
        &command_buffer,
        &sorted_values,
        &marks,
        &metadata,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Step 5: Prefix sum
    let unique_idx = device
        .new_buffer(num_els * std::mem::size_of::<i32>(), options)
        .unwrap();

    let command_buffer = create_command_buffer(&command_queue).unwrap();
    call_unique_prefix_sum(&kernels, &device, &command_buffer, &marks, &unique_idx, &metadata).unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Step 6: Build outputs
    let values_buffer = device
        .new_buffer(unique_count * std::mem::size_of::<i32>(), options)
        .unwrap();
    let inverse_buffer = device
        .new_buffer(num_els * std::mem::size_of::<i32>(), options)
        .unwrap();
    let counts_buffer = device
        .new_buffer(unique_count * std::mem::size_of::<i32>(), options)
        .unwrap();

    // Initialize counts to 0
    unsafe {
        let ptr = counts_buffer.contents() as *mut i32;
        for i in 0..unique_count {
            *ptr.add(i) = 0;
        }
    }

    let command_buffer = create_command_buffer(&command_queue).unwrap();
    call_unique_build(
        unique_build::I32,
        &kernels,
        &device,
        &command_buffer,
        &sorted_values,
        &sorted_indices,
        &marks,
        &unique_idx,
        &values_buffer,
        &inverse_buffer,
        &counts_buffer,
        &metadata,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let values = read_to_vec(&values_buffer, unique_count);
    let inverse = read_to_vec(&inverse_buffer, num_els);
    let counts = read_to_vec(&counts_buffer, unique_count);

    (unique_count, values, inverse, counts)
}

#[test]
fn test_unique_i32_basic() {
    // Input: [3, 1, 2, 1, 3, 2, 4]
    // Expected values (sorted): [1, 2, 3, 4]
    // Expected inverse: [2, 0, 1, 0, 2, 1, 3]
    // Expected counts: [2, 2, 2, 1]
    let input = vec![3i32, 1, 2, 1, 3, 2, 4];

    let (unique_count, values, inverse, counts) = run_unique_i32(&input);

    assert_eq!(unique_count, 4);
    assert_eq!(values, vec![1, 2, 3, 4]);
    assert_eq!(inverse, vec![2, 0, 1, 0, 2, 1, 3]);
    assert_eq!(counts, vec![2, 2, 2, 1]);
}

#[test]
fn test_unique_i32_all_same() {
    // Input: [5, 5, 5, 5]
    // Expected values: [5]
    // Expected inverse: [0, 0, 0, 0]
    // Expected counts: [4]
    let input = vec![5i32, 5, 5, 5];

    let (unique_count, values, inverse, counts) = run_unique_i32(&input);

    assert_eq!(unique_count, 1);
    assert_eq!(values, vec![5]);
    assert_eq!(inverse, vec![0, 0, 0, 0]);
    assert_eq!(counts, vec![4]);
}

#[test]
fn test_unique_i32_all_unique() {
    // Input: [4, 2, 1, 3]
    // Expected values (sorted): [1, 2, 3, 4]
    // Expected inverse: [3, 1, 0, 2]
    // Expected counts: [1, 1, 1, 1]
    let input = vec![4i32, 2, 1, 3];

    let (unique_count, values, inverse, counts) = run_unique_i32(&input);

    assert_eq!(unique_count, 4);
    assert_eq!(values, vec![1, 2, 3, 4]);
    assert_eq!(inverse, vec![3, 1, 0, 2]);
    assert_eq!(counts, vec![1, 1, 1, 1]);
}

#[test]
fn test_unique_i32_single_element() {
    // Input: [42]
    // Expected values: [42]
    // Expected inverse: [0]
    // Expected counts: [1]
    let input = vec![42i32];

    let (unique_count, values, inverse, counts) = run_unique_i32(&input);

    assert_eq!(unique_count, 1);
    assert_eq!(values, vec![42]);
    assert_eq!(inverse, vec![0]);
    assert_eq!(counts, vec![1]);
}
