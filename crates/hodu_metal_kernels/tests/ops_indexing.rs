use hodu_metal_kernels::{
    kernel::Kernels,
    kernels::{call_ops_onehot, onehot, Kernel},
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
