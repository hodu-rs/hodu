use hodu_cpu_kernels::*;

// Helper function to calculate strides from shape
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

// Resize modes
const RESIZE_MODE_NEAREST: usize = 0;
const RESIZE_MODE_LINEAR: usize = 1;
const RESIZE_MODE_CUBIC: usize = 2;

// Coordinate transformation modes
const RESIZE_COORD_HALF_PIXEL: usize = 0;
const RESIZE_COORD_ASYMMETRIC: usize = 1;
const RESIZE_COORD_ALIGN_CORNERS: usize = 2;

// Nearest rounding modes
const RESIZE_NEAREST_FLOOR: usize = 0;

// Helper function to build resize metadata
// Layout: [output_size, num_dims, input_shape..., input_strides..., offset,
//          output_shape..., mode, coord_transform, nearest_mode]
fn build_resize_metadata(
    input_shape: &[usize],
    input_strides: &[usize],
    offset: usize,
    output_shape: &[usize],
    mode: usize,
    coord_transform: usize,
    nearest_mode: usize,
) -> Vec<usize> {
    let output_size: usize = output_shape.iter().product();
    let num_dims = input_shape.len();
    let mut metadata = vec![output_size, num_dims];
    metadata.extend(input_shape);
    metadata.extend(input_strides);
    metadata.push(offset);
    metadata.extend(output_shape);
    metadata.push(mode);
    metadata.push(coord_transform);
    metadata.push(nearest_mode);
    metadata
}

// Test nearest neighbor 2x upsample (NCHW: 1x1x2x2 -> 1x1x4x4)
#[test]
fn test_resize_nearest_upsample_2x_f32() {
    // Input: 1x1x2x2
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let input_shape = vec![1, 1, 2, 2];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 4, 4];
    let mut output = vec![0.0f32; 16];

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_NEAREST,
        RESIZE_COORD_ASYMMETRIC,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(
        resize::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Nearest neighbor 2x upsample with asymmetric coord transform
    // Each pixel becomes 2x2 block
    let expected = [
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(output, expected);
}

// Test bilinear 2x upsample
#[test]
fn test_resize_bilinear_upsample_f32() {
    // Input: 1x1x2x2
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let input_shape = vec![1, 1, 2, 2];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 4, 4];
    let mut output = vec![0.0f32; 16];

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_LINEAR,
        RESIZE_COORD_ALIGN_CORNERS,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(
        resize::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // With align_corners, corners should match exactly
    // Top-left (1.0) and top-right (2.0), bottom-left (3.0), bottom-right (4.0)
    assert!((output[0] - 1.0).abs() < 1e-5); // top-left
    assert!((output[3] - 2.0).abs() < 1e-5); // top-right
    assert!((output[12] - 3.0).abs() < 1e-5); // bottom-left
    assert!((output[15] - 4.0).abs() < 1e-5); // bottom-right
}

// Test nearest downsample
#[test]
fn test_resize_nearest_downsample_f32() {
    // Input: 1x1x4x4
    let input = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let input_shape = vec![1, 1, 4, 4];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 2, 2];
    let mut output = vec![0.0f32; 4];

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_NEAREST,
        RESIZE_COORD_ASYMMETRIC,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(
        resize::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // With asymmetric and floor, should pick top-left of each 2x2 block
    assert_eq!(output, vec![1.0, 3.0, 9.0, 11.0]);
}

// Test bilinear with half_pixel coordinate transform
#[test]
fn test_resize_bilinear_half_pixel_f32() {
    // Input: 1x1x2x2
    let input = [0.0f32, 10.0, 10.0, 20.0];
    let input_shape = vec![1, 1, 2, 2];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 4, 4];
    let mut output = vec![0.0f32; 16];

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_LINEAR,
        RESIZE_COORD_HALF_PIXEL,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(
        resize::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Just check output is within expected range
    for &val in &output {
        assert!(val >= 0.0 && val <= 20.0);
    }
}

// Test bicubic interpolation
#[test]
fn test_resize_bicubic_f32() {
    // Input: 1x1x4x4
    let input = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let input_shape = vec![1, 1, 4, 4];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 8, 8];
    let mut output = vec![0.0f32; 64];

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_CUBIC,
        RESIZE_COORD_HALF_PIXEL,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(
        resize::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Check output is within expected range (cubic can overshoot slightly)
    for &val in &output {
        assert!(val >= 0.0 && val <= 20.0);
    }
}

// Test with batch and channel dimensions
#[test]
fn test_resize_batch_channel_f32() {
    // Input: 2x2x2x2 (batch=2, channel=2, H=2, W=2)
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input_shape = vec![2, 2, 2, 2];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![2, 2, 4, 4];
    let mut output = vec![0.0f32; 64];

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_NEAREST,
        RESIZE_COORD_ASYMMETRIC,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(
        resize::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Check that batch and channel are preserved
    assert!((output[0] - 1.0).abs() < 1e-5); // batch 0, channel 0
    assert!((output[16] - 5.0).abs() < 1e-5); // batch 0, channel 1
    assert!((output[32] - 9.0).abs() < 1e-5); // batch 1, channel 0
    assert!((output[48] - 13.0).abs() < 1e-5); // batch 1, channel 1
}

// Test identity resize (same size)
#[test]
fn test_resize_identity_f32() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let input_shape = vec![1, 1, 2, 2];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 2, 2];
    let mut output = vec![0.0f32; 4];

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_LINEAR,
        RESIZE_COORD_HALF_PIXEL,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(
        resize::F32,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    // Identity should preserve values
    for (a, b) in output.iter().zip(input.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

// Test f64 type
#[test]
fn test_resize_nearest_f64() {
    let input = [1.0f64, 2.0, 3.0, 4.0];
    let input_shape = vec![1, 1, 2, 2];
    let input_strides = calculate_strides(&input_shape);
    let output_shape = vec![1, 1, 4, 4];
    let mut output = vec![0.0f64; 16];

    let metadata = build_resize_metadata(
        &input_shape,
        &input_strides,
        0,
        &output_shape,
        RESIZE_MODE_NEAREST,
        RESIZE_COORD_ASYMMETRIC,
        RESIZE_NEAREST_FLOOR,
    );

    call_ops_resize(
        resize::F64,
        input.as_ptr() as *const core::ffi::c_void,
        output.as_mut_ptr() as *mut core::ffi::c_void,
        &metadata,
    )
    .unwrap();

    let expected = [
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    assert_eq!(output, expected);
}
