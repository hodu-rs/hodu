use hodu_cuda_kernels::{compat::*, kernel::Kernels, kernels::*};

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

fn run_binary<T: cudarc::driver::DeviceRepr + Clone>(kernel: Kernel, lhs: &[T], rhs: &[T]) -> Vec<T> {
    assert_eq!(lhs.len(), rhs.len());
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let lhs_dev = stream.memcpy_stod(lhs).unwrap();
    let rhs_dev = stream.memcpy_stod(rhs).unwrap();
    let mut output = unsafe { stream.alloc::<T>(lhs.len()).unwrap() };

    let shape = vec![lhs.len()];
    let strides = vec![1];
    let num_els = lhs.len();
    let num_dims = 1;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape); // lhs_shape
    metadata.extend(&shape); // rhs_shape
    metadata.extend(&strides); // lhs_strides
    metadata.extend(&strides); // rhs_strides
    metadata.push(0); // lhs_offset
    metadata.push(0); // rhs_offset

    call_ops_binary::<T, T>(kernel, &kernels, &device, &lhs_dev, &rhs_dev, &mut output, &metadata).unwrap();

    let mut result = vec![unsafe { core::mem::zeroed() }; lhs.len()];
    stream.memcpy_dtoh(&output, &mut result).unwrap();
    result
}

fn run_binary_logical<T: cudarc::driver::DeviceRepr + Clone, O: cudarc::driver::DeviceRepr + Clone>(
    kernel: Kernel,
    lhs: &[T],
    rhs: &[T],
) -> Vec<O> {
    assert_eq!(lhs.len(), rhs.len());
    let kernels = kernels();
    let device = device();
    let stream = device.default_stream();

    let lhs_dev = stream.memcpy_stod(lhs).unwrap();
    let rhs_dev = stream.memcpy_stod(rhs).unwrap();
    let mut output = unsafe { stream.alloc::<O>(lhs.len()).unwrap() };

    let shape = vec![lhs.len()];
    let strides = vec![1];
    let num_els = lhs.len();
    let num_dims = 1;

    let mut metadata = Vec::new();
    metadata.push(num_els);
    metadata.push(num_dims);
    metadata.extend(&shape); // lhs_shape
    metadata.extend(&shape); // rhs_shape
    metadata.extend(&strides); // lhs_strides
    metadata.extend(&strides); // rhs_strides
    metadata.push(0); // lhs_offset
    metadata.push(0); // rhs_offset

    call_ops_binary::<T, O>(kernel, &kernels, &device, &lhs_dev, &rhs_dev, &mut output, &metadata).unwrap();

    let mut result = vec![unsafe { core::mem::zeroed() }; lhs.len()];
    stream.memcpy_dtoh(&output, &mut result).unwrap();
    result
}

// Arithmetic operations
#[test]
fn binary_ops_f32() {
    let lhs: Vec<f32> = vec![1.1f32, 2.2, 3.3];
    let rhs: Vec<f32> = vec![4.2f32, 5.5f32, 6.91f32];

    macro_rules! binary_op {
        ($opname:ident, $opexpr:expr) => {{
            let results = run_binary($opname::F32, &lhs, &rhs);
            let expected: Vec<f32> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(x, y): (&f32, &f32)| $opexpr(*x, *y))
                .collect();
            assert_eq!(approx(results, 6), approx(expected, 6));
        }};
    }

    binary_op!(add, |x, y| x + y);
    binary_op!(sub, |x, y| x - y);
    binary_op!(mul, |x, y| x * y);
    binary_op!(div, |x, y| x / y);
    binary_op!(minimum, |x: f32, y| x.min(y));
    binary_op!(maximum, |x: f32, y| x.max(y));
    binary_op!(pow, |x: f32, y| x.powf(y));
}

// Comparison operations
#[test]
fn binary_cmp_f32() {
    let lhs: Vec<f32> = vec![1.0f32, 2.0, 3.0];
    let rhs: Vec<f32> = vec![1.0f32, 3.0, 2.0];

    let eq_results: Vec<bool> = run_binary_logical(eq::F32, &lhs, &rhs);
    let ne_results: Vec<bool> = run_binary_logical(ne::F32, &lhs, &rhs);
    let lt_results: Vec<bool> = run_binary_logical(lt::F32, &lhs, &rhs);
    let le_results: Vec<bool> = run_binary_logical(le::F32, &lhs, &rhs);
    let gt_results: Vec<bool> = run_binary_logical(gt::F32, &lhs, &rhs);
    let ge_results: Vec<bool> = run_binary_logical(ge::F32, &lhs, &rhs);

    assert_eq!(eq_results, vec![true, false, false]);
    assert_eq!(ne_results, vec![false, true, true]);
    assert_eq!(lt_results, vec![false, true, false]);
    assert_eq!(le_results, vec![true, true, false]);
    assert_eq!(gt_results, vec![false, false, true]);
    assert_eq!(ge_results, vec![true, false, true]);
}
