use crate::{
    error::MetalKernelError,
    kernel::Kernels,
    kernels::macros::ops,
    metal::{Buffer, ComputeCommandEncoder, Device},
    set_params,
    source::Source,
    utils::{linear_split, BufferOffset, EncoderProvider},
};
use objc2_metal::MTLResourceUsage;

ops!(
    eq_scalar,
    ne_scalar,
    lt_scalar,
    le_scalar,
    gt_scalar,
    ge_scalar,
    neg,
    abs,
    sign,
    square,
    sqrt,
    recip,
    relu,
    sigmoid,
    hardsigmoid,
    gelu,
    softplus,
    silu,
    hardsilu,
    mish,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
    exp,
    exp2,
    exp10,
    ln,
    log2,
    log10,
    ceil,
    floor,
    round,
    erf,
    logical_not,
    add_scalar,
    sub_scalar,
    mul_scalar,
    div_scalar,
    pow_scalar,
    maximum_scalar,
    minimum_scalar,
    leaky_relu,
    elu,
    prelu
);

/// Executes a unary operation on a tensor using Metal compute pipeline.
///
/// Applies an element-wise unary operation to each element of the input tensor.
/// Supports both contiguous and strided tensor layouts.
///
/// # Arguments
/// * `kernel` - Unary operation kernel (neg, abs, sign, square, sqrt, recip, relu, sigmoid,
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
///   tanh, gelu, softplus, silu, mish, sin, cos, tan, exp, exp2, exp10, ln, log2,
///   log10, logical_not, etc.)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer
/// * `metadata` - Metadata describing tensor shape, strides, and offset
///
/// # Metadata Layout
/// Total metadata length: `2 + num_dims * 2 + 1`
///
/// - `metadata[0]`: num_els (total number of elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: shape (dimensions of the tensor)
/// - `metadata[2+num_dims..2+2*num_dims]`: strides (stride for each dimension)
/// - `metadata[2+2*num_dims]`: offset (starting offset in input buffer)
///
/// # Supported Operations
/// ## Basic Math
/// - `neg`: Negation (-x)
/// - `abs`: Absolute value |x|
/// - `sign`: Sign function (-1, 0, or 1)
/// - `square`: Square (x²)
/// - `sqrt`: Square root (√x)
/// - `recip`: Reciprocal (1/x)
///
/// ## Activation Functions
/// - `relu`: Rectified Linear Unit max(0, x)
/// - `sigmoid`: Sigmoid 1/(1+e^(-x))
/// - `tanh`: Hyperbolic tangent
/// - `gelu`: Gaussian Error Linear Unit
/// - `softplus`: Smooth approximation of ReLU log(1+e^x)
/// - `silu`: Sigmoid Linear Unit x·σ(x)
/// - `mish`: Mish activation x·tanh(softplus(x))
/// - `leaky_relu`: Leaky ReLU (requires scalar parameter)
/// - `elu`: Exponential Linear Unit (requires scalar parameter)
/// - `prelu`: Parametric ReLU (requires scalar parameter)
/// - `rrelu`: Randomized ReLU (requires scalar parameter)
///
/// ## Trigonometric
/// - `sin`, `cos`, `tan`: Trigonometric functions
///
/// ## Exponential/Logarithmic
/// - `exp`: Natural exponential e^x
/// - `exp2`: Base-2 exponential 2^x
/// - `exp10`: Base-10 exponential 10^x
/// - `ln`: Natural logarithm
/// - `log2`: Base-2 logarithm
/// - `log10`: Base-10 logarithm
///
/// ## Logical
/// - `logical_not`: Logical NOT
///
/// # Example
/// ```ignore
/// // Apply ReLU to a 3x4 tensor
/// let metadata = vec![
///     12,     // num_els
///     2,      // num_dims
///     3, 4,   // shape
///     4, 1,   // strides
///     0,      // offset
/// ];
/// call_ops_unary(&device, &command_buffer, &kernels, relu::F32,
///            input_buffer, &output, &metadata)?;
/// ```
pub fn call_ops_unary(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input (can be nullptr for in-place)
    // buffer(1): output
    // buffer(2): metadata
    set_params!(encoder, (&input, output, metadata));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}

/// Executes a unary operation with a scalar parameter on a tensor using Metal compute pipeline.
///
/// Applies an element-wise unary operation that requires an additional scalar parameter.
/// This includes operations like scalar arithmetic (add, sub, mul, div, pow) and
/// parameterized activation functions (leaky_relu, elu, prelu, rrelu).
///
/// # Arguments
/// * `kernel` - Unary scalar operation kernel (add_scalar, sub_scalar, mul_scalar, div_scalar,
/// * `kernels` - Kernel cache
/// * `device` - Metal device to execute on
/// * `ep` - Encoder provider (command buffer)
///   pow_scalar, maximum_scalar, minimum_scalar, eq_scalar, ne_scalar, lt_scalar,
///   le_scalar, gt_scalar, ge_scalar, leaky_relu, elu, prelu, rrelu)
/// * `input` - Input tensor buffer
/// * `output` - Output buffer
/// * `metadata` - Metadata describing tensor shape, strides, and offset
/// * `scalar_val` - Scalar parameter for the operation
///
/// # Metadata Layout
/// Total metadata length: `2 + num_dims * 2 + 1`
///
/// - `metadata[0]`: num_els (total number of elements)
/// - `metadata[1]`: num_dims (number of dimensions)
/// - `metadata[2..2+num_dims]`: shape (dimensions of the tensor)
/// - `metadata[2+num_dims..2+2*num_dims]`: strides (stride for each dimension)
/// - `metadata[2+2*num_dims]`: offset (starting offset in input buffer)
///
/// # Type Parameter
/// * `T: EncoderParam` - The type of the scalar parameter (f32, i32, etc.)
///
/// # Supported Operations
/// ## Scalar Arithmetic
/// - `add_scalar`: Add scalar (x + c)
/// - `sub_scalar`: Subtract scalar (x - c)
/// - `mul_scalar`: Multiply by scalar (x * c)
/// - `div_scalar`: Divide by scalar (x / c)
/// - `pow_scalar`: Power with scalar exponent (x^c)
/// - `maximum_scalar`: Element-wise maximum with scalar max(x, c)
/// - `minimum_scalar`: Element-wise minimum with scalar min(x, c)
///
/// ## Scalar Comparisons
/// - `eq_scalar`: Equal to scalar (x == c)
/// - `ne_scalar`: Not equal to scalar (x != c)
/// - `lt_scalar`: Less than scalar (x < c)
/// - `le_scalar`: Less or equal to scalar (x <= c)
/// - `gt_scalar`: Greater than scalar (x > c)
/// - `ge_scalar`: Greater or equal to scalar (x >= c)
///
/// ## Parameterized Activations
/// - `leaky_relu`: Leaky ReLU with negative slope (x if x > 0 else α*x)
/// - `elu`: ELU with alpha (x if x > 0 else α*(e^x - 1))
/// - `prelu`: PReLU with slope (x if x > 0 else α*x)
/// - `rrelu`: Randomized ReLU (x if x > 0 else α*x where α is random)
///
/// # Example
/// ```ignore
/// // Add 5.0 to all elements of a tensor
/// let metadata = vec![
///     12,     // num_els
///     2,      // num_dims
///     3, 4,   // shape
///     4, 1,   // strides
///     0,      // offset
/// ];
/// call_ops_unary_scalar(&device, &command_buffer, &kernels, add_scalar::F32,
///                   input_buffer, &output, &metadata, 5.0f32)?;
/// ```
///
/// # Example with Leaky ReLU
/// ```ignore
/// // Apply Leaky ReLU with negative slope 0.01
/// call_ops_unary_scalar(&device, &command_buffer, &kernels, leaky_relu::F32,
///                   input_buffer, &output, &metadata, 0.01f32)?;
/// ```
#[allow(clippy::too_many_arguments)]
pub fn call_ops_unary_scalar<T: crate::utils::EncoderParam>(
    kernel: Kernel,
    kernels: &Kernels,
    device: &Device,
    ep: impl EncoderProvider,
    input: BufferOffset,
    output: &Buffer,
    metadata: &[usize],
    scalar_val: T,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel.0)?;

    let num_els = metadata[0];

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    // Metal kernel signature:
    // buffer(0): input
    // buffer(1): output
    // buffer(2): metadata
    // buffer(3): const_val (scalar)
    set_params!(encoder, (&input, output, metadata, scalar_val));

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, num_els);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

    Ok(())
}
