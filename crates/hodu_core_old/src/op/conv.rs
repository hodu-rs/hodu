#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv1D {
    pub(crate) batch_size: usize,
    pub(crate) length_input: usize,
    pub(crate) channels_output: usize,
    pub(crate) channels_input: usize,
    pub(crate) kernel_size: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose1D {
    pub(crate) batch_size: usize,
    pub(crate) length_input: usize,
    pub(crate) channels_output: usize,
    pub(crate) channels_input: usize,
    pub(crate) kernel_size: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv2D {
    pub(crate) batch_size: usize,
    pub(crate) input_height: usize,
    pub(crate) input_width: usize,
    pub(crate) kernel_height: usize,
    pub(crate) kernel_width: usize,
    pub(crate) channels_output: usize,
    pub(crate) channels_input: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose2D {
    pub(crate) batch_size: usize,
    pub(crate) input_height: usize,
    pub(crate) input_width: usize,
    pub(crate) kernel_height: usize,
    pub(crate) kernel_width: usize,
    pub(crate) channels_output: usize,
    pub(crate) channels_input: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConv3D {
    pub(crate) batch_size: usize,
    pub(crate) input_depth: usize,
    pub(crate) input_height: usize,
    pub(crate) input_width: usize,
    pub(crate) kernel_depth: usize,
    pub(crate) kernel_height: usize,
    pub(crate) kernel_width: usize,
    pub(crate) channels_output: usize,
    pub(crate) channels_input: usize,
    pub(crate) padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamsConvTranspose3D {
    pub(crate) batch_size: usize,
    pub(crate) input_depth: usize,
    pub(crate) input_height: usize,
    pub(crate) input_width: usize,
    pub(crate) kernel_depth: usize,
    pub(crate) kernel_height: usize,
    pub(crate) kernel_width: usize,
    pub(crate) channels_output: usize,
    pub(crate) channels_input: usize,
    pub(crate) padding: usize,
    pub(crate) output_padding: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}
