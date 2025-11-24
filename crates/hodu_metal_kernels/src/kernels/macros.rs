// Define Kernel type once, shared across all modules
pub struct Kernel(pub &'static str);

macro_rules! ops{
    ($($name:ident),+) => {
        // Import Kernel from the kernels::macros module
        use $crate::kernels::macros::Kernel;

        $(
        pub mod $name {
            use $crate::kernels::macros::Kernel;
            pub const BOOL: Kernel = Kernel(concat!("hodu_metal_", stringify!($name), "_bool"));
            pub const BF16: Kernel = Kernel(concat!("hodu_metal_", stringify!($name), "_bf16"));
            pub const F16: Kernel = Kernel(concat!("hodu_metal_", stringify!($name), "_f16"));
            pub const F32: Kernel = Kernel(concat!("hodu_metal_", stringify!($name), "_f32"));
            pub const U8: Kernel = Kernel(concat!("hodu_metal_", stringify!($name), "_u8"));
            pub const U16: Kernel = Kernel(concat!("hodu_metal_", stringify!($name), "_u16"));
            pub const U32: Kernel = Kernel(concat!("hodu_metal_", stringify!($name), "_u32"));
            pub const U64: Kernel = Kernel(concat!("hodu_metal_", stringify!($name), "_u64"));
            pub const I8: Kernel = Kernel(concat!("hodu_metal_", stringify!($name), "_i8"));
            pub const I16: Kernel = Kernel(concat!("hodu_metal_", stringify!($name), "_i16"));
            pub const I32: Kernel = Kernel(concat!("hodu_metal_", stringify!($name), "_i32"));
            pub const I64: Kernel = Kernel(concat!("hodu_metal_", stringify!($name), "_i64"));
        }
        )+
    };
}
pub(crate) use ops;
