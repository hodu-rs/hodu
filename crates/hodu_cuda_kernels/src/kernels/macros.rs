// Define Kernel type once, shared across all modules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Kernel(pub &'static str);

macro_rules! ops {
    ($($name:ident),+) => {
        $(
        pub mod $name {
            use $crate::kernels::macros::Kernel;
            pub const BOOL: Kernel = Kernel(concat!(stringify!($name), "_bool"));
            pub const F8E4M3: Kernel = Kernel(concat!(stringify!($name), "_f8e4m3"));
            pub const F8E5M2: Kernel = Kernel(concat!(stringify!($name), "_f8e5m2"));
            pub const BF16: Kernel = Kernel(concat!(stringify!($name), "_bf16"));
            pub const F16: Kernel = Kernel(concat!(stringify!($name), "_f16"));
            pub const F32: Kernel = Kernel(concat!(stringify!($name), "_f32"));
            pub const F64: Kernel = Kernel(concat!(stringify!($name), "_f64"));
            pub const U8: Kernel = Kernel(concat!(stringify!($name), "_u8"));
            pub const U16: Kernel = Kernel(concat!(stringify!($name), "_u16"));
            pub const U32: Kernel = Kernel(concat!(stringify!($name), "_u32"));
            pub const U64: Kernel = Kernel(concat!(stringify!($name), "_u64"));
            pub const I8: Kernel = Kernel(concat!(stringify!($name), "_i8"));
            pub const I16: Kernel = Kernel(concat!(stringify!($name), "_i16"));
            pub const I32: Kernel = Kernel(concat!(stringify!($name), "_i32"));
            pub const I64: Kernel = Kernel(concat!(stringify!($name), "_i64"));
        }
        )+
    };
}
pub(crate) use ops;
