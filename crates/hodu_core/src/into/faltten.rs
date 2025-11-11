use crate::{be_cpu::storage::CpuStorage, layer::compat::*, types::DType};
use float8::F8E4M3;
#[cfg(feature = "f8e5m2")]
use float8::F8E5M2;
use half::{bf16, f16};

pub trait IntoFlattened: Sized {
    type Elem: Clone;

    fn to_flatten_vec(self) -> Vec<Self::Elem>;
    fn get_shape_vec(&self) -> Vec<usize>;
    fn get_dtype(&self) -> DType;
    fn to_cpu_storage(self) -> CpuStorage;
}

macro_rules! into_flattened_impl {
    ($t:ty, $dtype:expr) => {
        // Scalar
        impl IntoFlattened for $t {
            type Elem = $t;

            fn to_flatten_vec(self) -> Vec<$t> {
                vec![self]
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                vec![]
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 1D Vector
        impl IntoFlattened for Vec<$t> {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                self
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                vec![self.len()]
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 2D Vector
        impl IntoFlattened for Vec<Vec<$t>> {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for row in self {
                    flat.extend(row);
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0]
                } else {
                    vec![self.len(), self[0].len()]
                }
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 3D Vector
        impl IntoFlattened for Vec<Vec<Vec<$t>>> {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for matrix in self {
                    for row in matrix {
                        flat.extend(row);
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0]
                } else {
                    vec![self.len(), self[0].len(), self[0][0].len()]
                }
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 4D Vector
        impl IntoFlattened for Vec<Vec<Vec<Vec<$t>>>> {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for tensor3d in self {
                    for matrix in tensor3d {
                        for row in matrix {
                            flat.extend(row);
                        }
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0]
                } else {
                    vec![self.len(), self[0].len(), self[0][0].len(), self[0][0][0].len()]
                }
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 5D Vector
        impl IntoFlattened for Vec<Vec<Vec<Vec<Vec<$t>>>>> {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for tensor4d in self {
                    for tensor3d in tensor4d {
                        for matrix in tensor3d {
                            for row in matrix {
                                flat.extend(row);
                            }
                        }
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0, 0]
                } else {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        self[0][0][0][0].len(),
                    ]
                }
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 6D Vector
        impl IntoFlattened for Vec<Vec<Vec<Vec<Vec<Vec<$t>>>>>> {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for tensor5d in self {
                    for tensor4d in tensor5d {
                        for tensor3d in tensor4d {
                            for matrix in tensor3d {
                                for row in matrix {
                                    flat.extend(row);
                                }
                            }
                        }
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0, 0, 0]
                } else {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        self[0][0][0][0].len(),
                        self[0][0][0][0][0].len(),
                    ]
                }
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 1D Fixed-size Array Reference
        impl<'a, const N: usize> IntoFlattened for &'a [$t; N] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                self.to_vec()
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                vec![N]
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 2D Fixed-size Array Reference
        impl<'a, const M: usize, const N: usize> IntoFlattened for &'a [[$t; N]; M] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for row in self {
                    flat.extend_from_slice(row);
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                vec![M, N]
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 3D Fixed-size Array Reference
        impl<'a, const L: usize, const M: usize, const N: usize> IntoFlattened for &'a [[[$t; N]; M]; L] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for matrix in self {
                    for row in matrix {
                        flat.extend_from_slice(row);
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                vec![L, M, N]
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 4D Fixed-size Array Reference
        impl<'a, const K: usize, const L: usize, const M: usize, const N: usize> IntoFlattened
            for &'a [[[[$t; N]; M]; L]; K]
        {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for tensor3d in self {
                    for matrix in tensor3d {
                        for row in matrix {
                            flat.extend_from_slice(row);
                        }
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                vec![K, L, M, N]
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 5D Fixed-size Array Reference
        impl<'a, const J: usize, const K: usize, const L: usize, const M: usize, const N: usize> IntoFlattened
            for &'a [[[[[$t; N]; M]; L]; K]; J]
        {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for tensor4d in self {
                    for tensor3d in tensor4d {
                        for matrix in tensor3d {
                            for row in matrix {
                                flat.extend_from_slice(row);
                            }
                        }
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                vec![J, K, L, M, N]
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 6D Fixed-size Array Reference
        impl<'a, const I: usize, const J: usize, const K: usize, const L: usize, const M: usize, const N: usize>
            IntoFlattened for &'a [[[[[[$t; N]; M]; L]; K]; J]; I]
        {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for tensor5d in self {
                    for tensor4d in tensor5d {
                        for tensor3d in tensor4d {
                            for matrix in tensor3d {
                                for row in matrix {
                                    flat.extend_from_slice(row);
                                }
                            }
                        }
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                vec![I, J, K, L, M, N]
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 1D Slice of Slices
        impl<'a> IntoFlattened for &'a [&'a [$t]] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for row in self {
                    flat.extend_from_slice(row);
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0]
                } else {
                    vec![self.len(), self[0].len()]
                }
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 2D Slice of Slices
        impl<'a> IntoFlattened for &'a [&'a [&'a [$t]]] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for matrix in self {
                    for row in *matrix {
                        flat.extend_from_slice(row);
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0]
                } else if self[0].is_empty() {
                    vec![self.len(), 0, 0]
                } else {
                    vec![self.len(), self[0].len(), self[0][0].len()]
                }
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 3D Slice of Slices
        impl<'a> IntoFlattened for &'a [&'a [&'a [&'a [$t]]]] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for tensor3d in self {
                    for matrix in *tensor3d {
                        for row in *matrix {
                            flat.extend_from_slice(row);
                        }
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0]
                } else if self[0].is_empty() {
                    vec![self.len(), 0, 0, 0]
                } else if self[0][0].is_empty() {
                    vec![self.len(), self[0].len(), 0, 0]
                } else {
                    vec![self.len(), self[0].len(), self[0][0].len(), self[0][0][0].len()]
                }
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 4D Slice of Slices
        impl<'a> IntoFlattened for &'a [&'a [&'a [&'a [&'a [$t]]]]] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for tensor4d in self {
                    for tensor3d in *tensor4d {
                        for matrix in *tensor3d {
                            for row in *matrix {
                                flat.extend_from_slice(row);
                            }
                        }
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0, 0]
                } else if self[0].is_empty() {
                    vec![self.len(), 0, 0, 0, 0]
                } else if self[0][0].is_empty() {
                    vec![self.len(), self[0].len(), 0, 0, 0]
                } else if self[0][0][0].is_empty() {
                    vec![self.len(), self[0].len(), self[0][0].len(), 0, 0]
                } else {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        self[0][0][0][0].len(),
                    ]
                }
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 5D Slice of Slices
        impl<'a> IntoFlattened for &'a [&'a [&'a [&'a [&'a [&'a [$t]]]]]] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for tensor5d in self {
                    for tensor4d in *tensor5d {
                        for tensor3d in *tensor4d {
                            for matrix in *tensor3d {
                                for row in *matrix {
                                    flat.extend_from_slice(row);
                                }
                            }
                        }
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0, 0, 0]
                } else if self[0].is_empty() {
                    vec![self.len(), 0, 0, 0, 0, 0]
                } else if self[0][0].is_empty() {
                    vec![self.len(), self[0].len(), 0, 0, 0, 0]
                } else if self[0][0][0].is_empty() {
                    vec![self.len(), self[0].len(), self[0][0].len(), 0, 0, 0]
                } else if self[0][0][0][0].is_empty() {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        0,
                        0,
                    ]
                } else {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        self[0][0][0][0].len(),
                        self[0][0][0][0][0].len(),
                    ]
                }
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }

        // 6D Slice of Slices
        impl<'a> IntoFlattened for &'a [&'a [&'a [&'a [&'a [&'a [&'a [$t]]]]]]] {
            type Elem = $t;
            fn to_flatten_vec(self) -> Vec<$t> {
                let mut flat = Vec::new();
                for tensor6d in self {
                    for tensor5d in *tensor6d {
                        for tensor4d in *tensor5d {
                            for tensor3d in *tensor4d {
                                for matrix in *tensor3d {
                                    for row in *matrix {
                                        flat.extend_from_slice(row);
                                    }
                                }
                            }
                        }
                    }
                }
                flat
            }
            fn get_shape_vec(&self) -> Vec<usize> {
                if self.is_empty() {
                    vec![0, 0, 0, 0, 0, 0, 0]
                } else if self[0].is_empty() {
                    vec![self.len(), 0, 0, 0, 0, 0, 0]
                } else if self[0][0].is_empty() {
                    vec![self.len(), self[0].len(), 0, 0, 0, 0, 0]
                } else if self[0][0][0].is_empty() {
                    vec![self.len(), self[0].len(), self[0][0].len(), 0, 0, 0, 0]
                } else if self[0][0][0][0].is_empty() {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        0,
                        0,
                        0,
                    ]
                } else if self[0][0][0][0][0].is_empty() {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        self[0][0][0][0].len(),
                        0,
                        0,
                    ]
                } else {
                    vec![
                        self.len(),
                        self[0].len(),
                        self[0][0].len(),
                        self[0][0][0].len(),
                        self[0][0][0][0].len(),
                        self[0][0][0][0][0].len(),
                        self[0][0][0][0][0][0].len(),
                    ]
                }
            }
            fn get_dtype(&self) -> DType {
                $dtype
            }
            fn to_cpu_storage(self) -> CpuStorage {
                let vec = self.to_flatten_vec();
                CpuStorage::from_vec(vec)
            }
        }
    };
}

into_flattened_impl!(bool, DType::BOOL);
into_flattened_impl!(F8E4M3, DType::F8E4M3);
#[cfg(feature = "f8e5m2")]
into_flattened_impl!(F8E5M2, DType::F8E5M2);
into_flattened_impl!(bf16, DType::BF16);
into_flattened_impl!(f16, DType::F16);
into_flattened_impl!(f32, DType::F32);
#[cfg(feature = "f64")]
into_flattened_impl!(f64, DType::F64);
into_flattened_impl!(u8, DType::U8);
#[cfg(feature = "u16")]
into_flattened_impl!(u16, DType::U16);
into_flattened_impl!(u32, DType::U32);
#[cfg(feature = "u64")]
into_flattened_impl!(u64, DType::U64);
into_flattened_impl!(i8, DType::I8);
#[cfg(feature = "i16")]
into_flattened_impl!(i16, DType::I16);
into_flattened_impl!(i32, DType::I32);
#[cfg(feature = "i64")]
into_flattened_impl!(i64, DType::I64);
