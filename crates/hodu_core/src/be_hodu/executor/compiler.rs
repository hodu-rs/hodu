use super::{CompiledNode, HoduExecutor};
use crate::{
    be_hodu::{metal::storage::MetalStorage, storage::HoduStorage},
    compat::*,
    error::{HoduError, HoduResult},
    script::{
        ir::{NodeId, ScriptIR},
        Script,
    },
    tensor::TensorId,
    types::{device::Device, dtype::DType, layout::Layout},
};

type SharedStorage = Arc<HoduStorage>;

/// Compiler for HoduExecutor
impl HoduExecutor {
    pub(super) fn convert_script_ir_to_compiled_nodes(&self, script_ir: &ScriptIR) -> HoduResult<Vec<CompiledNode>> {
        let mut compiled_nodes = Vec::with_capacity(script_ir.graph.topology.nodes.len());

        for node in &script_ir.graph.topology.nodes {
            let compiled_node = CompiledNode {
                id: node.id,
                operation: node.operation.clone(),
                input_tensors: node.input_tensors.clone(),
                output_tensors: node.output_tensors.clone(),
                input_layouts: node.input_layouts.clone(),
                output_layouts: node.output_layouts.clone(),
            };
            compiled_nodes.push(compiled_node);
        }

        // Build execution order lookup map for O(1) access
        #[cfg(feature = "std")]
        let execution_order_map: HashMap<NodeId, usize> = script_ir
            .graph
            .execution_plan
            .execution_order
            .iter()
            .enumerate()
            .map(|(index, &node_id)| (node_id, index))
            .collect();

        #[cfg(not(feature = "std"))]
        let execution_order_map: HashMap<NodeId, usize> = script_ir
            .graph
            .execution_plan
            .execution_order
            .iter()
            .enumerate()
            .map(|(index, &node_id)| (node_id, index))
            .collect();

        compiled_nodes.sort_by_key(|node| execution_order_map.get(&node.id).copied().unwrap_or(usize::MAX));

        Ok(compiled_nodes)
    }

    pub(super) fn collect_tensor_layouts(&self, script_ir: &ScriptIR) -> HashMap<TensorId, Layout> {
        #[cfg(feature = "std")]
        let mut tensor_layouts = {
            // Estimate capacity: node layouts + inputs + outputs
            let estimated_layout_count = script_ir.graph.topology.nodes.len() * 2
                + script_ir.graph.metadata.inputs.len()
                + script_ir.graph.metadata.outputs.len();
            HashMap::with_capacity(estimated_layout_count)
        };
        #[cfg(not(feature = "std"))]
        let mut tensor_layouts = HashMap::new();

        #[cfg(all(feature = "std", feature = "rayon"))]
        {
            let node_layouts: Vec<_> = script_ir
                .graph
                .topology
                .nodes
                .par_iter()
                .flat_map(|node| {
                    let mut layouts = Vec::new();
                    for (layout, &tensor_id) in node.input_layouts.iter().zip(&node.input_tensors) {
                        layouts.push((tensor_id, layout.clone()));
                    }
                    for (layout, &tensor_id) in node.output_layouts.iter().zip(&node.output_tensors) {
                        layouts.push((tensor_id, layout.clone()));
                    }
                    layouts
                })
                .collect();

            for (tensor_id, layout) in node_layouts {
                tensor_layouts.insert(tensor_id, layout);
            }
        }
        #[cfg(not(all(feature = "std", feature = "rayon")))]
        {
            for node in &script_ir.graph.topology.nodes {
                for (layout, &tensor_id) in node.input_layouts.iter().zip(&node.input_tensors) {
                    tensor_layouts.insert(tensor_id, layout.clone());
                }
                for (layout, &tensor_id) in node.output_layouts.iter().zip(&node.output_tensors) {
                    tensor_layouts.insert(tensor_id, layout.clone());
                }
            }
        }

        for input in &script_ir.graph.metadata.inputs {
            if let Some(tensor_info) = script_ir.graph.metadata.tensor_info.get(&input.tensor_id) {
                if let Some(ref shape) = tensor_info.shape {
                    // Avoid Vec allocation by collecting into a small stack array when possible
                    let shape_usize: Vec<usize> = shape.iter().map(|s| s.unwrap_or(1)).collect();
                    tensor_layouts.insert(input.tensor_id, Layout::from_shape(&shape_usize));
                }
            }
        }

        for output in &script_ir.graph.metadata.outputs {
            #[cfg(not(feature = "std"))]
            pub use alloc::collections::btree_map::Entry;
            #[cfg(feature = "std")]
            pub use std::collections::hash_map::Entry;
            // Don't overwrite output layouts if they already exist from node processing
            // This preserves transpose and other view operation layouts
            if let Entry::Vacant(e) = tensor_layouts.entry(output.tensor_id) {
                if let Some(tensor_info) = script_ir.graph.metadata.tensor_info.get(&output.tensor_id) {
                    if let Some(ref shape) = tensor_info.shape {
                        // Avoid Vec allocation by collecting into a small stack array when possible
                        let shape_usize: Vec<usize> = shape.iter().map(|s| s.unwrap_or(1)).collect();
                        e.insert(Layout::from_shape(&shape_usize));
                    }
                }
            }
        }

        tensor_layouts
    }

    pub(super) fn collect_tensor_dtypes(&self, script_ir: &ScriptIR, script: &Script) -> HashMap<TensorId, DType> {
        let mut tensor_dtypes = HashMap::new();

        #[cfg(all(feature = "std", feature = "rayon"))]
        {
            let dtypes: Vec<_> = script_ir
                .graph
                .metadata
                .tensor_info
                .par_iter()
                .filter_map(|(&tensor_id, tensor_info)| tensor_info.dtype.map(|dtype| (tensor_id, dtype)))
                .collect();

            for (tensor_id, dtype) in dtypes {
                tensor_dtypes.insert(tensor_id, dtype);
            }
        }
        #[cfg(not(all(feature = "std", feature = "rayon")))]
        {
            for (&tensor_id, tensor_info) in &script_ir.graph.metadata.tensor_info {
                if let Some(dtype) = tensor_info.dtype {
                    tensor_dtypes.insert(tensor_id, dtype);
                }
            }
        }

        // Override dtypes for input tensors with actual runtime input dtypes
        let runtime_inputs = script.get_inputs();
        for input in &script_ir.graph.metadata.inputs {
            if let Some(tensor) = runtime_inputs.get(&input.name) {
                let actual_dtype = tensor.get_dtype();
                tensor_dtypes.insert(input.tensor_id, actual_dtype);
            }
        }

        tensor_dtypes
    }

    pub(super) fn prepare_constant_storage(
        &self,
        script_ir: &ScriptIR,
    ) -> HoduResult<HashMap<TensorId, SharedStorage>> {
        #[cfg(feature = "std")]
        let mut constant_storage = HashMap::with_capacity(script_ir.graph.metadata.constants.len());
        #[cfg(not(feature = "std"))]
        let mut constant_storage = HashMap::new();

        #[cfg(all(feature = "std", feature = "rayon"))]
        {
            let constant_storages: Vec<_> = script_ir
                .graph
                .metadata
                .constants
                .par_iter()
                .map(|(tensor_id, constant_node)| {
                    let storage = match self.current_device {
                        Device::CPU => {
                            let storage = self.convert_constant_to_cpu_storage(constant_node)?;
                            Arc::new(HoduStorage::CPU(storage))
                        },
                        Device::CUDA(_) => {
                            return Err(HoduError::InternalError(
                                "CUDA constant conversion not implemented".to_string(),
                            ));
                        },
                        Device::Metal => {
                            let cpu_storage = self.convert_constant_to_cpu_storage(constant_node)?;
                            let metal_storage = MetalStorage::from_cpu_storage(&cpu_storage)?;
                            Arc::new(HoduStorage::Metal(metal_storage))
                        },
                    };
                    Ok((*tensor_id, storage))
                })
                .collect::<HoduResult<Vec<_>>>()?;

            for (tensor_id, storage) in constant_storages {
                constant_storage.insert(tensor_id, storage);
            }
        }
        #[cfg(not(all(feature = "std", feature = "rayon")))]
        {
            for (tensor_id, constant_node) in &script_ir.graph.metadata.constants {
                match self.current_device {
                    Device::CPU => {
                        let storage = self.convert_constant_to_cpu_storage(constant_node)?;
                        constant_storage.insert(*tensor_id, Arc::new(HoduStorage::CPU(storage)));
                    },
                    Device::CUDA(_) => {
                        return Err(HoduError::InternalError(
                            "CUDA constant conversion not implemented".to_string(),
                        ));
                    },
                    Device::Metal => {
                        let cpu_storage = self.convert_constant_to_cpu_storage(constant_node)?;
                        let metal_storage = MetalStorage::from_cpu_storage(&cpu_storage)?;
                        constant_storage.insert(*tensor_id, Arc::new(HoduStorage::Metal(metal_storage)));
                    },
                }
            }
        }

        Ok(constant_storage)
    }

    fn convert_constant_to_cpu_storage(
        &self,
        constant: &crate::script::ir::ConstantNode,
    ) -> HoduResult<crate::be_hodu::cpu::storage::CpuStorage> {
        use crate::be_hodu::cpu::storage::CpuStorage;
        use crate::script::ir::CompressionType;
        use crate::types::dtype::DType;
        use float8::{F8E4M3, F8E5M2};
        use half::{bf16, f16};

        // Handle decompression if needed
        let data = match &constant.compression {
            #[cfg(all(feature = "serde", feature = "std"))]
            Some(CompressionType::Gzip) => {
                use std::io::Read;
                let mut decoder = flate2::read::GzDecoder::new(&constant.data[..]);
                let mut decompressed = Vec::new();
                decoder
                    .read_to_end(&mut decompressed)
                    .map_err(|e| HoduError::DecompressionError(e.to_string()))?;
                decompressed
            },
            #[cfg(not(all(feature = "serde", feature = "std")))]
            Some(CompressionType::Gzip) => {
                return Err(HoduError::InternalError(
                    "Gzip decompression requires both 'serde' and 'std' features to be enabled".to_string(),
                ));
            },
            Some(CompressionType::Zstd) => {
                return Err(HoduError::InternalError(
                    "Zstd decompression not implemented".to_string(),
                ));
            },
            _ => constant.data.clone(),
        };

        let elem_count = constant.shape.iter().product::<usize>();

        let cpu_storage = match constant.dtype {
            DType::BOOL => {
                let values: Vec<bool> = data
                    .chunks_exact(1)
                    .take(elem_count)
                    .map(|chunk| chunk[0] != 0)
                    .collect();
                CpuStorage::BOOL(values)
            },
            DType::F8E4M3 => {
                let values: Vec<F8E4M3> = data
                    .chunks_exact(1)
                    .take(elem_count)
                    .map(|chunk| F8E4M3::from_bits(chunk[0]))
                    .collect();
                CpuStorage::F8E4M3(values)
            },
            DType::F8E5M2 => {
                let values: Vec<F8E5M2> = data
                    .chunks_exact(1)
                    .take(elem_count)
                    .map(|chunk| F8E5M2::from_bits(chunk[0]))
                    .collect();
                CpuStorage::F8E5M2(values)
            },
            DType::BF16 => {
                let values: Vec<bf16> = data
                    .chunks_exact(2)
                    .take(elem_count)
                    .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                CpuStorage::BF16(values)
            },
            DType::F16 => {
                let values: Vec<f16> = data
                    .chunks_exact(2)
                    .take(elem_count)
                    .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                CpuStorage::F16(values)
            },
            DType::F32 => {
                let values: Vec<f32> = data
                    .chunks_exact(4)
                    .take(elem_count)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                CpuStorage::F32(values)
            },
            DType::F64 => {
                let values: Vec<f64> = data
                    .chunks_exact(8)
                    .take(elem_count)
                    .map(|chunk| {
                        f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                        ])
                    })
                    .collect();
                CpuStorage::F64(values)
            },
            DType::U8 => {
                let values: Vec<u8> = data.into_iter().take(elem_count).collect();
                CpuStorage::U8(values)
            },
            DType::U16 => {
                let values: Vec<u16> = data
                    .chunks_exact(2)
                    .take(elem_count)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                CpuStorage::U16(values)
            },
            DType::U32 => {
                let values: Vec<u32> = data
                    .chunks_exact(4)
                    .take(elem_count)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                CpuStorage::U32(values)
            },
            DType::U64 => {
                let values: Vec<u64> = data
                    .chunks_exact(8)
                    .take(elem_count)
                    .map(|chunk| {
                        u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                        ])
                    })
                    .collect();
                CpuStorage::U64(values)
            },
            DType::I8 => {
                let values: Vec<i8> = data.into_iter().take(elem_count).map(|b| b as i8).collect();
                CpuStorage::I8(values)
            },
            DType::I16 => {
                let values: Vec<i16> = data
                    .chunks_exact(2)
                    .take(elem_count)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                CpuStorage::I16(values)
            },
            DType::I32 => {
                let values: Vec<i32> = data
                    .chunks_exact(4)
                    .take(elem_count)
                    .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                CpuStorage::I32(values)
            },
            DType::I64 => {
                let values: Vec<i64> = data
                    .chunks_exact(8)
                    .take(elem_count)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                        ])
                    })
                    .collect();
                CpuStorage::I64(values)
            },
        };

        Ok(cpu_storage)
    }
}
