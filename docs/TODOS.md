# TODOS.md

**Serialization:** (🟡 Important)
- To be determined

**Backend:** (🔴 Critical)
- [x] CPU SIMD support
- [x] CPU parallelization support
- [x] CUDA support
- [x] Metal support
- [ ] OS-provided BLAS support
  - [x] aarch64-apple-darwin (Accelerate framework)
  - [ ] x86_64-apple-darwin (Accelerate framework)
  - [ ] x86_64-unknown-linux-gnu (system BLAS)
  - [ ] aarch64-unknown-linux-gnu (system BLAS)

**Tensor Creation & Initialization:** (🔴 Critical)
- [ ] Implement initialization functions (xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal)
- [ ] Implement basic creation ops (eye, arange, linspace, uniform, normal)

**ONNX Compatibility - Tensor Operations:** (🔴 Critical)
- [x] Implement padding operations (pad_constant, pad_reflect, pad_replicate, pad_circular)
- [x] Implement flip operation
- [x] Implement repeat operation (ONNX: Tile)
- [x] Implement expand operation (broadcast)
- [ ] Implement ceil, floor, round operations (ONNX: Ceil, Floor, Round)
- [ ] Implement cumsum operation (ONNX: CumSum)

**ONNX Compatibility - Unary Operations:** (🟡 Important)
- [ ] Implement erf (ONNX: Erf) - used in accurate GELU
- [ ] Implement inverse trigonometric (asin, acos, atan)
- [ ] Implement hyperbolic (sinh, cosh, atanh)
- [ ] Implement hardswish, hardsigmoid (ONNX: HardSwish, HardSigmoid)

**ONNX Compatibility - Other Operations:** (🟡 Important)
- [ ] Implement einsum (ONNX: Einsum)
- [ ] Implement resize/upsample (ONNX: Resize)
- [ ] Implement topk (ONNX: TopK)
- [ ] Implement nonzero (ONNX: NonZero)
- [ ] Implement onehot (ONNX: OneHot)
- [ ] Implement mod/fmod (ONNX: Mod)

**Recurrent Layers:** (🔴 Critical)
- [ ] Implement RNN
- [ ] Implement LSTM
- [ ] Implement GRU

**Attention Layers:** (🔴 Critical)
- [ ] Implement MultiheadAttention
- [ ] Implement ScaledDotProductAttention

**Pooling Layers:**
- [x] Implement pooling layers
- [ ] Implement GlobalAvgPool, GlobalMaxPool (🟡 Important)
- [ ] Implement FractionalMaxPool (🟢 Nice-to-have)

**Normalization Layers:**
- [x] Implement normalization layers
- [ ] Implement GroupNorm, InstanceNorm (🟡 Important)
- [ ] Implement RMSNorm (🟡 Important)

**Activation Functions:** (🟡 Important)
- [x] Implement Swish/SiLU, Mish
- [x] Implement PReLU, RReLU

**Loss Functions:** (🟡 Important)
- [ ] Implement SmoothL1Loss
- [ ] Implement KLDivLoss
- [ ] Implement CosineEmbeddingLoss

**Optimizers:** (🟡 Important)
- [ ] Implement RMSprop
- [ ] Implement Adagrad

**DataSet**
- [x] Implement Dataset
