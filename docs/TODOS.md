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
- [x] Implement basic creation ops (eye, arange, linspace, uniform, normal)

**ONNX Compatibility - Tensor Operations:** (🔴 Critical)
- [x] Implement padding operations (pad_constant, pad_reflect, pad_replicate, pad_circular)
- [x] Implement flip operation
- [x] Implement repeat operation (ONNX: Tile)
- [x] Implement expand operation (broadcast)
- [x] Implement ceil, floor, round operations (ONNX: Ceil, Floor, Round)
- [x] Implement cumsum operation (ONNX: CumSum)

**ONNX Compatibility - Unary Operations:** (🟡 Important)
- [x] Implement erf (ONNX: Erf) - used in accurate GELU
- [x] Implement inverse trigonometric (asin, acos, atan)
- [x] Implement hyperbolic (sinh, cosh, tanh)
- [x] Implement inverse hyperbolic (asinh, acosh, atanh)
- [x] Implement hardswish(hardsilu), hardsigmoid (ONNX: HardSwish, HardSigmoid)

**ONNX Compatibility - Other Operations:** (🟡 Important)
- [x] Implement einsum (ONNX: Einsum)
- [x] Implement resize/upsample (ONNX: Resize)
- [ ] Implement topk (ONNX: TopK)
- [ ] Implement nonzero (ONNX: NonZero)
- [x] Implement onehot (ONNX: OneHot)
- [x] Implement rem (ONNX: Mod)

**Recurrent Layers:** (🔴 Critical)
- [x] Implement RNN
- [x] Implement LSTM
- [x] Implement GRU

**Attention Layers:** (🔴 Critical)
- [x] Implement MultiheadAttention
- [x] Implement ScaledDotProductAttention

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
