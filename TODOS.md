# TODOS.md
This document outlines goals for **Release**, not Pre-Release.
Items may be added at any time.

## hodu_core
**Serialization:** (🟡 Important)
- To be determined

**Embedded System Stabilization:** (🟢 Nice-to-have)
- [ ] Stabilize embedded platform support

**Backend:** (🔴 Critical)
- [ ] CPU SIMD support
- [ ] CUDA support
- [ ] Metal support

**Scripting:** (🟢 Nice-to-have)
- [ ] XLA optimization

**Tensor Creation & Initialization:** (🔴 Critical)
- [ ] Implement initialization functions (xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal)
- [ ] Implement basic creation ops (eye, arange, linspace, uniform, normal)

## hodu_nn
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
- [ ] Implement Swish/SiLU, Mish
- [ ] Implement PReLU, RReLU

**Loss Functions:** (🟡 Important)
- [ ] Implement SmoothL1Loss
- [ ] Implement KLDivLoss
- [ ] Implement CosineEmbeddingLoss

**Optimizers:** (🟡 Important)
- [ ] Implement RMSprop
- [ ] Implement Adagrad

**Tensor Operations:** (🟡 Important)
- [ ] Implement padding operations (pad, pad_constant, pad_reflect)
- [ ] Implement repeat, tile, expand operations (🟢 Nice-to-have)
- [ ] Implement one_hot, topk, sort, argsort (🟢 Nice-to-have)

## hodu_data (🟡 Important)
- [ ] Implement Dataset

## hodu_onnx (🟢 Nice-to-have)
- [ ] ONNX integration
