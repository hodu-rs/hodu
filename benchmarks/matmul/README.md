# Matrix Multiplication Benchmark

Compares matrix multiplication performance across **Hodu**, **PyTorch**, **JAX**, and **TensorFlow**.

## Quick Start

```bash
# CPU only
python3 run.py

# With Metal (macOS)
python3 run.py --metal

# With CUDA (NVIDIA GPUs)
python3 run.py --cuda

# With XLA
python3 run.py --xla
```

## Test Configurations

**Matrix sizes:** 128×128×128, 256×256×256, 512×512×512, 1024×1024×1024, 2048×2048×2048

**Settings:** 5 warmup iterations, 10 benchmark iterations, 1-second timeout

## Execution Modes

### Hodu (`_hodu.rs`)
- `dynamic-cpu`, `dynamic-metal`
- `static-cpu`, `static-metal`, `static-xla`

```bash
cargo run --release --bin hodu -- static-cpu
cargo run --release --bin hodu --features=metal -- static-metal
```

### PyTorch (`_torch.py`)
- `dynamic-cpu`, `dynamic-cuda`, `dynamic-mps`
- `static-cpu`, `static-cuda`, `static-mps` (uses `torch.compile`)

### JAX (`_jax.py`)
- `dynamic-cpu`, `dynamic-gpu`
- `static-cpu`, `static-gpu` (uses `jax.jit`)

### TensorFlow (`_tensorflow.py`)
- `dynamic-cpu`, `dynamic-gpu`
- `static-cpu`, `static-gpu` (uses `tf.function`)

## Prerequisites

```bash
pip install -r ../requirements.txt
```

Requires: Rust toolchain, Python 3.x, PyTorch, JAX, TensorFlow (optional)

## Output

Generates a comparison table with timing results and speedup ratios versus baseline (Hodu Static CPU).
