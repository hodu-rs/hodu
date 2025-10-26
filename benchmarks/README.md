# Benchmarks

Performance benchmarks comparing Hodu against popular deep learning frameworks.

## Requirements

- **Python**: 3.11.x
- **Rust**: 1.90.0 or higher

## Setup

```bash
python3 setup.py
```

## Running Benchmarks

Navigate to each benchmark directory and run:

```bash
cd <benchmark_folder>

# Run specific backends (combine flags as needed)
python3 run.py --cpu            # CPU only
python3 run.py --cuda           # CUDA only (NVIDIA GPUs)
python3 run.py --metal          # Metal only (macOS)
python3 run.py --xla            # XLA only
python3 run.py --cpu --cuda     # CPU + CUDA
python3 run.py --cpu --metal    # CPU + Metal (macOS)
```
