# Benchmarks

Performance benchmarks comparing Hodu against popular deep learning frameworks.

## Available Benchmarks

- **matmul/** - Matrix multiplication benchmarks

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
python3 run.py          # CPU only
python3 run.py --metal  # With Metal (macOS)
python3 run.py --cuda   # With CUDA (NVIDIA GPUs)
python3 run.py --xla    # With XLA
```
