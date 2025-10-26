# Benchmarks

Performance benchmarks comparing Hodu against popular deep learning frameworks.

## Available Benchmarks

- **matmul/** - Matrix multiplication benchmarks comparing Hodu, PyTorch, and TensorFlow

## Requirements

- **Python**: 3.11.14
- **Rust**: 1.90.0 or higher

## Setup

### Basic Installation

```bash
pip install -r requirements.txt
```

### Metal Support (macOS/Apple Silicon)

For GPU acceleration on macOS:

```bash
# TensorFlow Metal support (requires tensorflow-macos)
pip uninstall tensorflow
pip install tensorflow-macos tensorflow-metal
```

**Note:**
- PyTorch includes built-in MPS (Metal Performance Shaders) support
- TensorFlow on macOS requires `tensorflow-macos` instead of `tensorflow` for Metal support

## Running Benchmarks

See individual benchmark directories for specific instructions.
