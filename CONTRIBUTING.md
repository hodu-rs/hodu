# Contributing to Hodu

## Project Structure

```
hodu/
├── hodu-lib/              # Main library (re-exports all crates)
├── hodu-cli/              # Command-line interface
├── hodu-plugin-sdk/       # Plugin SDK for backends/formats
├── crates/
│   ├── hodu_core/         # Core tensor operations
│   ├── hodu_nn/           # Neural network layers
│   ├── hodu_datasets/     # Dataset loaders
│   ├── hodu_cpu_kernels/  # CPU SIMD kernels
│   ├── hodu_metal_kernels/# Metal GPU kernels (macOS)
│   ├── hodu_cuda_kernels/ # CUDA GPU kernels
│   ├── hodu_internal/     # Internal utilities
│   └── hodu_macro_utils/  # Macro utilities
└── tools/                 # Development scripts
```

## Requirements

- Rust 1.90.0 or higher
- (Optional) CUDA Toolkit, Xcode Command Line Tools, clang-format

## Getting Started

```bash
git clone https://github.com/daminstudio/hodu.git
cd hodu
cargo build
cargo test
./tools/format.sh
./tools/check-lib.sh        # Check hodu-lib
./tools/check-cli.sh        # Check hodu-cli
./tools/check-plugin-sdk.sh # Check hodu-plugin-sdk
```

## Commit Style

```
<type>(<scope>): <description>
```

**Types:** `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `chore`

**Examples:**
```
feat(nn): add batch normalization layer
fix(core): correct broadcasting for matmul
chore(deps): bump serde to 1.0.228
```

## Pull Request

1. Create a feature branch from `main`
2. Squash commits into one before submitting
3. Ensure tests pass and code is formatted
4. Add appropriate labels:
   - Category: `c-bug`, `c-feature`, `c-docs`, `c-performance`, `c-dependencies`
   - Component: `m-lib`, `m-cli`, `m-gui`, `m-plugin-sdk`
   - Crate: `h-core`, `h-nn`, `h-datasets`, `h-kernels(cpu)`, `h-kernels(cuda)`, `h-kernels(metal)`
   - OS: `o-linux`, `o-macos`, `o-windows`
