# Contribution Guide

## Development Environment

**Requirements:**
- Rust 1.90.0 or higher

**Optional (for specific features):**
- CUDA Toolkit (for `cuda` feature)
- Xcode Command Line Tools (for `metal` feature on macOS)
- LLVM/Clang (for `xla` feature - requires 8GB+ RAM, 20GB+ disk)
- clang-format (for Metal shader formatting)
- ruff (for Python formatting)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/hodu.git
   cd hodu
   ```
3. Build the project:
   ```bash
   cargo build
   ```
4. Run tests:
   ```bash
   cargo test
   ```
5. Check and format code before submitting:
   ```bash
   bash ./tools/check.sh   # Run lints and checks
   bash ./tools/format.sh  # Format code
   ```
