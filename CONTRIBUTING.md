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
   bash ./tools/format.sh  # Format code
   bash ./tools/check.sh   # Run lints and checks

   # To test additional features (e.g., metal):
   bash ./tools/check.sh -f metal
   bash ./tools/check.sh --features metal

   # Multiple features:
   bash ./tools/check.sh -f metal cuda
   ```

## Commit Guidelines

We follow a conventional commit style for clear and consistent commit history.

**Format:**
```
<type>: <description>
<type>(<scope>): <description>
```

**Types:**
- `feat`: New feature or functionality
- `fix`: Bug fix
- `refactor`: Code restructuring without changing functionality
- `chore`: Maintenance tasks, dependency updates, configuration changes
- `docs`: Documentation updates
- `merge`: Merge commits

**Scope (optional):**
- Specific component or area affected (e.g., `benchmark`, `deps`, `metal`, `simd`)

**Examples:**
```
feat: add blocking, packing, and parallelization to NEON SIMD matmul
fix: use Metal storage for Metal device in script executor
refactor(benchmark): remove redundant cargo build step before cargo run
chore(deps): update burn requirement from 0.16.0 to 0.18.0
docs: add contribution guide and related projects section
```

**Best Practices:**
- Use lowercase for type and description
- Keep descriptions concise and clear
- Start descriptions with a verb (add, fix, update, remove, etc.)
- Don't end with a period
