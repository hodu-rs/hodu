# hodu-cli

Command-line interface for the Hodu ML framework.

## Installation

```bash
cargo install hodu-cli
```

## Commands

### Model Operations

| Command | Description |
|---------|-------------|
| `hodu run <model> -i name=path` | Run model inference |
| `hodu build <model> -o output` | AOT compile model to native artifact |
| `hodu convert <input> -o output` | Convert models/tensors between formats |
| `hodu inspect <file>` | Inspect model or tensor file |
| `hodu version` | Show version information |

### Plugin Management

| Command | Description |
|---------|-------------|
| `hodu plugin list` | List installed plugins |
| `hodu plugin info <name>` | Show detailed plugin information |
| `hodu plugin install <name>` | Install plugin from official registry |
| `hodu plugin install --path <dir>` | Install plugin from local path |
| `hodu plugin install --git <url> [--subdir <path>]` | Install plugin from git repository |
| `hodu plugin remove <name>` | Remove installed plugin |
| `hodu plugin update [name]` | Update plugin(s) from source |
| `hodu plugin enable <name>` | Enable a disabled plugin |
| `hodu plugin disable <name>` | Disable a plugin without removing |
| `hodu plugin verify` | Verify plugin integrity |
| `hodu plugin create <name> -t <type>` | Create new plugin project |

## Usage Examples

### Run Inference

```bash
# Run with CPU backend
hodu run model.onnx -i input=data.hdt -d cpu

# Run with CUDA device
hodu run model.hdss -i x=input.hdt -d cuda::0

# Save outputs to directory
hodu run model.onnx -i input=data.hdt --save ./outputs
```

### Build Model

```bash
# Build shared library
hodu build model.hdss -o model.so -d cpu

# Build Metal library for Apple Silicon
hodu build model.hdss -o model.metallib -d metal

# Build with specific format
hodu build model.onnx -o model.a -f staticlib
```

### Convert Formats

```bash
# Convert ONNX to HDSS
hodu convert model.onnx -o model.hdss

# Convert tensor formats
hodu convert data.npy -o data.hdt

# Verbose output
hodu convert model.onnx -o model.hdss -v
```

### Inspect Files

```bash
# Inspect model
hodu inspect model.hdss

# Inspect with verbose output
hodu inspect model.onnx -v

# Output as JSON
hodu inspect model.hdss -f json
```

### Plugin Management

```bash
# Install from official registry (recommended)
hodu plugin install aot-cpu

# Install from local development
hodu plugin install --path ./my-plugin

# Install from git repository
hodu plugin install --git https://github.com/user/hodu-backend-cuda

# Install from git with subdirectory
hodu plugin install --git https://github.com/daminstudio/hodu-plugins --subdir hodu-backend-aot-cpu-plugin

# Install specific tag/branch
hodu plugin install --git https://github.com/user/plugin --tag v1.0.0

# Create new backend plugin
hodu plugin create my-backend -t backend

# Create new model format plugin (e.g., ONNX loader)
hodu plugin create my-format -t model_format

# Create new tensor format plugin (e.g., NPY loader)
hodu plugin create my-tensor -t tensor_format

# Enable/disable plugins
hodu plugin disable aot-cpu
hodu plugin enable aot-cpu
```

### Official Plugins

Official plugins are available at [hodu-plugins](https://github.com/daminstudio/hodu-plugins).

| Plugin | Description |
|--------|-------------|
| `aot-cpu` | AOT compiler for CPU via C code generation |

## Plugin Types

| Type | Description | Capabilities |
|------|-------------|--------------|
| `backend` | Execute/compile models | `backend.run`, `backend.build` |
| `model_format` | Load/save model files | `format.load_model`, `format.save_model` |
| `tensor_format` | Load/save tensor files | `format.load_tensor`, `format.save_tensor` |

## Plugin Dependencies

Plugins can declare dependencies on other plugins in their `manifest.json`:

```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "capabilities": ["backend.run"],
  "dependencies": ["hodu-format-onnx"]
}
```

When installing a plugin with dependencies, missing dependencies will be reported as warnings. When disabling a plugin, it will be blocked if other enabled plugins depend on it.

Additional metadata fields supported in `manifest.json`:
- `description`: Short description of the plugin
- `license`: License identifier (e.g., "MIT", "Apache-2.0")

## Device Naming Convention

Devices use lowercase names with `::` separator for device index:

| Device | Format |
|--------|--------|
| CPU | `cpu` |
| CUDA | `cuda`, `cuda::0`, `cuda::1` |
| Metal | `metal` |
| ROCm | `rocm`, `rocm::0` |
| Vulkan | `vulkan` |
| WebGPU | `webgpu` |

## Supported Formats

### Builtin

| Extension | Type | Description |
|-----------|------|-------------|
| `.hdss` | Model | Hodu model snapshot |
| `.hdt` | Tensor | Hodu tensor data |
| `.json` | Tensor | JSON tensor format |

### Via Plugins

Install format plugins to support additional formats (ONNX, TensorFlow, NumPy, etc.)

## License

BSD-3-Clause
