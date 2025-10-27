import torch
import torch.nn as nn
import time
import sys


class BenchMode:
    DYNAMIC_CPU = "dynamic-cpu"
    DYNAMIC_CUDA = "dynamic-cuda"
    DYNAMIC_METAL = "dynamic-metal"
    STATIC_CPU = "static-cpu"
    STATIC_CUDA = "static-cuda"
    STATIC_METAL = "static-metal"

    @staticmethod
    def get_name(mode):
        names = {
            BenchMode.DYNAMIC_CPU: "Dynamic CPU",
            BenchMode.DYNAMIC_CUDA: "Dynamic CUDA",
            BenchMode.DYNAMIC_METAL: "Dynamic Metal",
            BenchMode.STATIC_CPU: "Static CPU",
            BenchMode.STATIC_CUDA: "Static CUDA",
            BenchMode.STATIC_METAL: "Static Metal",
        }
        return names.get(mode, mode)

    @staticmethod
    def get_device(mode):
        if "cpu" in mode:
            return torch.device("cpu")
        elif "cuda" in mode:
            try:
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    return torch.device("cuda")
            except Exception:
                pass
            return torch.device("cpu")
        elif "metal" in mode:
            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return torch.device("mps")
            except Exception:
                pass
            return torch.device("cpu")
        return torch.device("cpu")

    @staticmethod
    def synchronize(device):
        """Synchronize device if needed"""
        if device.type == "cuda":
            try:
                if hasattr(torch, "cuda"):
                    torch.cuda.synchronize()
            except Exception:
                pass
        elif device.type == "mps":
            try:
                if hasattr(torch, "mps"):
                    torch.mps.synchronize()
            except Exception:
                pass


class MLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.fc2 = nn.Linear(hidden_features, hidden_features, bias=True)
        self.fc3 = nn.Linear(hidden_features, out_features, bias=True)
        self.gelu = nn.GELU()

        # Projection for residual connection if dimensions don't match
        self.projection = None
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        # Save input for residual connection
        identity = x if self.projection is None else self.projection(x)

        # Forward through layers
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)

        # Residual connection
        return x + identity


def benchmark_dynamic(
    mode, batch_size, in_features, hidden_features, out_features, warmup, iterations
):
    device = BenchMode.get_device(mode)

    # Create model and input
    model = MLPBlock(in_features, hidden_features, out_features).to(device)
    model.eval()
    x = torch.randn(batch_size, in_features, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            BenchMode.synchronize(device)

    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
            BenchMode.synchronize(device)
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_static(
    mode, batch_size, in_features, hidden_features, out_features, warmup, iterations
):
    device = BenchMode.get_device(mode)

    # Create model and input
    model = MLPBlock(in_features, hidden_features, out_features).to(device)
    model.eval()
    x = torch.randn(batch_size, in_features, device=device)

    # Compile the model (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    # Warmup (first call triggers compilation)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            BenchMode.synchronize(device)

    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
            BenchMode.synchronize(device)
    elapsed = time.time() - start

    return elapsed / iterations


def run_benchmark(mode, configs, warmup, iterations):
    print(f"mode={BenchMode.get_name(mode)}")
    print(f"warmup={warmup}")
    print(f"iterations={iterations}")

    timed_out = False

    for batch_size, in_features, hidden_features, out_features in configs:
        if timed_out:
            print(
                f"{batch_size}x{in_features}x{hidden_features}x{out_features},TIMEOUT"
            )
            continue

        try:
            if "static" in mode:
                time_seconds = benchmark_static(
                    mode,
                    batch_size,
                    in_features,
                    hidden_features,
                    out_features,
                    warmup,
                    iterations,
                )
            else:
                time_seconds = benchmark_dynamic(
                    mode,
                    batch_size,
                    in_features,
                    hidden_features,
                    out_features,
                    warmup,
                    iterations,
                )

            time_ms = time_seconds * 1000
            print(
                f"{batch_size}x{in_features}x{hidden_features}x{out_features},time_ms={time_ms:.6f}ms"
            )
        except Exception as e:
            if "timeout" in str(e).lower():
                print(
                    f"{batch_size}x{in_features}x{hidden_features}x{out_features},TIMEOUT"
                )
                timed_out = True
            else:
                print(
                    f"{batch_size}x{in_features}x{hidden_features}x{out_features},ERROR"
                )


def main():
    if len(sys.argv) != 2:
        print("Usage: python _torch.py <mode>")
        print("\nAvailable modes:")
        print("  dynamic-cpu     - Dynamic execution on CPU")
        print("  dynamic-cuda    - Dynamic execution on CUDA")
        print("  dynamic-metal   - Dynamic execution on Metal (MPS)")
        print("  static-cpu      - Static graph (torch.compile) on CPU")
        print("  static-cuda     - Static graph (torch.compile) on CUDA")
        print("  static-metal    - Static graph (torch.compile) on Metal (MPS)")
        sys.exit(1)

    mode = sys.argv[1]

    # MLP configs: (batch_size, in_features, hidden_features, out_features)
    configs = [
        (32, 256, 512, 256),
        (64, 512, 1024, 512),
        (128, 768, 2048, 1024),
    ]

    warmup = 5
    iterations = 10

    run_benchmark(mode, configs, warmup, iterations)


if __name__ == "__main__":
    main()
