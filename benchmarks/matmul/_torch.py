import torch
import time
import sys


class BenchMode:
    DYNAMIC_CPU = "dynamic-cpu"
    DYNAMIC_CUDA = "dynamic-cuda"
    DYNAMIC_MPS = "dynamic-mps"
    STATIC_CPU = "static-cpu"
    STATIC_CUDA = "static-cuda"
    STATIC_MPS = "static-mps"

    @staticmethod
    def get_name(mode):
        names = {
            BenchMode.DYNAMIC_CPU: "Dynamic CPU",
            BenchMode.DYNAMIC_CUDA: "Dynamic CUDA",
            BenchMode.DYNAMIC_MPS: "Dynamic MPS",
            BenchMode.STATIC_CPU: "Static CPU",
            BenchMode.STATIC_CUDA: "Static CUDA",
            BenchMode.STATIC_MPS: "Static MPS",
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
        elif "mps" in mode:
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


def benchmark_dynamic(mode, m, k, n, warmup, iterations):
    device = BenchMode.get_device(mode)

    # Create random tensors
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)

    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(a, b)
        BenchMode.synchronize(device)

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(a, b)
        BenchMode.synchronize(device)
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_static(mode, m, k, n, warmup, iterations):
    device = BenchMode.get_device(mode)

    # Create random tensors
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)

    # Compile the matmul operation (PyTorch 2.0+)
    if hasattr(torch, "compile"):

        @torch.compile
        def matmul_compiled(x, y):
            return torch.matmul(x, y)
    else:
        # Fallback for older PyTorch versions
        def matmul_compiled(x, y):
            return torch.matmul(x, y)

    # Warmup
    for _ in range(warmup):
        _ = matmul_compiled(a, b)
        BenchMode.synchronize(device)

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        _ = matmul_compiled(a, b)
        BenchMode.synchronize(device)
    elapsed = time.time() - start

    return elapsed / iterations


def run_benchmark(mode, configs, warmup, iterations):
    print(f"mode={BenchMode.get_name(mode)}")
    print(f"warmup={warmup}")
    print(f"iterations={iterations}")

    for m, k, n in configs:
        if "static" in mode:
            time_sec = benchmark_static(mode, m, k, n, warmup, iterations)
        else:
            time_sec = benchmark_dynamic(mode, m, k, n, warmup, iterations)

        print(f"{m}x{k}x{n},{time_sec * 1000:.6f}ms")


def print_usage():
    print("Usage: python _torch.py <mode>")
    print("\nAvailable modes:")
    print("  dynamic-cpu     - Dynamic execution on CPU")

    # Check CUDA availability
    cuda_available = False
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            cuda_available = True
            print("  dynamic-cuda    - Dynamic execution on CUDA")
    except Exception:
        pass

    # Check MPS availability
    mps_available = False
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mps_available = True
            print("  dynamic-mps     - Dynamic execution on MPS (Apple Silicon)")
    except Exception:
        pass

    print("  static-cpu      - Static computation graph on CPU (torch.compile)")

    if cuda_available:
        print("  static-cuda     - Static computation graph on CUDA (torch.compile)")
    if mps_available:
        print("  static-mps      - Static computation graph on MPS (torch.compile)")


def get_valid_modes():
    """Get list of valid modes based on available hardware"""
    valid_modes = [
        BenchMode.DYNAMIC_CPU,
        BenchMode.STATIC_CPU,
    ]

    # Check CUDA
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            valid_modes.extend([BenchMode.DYNAMIC_CUDA, BenchMode.STATIC_CUDA])
    except Exception:
        pass

    # Check MPS
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            valid_modes.extend([BenchMode.DYNAMIC_MPS, BenchMode.STATIC_MPS])
    except Exception:
        pass

    return valid_modes


def main():
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1]
    valid_modes = get_valid_modes()

    if mode not in valid_modes:
        print(f"Error: Invalid mode '{mode}'")
        print_usage()
        sys.exit(1)

    configs = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    warmup = 5
    iterations = 10

    run_benchmark(mode, configs, warmup, iterations)


if __name__ == "__main__":
    main()
