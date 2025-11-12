import torch
import numpy as np
import time
import sys
import gc

# Try to import psutil for memory tracking
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return None
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def trimmed_mean(times, trim_ratio=0.1):
    """Calculate trimmed mean by removing top and bottom percentiles"""
    times = np.array(times)
    times.sort()
    trim_count = int(len(times) * trim_ratio)
    if trim_count > 0:
        trimmed = times[trim_count:-trim_count]
    else:
        trimmed = times
    return np.mean(trimmed)


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


def benchmark_dynamic(mode, m, k, n, warmup, iterations):
    gc.collect()
    device = BenchMode.get_device(mode)

    # Create random tensors
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)

    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(a, b)
        BenchMode.synchronize(device)

    gc.collect()
    mem_before = get_memory_usage_mb()
    peak_mem = mem_before if mem_before else 0

    # Benchmark - collect individual iteration times
    times = []
    bench_start = time.time()
    for i in range(iterations):
        start = time.time()
        _ = torch.matmul(a, b)
        BenchMode.synchronize(device)
        times.append(time.time() - start)

        # Track peak memory
        current_mem = get_memory_usage_mb()
        if current_mem and current_mem > peak_mem:
            peak_mem = current_mem

        # Check timeout after each iteration
        total_elapsed = time.time() - bench_start
        if total_elapsed > 10.0:
            raise TimeoutError(f"TIMEOUT: Exceeded 10 seconds after {i + 1} iterations")

    mem_after = get_memory_usage_mb()
    mem_used_mb = None
    if mem_before and mem_after:
        mem_used_mb = peak_mem - mem_before

    return trimmed_mean(times), mem_used_mb


def benchmark_static(mode, m, k, n, warmup, iterations):
    gc.collect()
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

    gc.collect()
    mem_before = get_memory_usage_mb()
    peak_mem = mem_before if mem_before else 0

    # Benchmark - collect individual iteration times
    times = []
    bench_start = time.time()
    for i in range(iterations):
        start = time.time()
        _ = matmul_compiled(a, b)
        BenchMode.synchronize(device)
        times.append(time.time() - start)

        # Track peak memory
        current_mem = get_memory_usage_mb()
        if current_mem and current_mem > peak_mem:
            peak_mem = current_mem

        # Check timeout after each iteration
        total_elapsed = time.time() - bench_start
        if total_elapsed > 10.0:
            raise TimeoutError(f"TIMEOUT: Exceeded 10 seconds after {i + 1} iterations")

    mem_after = get_memory_usage_mb()
    mem_used_mb = None
    if mem_before and mem_after:
        mem_used_mb = peak_mem - mem_before

    return trimmed_mean(times), mem_used_mb


def run_benchmark(mode, configs, warmup, iterations):
    print(f"mode={BenchMode.get_name(mode)}")
    print(f"warmup={warmup}")
    print(f"iterations={iterations}")

    timed_out = False

    for m, k, n in configs:
        # If we already timed out, skip remaining benchmarks
        if timed_out:
            print(f"{m}x{k}x{n},TIMEOUT")
            continue

        try:
            if "static" in mode:
                time_sec, mem_mb = benchmark_static(mode, m, k, n, warmup, iterations)
            else:
                time_sec, mem_mb = benchmark_dynamic(mode, m, k, n, warmup, iterations)

            if mem_mb is not None:
                print(
                    f"{m}x{k}x{n},time_ms={time_sec * 1000:.6f}ms,mem_mb={mem_mb:.2f}"
                )
            else:
                print(f"{m}x{k}x{n},time_ms={time_sec * 1000:.6f}ms")
        except TimeoutError:
            print(f"{m}x{k}x{n},TIMEOUT")
            timed_out = True
        except Exception:
            print(f"{m}x{k}x{n},ERROR")


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

    # Check Metal/MPS availability
    metal_available = False
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            metal_available = True
            print("  dynamic-metal   - Dynamic execution on Metal (Apple Silicon)")
    except Exception:
        pass

    print("  static-cpu      - Static computation graph on CPU (torch.compile)")

    if cuda_available:
        print("  static-cuda     - Static computation graph on CUDA (torch.compile)")
    if metal_available:
        print("  static-metal    - Static computation graph on Metal (torch.compile)")


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

    # Check Metal/MPS
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            valid_modes.extend([BenchMode.DYNAMIC_METAL, BenchMode.STATIC_METAL])
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
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]

    warmup = 100
    iterations = 100

    run_benchmark(mode, configs, warmup, iterations)


if __name__ == "__main__":
    main()
