import sys
import numpy as np
import time
import gc

import jax
import jax.numpy as jnp


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
    def set_device(mode):
        """Set JAX default device based on mode"""
        if "cpu" in mode:
            jax.config.update("jax_default_device", jax.devices("cpu")[0])
        elif "cuda" in mode:
            # Try to get CUDA device
            all_devices = jax.devices()
            cuda_devices = [d for d in all_devices if d.platform == "gpu"]

            if not cuda_devices:
                print("Error: CUDA mode requested but no CUDA device detected")
                print(f"Available devices: {all_devices}")
                raise RuntimeError("CUDA not available for JAX")

            jax.config.update("jax_default_device", cuda_devices[0])
        elif "metal" in mode:
            # Try to get Metal device
            all_devices = jax.devices()
            metal_devices = [d for d in all_devices if d.platform == "METAL"]

            if not metal_devices:
                print("Error: Metal mode requested but no Metal device detected")
                print(f"Available devices: {all_devices}")
                raise RuntimeError("Metal not available for JAX")

            jax.config.update("jax_default_device", metal_devices[0])


def benchmark_dynamic(mode, m, k, n, warmup, iterations):
    gc.collect()
    BenchMode.set_device(mode)

    # Create random tensors
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    a = jax.random.normal(key1, (m, k))
    b = jax.random.normal(key2, (k, n))

    # Warmup
    for _ in range(warmup):
        result = jnp.matmul(a, b)
        result.block_until_ready()  # Ensure computation completes

    # Benchmark - collect individual iteration times
    times = []
    bench_start = time.time()
    for i in range(iterations):
        start = time.time()
        result = jnp.matmul(a, b)
        result.block_until_ready()  # Ensure computation completes
        times.append(time.time() - start)

        # Check timeout after each iteration
        total_elapsed = time.time() - bench_start
        if total_elapsed > 10.0:
            raise TimeoutError(f"TIMEOUT: Exceeded 10 seconds after {i + 1} iterations")

    return trimmed_mean(times)


def benchmark_static(mode, m, k, n, warmup, iterations):
    gc.collect()
    BenchMode.set_device(mode)

    # Create random tensors
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    a = jax.random.normal(key1, (m, k))
    b = jax.random.normal(key2, (k, n))

    # JIT compile the matmul operation
    @jax.jit
    def matmul_compiled(x, y):
        return jnp.matmul(x, y)

    # Warmup (first call triggers compilation)
    for _ in range(warmup):
        result = matmul_compiled(a, b)
        result.block_until_ready()

    # Benchmark - collect individual iteration times
    times = []
    bench_start = time.time()
    for i in range(iterations):
        start = time.time()
        result = matmul_compiled(a, b)
        result.block_until_ready()
        times.append(time.time() - start)

        # Check timeout after each iteration
        total_elapsed = time.time() - bench_start
        if total_elapsed > 10.0:
            raise TimeoutError(f"TIMEOUT: Exceeded 10 seconds after {i + 1} iterations")

    return trimmed_mean(times)


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
                time_sec = benchmark_static(mode, m, k, n, warmup, iterations)
            else:
                time_sec = benchmark_dynamic(mode, m, k, n, warmup, iterations)

            print(f"{m}x{k}x{n},time_ms={time_sec * 1000:.6f}ms")
        except TimeoutError:
            print(f"{m}x{k}x{n},TIMEOUT")
            timed_out = True
        except Exception:
            print(f"{m}x{k}x{n},ERROR")


def print_usage():
    print("Usage: python _jax.py <mode>")
    print("\nAvailable modes:")
    print("  dynamic-cpu     - Dynamic execution on CPU")
    print("  dynamic-cuda    - Dynamic execution on CUDA")
    print("  dynamic-metal   - Dynamic execution on Metal")
    print("  static-cpu      - Static computation graph on CPU (jax.jit)")
    print("  static-cuda     - Static computation graph on CUDA (jax.jit)")
    print("  static-metal    - Static computation graph on Metal (jax.jit)")


def get_valid_modes():
    """Get list of valid modes based on available hardware"""
    return [
        BenchMode.DYNAMIC_CPU,
        BenchMode.DYNAMIC_CUDA,
        BenchMode.DYNAMIC_METAL,
        BenchMode.STATIC_CPU,
        BenchMode.STATIC_CUDA,
        BenchMode.STATIC_METAL,
    ]


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1]
    valid_modes = get_valid_modes()

    if mode not in valid_modes:
        print(f"Error: Invalid mode '{mode}'")
        print_usage()
        sys.exit(1)

    # Parse warmup and iterations from command line, with defaults
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    configs = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]

    run_benchmark(mode, configs, warmup, iterations)


if __name__ == "__main__":
    main()
