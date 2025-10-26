import jax
import jax.numpy as jnp
import time
import sys


class BenchMode:
    DYNAMIC_CPU = "dynamic-cpu"
    DYNAMIC_GPU = "dynamic-gpu"
    STATIC_CPU = "static-cpu"
    STATIC_GPU = "static-gpu"

    @staticmethod
    def get_name(mode):
        names = {
            BenchMode.DYNAMIC_CPU: "Dynamic CPU",
            BenchMode.DYNAMIC_GPU: "Dynamic GPU",
            BenchMode.STATIC_CPU: "Static CPU",
            BenchMode.STATIC_GPU: "Static GPU",
        }
        return names.get(mode, mode)

    @staticmethod
    def set_device(mode):
        """Set JAX default device based on mode"""
        if "cpu" in mode:
            jax.config.update("jax_platform_name", "cpu")
        elif "gpu" in mode:
            # Check if GPU is available
            try:
                devices = jax.devices("gpu")
                if devices:
                    jax.config.update("jax_platform_name", "gpu")
                else:
                    jax.config.update("jax_platform_name", "cpu")
            except Exception:
                jax.config.update("jax_platform_name", "cpu")


def benchmark_dynamic(mode, m, k, n, warmup, iterations):
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

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        result = jnp.matmul(a, b)
        result.block_until_ready()  # Ensure computation completes
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_static(mode, m, k, n, warmup, iterations):
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

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        result = matmul_compiled(a, b)
        result.block_until_ready()
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
    print("Usage: python bench_jax.py <mode>")
    print("\nAvailable modes:")
    print("  dynamic-cpu     - Dynamic execution on CPU")

    # Check GPU availability
    try:
        gpus = jax.devices("gpu")
        if gpus:
            print("  dynamic-gpu     - Dynamic execution on GPU")
    except Exception:
        pass

    print("  static-cpu      - Static computation graph on CPU (jax.jit)")

    try:
        gpus = jax.devices("gpu")
        if gpus:
            print("  static-gpu      - Static computation graph on GPU (jax.jit)")
    except Exception:
        pass


def get_valid_modes():
    """Get list of valid modes based on available hardware"""
    valid_modes = [
        BenchMode.DYNAMIC_CPU,
        BenchMode.STATIC_CPU,
    ]

    # Check GPU
    try:
        gpus = jax.devices("gpu")
        if gpus:
            valid_modes.extend([BenchMode.DYNAMIC_GPU, BenchMode.STATIC_GPU])
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
