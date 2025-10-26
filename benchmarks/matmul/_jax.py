import sys
import jax
import jax.numpy as jnp
import time


class BenchMode:
    DYNAMIC_CPU = "dynamic-cpu"
    DYNAMIC_METAL = "dynamic-metal"
    STATIC_CPU = "static-cpu"
    STATIC_METAL = "static-metal"

    @staticmethod
    def get_name(mode):
        names = {
            BenchMode.DYNAMIC_CPU: "Dynamic CPU",
            BenchMode.DYNAMIC_METAL: "Dynamic Metal",
            BenchMode.STATIC_CPU: "Static CPU",
            BenchMode.STATIC_METAL: "Static Metal",
        }
        return names.get(mode, mode)

    @staticmethod
    def set_device(mode):
        """Set JAX default device based on mode"""
        if "cpu" in mode:
            jax.config.update("jax_default_device", jax.devices("cpu")[0])
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
        try:
            if "static" in mode:
                time_sec = benchmark_static(mode, m, k, n, warmup, iterations)
            else:
                time_sec = benchmark_dynamic(mode, m, k, n, warmup, iterations)

            print(f"{m}x{k}x{n},{time_sec * 1000:.6f}ms")
        except Exception as e:
            print(f"{m}x{k}x{n},ERROR")
            # Optionally print error to stderr for debugging
            import sys

            print(f"Error for {m}x{k}x{n}: {e}", file=sys.stderr)


def print_usage():
    print("Usage: python _jax.py <mode>")
    print("\nAvailable modes:")
    print("  dynamic-cpu     - Dynamic execution on CPU")
    print("  dynamic-metal   - Dynamic execution on Metal")
    print("  static-cpu      - Static computation graph on CPU (jax.jit)")
    print("  static-metal    - Static computation graph on Metal (jax.jit)")


def get_valid_modes():
    """Get list of valid modes based on available hardware"""
    return [
        BenchMode.DYNAMIC_CPU,
        BenchMode.DYNAMIC_METAL,
        BenchMode.STATIC_CPU,
        BenchMode.STATIC_METAL,
    ]


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

    warmup = 5
    iterations = 10

    run_benchmark(mode, configs, warmup, iterations)


if __name__ == "__main__":
    main()
