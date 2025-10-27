import sys
import jax
import jax.numpy as jnp
from jax import random
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


def init_mlp_params(key, in_features, hidden_features, out_features):
    """Initialize MLP parameters"""
    k1, k2, k3, k4 = random.split(key, 4)

    # Xavier initialization
    k_in = jnp.sqrt(1.0 / in_features)
    k_hidden = jnp.sqrt(1.0 / hidden_features)

    params = {
        "fc1": {
            "w": random.normal(k1, (in_features, hidden_features)) * k_in,
            "b": jnp.zeros(hidden_features),
        },
        "fc2": {
            "w": random.normal(k2, (hidden_features, hidden_features)) * k_hidden,
            "b": jnp.zeros(hidden_features),
        },
        "fc3": {
            "w": random.normal(k3, (hidden_features, out_features)) * k_hidden,
            "b": jnp.zeros(out_features),
        },
    }

    # Add projection if needed
    if in_features != out_features:
        params["projection"] = {
            "w": random.normal(k4, (in_features, out_features)) * k_in
        }

    return params


def mlp_forward(params, x):
    """MLP forward pass with GELU and residual connection"""
    # Save input for residual
    if "projection" in params:
        identity = jnp.dot(x, params["projection"]["w"])
    else:
        identity = x

    # Forward through layers
    x = jnp.dot(x, params["fc1"]["w"]) + params["fc1"]["b"]
    x = jax.nn.gelu(x)
    x = jnp.dot(x, params["fc2"]["w"]) + params["fc2"]["b"]
    x = jax.nn.gelu(x)
    x = jnp.dot(x, params["fc3"]["w"]) + params["fc3"]["b"]

    # Residual connection
    return x + identity


def benchmark_dynamic(
    mode, batch_size, in_features, hidden_features, out_features, warmup, iterations
):
    BenchMode.set_device(mode)

    # Initialize parameters and input
    key = random.PRNGKey(0)
    key1, key2 = random.split(key)

    params = init_mlp_params(key1, in_features, hidden_features, out_features)
    x = random.normal(key2, (batch_size, in_features))

    # Warmup
    for _ in range(warmup):
        result = mlp_forward(params, x)
        result.block_until_ready()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        result = mlp_forward(params, x)
        result.block_until_ready()
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_static(
    mode, batch_size, in_features, hidden_features, out_features, warmup, iterations
):
    BenchMode.set_device(mode)

    # Initialize parameters and input
    key = random.PRNGKey(0)
    key1, key2 = random.split(key)

    params = init_mlp_params(key1, in_features, hidden_features, out_features)
    x = random.normal(key2, (batch_size, in_features))

    # JIT compile the forward pass
    @jax.jit
    def mlp_compiled(params, x):
        return mlp_forward(params, x)

    # Warmup (first call triggers compilation)
    for _ in range(warmup):
        result = mlp_compiled(params, x)
        result.block_until_ready()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        result = mlp_compiled(params, x)
        result.block_until_ready()
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
        print("Usage: python _jax.py <mode>")
        print("\nAvailable modes:")
        print("  dynamic-cpu     - Dynamic execution on CPU")
        print("  dynamic-metal   - Dynamic execution on Metal")
        print("  static-cpu      - JIT compiled on CPU")
        print("  static-metal    - JIT compiled on Metal")
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
