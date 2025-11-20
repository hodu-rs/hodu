import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax
import time
import sys


def trimmed_mean(times, trim_ratio=0.1):
    """Calculate trimmed mean by removing outliers."""
    times = sorted(times)
    trim_count = int(len(times) * trim_ratio)
    if trim_count > 0:
        trimmed = times[trim_count:-trim_count]
        return sum(trimmed) / len(trimmed)
    return sum(times) / len(times)


def simple_cnn_forward(params, x):
    """Simple CNN forward pass using JAX."""
    # Conv1: 3 -> 32
    x = (
        jnp.dot(x.reshape(x.shape[0], -1, x.shape[-1]), params["conv1_w"])
        + params["conv1_b"]
    )
    x = x.reshape(x.shape[0], 32, 32, 32).transpose(0, 3, 1, 2)  # NHWC -> NCHW
    x = jax.nn.relu(x)
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (1, 1, 2, 2), (1, 1, 2, 2), "VALID"
    )  # MaxPool

    # Conv2: 32 -> 64
    x = x.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    x = (
        jnp.dot(x.reshape(x.shape[0], -1, x.shape[-1]), params["conv2_w"])
        + params["conv2_b"]
    )
    x = x.reshape(x.shape[0], 16, 16, 64).transpose(0, 3, 1, 2)  # NHWC -> NCHW
    x = jax.nn.relu(x)
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (1, 1, 2, 2), (1, 1, 2, 2), "VALID"
    )  # MaxPool

    # Flatten
    x = x.reshape(x.shape[0], -1)  # [batch, 64*8*8]

    # FC1
    x = jnp.dot(x, params["fc1_w"]) + params["fc1_b"]  # [batch, 128]
    x = jax.nn.relu(x)

    # FC2
    x = jnp.dot(x, params["fc2_w"]) + params["fc2_b"]  # [batch, 10]

    return x


def cross_entropy_loss(params, x, target):
    """Cross entropy loss."""
    logits = simple_cnn_forward(params, x)
    return -jnp.mean(
        jnp.sum(jax.nn.log_softmax(logits) * jax.nn.one_hot(target, 10), axis=-1)
    )


def init_params(key):
    """Initialize model parameters."""
    keys = jax.random.split(key, 6)

    params = {
        "conv1_w": jax.random.normal(keys[0], (3 * 3 * 3, 32)) * 0.1,
        "conv1_b": jnp.zeros(32),
        "conv2_w": jax.random.normal(keys[1], (3 * 3 * 32, 64)) * 0.1,
        "conv2_b": jnp.zeros(64),
        "fc1_w": jax.random.normal(keys[2], (64 * 8 * 8, 128)) * 0.1,
        "fc1_b": jnp.zeros(128),
        "fc2_w": jax.random.normal(keys[3], (128, 10)) * 0.1,
        "fc2_b": jnp.zeros(10),
    }

    return params


def benchmark_training(device_str, batch_size, warmup, iterations):
    """Benchmark CNN training with forward and backward passes."""
    # Set device
    if device_str == "cuda":
        if jax.devices()[0].platform != "gpu":
            return None, None, "ERROR: CUDA not available"

    # Initialize
    key = jax.random.PRNGKey(0)
    params = init_params(key)

    # Optimizer
    optimizer = optax.adam(0.001)
    opt_state = optimizer.init(params)

    # Random data
    x = jax.random.normal(key, (batch_size, 3, 32, 32))
    target = jax.random.randint(key, (batch_size,), 0, 10)

    # JIT compile functions
    forward_fn = jit(lambda p, x: simple_cnn_forward(p, x))
    loss_and_grad_fn = jit(value_and_grad(cross_entropy_loss))

    # Training step
    @jit
    def train_step(params, opt_state, x, target):
        loss, grads = loss_and_grad_fn(params, x, target)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Warmup
    for _ in range(warmup):
        params, opt_state, _ = train_step(params, opt_state, x, target)

    jax.block_until_ready(params)

    # Benchmark full training step (forward + backward + optimizer step)
    step_times = []
    bench_start = time.time()

    for i in range(iterations):
        start = time.time()
        params, opt_state, loss = train_step(params, opt_state, x, target)
        jax.block_until_ready(params)
        step_times.append(time.time() - start)

        if time.time() - bench_start > 60.0:
            return None, f"TIMEOUT: Exceeded 60 seconds after {i + 1} iterations"

    return trimmed_mean(step_times, 0.1), None


def run_benchmark(mode, configs, warmup, iterations):
    """Run benchmarks for all configurations."""
    print(f"mode={mode}")
    print(f"warmup={warmup}")
    print(f"iterations={iterations}")

    # Determine device
    if mode == "cpu":
        jax.config.update("jax_platform_name", "cpu")
        device_str = "cpu"
    elif mode == "cuda":
        device_str = "cuda"
    else:
        print(f"ERROR: Unknown mode '{mode}'")
        sys.exit(1)

    timed_out = False

    for batch_size in configs:
        if timed_out:
            print(f"{batch_size},TIMEOUT")
            continue

        try:
            step_time, error = benchmark_training(
                device_str, batch_size, warmup, iterations
            )

            if error:
                if "TIMEOUT" in error:
                    print(f"{batch_size},TIMEOUT")
                    timed_out = True
                else:
                    print(f"{batch_size},ERROR")
            else:
                print(f"{batch_size},time_ms={step_time * 1000:.6f}")
        except Exception:
            print(f"{batch_size},ERROR")


def print_usage():
    print("Usage: python _jax.py <mode> [warmup] [iterations]")
    print("\nAvailable modes:")
    print("  cpu   - CPU execution")
    print("  cuda  - CUDA GPU execution")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1]
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    # CNN training configs: batch sizes for 32x32 RGB images
    configs = [16, 32, 64]

    run_benchmark(mode, configs, warmup, iterations)
