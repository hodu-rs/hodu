import tensorflow as tf
import numpy as np
import time
import sys
import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
    def get_device(mode):
        if "cpu" in mode:
            return "/CPU:0"
        elif "gpu" in mode:
            # Check if GPU is available
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                print("Error: GPU mode requested but no GPU detected")
                print(f"Available devices: {tf.config.list_physical_devices()}")
                raise RuntimeError(
                    "GPU not available. Install tensorflow-metal for Metal support."
                )
            return "/GPU:0"
        return "/CPU:0"


def benchmark_dynamic(mode, m, k, n, warmup, iterations):
    device = BenchMode.get_device(mode)

    with tf.device(device):
        # Create random tensors
        a = tf.random.normal([m, k])
        b = tf.random.normal([k, n])

        # Warmup
        for _ in range(warmup):
            result = tf.matmul(a, b)
            _ = result.numpy()  # Ensure computation completes

        # Benchmark - collect individual iteration times
        times = []
        for _ in range(iterations):
            start = time.time()
            result = tf.matmul(a, b)
            _ = result.numpy()  # Ensure computation completes
            times.append(time.time() - start)

    return trimmed_mean(times)


def benchmark_static(mode, m, k, n, warmup, iterations):
    device = BenchMode.get_device(mode)

    with tf.device(device):
        # Create random tensors
        a = tf.random.normal([m, k])
        b = tf.random.normal([k, n])

        # Compile the matmul operation with tf.function
        @tf.function
        def matmul_compiled(x, y):
            return tf.matmul(x, y)

        # Warmup
        for _ in range(warmup):
            result = matmul_compiled(a, b)
            _ = result.numpy()  # Ensure computation completes

        # Benchmark - collect individual iteration times
        times = []
        for _ in range(iterations):
            start = time.time()
            result = matmul_compiled(a, b)
            _ = result.numpy()  # Ensure computation completes
            times.append(time.time() - start)

    return trimmed_mean(times)


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
    print("Usage: python _tensorflow.py <mode>")
    print("\nAvailable modes:")
    print("  dynamic-cpu     - Dynamic execution on CPU")
    print("  dynamic-gpu     - Dynamic execution on GPU")
    print("  static-cpu      - Static computation graph on CPU (tf.function)")
    print("  static-gpu      - Static computation graph on GPU (tf.function)")


def get_valid_modes():
    """Get list of valid modes based on available hardware"""
    return [
        BenchMode.DYNAMIC_CPU,
        BenchMode.DYNAMIC_GPU,
        BenchMode.STATIC_CPU,
        BenchMode.STATIC_GPU,
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

    warmup = 10
    iterations = 30

    run_benchmark(mode, configs, warmup, iterations)


if __name__ == "__main__":
    main()
