import tensorflow as tf
import time
import sys
import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
            if gpus:
                return "/GPU:0"
            return "/CPU:0"
        return "/CPU:0"


def benchmark_dynamic(mode, m, k, n, warmup, iterations):
    device = BenchMode.get_device(mode)

    with tf.device(device):
        # Create random tensors
        a = tf.random.normal([m, k])
        b = tf.random.normal([k, n])

        # Warmup
        for _ in range(warmup):
            _ = tf.matmul(a, b)

        # Benchmark
        start = time.time()
        for _ in range(iterations):
            _ = tf.matmul(a, b)
        elapsed = time.time() - start

    return elapsed / iterations


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
            _ = matmul_compiled(a, b)

        # Benchmark
        start = time.time()
        for _ in range(iterations):
            _ = matmul_compiled(a, b)
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
    print("Usage: python _tf.py <mode>")
    print("\nAvailable modes:")
    print("  dynamic-cpu     - Dynamic execution on CPU")

    # Check GPU availability
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("  dynamic-gpu     - Dynamic execution on GPU")

    print("  static-cpu      - Static computation graph on CPU (tf.function)")

    if gpus:
        print("  static-gpu      - Static computation graph on GPU (tf.function)")


def get_valid_modes():
    """Get list of valid modes based on available hardware"""
    valid_modes = [
        BenchMode.DYNAMIC_CPU,
        BenchMode.STATIC_CPU,
    ]

    # Check GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        valid_modes.extend([BenchMode.DYNAMIC_GPU, BenchMode.STATIC_GPU])

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
