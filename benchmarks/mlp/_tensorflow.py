import sys
import numpy as np
import tensorflow as tf
import time
import gc


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
            if tf.config.list_physical_devices("GPU"):
                return "/GPU:0"
            else:
                print("Warning: GPU requested but not available, falling back to CPU")
                return "/CPU:0"
        return "/CPU:0"


class MLPBlock(tf.keras.Model):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(
            hidden_features, activation=None, use_bias=True
        )
        self.fc2 = tf.keras.layers.Dense(
            hidden_features, activation=None, use_bias=True
        )
        self.fc3 = tf.keras.layers.Dense(out_features, activation=None, use_bias=True)

        # Projection for residual connection if dimensions don't match
        self.projection = None
        if in_features != out_features:
            self.projection = tf.keras.layers.Dense(out_features, use_bias=False)

    def call(self, x):
        # Save input for residual connection
        identity = x if self.projection is None else self.projection(x)

        # Forward through layers
        out = self.fc1(x)
        out = tf.nn.gelu(out)
        out = self.fc2(out)
        out = tf.nn.gelu(out)
        out = self.fc3(out)

        # Residual connection
        return out + identity


def benchmark_dynamic(
    mode, batch_size, in_features, hidden_features, out_features, warmup, iterations
):
    gc.collect()
    device = BenchMode.get_device(mode)

    with tf.device(device):
        # Create model and input
        model = MLPBlock(in_features, hidden_features, out_features)
        x = tf.random.normal([batch_size, in_features])

        # Build model
        _ = model(x)

        # Warmup
        for _ in range(warmup):
            _ = model(x, training=False)

        # Benchmark - collect individual iteration times
        times = []
        bench_start = time.time()
        for i in range(iterations):
            start = time.time()
            _ = model(x, training=False)
            times.append(time.time() - start)

            # Check timeout after each iteration
            total_elapsed = time.time() - bench_start
            if total_elapsed > 10.0:
                raise TimeoutError(
                    f"TIMEOUT: Exceeded 10 seconds after {i + 1} iterations"
                )

    return trimmed_mean(times)


def benchmark_static(
    mode, batch_size, in_features, hidden_features, out_features, warmup, iterations
):
    gc.collect()
    device = BenchMode.get_device(mode)

    with tf.device(device):
        # Create model and input
        model = MLPBlock(in_features, hidden_features, out_features)
        x = tf.random.normal([batch_size, in_features])

        # Build model
        _ = model(x)

        # Compile the model with tf.function
        @tf.function
        def model_compiled(x):
            return model(x, training=False)

        # Warmup (first call triggers tracing)
        for _ in range(warmup):
            _ = model_compiled(x)

        # Benchmark - collect individual iteration times
        times = []
        bench_start = time.time()
        for i in range(iterations):
            start = time.time()
            _ = model_compiled(x)
            times.append(time.time() - start)

            # Check timeout after each iteration
            total_elapsed = time.time() - bench_start
            if total_elapsed > 10.0:
                raise TimeoutError(
                    f"TIMEOUT: Exceeded 10 seconds after {i + 1} iterations"
                )

    return trimmed_mean(times)


def run_benchmark(mode, configs, warmup, iterations):
    print(f"mode={BenchMode.get_name(mode)}")
    print(f"warmup={warmup}")
    print(f"iterations={iterations}")

    timed_out = False

    for batch_size, in_features, hidden_features, out_features in configs:
        # If we already timed out, skip remaining benchmarks
        if timed_out:
            print(
                f"{batch_size}x{in_features}x{hidden_features}x{out_features},TIMEOUT"
            )
            continue

        try:
            if "static" in mode:
                time_sec = benchmark_static(
                    mode,
                    batch_size,
                    in_features,
                    hidden_features,
                    out_features,
                    warmup,
                    iterations,
                )
            else:
                time_sec = benchmark_dynamic(
                    mode,
                    batch_size,
                    in_features,
                    hidden_features,
                    out_features,
                    warmup,
                    iterations,
                )

            print(
                f"{batch_size}x{in_features}x{hidden_features}x{out_features},time_ms={time_sec * 1000:.6f}ms"
            )
        except TimeoutError:
            print(
                f"{batch_size}x{in_features}x{hidden_features}x{out_features},TIMEOUT"
            )
            timed_out = True
        except Exception:
            print(f"{batch_size}x{in_features}x{hidden_features}x{out_features},ERROR")


def main():
    if len(sys.argv) < 2:
        print("Usage: python _tensorflow.py <mode> [warmup] [iterations]")
        print("\nAvailable modes:")
        print("  dynamic-cpu     - Dynamic execution on CPU")
        print("  dynamic-gpu     - Dynamic execution on GPU")
        print("  static-cpu      - Static graph (tf.function) on CPU")
        print("  static-gpu      - Static graph (tf.function) on GPU")
        sys.exit(1)

    mode = sys.argv[1]

    # Parse warmup and iterations from command line, with defaults
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    # MLP configs: (batch_size, in_features, hidden_features, out_features)
    configs = [
        (32, 256, 512, 256),
        (64, 512, 1024, 512),
        (128, 768, 2048, 1024),
    ]

    run_benchmark(mode, configs, warmup, iterations)


if __name__ == "__main__":
    main()
