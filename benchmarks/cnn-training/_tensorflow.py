import tensorflow as tf
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


class SimpleCNN(tf.keras.Model):
    """Simple CNN for image classification."""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv layers: 3 -> 32 -> 64 channels
        self.conv1 = tf.keras.layers.Conv2D(
            32, 3, padding="same", data_format="channels_first"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            64, 3, padding="same", data_format="channels_first"
        )

        # FC layers: 64*8*8 -> 128 -> 10
        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = tf.keras.layers.Dense(10)

        self.pool = tf.keras.layers.MaxPooling2D(2, data_format="channels_first")

    def call(self, x):
        # Input: [batch, 3, 32, 32]
        x = self.conv1(x)  # [batch, 32, 32, 32]
        x = tf.nn.relu(x)
        x = self.pool(x)  # [batch, 32, 16, 16]

        x = self.conv2(x)  # [batch, 64, 16, 16]
        x = tf.nn.relu(x)
        x = self.pool(x)  # [batch, 64, 8, 8]

        # Flatten
        x = tf.reshape(x, [tf.shape(x)[0], -1])  # [batch, 64*8*8]

        x = self.fc1(x)  # [batch, 128]
        x = tf.nn.relu(x)

        x = self.fc2(x)  # [batch, 10]
        return x


def benchmark_training(device_str, batch_size, warmup, iterations):
    """Benchmark CNN training with forward and backward passes."""
    with tf.device(device_str):
        # Create model
        model = SimpleCNN()

        # Optimizer and loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Random data
        x = tf.random.normal([batch_size, 3, 32, 32])
        target = tf.random.uniform([batch_size], 0, 10, dtype=tf.int32)

        # Build model
        _ = model(x)

        # Training step function
        @tf.function
        def train_step(x, target):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = loss_fn(target, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        # Forward only function
        @tf.function
        def forward_step(x, target):
            logits = model(x, training=False)
            loss = loss_fn(target, logits)
            return loss

        # Warmup
        for _ in range(warmup):
            _ = train_step(x, target)

        # Benchmark full training step (forward + backward + optimizer step)
        step_times = []
        bench_start = time.time()

        for i in range(iterations):
            start = time.time()
            _ = train_step(x, target)
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
        device_str = "/CPU:0"
    elif mode == "cuda":
        if not tf.config.list_physical_devices("GPU"):
            print("ERROR: CUDA not available")
            sys.exit(1)
        device_str = "/GPU:0"
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
    print("Usage: python _tensorflow.py <mode> [warmup] [iterations]")
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
