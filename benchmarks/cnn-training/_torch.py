import torch
import torch.nn as nn
import torch.optim as optim
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


class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv layers: 3 -> 32 -> 64 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # FC layers: 64*8*8 -> 128 -> 10
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Input: [batch, 3, 32, 32]
        x = self.conv1(x)  # [batch, 32, 32, 32]
        x = self.relu(x)
        x = self.pool(x)  # [batch, 32, 16, 16]

        x = self.conv2(x)  # [batch, 64, 16, 16]
        x = self.relu(x)
        x = self.pool(x)  # [batch, 64, 8, 8]

        # Flatten
        x = x.view(x.size(0), -1)  # [batch, 64*8*8]

        x = self.fc1(x)  # [batch, 128]
        x = self.relu(x)

        x = self.fc2(x)  # [batch, 10]
        return x


def benchmark_training(device_str, batch_size, warmup, iterations):
    """Benchmark CNN training with full training step."""
    device = torch.device(device_str)

    # Create model
    model = SimpleCNN().to(device)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Random data
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

    # Benchmark full training step (forward + backward + optimizer step)
    step_times = []
    bench_start = time.time()

    for i in range(iterations):
        start = time.time()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        if device_str == "cuda":
            torch.cuda.synchronize()

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
        device_str = "cpu"
    elif mode == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            sys.exit(1)
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
    print("Usage: python _torch.py <mode> [warmup] [iterations]")
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
