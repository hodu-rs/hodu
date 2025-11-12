#!/usr/bin/env python3
"""Common utilities for benchmarking."""

import time
import gc
import sys
import numpy as np

# Try to import psutil for memory tracking
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_memory_usage_mb():
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return None

    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB
    except Exception:
        return None


def trimmed_mean(times, trim_ratio=0.2):
    """Calculate trimmed mean by removing top and bottom percentiles."""
    times = np.array(times)
    times.sort()
    trim_count = int(len(times) * trim_ratio)
    if trim_count > 0:
        trimmed = times[trim_count:-trim_count]
    else:
        trimmed = times
    return np.mean(trimmed)


class BenchmarkRunner:
    """Base class for running benchmarks with memory tracking."""

    def __init__(self, warmup=50, iterations=30, timeout=2.0):
        self.warmup = warmup
        self.iterations = iterations
        self.timeout = timeout

    def run_with_memory(self, benchmark_fn, *args, **kwargs):
        """Run benchmark function and track memory usage.

        Returns:
            tuple: (avg_time_seconds, peak_memory_mb)
        """
        gc.collect()

        # Warmup
        for _ in range(self.warmup):
            benchmark_fn(*args, **kwargs)

        gc.collect()
        mem_before = get_memory_usage_mb()
        peak_mem = mem_before if mem_before else 0

        # Benchmark
        times = []
        bench_start = time.time()

        for i in range(self.iterations):
            start = time.time()
            benchmark_fn(*args, **kwargs)
            elapsed = time.time() - start
            times.append(elapsed)

            # Track peak memory
            current_mem = get_memory_usage_mb()
            if current_mem and current_mem > peak_mem:
                peak_mem = current_mem

            # Check timeout
            total_elapsed = time.time() - bench_start
            if total_elapsed > self.timeout:
                raise TimeoutError(
                    f"TIMEOUT: Exceeded {self.timeout} seconds after {i + 1} iterations"
                )

        avg_time = trimmed_mean(times)

        mem_after = get_memory_usage_mb()
        if mem_before and mem_after:
            mem_used = peak_mem - mem_before
        else:
            mem_used = None

        return avg_time, mem_used
