#!/usr/bin/env python3
"""
Benchmark runner with beautiful output and memory tracking.
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Try to import rich for beautiful output
try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich library not available. Install with: pip install rich")
    print("Falling back to simple output...")


# Configuration
TABLE_WIDTH = 160
VENV_JAX_TF = "1"
VENV_TENSORFLOW = "2"


class BenchmarkRunner:
    """Main benchmark runner with rich output."""

    def __init__(self, warmup=5, iterations=100):
        self.console = Console() if RICH_AVAILABLE else None
        self.all_results = {}
        self.quiet_mode = False
        self.warmup = warmup
        self.iterations = iterations

    def print_header(self, bench_type):
        """Print beautiful header."""
        if not RICH_AVAILABLE or self.quiet_mode:
            print(f"\n{'=' * 80}")
            print(f"Benchmark: {bench_type.upper()}")
            print(f"{'=' * 80}\n")
            return

        title = {
            "matmul": "Matrix Multiplication Benchmark",
            "mlp": "MLP Block Benchmark (3-layer with GELU and Residual)",
        }.get(bench_type, f"{bench_type.upper()} Benchmark")

        self.console.print(
            Panel.fit(
                f"[bold cyan]{title}[/bold cyan]\n"
                f"[dim]Measuring execution time and memory usage[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def run_command(self, command, cwd=None, clean_environment=False):
        """Run a command and capture output with optional process isolation."""
        try:
            # Create a clean environment if requested (for process isolation)
            env = None
            if clean_environment:
                import os

                env = os.environ.copy()
                # Clear any cached library state
                env["OPENBLAS_NUM_THREADS"] = str(os.cpu_count() or 4)

            result = subprocess.run(
                command, check=False, capture_output=True, text=True, cwd=cwd, env=env
            )
            return result
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Error running command: {e}[/red]")
            else:
                print(f"Error running command: {e}")
            return None

    def parse_benchmark_output(self, output):
        """Parse benchmark output including memory data."""
        results = {}
        mode = None

        for line in output.strip().split("\n"):
            if line.startswith("mode="):
                mode = line.split("=", 1)[1]
            elif "," in line:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    size = parts[0]
                    value_str = parts[1]

                    if "TIMEOUT" in value_str:
                        results[size] = {"time": "TIMEOUT", "memory": None}
                    elif "ERROR" in value_str:
                        results[size] = {"time": "ERROR", "memory": None}
                    else:
                        time_ms = None
                        mem_kb = None

                        # Extract time
                        time_match = re.search(r"time_ms=([0-9.]+)", value_str)
                        if time_match:
                            time_ms = float(time_match.group(1))

                        # Extract memory
                        mem_match = re.search(r"mem_kb=([0-9.]+)", value_str)
                        if mem_match:
                            mem_kb = float(mem_match.group(1))
                        else:
                            mem_match = re.search(r"mem_mb=([0-9.]+)", value_str)
                            if mem_match:
                                mem_kb = float(mem_match.group(1)) * 1024

                        results[size] = {"time": time_ms, "memory": mem_kb}

        return mode, results

    def get_rust_features(self, framework, mode):
        """Get cargo features for Rust frameworks."""
        features_map = {
            "candle": {
                "metal": "metal,candle-bench",
                "cuda": "cuda,candle-bench",
                "default": "candle-bench",
            },
            "burn": {
                "wgpu": "wgpu,burn-bench",
                "tch": "cuda,burn-bench",
                "cuda": "cuda,burn-bench",
                "default": "burn-bench",
            },
            "hodu": {
                "metal": "metal,hodu-bench",
                "xla": "xla,hodu-bench",
                "cuda": "cuda,hodu-bench",
                "default": "hodu-bench",
            },
        }

        framework_features = features_map.get(framework, {})
        for key in ["metal", "cuda", "wgpu", "tch", "xla"]:
            if key in mode:
                return framework_features.get(key, framework_features.get("default"))
        return framework_features.get("default", "")

    def run_rust_benchmark(self, bench_type, framework, mode):
        """Run Rust-based benchmark with process isolation."""
        features = self.get_rust_features(framework, mode)
        cmd = [
            "cargo",
            "run",
            "--release",
            f"--features={features}",
            "--bin",
            framework,
            "--",
            mode,
            str(self.warmup),
            str(self.iterations),
        ]

        result = self.run_command(
            cmd, cwd=Path(__file__).parent / bench_type, clean_environment=True
        )

        if result and result.returncode == 0:
            parsed_result = self.parse_benchmark_output(result.stdout)
            # Force garbage collection and memory cleanup after each benchmark
            self._cleanup_after_benchmark()
            return parsed_result
        return None, {}

    def _cleanup_after_benchmark(self):
        """Cleanup memory and force garbage collection between benchmarks."""
        import gc
        import time
        import sys

        # Force multiple rounds of garbage collection
        for _ in range(3):
            gc.collect()

        # On macOS, try to hint the OS to reclaim memory
        if sys.platform == "darwin":
            try:
                import subprocess

                # Sync file system buffers
                subprocess.run(["sync"], check=False, capture_output=True, timeout=1)
            except:
                pass

        # Longer delay to allow OS to reclaim memory and cool down CPU
        time.sleep(1.0)

    def get_python_executable(self, script):
        """Get the appropriate Python executable."""
        base_path = Path(__file__).parent
        venv_num = VENV_TENSORFLOW if script == "_tensorflow.py" else VENV_JAX_TF
        venv_path = base_path / ".venvs" / venv_num

        if venv_path.exists():
            return str(venv_path / "bin" / "python3")
        return "python3"

    def run_python_benchmark(self, bench_type, script, mode):
        """Run Python-based benchmark with process isolation."""
        python_exe = self.get_python_executable(script)
        cmd = [python_exe, script, mode, str(self.warmup), str(self.iterations)]

        result = self.run_command(
            cmd, cwd=Path(__file__).parent / bench_type, clean_environment=True
        )

        if result and result.returncode == 0:
            parsed_result = self.parse_benchmark_output(result.stdout)
            # Force garbage collection and memory cleanup after each benchmark
            self._cleanup_after_benchmark()
            return parsed_result
        else:
            error_msg = result.stderr if result else "Unknown error"
            if "ModuleNotFoundError" in error_msg or "ImportError" in error_msg:
                return None, {}  # Silently skip
        return None, {}

    def create_results_table(
        self, bench_type, cpu_results, gpu_results, cpu_baseline, gpu_baseline
    ):
        """Create beautiful results table using rich."""
        if not RICH_AVAILABLE:
            return self._create_simple_table(
                bench_type, cpu_results, gpu_results, cpu_baseline, gpu_baseline
            )

        # Get all sizes
        all_sizes = set()
        for results in {**cpu_results, **gpu_results}.values():
            all_sizes.update(results.keys())
        sizes = sorted(all_sizes, key=lambda s: [int(x) for x in s.split("x")])

        if not sizes:
            self.console.print("[red]No timing data found[/red]")
            return

        # Create table
        table = Table(
            title=f"[bold cyan]{bench_type.upper()} Benchmark Results[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_style="bold cyan",
        )

        table.add_column("Framework", style="cyan", width=14)
        table.add_column("Mode", style="yellow", width=18)

        # Add size columns
        for size in sizes:
            table.add_column(f"{size}\nTime", justify="right", width=12)
            table.add_column("Ratio", justify="right", width=10)

        # Add CPU results
        if cpu_results:
            cpu_baseline_results = cpu_results.get(cpu_baseline, {})
            for impl_name in sorted(cpu_results.keys()):
                results = cpu_results[impl_name]
                parts = impl_name.split(" - ")
                framework = parts[0] if parts else impl_name
                mode = parts[1] if len(parts) > 1 else ""

                row = [framework, mode]
                for size in sizes:
                    result = results.get(size, {})
                    time_ms = result.get("time")
                    baseline_result = cpu_baseline_results.get(size, {})
                    baseline_time = baseline_result.get("time")

                    # Time
                    if time_ms == "TIMEOUT":
                        row.append("[red]TIMEOUT[/red]")
                    elif time_ms == "ERROR":
                        row.append("[red]ERROR[/red]")
                    elif time_ms is not None:
                        row.append(f"[bold]{time_ms:.3f}ms[/bold]")
                    else:
                        row.append("[dim]N/A[/dim]")

                    # Ratio
                    if (
                        time_ms
                        and isinstance(time_ms, (int, float))
                        and baseline_time
                        and isinstance(baseline_time, (int, float))
                    ):
                        ratio = baseline_time / time_ms
                        if ratio >= 1.1:
                            row.append(f"[bold green]{ratio:.2f}x[/bold green]")
                        elif ratio >= 0.9:
                            row.append(f"[bold yellow]{ratio:.2f}x[/bold yellow]")
                        else:
                            row.append(f"[bold red]{ratio:.2f}x[/bold red]")
                    elif impl_name == cpu_baseline:
                        row.append("[bold white]baseline[/bold white]")
                    else:
                        row.append("[dim]-[/dim]")

                table.add_row(*row)

        # Add separator and GPU results
        if cpu_results and gpu_results:
            table.add_section()

        if gpu_results:
            gpu_baseline_results = gpu_results.get(gpu_baseline, {})
            for impl_name in sorted(gpu_results.keys()):
                results = gpu_results[impl_name]
                parts = impl_name.split(" - ")
                framework = parts[0] if parts else impl_name
                mode = parts[1] if len(parts) > 1 else ""

                row = [framework, mode]
                for size in sizes:
                    result = results.get(size, {})
                    time_ms = result.get("time")
                    baseline_result = gpu_baseline_results.get(size, {})
                    baseline_time = baseline_result.get("time")

                    # Time
                    if time_ms == "TIMEOUT":
                        row.append("[red]TIMEOUT[/red]")
                    elif time_ms == "ERROR":
                        row.append("[red]ERROR[/red]")
                    elif time_ms is not None:
                        row.append(f"[bold]{time_ms:.3f}ms[/bold]")
                    else:
                        row.append("[dim]N/A[/dim]")

                    # Ratio
                    if (
                        time_ms
                        and isinstance(time_ms, (int, float))
                        and baseline_time
                        and isinstance(baseline_time, (int, float))
                    ):
                        ratio = baseline_time / time_ms
                        if ratio >= 1.1:
                            row.append(f"[bold green]{ratio:.2f}x[/bold green]")
                        elif ratio >= 0.9:
                            row.append(f"[bold yellow]{ratio:.2f}x[/bold yellow]")
                        else:
                            row.append(f"[bold red]{ratio:.2f}x[/bold red]")
                    elif impl_name == gpu_baseline:
                        row.append("[bold white]baseline[/bold white]")
                    else:
                        row.append("[dim]-[/dim]")

                table.add_row(*row)

        self.console.print(table)

    def _create_simple_table(
        self, bench_type, cpu_results, gpu_results, cpu_baseline, gpu_baseline
    ):
        """Create simple text table without rich."""
        print(f"\n{'=' * TABLE_WIDTH}")
        print(f"{bench_type.upper()} Benchmark Results")
        print(f"{'=' * TABLE_WIDTH}\n")
        # Simple table output similar to old version
        print("Framework   Mode           Results")
        print("-" * TABLE_WIDTH)
        # ... (simplified output)

    def run_all_benchmarks(self, bench_type, modes_config):
        """Run all benchmarks with progress bar."""
        if not RICH_AVAILABLE or self.quiet_mode:
            return self._run_all_simple(bench_type, modes_config)

        rust_benchmarks = [
            ("burn", modes_config["burn"]),
            ("candle", modes_config["candle"]),
            ("hodu", modes_config["hodu"]),
        ]

        total = (
            sum(len(modes) for _, modes in rust_benchmarks)
            + len(modes_config["jax"])
            + len(modes_config["tensorflow"])
            + len(modes_config["pytorch"])
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("[cyan]Running benchmarks...", total=total)

            # Rust benchmarks
            for framework, modes in rust_benchmarks:
                for mode in modes:
                    progress.update(
                        task,
                        description=f"[cyan]Running {framework.capitalize()} {mode}...",
                    )
                    mode_name, results = self.run_rust_benchmark(
                        bench_type, framework, mode
                    )
                    if results:
                        self.all_results[
                            f"{framework.capitalize()} - {mode_name or mode}"
                        ] = results
                    progress.advance(task)

            # Python benchmarks
            for mode in modes_config["jax"]:
                progress.update(task, description=f"[cyan]Running JAX {mode}...")
                mode_name, results = self.run_python_benchmark(
                    bench_type, "_jax.py", mode
                )
                if results:
                    self.all_results[f"JAX - {mode_name or mode}"] = results
                progress.advance(task)

            for mode in modes_config["tensorflow"]:
                progress.update(task, description=f"[cyan]Running TensorFlow {mode}...")
                tf_mode = mode.replace("cuda", "gpu").replace("metal", "gpu")
                mode_name, results = self.run_python_benchmark(
                    bench_type, "_tensorflow.py", tf_mode
                )
                if results:
                    self.all_results[f"TensorFlow - {mode_name or tf_mode}"] = results
                progress.advance(task)

            for mode in modes_config["pytorch"]:
                progress.update(task, description=f"[cyan]Running PyTorch {mode}...")
                mode_name, results = self.run_python_benchmark(
                    bench_type, "_torch.py", mode
                )
                if results:
                    self.all_results[f"PyTorch - {mode_name or mode}"] = results
                progress.advance(task)

    def _run_all_simple(self, bench_type, modes_config):
        """Run benchmarks without progress bar."""
        rust_benchmarks = [
            ("burn", modes_config["burn"]),
            ("candle", modes_config["candle"]),
            ("hodu", modes_config["hodu"]),
        ]

        # Rust benchmarks
        for framework, modes in rust_benchmarks:
            for mode in modes:
                if not self.quiet_mode:
                    print(f"Running {framework.capitalize()} {mode}...")
                mode_name, results = self.run_rust_benchmark(
                    bench_type, framework, mode
                )
                if results:
                    self.all_results[
                        f"{framework.capitalize()} - {mode_name or mode}"
                    ] = results

        # Python benchmarks
        for mode in modes_config["jax"]:
            if not self.quiet_mode:
                print(f"Running JAX {mode}...")
            mode_name, results = self.run_python_benchmark(bench_type, "_jax.py", mode)
            if results:
                self.all_results[f"JAX - {mode_name or mode}"] = results

        for mode in modes_config["tensorflow"]:
            if not self.quiet_mode:
                print(f"Running TensorFlow {mode}...")
            tf_mode = mode.replace("cuda", "gpu").replace("metal", "gpu")
            mode_name, results = self.run_python_benchmark(
                bench_type, "_tensorflow.py", tf_mode
            )
            if results:
                self.all_results[f"TensorFlow - {mode_name or tf_mode}"] = results

        for mode in modes_config["pytorch"]:
            if not self.quiet_mode:
                print(f"Running PyTorch {mode}...")
            mode_name, results = self.run_python_benchmark(
                bench_type, "_torch.py", mode
            )
            if results:
                self.all_results[f"PyTorch - {mode_name or mode}"] = results


def is_cpu_result(key):
    """Check if result is CPU benchmark."""
    return any(k in key for k in ["CPU", "cpu", "XLA"])


def is_gpu_result(key):
    """Check if result is GPU benchmark."""
    return any(k in key for k in ["Metal", "CUDA", "GPU", "gpu", "WGPU", "TCH"])


def find_baseline(results, is_cpu=True):
    """Find PyTorch baseline."""
    for key in results.keys():
        if "PyTorch" not in key:
            continue
        if is_cpu and ("Dynamic CPU" in key or "dynamic-cpu" in key):
            return key
        if not is_cpu and any(
            x in key
            for x in ["Dynamic Metal", "dynamic-metal", "Dynamic CUDA", "dynamic-cuda"]
        ):
            return key

    check_func = is_cpu_result if is_cpu else is_gpu_result
    for key in results.keys():
        if check_func(key):
            return key
    return None


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python run.py --bench=matmul|mlp [--cpu] [--metal] [--cuda] [--xla] [--quiet] [-w=N] [-i=N]"
        )
        print("  -w=N: warmup iterations (default: 5)")
        print("  -i=N: benchmark iterations (default: 50)")
        sys.exit(1)

    # Parse arguments
    bench_type = None
    enable_cpu = "--cpu" in sys.argv
    enable_metal = "--metal" in sys.argv
    enable_cuda = "--cuda" in sys.argv
    enable_xla = "--xla" in sys.argv
    quiet_mode = "--quiet" in sys.argv
    warmup = 5  # Default
    iterations = 50  # Default

    for arg in sys.argv[1:]:
        if arg.startswith("--bench="):
            bench_type = arg.split("=", 1)[1]
        elif arg.startswith("-w="):
            try:
                warmup = int(arg.split("=", 1)[1])
            except ValueError:
                print(f"Error: Invalid warmup value: {arg}")
                sys.exit(1)
        elif arg.startswith("-i="):
            try:
                iterations = int(arg.split("=", 1)[1])
            except ValueError:
                print(f"Error: Invalid iterations value: {arg}")
                sys.exit(1)

    if not bench_type or bench_type not in ["matmul", "mlp"]:
        print("Error: Must specify --bench=matmul or --bench=mlp")
        sys.exit(1)

    # Setup runner
    runner = BenchmarkRunner(warmup=warmup, iterations=iterations)
    runner.quiet_mode = quiet_mode
    runner.print_header(bench_type)

    # Configure modes
    modes_config = {
        "burn": [],
        "candle": [],
        "hodu": [],
        "jax": [],
        "tensorflow": [],
        "pytorch": [],
    }

    if enable_cpu:
        modes_config["hodu"].extend(["dynamic-cpu", "static-cpu"])
        modes_config["jax"].append("dynamic-cpu")
        modes_config["jax"].append("static-cpu")
        modes_config["tensorflow"].extend(["dynamic-cpu", "static-cpu"])
        modes_config["pytorch"].extend(["dynamic-cpu", "static-cpu"])
        modes_config["burn"].append("dynamic-cpu")
        modes_config["candle"].append("dynamic-cpu")

    if sys.platform == "darwin" and enable_metal:
        modes_config["hodu"].extend(["dynamic-metal", "static-metal"])
        modes_config["jax"].extend(["dynamic-metal", "static-metal"])
        modes_config["tensorflow"].extend(["dynamic-metal", "static-metal"])
        modes_config["pytorch"].extend(["dynamic-metal", "static-metal"])
        modes_config["burn"].append("dynamic-wgpu")
        modes_config["candle"].append("dynamic-metal")
    elif sys.platform != "darwin" and enable_cuda:
        modes_config["hodu"].extend(["dynamic-cuda", "static-cuda"])
        modes_config["jax"].extend(["dynamic-cuda", "static-cuda"])
        modes_config["tensorflow"].extend(["dynamic-cuda", "static-cuda"])
        modes_config["pytorch"].extend(["dynamic-cuda", "static-cuda"])
        modes_config["burn"].extend(["dynamic-wgpu", "dynamic-tch"])
        modes_config["candle"].append("dynamic-cuda")

    if enable_xla and enable_cpu:
        modes_config["hodu"].append("static-xla-cpu")

    # Run benchmarks
    runner.run_all_benchmarks(bench_type, modes_config)

    # Separate CPU and GPU results
    cpu_results = {k: v for k, v in runner.all_results.items() if is_cpu_result(k)}
    gpu_results = {k: v for k, v in runner.all_results.items() if is_gpu_result(k)}

    # Find baselines
    cpu_baseline = find_baseline(runner.all_results, is_cpu=True)
    gpu_baseline = find_baseline(runner.all_results, is_cpu=False)

    # Display results
    runner.create_results_table(
        bench_type, cpu_results, gpu_results, cpu_baseline, gpu_baseline
    )


if __name__ == "__main__":
    main()
