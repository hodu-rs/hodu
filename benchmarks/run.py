#!/usr/bin/env python3

import subprocess
import sys
import re
import json
from pathlib import Path
from datetime import datetime

# --- Colors (ANSI escape codes) ---
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
CYAN = "\033[0;36m"
MAGENTA = "\033[0;35m"
WHITE = "\033[1;37m"
LIGHT_PINK = "\033[38;5;218m"
NC = "\033[0m"  # No Color

# Cursor control
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

# Global quiet mode flag
QUIET_MODE = False


def print_color(color, text):
    """Prints text in the specified color."""
    if not QUIET_MODE:
        print(f"{color}{text}{NC}")


def print_progress(current, total):
    """Print progress percentage in quiet mode (updates same line)."""
    if QUIET_MODE:
        if current == 1:
            # Hide cursor at the start
            print(HIDE_CURSOR, end="", flush=True)
        percentage = int((current / total) * 100)
        bar_length = 20
        filled = int(bar_length * current / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\rProgress: [{bar}] {percentage}%    ", end="", flush=True)


def run_command(command, cwd=None, env=None):
    """Runs a command and captures output."""
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, cwd=cwd, env=env
        )
        return result
    except Exception as e:
        print_color(RED, f"Error running command {' '.join(command)}: {e}")
        return None


def parse_benchmark_output(output):
    """Parse benchmark output and extract timing data."""
    results = {}
    mode = None

    for line in output.strip().split("\n"):
        if line.startswith("mode="):
            mode = line.split("=", 1)[1]
        elif "," in line:
            # Parse generic format: <size>,<value>
            # where size can be any dimension (3D, 4D, etc.)
            parts = line.split(",", 1)
            if len(parts) == 2:
                size = parts[0]
                value = parts[1]

                if "TIMEOUT" in value:
                    results[size] = "TIMEOUT"
                elif "ERROR" in value:
                    results[size] = "ERROR"
                elif "ms" in value:
                    # Extract time value from formats like "0.123456ms" or "time_ms=0.123456ms"
                    match = re.search(r"(?:time_ms=)?([0-9.]+)", value)
                    if match:
                        results[size] = float(match.group(1))

    return mode, results


def run_candle_benchmark(bench_type, mode):
    """Run Candle (Rust) benchmark."""
    print_color(CYAN, f"\n--- Running Candle {mode} ---")

    # Run benchmark (cargo run will build if needed)
    run_cmd = ["cargo", "run", "--release", "--bin", "candle", "--"]
    if "metal" in mode:
        run_cmd.insert(3, "--features=metal")
    elif "cuda" in mode:
        run_cmd.insert(3, "--features=cuda")
    run_cmd.append(mode)

    print_color(YELLOW, f"Running: {' '.join(run_cmd)}")
    result = run_command(run_cmd, cwd=Path(__file__).parent / bench_type)

    if result and result.returncode == 0:
        return parse_benchmark_output(result.stdout)
    else:
        error_msg = result.stderr if result else "Unknown error"
        print_color(RED, f"Candle benchmark failed: {error_msg}")
        return None, {}


def run_burn_benchmark(bench_type, mode):
    """Run Burn (Rust) benchmark."""
    print_color(CYAN, f"\n--- Running Burn {mode} ---")

    # Run benchmark (cargo run will build if needed)
    run_cmd = ["cargo", "run", "--release", "--bin", "burn", "--"]
    if "wgpu" in mode:
        run_cmd.insert(3, "--features=wgpu")
    elif "tch" in mode or "cuda" in mode:
        run_cmd.insert(3, "--features=cuda")
    run_cmd.append(mode)

    print_color(YELLOW, f"Running: {' '.join(run_cmd)}")
    result = run_command(run_cmd, cwd=Path(__file__).parent / bench_type)

    if result and result.returncode == 0:
        return parse_benchmark_output(result.stdout)
    else:
        error_msg = result.stderr if result else "Unknown error"
        print_color(RED, f"Burn benchmark failed: {error_msg}")
        return None, {}


def run_hodu_benchmark(bench_type, mode):
    """Run Hodu (Rust) benchmark."""
    print_color(CYAN, f"\n--- Running Hodu {mode} ---")

    # Run benchmark (cargo run will build if needed)
    run_cmd = ["cargo", "run", "--release", "--bin", "hodu", "--"]
    if "metal" in mode:
        run_cmd.insert(3, "--features=metal,hodu-bench")
    elif "xla" in mode:
        run_cmd.insert(3, "--features=xla,hodu-bench")
    else:
        run_cmd.insert(3, "--features=hodu-bench")
    run_cmd.append(mode)

    print_color(YELLOW, f"Running: {' '.join(run_cmd)}")
    result = run_command(run_cmd, cwd=Path(__file__).parent / bench_type)

    if result and result.returncode == 0:
        return parse_benchmark_output(result.stdout)
    else:
        error_msg = result.stderr if result else "Unknown error"
        print_color(RED, f"Hodu benchmark failed: {error_msg}")
        return None, {}


def run_python_benchmark(bench_type, script, mode):
    """Run Python-based benchmark (PyTorch, JAX, TensorFlow)."""
    script_name = Path(script).stem.replace("_", "")
    print_color(CYAN, f"\n--- Running {script_name.upper()} {mode} ---")

    # Determine which Python executable to use
    base_path = Path(__file__).parent

    if script == "_jax.py":
        # Use .venvs/1 for JAX
        venv_path = base_path / ".venvs" / "1"
        if venv_path.exists():
            python_exe = venv_path / "bin" / "python3"
            cmd = [str(python_exe), script, mode]
        else:
            print_color(YELLOW, "Warning: .venvs/1 not found, using system python3")
            cmd = ["python3", script, mode]
    elif script == "_tensorflow.py":
        # Use .venvs/2 for TensorFlow
        venv_path = base_path / ".venvs" / "2"
        if venv_path.exists():
            python_exe = venv_path / "bin" / "python3"
            cmd = [str(python_exe), script, mode]
        else:
            print_color(YELLOW, "Warning: .venvs/2 not found, using system python3")
            cmd = ["python3", script, mode]
    else:
        # Use .venvs/1 for PyTorch (same as JAX)
        venv_path = base_path / ".venvs" / "1"
        if venv_path.exists():
            python_exe = venv_path / "bin" / "python3"
            cmd = [str(python_exe), script, mode]
        else:
            print_color(YELLOW, "Warning: .venvs/1 not found, using system python3")
            cmd = ["python3", script, mode]

    print_color(YELLOW, f"Running: {' '.join(cmd)}")

    result = run_command(cmd, cwd=Path(__file__).parent / bench_type)

    if result and result.returncode == 0:
        return parse_benchmark_output(result.stdout)
    else:
        error_msg = result.stderr if result else "Unknown error"
        # Check if it's just a missing dependency
        if "ModuleNotFoundError" in error_msg or "ImportError" in error_msg:
            print_color(
                YELLOW, f"{script_name.upper()} not available (dependency missing)"
            )
        else:
            print_color(RED, f"{script_name.upper()} benchmark failed: {error_msg}")
        return None, {}


def get_ratio_color(ratio):
    """Get color based on ratio relative to baseline (1.0).

    Color intensity increases as the difference from 1.0 increases:
    - ratio > 1.0 (faster): Green shades (lighter near 1.0, darker further away)
    - ratio = 1.0 (same): White
    - ratio < 1.0 (slower): Red shades (lighter near 1.0, darker further away)
    """
    if ratio > 1.0:
        # Faster than baseline - Green with varying intensity
        if ratio >= 2.0:
            return "\033[38;5;34m"  # Dark green (very fast)
        elif ratio >= 1.5:
            return "\033[38;5;40m"  # Medium-dark green
        elif ratio >= 1.2:
            return "\033[38;5;46m"  # Medium green
        elif ratio >= 1.1:
            return "\033[38;5;82m"  # Light green
        else:
            return "\033[38;5;120m"  # Very light green (barely faster)
    elif ratio < 1.0:
        # Slower than baseline - Red with varying intensity
        if ratio <= 0.5:
            return "\033[38;5;160m"  # Dark red (very slow)
        elif ratio <= 0.7:
            return "\033[38;5;196m"  # Medium-dark red
        elif ratio <= 0.85:
            return "\033[38;5;202m"  # Medium red
        elif ratio <= 0.95:
            return "\033[38;5;208m"  # Light red/orange
        else:
            return "\033[38;5;214m"  # Very light orange (barely slower)
    else:
        # Same as baseline - White
        return WHITE


def print_comparison_table(bench_type, all_results, cpu_baseline, gpu_baseline):
    """Print a unified comparison table with CPU and GPU sections."""
    if not all_results:
        print_color(RED, "No results to display")
        return

    # Separate CPU and GPU results
    cpu_results = {}
    gpu_results = {}

    for key, results in all_results.items():
        # Check if this is a CPU or GPU benchmark based on the mode name
        if any(x in key for x in ["CPU", "cpu", "XLA"]):
            cpu_results[key] = results
        elif any(x in key for x in ["Metal", "CUDA", "GPU", "gpu", "WGPU", "TCH"]):
            gpu_results[key] = results
        else:
            # Default to CPU if unclear
            cpu_results[key] = results

    # Get all sizes
    sizes = set()
    for _, results in all_results.items():
        sizes.update(results.keys())
    sizes = sorted(sizes, key=lambda s: [int(x) for x in s.split("x")])

    if not sizes:
        print_color(RED, "No timing data found")
        return

    print("\n" + "=" * 140)
    if bench_type == "matmul":
        print("Matrix Multiplication Benchmark Results")
    else:  # mlp
        print("MLP Block Benchmark Results (3-layer MLP with GELU and Residual)")
    print("=" * 140)

    # Print sizes as metadata
    sizes_str = ", ".join(sizes)
    print(f"\nSizes: {sizes_str}\n")

    # Build simple header
    if bench_type == "matmul":
        # For matmul: Time(ms) over all time values, Ratio over all ratio values
        time_width = len(sizes) * 10 + (len(sizes) - 1) * 3
        ratio_width = len(sizes) * 12 + (len(sizes) - 1) * 3
        print(
            f"{'Framework':>12} {'Mode':>15} {'Time(ms)':>{time_width}} {'Ratio':>{ratio_width}}"
        )
    else:  # mlp
        # For mlp: Build header with Time/Ratio labels
        label_parts = []
        # First add all Time columns
        for i, size in enumerate(sizes):
            if i == 0:
                label_parts.append(f"{'Time(ms)':>10}")
            else:
                label_parts.append(f"{'':>10}")
        # Then add all Ratio columns
        for i, size in enumerate(sizes):
            if i == 0:
                label_parts.append(f"{'Ratio':>12}")
            else:
                label_parts.append(f"{'':>12}")
        label_header = "  ".join(label_parts)
        header_line = f"{'Framework':>12} {'Mode':>15} {label_header}"
        print(header_line)

    print("-" * 140)

    # Print CPU results first
    if cpu_results:
        cpu_baseline_results = all_results.get(cpu_baseline, {})
        impl_names = sorted(cpu_results.keys())

        for impl_name in impl_names:
            results = cpu_results[impl_name]

            # Extract framework and mode
            parts = impl_name.split(" - ")
            framework = parts[0] if parts else impl_name
            mode = parts[1] if len(parts) > 1 else ""

            if impl_name == cpu_baseline:
                fw_color = BLUE
            elif framework == "Hodu":
                fw_color = LIGHT_PINK
            else:
                fw_color = CYAN

            # Build time string
            time_parts = []
            ratio_parts = []

            for size in sizes:
                time_ms = results.get(size)
                cpu_baseline_time = cpu_baseline_results.get(size)

                if time_ms is None:
                    time_parts.append(f"{'N/A':>10}")
                    ratio_parts.append(f"{'':>12}")
                elif time_ms == "TIMEOUT":
                    time_parts.append(f"{'TIMEOUT':>10}")
                    if (
                        cpu_baseline_time
                        and cpu_baseline_time != "TIMEOUT"
                        and cpu_baseline_time != "ERROR"
                        and isinstance(cpu_baseline_time, (int, float))
                        and cpu_baseline_time > 0
                    ):
                        max_ratio = cpu_baseline_time / 1000.0
                        ratio_parts.append(f"{RED}{f'<{max_ratio:.2f}x':>12}{NC}")
                    else:
                        ratio_parts.append(f"{RED}{'TIMEOUT':>12}{NC}")
                elif time_ms == "ERROR":
                    time_parts.append(f"{'ERROR':>10}")
                    ratio_parts.append(f"{'N/A':>12}")
                else:
                    time_parts.append(f"{time_ms:>10.4f}")
                    # Calculate ratio vs CPU baseline
                    if cpu_baseline_time == "TIMEOUT" or cpu_baseline_time == "ERROR":
                        ratio_parts.append(f"{'N/A':>12}")
                    elif (
                        cpu_baseline_time
                        and isinstance(cpu_baseline_time, (int, float))
                        and cpu_baseline_time > 0
                    ):
                        ratio = cpu_baseline_time / time_ms
                        ratio_color = get_ratio_color(ratio)
                        ratio_parts.append(f"{ratio_color}{f'{ratio:.2f}x':>12}{NC}")
                    elif impl_name == cpu_baseline:
                        ratio_parts.append(f"{WHITE}{'baseline':>12}{NC}")
                    else:
                        ratio_parts.append(f"{'':>12}")

            if bench_type == "matmul":
                time_str = ",  ".join(time_parts)
                ratio_str = ",  ".join(ratio_parts)
                print(
                    f"{fw_color}{framework:>12}{NC} {mode:>15} {time_str} {ratio_str}"
                )
            else:  # mlp
                output_str = "  ".join(time_parts + ratio_parts)
                print(f"{fw_color}{framework:>12}{NC} {mode:>15} {output_str}")

    # Print separator between CPU and GPU
    if cpu_results and gpu_results:
        print("-" * 140)

    # Print GPU results
    if gpu_results:
        gpu_baseline_results = all_results.get(gpu_baseline, {})
        impl_names = sorted(gpu_results.keys())

        for impl_name in impl_names:
            results = gpu_results[impl_name]

            # Extract framework and mode
            parts = impl_name.split(" - ")
            framework = parts[0] if parts else impl_name
            mode = parts[1] if len(parts) > 1 else ""

            if impl_name == gpu_baseline:
                fw_color = BLUE
            elif framework == "Hodu":
                fw_color = LIGHT_PINK
            else:
                fw_color = CYAN

            # Build time string
            time_parts = []
            ratio_parts = []

            for size in sizes:
                time_ms = results.get(size)
                gpu_baseline_time = gpu_baseline_results.get(size)

                if time_ms is None:
                    time_parts.append(f"{'N/A':>10}")
                    ratio_parts.append(f"{'':>12}")
                elif time_ms == "TIMEOUT":
                    time_parts.append(f"{'TIMEOUT':>10}")
                    if (
                        gpu_baseline_time
                        and gpu_baseline_time != "TIMEOUT"
                        and gpu_baseline_time != "ERROR"
                        and isinstance(gpu_baseline_time, (int, float))
                        and gpu_baseline_time > 0
                    ):
                        max_ratio = gpu_baseline_time / 1000.0
                        ratio_parts.append(f"{RED}{f'<{max_ratio:.2f}x':>12}{NC}")
                    else:
                        ratio_parts.append(f"{RED}{'TIMEOUT':>12}{NC}")
                elif time_ms == "ERROR":
                    time_parts.append(f"{'ERROR':>10}")
                    ratio_parts.append(f"{'N/A':>12}")
                else:
                    time_parts.append(f"{time_ms:>10.4f}")
                    # Calculate ratio vs GPU baseline
                    if gpu_baseline_time == "TIMEOUT" or gpu_baseline_time == "ERROR":
                        ratio_parts.append(f"{'N/A':>12}")
                    elif (
                        gpu_baseline_time
                        and isinstance(gpu_baseline_time, (int, float))
                        and gpu_baseline_time > 0
                    ):
                        ratio = gpu_baseline_time / time_ms
                        ratio_color = get_ratio_color(ratio)
                        ratio_parts.append(f"{ratio_color}{f'{ratio:.2f}x':>12}{NC}")
                    elif impl_name == gpu_baseline:
                        ratio_parts.append(f"{WHITE}{'baseline':>12}{NC}")
                    else:
                        ratio_parts.append(f"{'':>12}")

            if bench_type == "matmul":
                time_str = ",  ".join(time_parts)
                ratio_str = ",  ".join(ratio_parts)
                print(
                    f"{fw_color}{framework:>12}{NC} {mode:>15} {time_str} {ratio_str}"
                )
            else:  # mlp
                output_str = "  ".join(time_parts + ratio_parts)
                print(f"{fw_color}{framework:>12}{NC} {mode:>15} {output_str}")

    print("\n" + "=" * 140)


def save_results_json(bench_type, all_results, output_path):
    """Save benchmark results to JSON file."""
    data = {
        "benchmark_type": bench_type,
        "timestamp": datetime.now().isoformat(),
        "results": {},
    }

    for impl_name, results in all_results.items():
        data["results"][impl_name] = {}
        for size, value in results.items():
            if isinstance(value, (int, float)):
                data["results"][impl_name][size] = value
            else:
                data["results"][impl_name][size] = str(value)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print_color(GREEN, f"\nResults saved to: {output_path}")


def plot_results(json_path, save_plot=False):
    """Generate plots from benchmark results JSON."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print_color(RED, "Error: pandas and matplotlib are required for plotting")
        print_color(YELLOW, "Install with: pip install pandas matplotlib")
        return

    # Load results
    with open(json_path, "r") as f:
        data = json.load(f)

    bench_type = data["benchmark_type"]
    results = data["results"]

    # Convert to DataFrame
    df_data = []
    for impl_name, sizes_data in results.items():
        for size, time_ms in sizes_data.items():
            if isinstance(time_ms, (int, float)):
                parts = impl_name.split(" - ")
                framework = parts[0] if parts else impl_name
                mode = parts[1] if len(parts) > 1 else ""
                df_data.append(
                    {
                        "framework": framework,
                        "mode": mode,
                        "impl": impl_name,
                        "size": size,
                        "time_ms": time_ms,
                    }
                )

    df = pd.DataFrame(df_data)

    if df.empty:  # type: ignore
        print_color(RED, "No valid data to plot")
        return

    # Separate CPU and GPU results
    cpu_mask = df["impl"].str.contains("CPU|cpu|XLA", case=False, na=False)
    gpu_mask = df["impl"].str.contains(
        "Metal|CUDA|GPU|gpu|WGPU|TCH", case=False, na=False
    )

    df_cpu = df[cpu_mask]
    df_gpu = df[gpu_mask]

    # Find baselines (PyTorch dynamic)
    cpu_baseline = None
    gpu_baseline = None

    for impl in df["impl"].unique():  # type: ignore
        if "PyTorch" in impl and ("Dynamic CPU" in impl or "dynamic-cpu" in impl):
            cpu_baseline = impl
        if "PyTorch" in impl and (
            "Dynamic Metal" in impl
            or "dynamic-metal" in impl
            or "Dynamic CUDA" in impl
            or "dynamic-cuda" in impl
        ):
            gpu_baseline = impl

    # Calculate ratios for CPU
    if cpu_baseline and not df_cpu.empty:  # type: ignore
        baseline_data = df_cpu[df_cpu["impl"] == cpu_baseline]
        for size in df_cpu["size"].unique():  # type: ignore
            baseline_time = baseline_data[baseline_data["size"] == size]["time_ms"]
            if not baseline_time.empty:  # type: ignore
                baseline_time = baseline_time.values[0]  # type: ignore
                mask = df_cpu["size"] == size
                df_cpu.loc[mask, "ratio"] = baseline_time / df_cpu.loc[mask, "time_ms"]
            else:
                mask = df_cpu["size"] == size
                df_cpu.loc[mask, "ratio"] = 1.0
    else:
        df_cpu["ratio"] = 1.0

    # Calculate ratios for GPU
    if gpu_baseline and not df_gpu.empty:  # type: ignore
        baseline_data = df_gpu[df_gpu["impl"] == gpu_baseline]
        for size in df_gpu["size"].unique():  # type: ignore
            baseline_time = baseline_data[baseline_data["size"] == size]["time_ms"]
            if not baseline_time.empty:  # type: ignore
                baseline_time = baseline_time.values[0]  # type: ignore
                mask = df_gpu["size"] == size
                df_gpu.loc[mask, "ratio"] = baseline_time / df_gpu.loc[mask, "time_ms"]
            else:
                mask = df_gpu["size"] == size
                df_gpu.loc[mask, "ratio"] = 1.0
    else:
        df_gpu["ratio"] = 1.0

    # Modern plot style
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.facecolor": "#f8f9fa",
            "figure.facecolor": "white",
            "axes.edgecolor": "#dee2e6",
            "axes.linewidth": 1.2,
            "grid.color": "#dee2e6",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.5,
        }
    )

    # Color palette - more vibrant
    framework_colors = {
        "Hodu": "#E91E63",  # Material Pink (more vibrant)
        "PyTorch": "#EE4C2C",  # PyTorch orange
        "JAX": "#4285F4",  # Google blue
        "TensorFlow": "#FF6F00",  # TensorFlow orange
        "Burn": "#D84315",  # Deep orange
        "Candle": "#8D6E63",  # Brown
    }

    # Create figure with subplots
    has_cpu = not df_cpu.empty  # type: ignore
    has_gpu = not df_gpu.empty  # type: ignore
    n_plots = (1 if has_cpu else 0) + (1 if has_gpu else 0)

    if n_plots == 0:
        print_color(RED, "No data to plot")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot CPU results
    if not df_cpu.empty:  # type: ignore
        ax = axes[plot_idx]
        plot_idx += 1

        sizes = sorted(
            df_cpu["size"].unique(),  # type: ignore
            key=lambda s: [int(x) for x in s.split("x")],  # type: ignore
        )

        impls = sorted(df_cpu["impl"].unique())  # type: ignore
        n_impls = len(impls)
        n_sizes = len(sizes)

        # Bar width and positions
        bar_width = 0.8 / n_impls
        x = range(n_sizes)

        for idx, impl in enumerate(impls):
            impl_data = df_cpu[df_cpu["impl"] == impl]
            ratios = [
                impl_data[impl_data["size"] == s]["ratio"].values[0]  # type: ignore
                if s in impl_data["size"].values  # type: ignore
                else 0
                for s in sizes
            ]

            framework = impl_data["framework"].iloc[0]  # type: ignore
            color = framework_colors.get(framework, "#607D8B")

            # Style differentiation
            is_dynamic = "Dynamic" in impl or "dynamic" in impl
            hatch = None if is_dynamic else "//"
            alpha = 0.9 if framework == "Hodu" else 0.85

            # Cleaner label
            mode = impl.split(" - ")[1] if " - " in impl else impl
            label = f"{framework} ({mode})"

            # Calculate x positions for this group
            x_pos = [i + (idx - n_impls / 2 + 0.5) * bar_width for i in x]

            # Filter out zero values for plotting
            x_pos_filtered = [x_pos[i] for i in range(len(ratios)) if ratios[i] > 0]
            ratios_filtered = [r for r in ratios if r > 0]

            if ratios_filtered:  # Only plot if there's data
                ax.bar(
                    x_pos_filtered,
                    ratios_filtered,
                    bar_width,
                    label=label,
                    color=color,
                    alpha=alpha,
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=1.5,
                )

        # Add baseline reference line at y=1.0
        ax.axhline(
            y=1.0,
            color="#dc3545",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Baseline (PyTorch)",
            zorder=0,
        )

        ax.set_xlabel("Problem Size", fontsize=14, fontweight="600", color="#2c3e50")
        ax.set_ylabel(
            "Speedup (relative to baseline)",
            fontsize=14,
            fontweight="600",
            color="#2c3e50",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=45, ha="right", fontsize=11)
        ax.tick_params(axis="both", labelsize=11, colors="#495057")
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=9,
            framealpha=0.95,
            edgecolor="#dee2e6",
            shadow=True,
            fancybox=True,
        )
        ax.set_yscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y")

        # Add annotations for values > 1 (better than baseline)
        ax.text(
            0.02,
            0.98,
            "Higher is better ↑",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            color="#28a745",
            fontweight="600",
        )

    # Plot GPU results
    if not df_gpu.empty:  # type: ignore
        ax = axes[plot_idx]

        sizes = sorted(
            df_gpu["size"].unique(),  # type: ignore
            key=lambda s: [int(x) for x in s.split("x")],  # type: ignore
        )

        impls = sorted(df_gpu["impl"].unique())  # type: ignore
        n_impls = len(impls)
        n_sizes = len(sizes)

        # Bar width and positions
        bar_width = 0.8 / n_impls
        x = range(n_sizes)

        for idx, impl in enumerate(impls):
            impl_data = df_gpu[df_gpu["impl"] == impl]
            ratios = [
                impl_data[impl_data["size"] == s]["ratio"].values[0]  # type: ignore
                if s in impl_data["size"].values  # type: ignore
                else 0
                for s in sizes
            ]

            framework = impl_data["framework"].iloc[0]  # type: ignore
            color = framework_colors.get(framework, "#607D8B")

            # Style differentiation
            is_dynamic = "Dynamic" in impl or "dynamic" in impl
            hatch = None if is_dynamic else "//"
            alpha = 0.9 if framework == "Hodu" else 0.85

            # Cleaner label
            mode = impl.split(" - ")[1] if " - " in impl else impl
            label = f"{framework} ({mode})"

            # Calculate x positions for this group
            x_pos = [i + (idx - n_impls / 2 + 0.5) * bar_width for i in x]

            # Filter out zero values for plotting
            x_pos_filtered = [x_pos[i] for i in range(len(ratios)) if ratios[i] > 0]
            ratios_filtered = [r for r in ratios if r > 0]

            if ratios_filtered:  # Only plot if there's data
                ax.bar(
                    x_pos_filtered,
                    ratios_filtered,
                    bar_width,
                    label=label,
                    color=color,
                    alpha=alpha,
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=1.5,
                )

        # Add baseline reference line at y=1.0
        ax.axhline(
            y=1.0,
            color="#dc3545",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Baseline (PyTorch)",
            zorder=0,
        )

        # Determine GPU type
        gpu_type = "Metal" if "Metal" in df_gpu["impl"].iloc[0] else "CUDA"  # type: ignore
        ax.set_xlabel("Problem Size", fontsize=14, fontweight="600", color="#2c3e50")
        ax.set_ylabel(
            "Speedup (relative to baseline)",
            fontsize=14,
            fontweight="600",
            color="#2c3e50",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=45, ha="right", fontsize=11)
        ax.tick_params(axis="both", labelsize=11, colors="#495057")
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=9,
            framealpha=0.95,
            edgecolor="#dee2e6",
            shadow=True,
            fancybox=True,
        )
        ax.set_yscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y")

        # Add annotations for values > 1 (better than baseline)
        ax.text(
            0.02,
            0.98,
            "Higher is better ↑",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            color="#28a745",
            fontweight="600",
        )

    plt.tight_layout(pad=2.0)

    # Save plot if requested
    if save_plot:
        output_dir = Path(json_path).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = output_dir / f"benchmark_{bench_type}_{timestamp}.png"
        plt.savefig(
            plot_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print_color(GREEN, f"Plot saved to: {plot_path}")

    plt.show()


def main():
    global QUIET_MODE

    # Parse command line arguments
    bench_type = None
    enable_cpu = "--cpu" in sys.argv
    enable_metal = "--metal" in sys.argv
    enable_cuda = "--cuda" in sys.argv
    enable_xla = "--xla" in sys.argv
    QUIET_MODE = "--quiet" in sys.argv
    enable_plot = "--plot" in sys.argv
    enable_save = "--save" in sys.argv

    # Check if only plotting
    plot_only = None
    for arg in sys.argv[1:]:
        if arg.startswith("--plot="):
            plot_only = arg.split("=", 1)[1]
            break

    # If plot only mode, just plot and exit
    if plot_only:
        plot_results(plot_only)
        return

    # Find --bench argument
    for arg in sys.argv[1:]:
        if arg.startswith("--bench="):
            bench_type = arg.split("=", 1)[1]
            break

    if not bench_type:
        print_color(RED, "Error: Must specify --bench=matmul or --bench=mlp")
        print_color(
            YELLOW,
            "Usage: python run.py --bench=matmul|mlp [--cpu] [--metal] [--cuda] [--xla] [--quiet] [--plot] [--save]",
        )
        print_color(
            YELLOW,
            "   Or: python run.py --plot=<json_file>  # Plot from existing results",
        )
        sys.exit(1)

    if bench_type not in ["matmul", "mlp"]:
        print_color(RED, f"Error: Unknown benchmark type '{bench_type}'")
        print_color(YELLOW, "Valid types: matmul, mlp")
        sys.exit(1)

    # Check if benchmark directory exists
    bench_dir = Path(__file__).parent / bench_type
    if not bench_dir.exists():
        print_color(RED, f"Error: Benchmark directory '{bench_type}' not found")
        sys.exit(1)

    # Determine which modes to test
    test_modes = []
    hodu_modes = []
    jax_modes = []

    # Add CPU modes if requested
    if enable_cpu:
        test_modes.extend(
            [
                ("dynamic-cpu", "CPU"),
                ("static-cpu", "CPU"),
            ]
        )
        hodu_modes.extend(["dynamic-cpu", "static-cpu"])
        jax_modes.extend([("dynamic-cpu", "CPU"), ("static-cpu", "CPU")])

    # Add GPU modes if requested
    if sys.platform == "darwin" and enable_metal:
        test_modes.extend(
            [
                ("dynamic-metal", "Metal"),
                ("static-metal", "Metal"),
            ]
        )
        hodu_modes.extend(["dynamic-metal", "static-metal"])
        jax_modes.extend([("dynamic-metal", "Metal"), ("static-metal", "Metal")])
    elif sys.platform != "darwin" and enable_cuda:
        test_modes.extend(
            [
                ("dynamic-cuda", "CUDA"),
                ("static-cuda", "CUDA"),
            ]
        )
        # Note: Hodu CUDA support would need to be added here
        # hodu_modes.extend(["dynamic-cuda", "static-cuda"])

    # Add XLA mode if requested (requires CPU to be enabled)
    if enable_xla and enable_cpu:
        hodu_modes.append("static-xla-cpu")

    # Determine Burn modes (dynamic only)
    burn_modes = []
    if enable_cpu:
        burn_modes.append("dynamic-cpu")
    # WGPU works on both Metal (macOS) and CUDA (Linux/Windows)
    if sys.platform == "darwin" and enable_metal:
        burn_modes.append("dynamic-wgpu")
    if sys.platform != "darwin" and enable_cuda:
        burn_modes.append("dynamic-wgpu")
        # LibTorch CUDA backend (only on CUDA platforms)
        burn_modes.append("dynamic-tch")

    # Determine Candle modes (dynamic only)
    candle_modes = []
    if enable_cpu:
        candle_modes.append("dynamic-cpu")
    if sys.platform == "darwin" and enable_metal:
        candle_modes.append("dynamic-metal")
    elif sys.platform != "darwin" and enable_cuda:
        candle_modes.append("dynamic-cuda")

    all_results = {}

    # Calculate total number of benchmarks for progress tracking
    total_benchmarks = (
        len(burn_modes)
        + len(candle_modes)
        + len(hodu_modes)
        + len(jax_modes)
        + len(test_modes) * 2  # TensorFlow + PyTorch
    )
    current_benchmark = 0

    # Run Burn benchmarks
    for mode in burn_modes:
        current_benchmark += 1
        print_progress(current_benchmark, total_benchmarks)
        mode_name, results = run_burn_benchmark(bench_type, mode)
        if results:
            all_results[f"Burn - {mode_name or mode}"] = results

    # Run Candle benchmarks
    for mode in candle_modes:
        current_benchmark += 1
        print_progress(current_benchmark, total_benchmarks)
        mode_name, results = run_candle_benchmark(bench_type, mode)
        if results:
            all_results[f"Candle - {mode_name or mode}"] = results

    # Run Hodu benchmarks
    for mode in hodu_modes:
        current_benchmark += 1
        print_progress(current_benchmark, total_benchmarks)
        mode_name, results = run_hodu_benchmark(bench_type, mode)
        if results:
            all_results[f"Hodu - {mode_name or mode}"] = results

    # Run JAX benchmarks
    for mode, device in jax_modes:
        current_benchmark += 1
        print_progress(current_benchmark, total_benchmarks)
        mode_name, results = run_python_benchmark(bench_type, "_jax.py", mode)
        if results:
            all_results[f"JAX - {mode_name or mode}"] = results

    # Run TensorFlow benchmarks
    for mode, device in test_modes:
        current_benchmark += 1
        print_progress(current_benchmark, total_benchmarks)
        # TensorFlow uses 'gpu' instead of 'cuda'/'metal'
        tf_mode = mode.replace("cuda", "gpu").replace("metal", "gpu")
        mode_name, results = run_python_benchmark(bench_type, "_tensorflow.py", tf_mode)
        if results:
            all_results[f"TensorFlow - {mode_name or tf_mode}"] = results

    # Run PyTorch benchmarks
    for mode, device in test_modes:
        current_benchmark += 1
        print_progress(current_benchmark, total_benchmarks)
        mode_name, results = run_python_benchmark(bench_type, "_torch.py", mode)
        if results:
            all_results[f"PyTorch - {mode_name or mode}"] = results

    # Print newline after progress indicator in quiet mode
    if QUIET_MODE:
        print(SHOW_CURSOR)  # Show cursor and newline

    # Find CPU and GPU baselines (PyTorch dynamic-cpu)
    cpu_baseline = None
    gpu_baseline = None

    for key in all_results.keys():
        if "PyTorch" in key and ("Dynamic CPU" in key or "dynamic-cpu" in key):
            cpu_baseline = key
            break

    for key in all_results.keys():
        if "PyTorch" in key and (
            "Dynamic Metal" in key
            or "dynamic-metal" in key
            or "Dynamic CUDA" in key
            or "dynamic-cuda" in key
        ):
            if gpu_baseline is None:  # Use first GPU PyTorch result found
                gpu_baseline = key

    # Fallback to first result if PyTorch not found
    if not cpu_baseline:
        for key in all_results.keys():
            if any(x in key for x in ["CPU", "cpu", "XLA"]):
                cpu_baseline = key
                break

    if not gpu_baseline:
        for key in all_results.keys():
            if any(x in key for x in ["Metal", "CUDA", "GPU", "gpu", "WGPU", "TCH"]):
                gpu_baseline = key
                break

    # Print unified comparison table
    print_comparison_table(bench_type, all_results, cpu_baseline, gpu_baseline)

    # Handle save and plot
    if enable_save or enable_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = Path(__file__).parent / f"results_{bench_type}_{timestamp}.json"

        # Save JSON only if --save is specified
        if enable_save:
            save_results_json(bench_type, all_results, json_path)

        # Generate plot if requested
        if enable_plot:
            # For plot-only mode, create temporary JSON
            if not enable_save:
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(
                        {
                            "benchmark_type": bench_type,
                            "timestamp": datetime.now().isoformat(),
                            "results": {
                                impl_name: {
                                    size: value
                                    if isinstance(value, (int, float))
                                    else str(value)
                                    for size, value in results.items()
                                }
                                for impl_name, results in all_results.items()
                            },
                        },
                        f,
                        indent=2,
                    )
                    temp_json_path = f.name
                plot_results(temp_json_path, save_plot=False)
                # Clean up temp file
                import os

                os.unlink(temp_json_path)
            else:
                plot_results(json_path, save_plot=True)


if __name__ == "__main__":
    main()
