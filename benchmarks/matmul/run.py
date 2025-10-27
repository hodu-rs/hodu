import subprocess
import sys
import re
from pathlib import Path

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


def print_color(color, text):
    """Prints text in the specified color."""
    print(f"{color}{text}{NC}")


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
            # Check for TIMEOUT first
            if "TIMEOUT" in line:
                # Parse: 128x128x128,TIMEOUT
                match = re.match(r"(\d+)x(\d+)x(\d+),TIMEOUT", line)
                if match:
                    m, k, n = match.groups()
                    size = f"{m}x{k}x{n}"
                    results[size] = "TIMEOUT"
            elif "ERROR" in line:
                # Parse: 128x128x128,ERROR
                match = re.match(r"(\d+)x(\d+)x(\d+),ERROR", line)
                if match:
                    m, k, n = match.groups()
                    size = f"{m}x{k}x{n}"
                    results[size] = "ERROR"
            elif "ms" in line:
                # Parse: 128x128x128,0.123456ms or 128x128x128,time_ms=0.123456ms
                match = re.match(
                    r"(\d+)x(\d+)x(\d+),(?:time_ms=)?([0-9.]+)(?:ms)?", line
                )
                if match:
                    m, k, n, time_ms = match.groups()
                    size = f"{m}x{k}x{n}"
                    results[size] = float(time_ms)

    return mode, results


def run_candle_benchmark(mode):
    """Run Candle (Rust) benchmark."""
    print_color(CYAN, f"\n--- Running Candle {mode} ---")

    # Build first
    build_cmd = ["cargo", "build", "--release", "--bin", "candle"]
    if "metal" in mode:
        build_cmd.append("--features=metal")
    elif "cuda" in mode:
        build_cmd.append("--features=cuda")

    print_color(YELLOW, f"Building: {' '.join(build_cmd)}")
    build_result = run_command(build_cmd, cwd=Path(__file__).parent)

    if build_result and build_result.returncode != 0:
        print_color(RED, f"Candle build failed: {build_result.stderr}")
        return None, {}

    # Run benchmark
    run_cmd = ["cargo", "run", "--release", "--bin", "candle", "--"]
    if "metal" in mode:
        run_cmd.insert(3, "--features=metal")
    elif "cuda" in mode:
        run_cmd.insert(3, "--features=cuda")
    run_cmd.append(mode)

    print_color(YELLOW, f"Running: {' '.join(run_cmd)}")
    result = run_command(run_cmd, cwd=Path(__file__).parent)

    if result and result.returncode == 0:
        return parse_benchmark_output(result.stdout)
    else:
        error_msg = result.stderr if result else "Unknown error"
        print_color(RED, f"Candle benchmark failed: {error_msg}")
        return None, {}


def run_burn_benchmark(mode):
    """Run Burn (Rust) benchmark."""
    print_color(CYAN, f"\n--- Running Burn {mode} ---")

    # Build first
    build_cmd = ["cargo", "build", "--release", "--bin", "burn"]
    if "wgpu" in mode:
        build_cmd.append("--features=wgpu")
    elif "tch" in mode or "cuda" in mode:
        build_cmd.append("--features=cuda")

    print_color(YELLOW, f"Building: {' '.join(build_cmd)}")
    build_result = run_command(build_cmd, cwd=Path(__file__).parent)

    if build_result and build_result.returncode != 0:
        print_color(RED, f"Burn build failed: {build_result.stderr}")
        return None, {}

    # Run benchmark
    run_cmd = ["cargo", "run", "--release", "--bin", "burn", "--"]
    if "wgpu" in mode:
        run_cmd.insert(3, "--features=wgpu")
    elif "tch" in mode or "cuda" in mode:
        run_cmd.insert(3, "--features=cuda")
    run_cmd.append(mode)

    print_color(YELLOW, f"Running: {' '.join(run_cmd)}")
    result = run_command(run_cmd, cwd=Path(__file__).parent)

    if result and result.returncode == 0:
        return parse_benchmark_output(result.stdout)
    else:
        error_msg = result.stderr if result else "Unknown error"
        print_color(RED, f"Burn benchmark failed: {error_msg}")
        return None, {}


def run_hodu_benchmark(mode):
    """Run Hodu (Rust) benchmark."""
    print_color(CYAN, f"\n--- Running Hodu {mode} ---")

    # Build first
    build_cmd = ["cargo", "build", "--release", "--bin", "hodu"]
    if "metal" in mode:
        build_cmd.append("--features=metal")
    elif "xla" in mode:
        build_cmd.append("--features=xla")

    print_color(YELLOW, f"Building: {' '.join(build_cmd)}")
    build_result = run_command(build_cmd, cwd=Path(__file__).parent)

    if build_result and build_result.returncode != 0:
        print_color(RED, f"Hodu build failed: {build_result.stderr}")
        return None, {}

    # Run benchmark
    run_cmd = ["cargo", "run", "--release", "--bin", "hodu", "--"]
    if "metal" in mode:
        run_cmd.insert(3, "--features=metal")
    elif "xla" in mode:
        run_cmd.insert(3, "--features=xla")
    run_cmd.append(mode)

    print_color(YELLOW, f"Running: {' '.join(run_cmd)}")
    result = run_command(run_cmd, cwd=Path(__file__).parent)

    if result and result.returncode == 0:
        return parse_benchmark_output(result.stdout)
    else:
        error_msg = result.stderr if result else "Unknown error"
        print_color(RED, f"Hodu benchmark failed: {error_msg}")
        return None, {}


def run_python_benchmark(script, mode):
    """Run Python-based benchmark (PyTorch, JAX, TensorFlow)."""
    script_name = Path(script).stem.replace("_", "")
    print_color(CYAN, f"\n--- Running {script_name.upper()} {mode} ---")

    # Determine which Python executable to use
    base_path = Path(__file__).parent.parent

    if script == "_jax.py":
        # Use venvs/1 for JAX
        venv_path = base_path / "venvs" / "1"
        if venv_path.exists():
            python_exe = venv_path / "bin" / "python3"
            cmd = [str(python_exe), script, mode]
        else:
            print_color(YELLOW, "Warning: venvs/1 not found, using system python3")
            cmd = ["python3", script, mode]
    elif script == "_tensorflow.py":
        # Use venvs/2 for TensorFlow
        venv_path = base_path / "venvs" / "2"
        if venv_path.exists():
            python_exe = venv_path / "bin" / "python3"
            cmd = [str(python_exe), script, mode]
        else:
            print_color(YELLOW, "Warning: venvs/2 not found, using system python3")
            cmd = ["python3", script, mode]
    else:
        # Use venvs/1 for PyTorch (same as JAX)
        venv_path = base_path / "venvs" / "1"
        if venv_path.exists():
            python_exe = venv_path / "bin" / "python3"
            cmd = [str(python_exe), script, mode]
        else:
            print_color(YELLOW, "Warning: venvs/1 not found, using system python3")
            cmd = ["python3", script, mode]

    print_color(YELLOW, f"Running: {' '.join(cmd)}")

    result = run_command(cmd, cwd=Path(__file__).parent)

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


def print_comparison_table(all_results, cpu_baseline, gpu_baseline):
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

    print("\n" + "=" * 80)
    print("Matrix Multiplication Benchmark Results")
    print("=" * 80)

    for size in sizes:
        print(f"\n[{size}]")
        print(f"{'Framework':<25} {'Mode':<15} {'Time (ms)':>12} {'Faster than':>12}")
        print("-" * 80)

        # Print CPU results first
        if cpu_results:
            cpu_baseline_results = all_results.get(cpu_baseline, {})
            cpu_baseline_time = cpu_baseline_results.get(size)

            impl_names = sorted(cpu_results.keys())

            for impl_name in impl_names:
                results = cpu_results[impl_name]
                time_ms = results.get(size)

                if time_ms is None:
                    continue

                # Extract framework and mode
                parts = impl_name.split(" - ")
                framework = parts[0] if parts else impl_name
                mode = parts[1] if len(parts) > 1 else ""

                # Handle TIMEOUT cases
                if time_ms == "TIMEOUT":
                    ratio_str = ""
                    ratio_color = RED
                    if (
                        cpu_baseline_time
                        and cpu_baseline_time != "TIMEOUT"
                        and cpu_baseline_time != "ERROR"
                    ):
                        if (
                            isinstance(cpu_baseline_time, (int, float))
                            and cpu_baseline_time > 0
                        ):
                            max_ratio = cpu_baseline_time / 1000.0
                            ratio_str = f"<{max_ratio:.2f}x"
                        else:
                            ratio_str = "<0.01x"
                    else:
                        ratio_str = "TIMEOUT"

                    if impl_name == cpu_baseline:
                        fw_color = BLUE
                    elif framework == "Hodu":
                        fw_color = LIGHT_PINK
                    else:
                        fw_color = CYAN
                    framework_str = f"{framework:<25}"
                    mode_str = f"{mode:<15}"
                    time_str = f"{'TIMEOUT':>12}"
                    ratio_str_formatted = f"{ratio_str:>12}"

                    print(
                        f"{fw_color}{framework_str}{NC} {mode_str} {RED}{time_str}{NC} {ratio_color}{ratio_str_formatted}{NC}"
                    )
                    continue

                # Handle ERROR cases
                if time_ms == "ERROR":
                    if impl_name == cpu_baseline:
                        fw_color = BLUE
                    elif framework == "Hodu":
                        fw_color = LIGHT_PINK
                    else:
                        fw_color = CYAN
                    framework_str = f"{framework:<25}"
                    mode_str = f"{mode:<15}"
                    time_str = f"{'ERROR':>12}"
                    ratio_str_formatted = f"{'N/A':>12}"

                    print(
                        f"{fw_color}{framework_str}{NC} {mode_str} {RED}{time_str}{NC} {ratio_str_formatted}"
                    )
                    continue

                # Calculate ratio vs CPU baseline for normal cases
                ratio_str = ""
                ratio_color = NC
                if cpu_baseline_time == "TIMEOUT" or cpu_baseline_time == "ERROR":
                    ratio_str = "N/A"
                    ratio_color = NC
                elif (
                    cpu_baseline_time
                    and isinstance(cpu_baseline_time, (int, float))
                    and cpu_baseline_time > 0
                ):
                    ratio = cpu_baseline_time / time_ms
                    ratio_str = f"{ratio:.2f}x"
                    ratio_color = get_ratio_color(ratio)
                elif impl_name == cpu_baseline:
                    ratio_str = "baseline"
                    ratio_color = WHITE

                if impl_name == cpu_baseline:
                    fw_color = BLUE
                elif framework == "Hodu":
                    fw_color = LIGHT_PINK
                else:
                    fw_color = CYAN
                framework_str = f"{framework:<25}"
                mode_str = f"{mode:<15}"
                time_str = f"{time_ms:>12.4f}"
                ratio_str_formatted = f"{ratio_str:>12}"

                print(
                    f"{fw_color}{framework_str}{NC} {mode_str} {time_str} {ratio_color}{ratio_str_formatted}{NC}"
                )

        # Print separator between CPU and GPU
        if cpu_results and gpu_results:
            print("-" * 80)

        # Print GPU results
        if gpu_results:
            gpu_baseline_results = all_results.get(gpu_baseline, {})
            gpu_baseline_time = gpu_baseline_results.get(size)

            impl_names = sorted(gpu_results.keys())

            for impl_name in impl_names:
                results = gpu_results[impl_name]
                time_ms = results.get(size)

                if time_ms is None:
                    continue

                # Extract framework and mode
                parts = impl_name.split(" - ")
                framework = parts[0] if parts else impl_name
                mode = parts[1] if len(parts) > 1 else ""

                # Handle TIMEOUT cases
                if time_ms == "TIMEOUT":
                    ratio_str = ""
                    ratio_color = RED
                    if (
                        gpu_baseline_time
                        and gpu_baseline_time != "TIMEOUT"
                        and gpu_baseline_time != "ERROR"
                    ):
                        if (
                            isinstance(gpu_baseline_time, (int, float))
                            and gpu_baseline_time > 0
                        ):
                            max_ratio = gpu_baseline_time / 1000.0
                            ratio_str = f"<{max_ratio:.2f}x"
                        else:
                            ratio_str = "<0.01x"
                    else:
                        ratio_str = "TIMEOUT"

                    if impl_name == gpu_baseline:
                        fw_color = BLUE
                    elif framework == "Hodu":
                        fw_color = LIGHT_PINK
                    else:
                        fw_color = CYAN
                    framework_str = f"{framework:<25}"
                    mode_str = f"{mode:<15}"
                    time_str = f"{'TIMEOUT':>12}"
                    ratio_str_formatted = f"{ratio_str:>12}"

                    print(
                        f"{fw_color}{framework_str}{NC} {mode_str} {RED}{time_str}{NC} {ratio_color}{ratio_str_formatted}{NC}"
                    )
                    continue

                # Handle ERROR cases
                if time_ms == "ERROR":
                    if impl_name == gpu_baseline:
                        fw_color = BLUE
                    elif framework == "Hodu":
                        fw_color = LIGHT_PINK
                    else:
                        fw_color = CYAN
                    framework_str = f"{framework:<25}"
                    mode_str = f"{mode:<15}"
                    time_str = f"{'ERROR':>12}"
                    ratio_str_formatted = f"{'N/A':>12}"

                    print(
                        f"{fw_color}{framework_str}{NC} {mode_str} {RED}{time_str}{NC} {ratio_str_formatted}"
                    )
                    continue

                # Calculate ratio vs GPU baseline for normal cases
                ratio_str = ""
                ratio_color = NC
                if gpu_baseline_time == "TIMEOUT" or gpu_baseline_time == "ERROR":
                    ratio_str = "N/A"
                    ratio_color = NC
                elif (
                    gpu_baseline_time
                    and isinstance(gpu_baseline_time, (int, float))
                    and gpu_baseline_time > 0
                ):
                    ratio = gpu_baseline_time / time_ms
                    ratio_str = f"{ratio:.2f}x"
                    ratio_color = get_ratio_color(ratio)
                elif impl_name == gpu_baseline:
                    ratio_str = "baseline"
                    ratio_color = WHITE

                if impl_name == gpu_baseline:
                    fw_color = BLUE
                elif framework == "Hodu":
                    fw_color = LIGHT_PINK
                else:
                    fw_color = CYAN
                framework_str = f"{framework:<25}"
                mode_str = f"{mode:<15}"
                time_str = f"{time_ms:>12.4f}"
                ratio_str_formatted = f"{ratio_str:>12}"

                print(
                    f"{fw_color}{framework_str}{NC} {mode_str} {time_str} {ratio_color}{ratio_str_formatted}{NC}"
                )

    print("\n" + "=" * 80)


def main():
    # Check if we're in the right directory
    if not Path("_hodu.rs").exists():
        print_color(RED, "Error: Must be run from benchmarks/matmul directory")
        sys.exit(1)

    # Parse command line arguments
    enable_cpu = "--cpu" in sys.argv
    enable_metal = "--metal" in sys.argv
    enable_cuda = "--cuda" in sys.argv
    enable_xla = "--xla" in sys.argv

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

    # Run Burn benchmarks
    for mode in burn_modes:
        mode_name, results = run_burn_benchmark(mode)
        if results:
            all_results[f"Burn - {mode_name or mode}"] = results

    # Run Candle benchmarks
    for mode in candle_modes:
        mode_name, results = run_candle_benchmark(mode)
        if results:
            all_results[f"Candle - {mode_name or mode}"] = results

    # Run Hodu benchmarks
    for mode in hodu_modes:
        mode_name, results = run_hodu_benchmark(mode)
        if results:
            all_results[f"Hodu - {mode_name or mode}"] = results

    # Run JAX benchmarks
    for mode, device in jax_modes:
        mode_name, results = run_python_benchmark("_jax.py", mode)
        if results:
            all_results[f"JAX - {mode_name or mode}"] = results

    # Run TensorFlow benchmarks
    for mode, device in test_modes:
        # TensorFlow uses 'gpu' instead of 'cuda'/'metal'
        tf_mode = mode.replace("cuda", "gpu").replace("metal", "gpu")
        mode_name, results = run_python_benchmark("_tensorflow.py", tf_mode)
        if results:
            all_results[f"TensorFlow - {mode_name or tf_mode}"] = results

    # Run PyTorch benchmarks
    for mode, device in test_modes:
        mode_name, results = run_python_benchmark("_torch.py", mode)
        if results:
            all_results[f"PyTorch - {mode_name or mode}"] = results

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
    print_comparison_table(all_results, cpu_baseline, gpu_baseline)


if __name__ == "__main__":
    main()
