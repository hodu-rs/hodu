import sys
import subprocess
import re
import platform
from pathlib import Path


# Colors (ANSI escape codes)
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
GREEN = "\033[0;32m"
NC = "\033[0m"  # No Color


def get_venv_configs(cuda_version=None):
    """Get virtual environment configurations based on platform and CUDA version

    Args:
        cuda_version: None, "12", or "13" for CUDA support
    """
    if platform.system() == "Darwin":
        # macOS: use Metal for JAX
        venv1_packages = [
            "numpy==2.3.4",
            "torch",
            "jax-metal",
            "jaxlib",
        ]
    elif cuda_version == "12":
        # Linux with CUDA 12: install JAX with CUDA 12 support
        venv1_packages = [
            "numpy==2.3.4",
            "torch",
            "jax[cuda12]",
        ]
    elif cuda_version == "13":
        # Linux with CUDA 13: install JAX with CUDA 13 support
        venv1_packages = [
            "numpy==2.3.4",
            "torch",
            "jax[cuda13]",
        ]
    else:
        # Linux without CUDA: CPU-only JAX
        venv1_packages = [
            "numpy==2.3.4",
            "torch",
            "jax",
            "jaxlib",
        ]

    if platform.system() == "Darwin":
        venv2_packages = [
            "numpy==1.26.4",
            "tensorflow-macos",
            "tensorflow-metal",
        ]
    else:
        venv2_packages = [
            "numpy==1.26.4",
            "tensorflow",
        ]

    return [
        {
            "name": ".venvs/1",
            "packages": venv1_packages,
        },
        {
            "name": ".venvs/2",
            "packages": venv2_packages,
        },
    ]


def check_python():
    """Check if python is available and get version"""
    try:
        result = subprocess.run(
            ["python3", "--version"], capture_output=True, text=True, check=True
        )
        version_str = result.stdout.strip()

        # Extract version number (e.g., "Python 3.11.14" -> "3.11.14")
        match = re.search(r"Python (\d+\.\d+\.\d+)", version_str)
        if match:
            return match.group(1)
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_version(version_str):
    """Parse version string into tuple of integers"""
    parts = version_str.split(".")
    return tuple(int(p) for p in parts[:3])


def get_pip_path(venv_path):
    """Get pip executable path for the virtual environment"""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"


def get_activate_command(venv_name):
    """Get activation command based on platform and shell"""
    system = platform.system()

    if system == "Windows":
        return [
            f"   {venv_name}\\Scripts\\Activate.ps1    (PowerShell)",
            f"   {venv_name}\\Scripts\\activate.bat    (CMD)",
        ]
    else:
        return [
            f"   source {venv_name}/bin/activate       (bash/zsh)",
            f"   source {venv_name}/bin/activate.fish  (fish)",
            f"   source {venv_name}/bin/activate.csh   (csh/tcsh)",
        ]


def main():
    # Parse command-line arguments
    cuda_version = None
    if "--cuda12" in sys.argv:
        cuda_version = "12"
    elif "--cuda13" in sys.argv:
        cuda_version = "13"

    # 1. Check if python exists
    version = check_python()
    if version is None:
        print(f"{RED}Error: Python 3 is not installed or not found in PATH{NC}")
        print("Please install Python 3.11.x to continue")
        sys.exit(1)

    print(f"{GREEN}Found Python {version}{NC}")
    if cuda_version:
        print(
            f"{YELLOW}CUDA {cuda_version} support enabled - will install JAX with CUDA{NC}"
        )

    # 2. Check if version is 3.11.x
    major, minor, patch = parse_version(version)
    if major != 3 or minor != 11:
        print(
            f"\n{YELLOW}Warning: Python 3.11.x is recommended, but found {version}{NC}"
        )
        print(
            f"{YELLOW}Some dependencies may not work correctly with this version{NC}\n"
        )
    else:
        print(f"{GREEN}Python version OK{NC}\n")

    # 3. Create virtual environments and install packages
    base_path = Path(__file__).parent
    venv_configs = get_venv_configs(cuda_version)
    for config in venv_configs:
        venv_name = config["name"]
        packages = [pkg for pkg in config["packages"] if pkg is not None]

        print(f"\n--- Setting up {venv_name} ---")

        # Create virtual environment
        venv_path = base_path / venv_name
        print(f"Creating virtual environment: {venv_name}")
        try:
            subprocess.run(
                ["python3", "-m", "venv", venv_name], check=True, cwd=base_path
            )
            print(f"{GREEN}✓ {venv_name} created successfully{NC}")
        except subprocess.CalledProcessError as e:
            print(f"{RED}✗ Failed to create {venv_name}: {e}{NC}")
            sys.exit(1)

        # Install packages
        if packages:
            print(f"Installing packages: {', '.join(packages)}")
            pip_path = get_pip_path(venv_path)
            try:
                subprocess.run(
                    [str(pip_path), "install"] + packages, check=True, cwd=base_path
                )
                print(f"{GREEN}✓ Packages installed successfully{NC}")
            except subprocess.CalledProcessError as e:
                print(f"{RED}✗ Failed to install packages: {e}{NC}")
                sys.exit(1)


if __name__ == "__main__":
    main()
