"""
install_dependencies.py — ShadowMap Dependency Installer

Checks each required package and installs only what is missing.
Does NOT reinstall existing packages.

Architecture: Multi-threaded, hardware-agnostic inference
optimized for AMD EPYC server-class CPUs.
"""

import importlib
import subprocess
import sys


# Mapping of pip package names to their import module names
REQUIRED_PACKAGES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "xgboost": "xgboost",
    "scikit-learn": "sklearn",
    "pandas": "pandas",
    "numpy": "numpy",
    "joblib": "joblib",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "onnxmltools": "onnxmltools",
    "skl2onnx": "skl2onnx",
}


def check_and_install():
    """Check each required package and install if missing."""
    print("=" * 60)
    print("ShadowMap — Dependency Installer (CPU-Only)")
    print("=" * 60)
    print()

    installed_count = 0
    newly_installed_count = 0
    failed_count = 0

    for pip_name, import_name in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {pip_name:20s} — Already installed (v{version})")
            installed_count += 1
        except ImportError:
            print(f"  [--] {pip_name:20s} — Not found. Installing...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pip_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"  [OK] {pip_name:20s} — Successfully installed")
                newly_installed_count += 1
            except subprocess.CalledProcessError:
                print(f"  [!!] {pip_name:20s} — FAILED to install")
                failed_count += 1

    print()
    print("=" * 60)
    print("Installation Summary")
    print("=" * 60)
    print(f"  Already installed:  {installed_count}")
    print(f"  Newly installed:    {newly_installed_count}")
    print(f"  Failed:             {failed_count}")
    print()

    if failed_count > 0:
        print("  [WARNING] Some packages failed to install.")
        print("  Try running manually: pip install -r requirements.txt")
    else:
        print("  All dependencies are ready!")
        print()
        print("  Next steps:")
        print("    python train_model.py")
        print("    python export_onnx.py")
        print("    python benchmark.py")
        print("    uvicorn main:app --reload --port 8000")

    print()


if __name__ == "__main__":
    check_and_install()
