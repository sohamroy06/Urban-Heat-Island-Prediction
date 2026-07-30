"""
benchmark.py — ShadowMap Inference Benchmarking (CPU-Optimized Edition)

Benchmarks single and batch inference latency for XGBoost and ONNX Runtime.
Reports CPU core count and latency comparisons.

Architecture: Multi-threaded, hardware-agnostic inference
optimized for AMD EPYC server-class CPUs.
"""

import os
import sys
import time
import platform
import multiprocessing

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts")


# --- AMD EPYC OPTIMIZED CPU MULTI-THREAD SECTION ---
def print_system_info():
    """
    Print comprehensive system and CPU information.
    """
    cpu_count = multiprocessing.cpu_count()

    print("=" * 60)
    print("SYSTEM & CPU INFORMATION")
    print("=" * 60)
    print(f"  OS:             {platform.system()} {platform.release()}")
    print(f"  Platform:       {platform.platform()}")
    print(f"  Processor:      {platform.processor()}")
    print(f"  Architecture:   {platform.machine()}")
    print(f"  Python:         {platform.python_version()}")
    print(f"  CPU Core Count: {cpu_count}")
    print(f"  Parallelism:    n_jobs=-1 (all {cpu_count} cores)")

    try:
        import xgboost as xgb
        print(f"  XGBoost:        {xgb.__version__}")
    except ImportError:
        print(f"  XGBoost:        not installed")

    try:
        import onnxruntime as ort
        print(f"  ONNX Runtime:   {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"  ORT Providers:  {providers}")
    except ImportError:
        print(f"  ONNX Runtime:   not installed")

    print()
# --- END AMD EPYC OPTIMIZED CPU MULTI-THREAD SECTION ---


def benchmark_xgboost_single(models, X_single, n_iterations=100):
    """
    Benchmark single-block XGBoost inference.

    Args:
        models: Loaded model dict.
        X_single: Single sample feature array (1, n_features).
        n_iterations: Number of iterations to average.

    Returns:
        Average latency in milliseconds.
    """
    # Warm-up
    for _ in range(10):
        models["mean_model"].predict(X_single)

    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        models["mean_model"].predict(X_single)
        models["lower_model"].predict(X_single)
        models["upper_model"].predict(X_single)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def benchmark_xgboost_batch(models, X_batch, n_iterations=50):
    """
    Benchmark batch XGBoost inference (300 blocks).

    Args:
        models: Loaded model dict.
        X_batch: Batch feature array (n_blocks, n_features).
        n_iterations: Number of iterations to average.

    Returns:
        Average latency in milliseconds.
    """
    # Warm-up
    for _ in range(5):
        models["mean_model"].predict(X_batch)

    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        models["mean_model"].predict(X_batch)
        models["lower_model"].predict(X_batch)
        models["upper_model"].predict(X_batch)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def benchmark_onnx_single(session, input_name, X_single, n_iterations=100):
    """
    Benchmark single-block ONNX Runtime inference (CPU).

    Args:
        session: ONNX Runtime InferenceSession.
        input_name: Name of the input tensor.
        X_single: Single sample (1, n_features) float32 array.
        n_iterations: Number of iterations.

    Returns:
        Average latency in milliseconds.
    """
    X_f32 = X_single.astype(np.float32)

    # Warm-up
    for _ in range(10):
        session.run(None, {input_name: X_f32})

    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        session.run(None, {input_name: X_f32})
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def benchmark_onnx_batch(session, input_name, X_batch, n_iterations=50):
    """
    Benchmark batch ONNX Runtime inference — 300 blocks (CPU).

    Args:
        session: ONNX Runtime InferenceSession.
        input_name: Name of the input tensor.
        X_batch: Batch array (n_blocks, n_features) float32.
        n_iterations: Number of iterations.

    Returns:
        Average latency in milliseconds.
    """
    X_f32 = X_batch.astype(np.float32)

    # Warm-up
    for _ in range(5):
        session.run(None, {input_name: X_f32})

    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        session.run(None, {input_name: X_f32})
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def run_benchmarks():
    """Run all inference benchmarks and print results."""
    from model import load_models
    from feature_engineering import ALL_FEATURES, prepare_features

    print_system_info()

    # ---- Load data and models ----
    print("=" * 60)
    print("LOADING MODELS & DATA")
    print("=" * 60)

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data.csv")
    if not os.path.exists(csv_path):
        print("[ERROR] sample_data.csv not found. Run train_model.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    _, scaler, X_all, _ = prepare_features(df, fit=False)
    print(f"  Loaded {len(df)} blocks, {X_all.shape[1]} features")

    models = load_models()
    print(f"  Models loaded (CPU multi-threaded, n_jobs=-1)")

    X_single = X_all[:1]          # 1 block
    X_batch = X_all[:300]         # 300 blocks (or all if fewer)
    n_batch = len(X_batch)
    print(f"  Single sample shape: {X_single.shape}")
    print(f"  Batch sample shape:  {X_batch.shape}")

    # --- AMD EPYC OPTIMIZED CPU MULTI-THREAD SECTION ---
    cpu_count = multiprocessing.cpu_count()

    # ---- XGBoost Benchmarks ----
    print(f"\n{'=' * 60}")
    print(f"XGBOOST INFERENCE BENCHMARKS (CPU, {cpu_count} cores)")
    print(f"{'=' * 60}")

    xgb_single_mean, xgb_single_std = benchmark_xgboost_single(models, X_single)
    print(f"  Single inference (3 models):  {xgb_single_mean:.3f} +/- {xgb_single_std:.3f} ms")

    xgb_batch_mean, xgb_batch_std = benchmark_xgboost_batch(models, X_batch)
    print(f"  Batch inference ({n_batch} blocks, 3 models): {xgb_batch_mean:.3f} +/- {xgb_batch_std:.3f} ms")
    print(f"  Per-block latency (batch):    {xgb_batch_mean / n_batch:.4f} ms")

    # ---- ONNX Runtime Benchmarks (CPU only) ----
    onnx_path = os.path.join(ARTIFACTS_DIR, "uhi_model.onnx")
    ort_single_mean = ort_single_std = ort_batch_mean = ort_batch_std = None

    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort

            # CPU-only execution provider
            providers = ["CPUExecutionProvider"]

            session = ort.InferenceSession(onnx_path, providers=providers)
            input_name = session.get_inputs()[0].name

            print(f"\n{'=' * 60}")
            print(f"ONNX RUNTIME BENCHMARKS (CPU, {cpu_count} cores)")
            print(f"{'=' * 60}")

            ort_single_mean, ort_single_std = benchmark_onnx_single(
                session, input_name, X_single
            )
            print(f"  Single inference (mean model): {ort_single_mean:.3f} +/- {ort_single_std:.3f} ms")

            ort_batch_mean, ort_batch_std = benchmark_onnx_batch(
                session, input_name, X_batch
            )
            print(f"  Batch inference ({n_batch} blocks):   {ort_batch_mean:.3f} +/- {ort_batch_std:.3f} ms")
            print(f"  Per-block latency (batch):     {ort_batch_mean / n_batch:.4f} ms")

        except ImportError:
            print("\n[WARN] onnxruntime not installed — skipping ONNX benchmarks")
    else:
        print(f"\n[WARN] ONNX model not found at {onnx_path} — skipping ONNX benchmarks")
        print("       Run export_onnx.py first to generate the ONNX model.")

    # ---- Summary Table ----
    print(f"\n{'=' * 60}")
    print("LATENCY COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"  CPU Core Count: {cpu_count}")
    print(f"  Average Single Inference Latency (XGBoost): {xgb_single_mean:.3f} ms")
    print(f"  Average Batch  Inference Latency (XGBoost): {xgb_batch_mean:.3f} ms")

    if ort_single_mean is not None:
        speedup_single = xgb_single_mean / ort_single_mean if ort_single_mean > 0 else 0
        speedup_batch = xgb_batch_mean / ort_batch_mean if ort_batch_mean > 0 else 0
        print(f"  Average Single Inference Latency (ONNX):    {ort_single_mean:.3f} ms")
        print(f"  Average Batch  Inference Latency (ONNX):    {ort_batch_mean:.3f} ms")
        print(f"  ONNX vs XGBoost single speedup: {speedup_single:.2f}x")
        print(f"  ONNX vs XGBoost batch speedup:  {speedup_batch:.2f}x")
    # --- END AMD EPYC OPTIMIZED CPU MULTI-THREAD SECTION ---

    print(f"\n{'=' * 60}")
    print("Benchmarking Complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_benchmarks()
