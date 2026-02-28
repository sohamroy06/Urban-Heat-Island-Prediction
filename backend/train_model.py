"""
train_model.py — ShadowMap Model Training Script (AMD GPU-Ready Edition)

Standalone script to generate data, engineer features, train the XGBoost
model suite, and save all artifacts. Run this before starting the API server.

Compatible with: CPU, NVIDIA CUDA, AMD ROCm
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import generate_synthetic_data, generate_geojson
from model import train_model, XGB_DEVICE


def main():
    """Run the full training pipeline."""
    print("=" * 60)
    print("ShadowMap — Model Training Pipeline (AMD GPU-Ready)")
    print("=" * 60)

    # --- AMD GPU READY SECTION --- Device Info ---
    print(f"\n[DEVICE] Compute device: {XGB_DEVICE}")
    print(f"[DEVICE] XGBoost tree_method: hist (hardware-agnostic)")
    if XGB_DEVICE == "cuda":
        print("[DEVICE] GPU acceleration active (CUDA / ROCm)")
    else:
        print("[DEVICE] Running on CPU (GPU not available)")
    # --- END AMD GPU READY SECTION ---

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "sample_data.csv")
    geojson_path = os.path.join(base_dir, "delhi_blocks.geojson")

    if os.path.exists(csv_path):
        print(f"\n[STEP 1] Loading existing data from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} blocks.")
    else:
        print(f"\n[STEP 1] Generating synthetic Delhi data...")
        df = generate_synthetic_data(n_blocks=300)
        df.to_csv(csv_path, index=False)
        print(f"  Generated {len(df)} blocks, saved to {csv_path}")

    if not os.path.exists(geojson_path):
        print(f"\n[STEP 2] Generating GeoJSON...")
        generate_geojson(df, geojson_path)
        print(f"  Saved GeoJSON to {geojson_path}")
    else:
        print(f"\n[STEP 2] GeoJSON already exists at {geojson_path}")

    print(f"\n[STEP 3] Training XGBoost model suite...")
    print(f"  Data shape: {df.shape}")
    print(f"  LST range: {df['lst'].min():.1f}°C – {df['lst'].max():.1f}°C")
    print(f"  Mean LST: {df['lst'].mean():.1f}°C")

    results = train_model(df)

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    print(f"\nModel Performance:")
    print(f"  RMSE:           {results['metrics']['rmse']:.4f}°C")
    print(f"  MAE:            {results['metrics']['mae']:.4f}°C")
    print(f"  R²:             {results['metrics']['r2']:.4f}")
    print(f"  Spatial CV R²:  {results['metrics']['spatial_cv_r2']:.4f}")
    print(f"  Device:         {results['metrics']['device']}")
    print(f"  Tree Method:    {results['metrics']['tree_method']}")
    print(f"\nTop Features:")
    importances = results["metrics"]["feature_importance"]
    for fname, pct in sorted(importances.items(), key=lambda x: -x[1])[:5]:
        print(f"  {fname:35s}: {pct:.2f}%")
    print(f"\nArtifacts saved to: {os.path.join(base_dir, 'model_artifacts')}")
    print(f"  - uhi_model.json          (mean predictor)")
    print(f"  - uhi_model_lower.json    (P10 quantile)")
    print(f"  - uhi_model_upper.json    (P90 quantile)")
    print(f"  - scaler.pkl")
    print(f"  - metrics.json")
    print(f"  - feature_importance.json")


if __name__ == "__main__":
    main()
