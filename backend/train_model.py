"""
train_model.py — ShadowMap Model Training Script (CPU-Optimized Edition)

Standalone script to generate data, engineer features, train the XGBoost
model suite with GridSearchCV, and save all artifacts.

Architecture: Multi-threaded, hardware-agnostic inference
optimized for AMD EPYC server-class CPUs.
"""

import os
import sys
import json
import multiprocessing

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import generate_synthetic_data, generate_geojson
from model import train_model

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts")


def main():
    """Run the full training pipeline with GridSearchCV hyperparameter tuning."""
    print("=" * 60)
    print("ShadowMap — Model Training Pipeline (CPU-Optimized)")
    print("=" * 60)

    # --- AMD EPYC OPTIMIZED CPU MULTI-THREAD SECTION ---
    cpu_count = multiprocessing.cpu_count()
    print(f"\n[SYSTEM] CPU cores available: {cpu_count}")
    print(f"[SYSTEM] Parallelism: n_jobs=-1 (all {cpu_count} cores)")
    print(f"[SYSTEM] XGBoost tree_method: hist (CPU multi-threaded)")
    print(f"[SYSTEM] Architecture: Hardware-agnostic CPU inference")
    # --- END AMD EPYC OPTIMIZED CPU MULTI-THREAD SECTION ---

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

    print(f"\n[STEP 3] Training XGBoost model suite (CPU, n_jobs=-1)...")
    print(f"  Data shape: {df.shape}")
    print(f"  LST range: {df['lst'].min():.1f}°C – {df['lst'].max():.1f}°C")
    print(f"  Mean LST: {df['lst'].mean():.1f}°C")

    results = train_model(df)

    # Verify all artifacts are saved
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Save feature importance separately
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.json")
    if not os.path.exists(fi_path):
        with open(fi_path, "w") as f:
            json.dump(results["feature_importance"], f, indent=2)

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    print(f"\nModel Performance:")
    print(f"  RMSE:           {results['metrics']['rmse']:.4f}°C")
    print(f"  MAE:            {results['metrics']['mae']:.4f}°C")
    print(f"  R²:             {results['metrics']['r2']:.4f}")
    print(f"  Spatial CV R²:  {results['metrics']['spatial_cv_r2']:.4f}")
    print(f"  Architecture:   CPU multi-threaded (n_jobs=-1)")
    print(f"  Tree Method:    hist")
    print(f"\nTop Features:")
    importances = results["metrics"]["feature_importance"]
    for fname, pct in sorted(importances.items(), key=lambda x: -x[1])[:5]:
        print(f"  {fname:35s}: {pct:.2f}%")
    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")
    print(f"  - uhi_model.json          (mean predictor)")
    print(f"  - uhi_model_lower.json    (P10 quantile)")
    print(f"  - uhi_model_upper.json    (P90 quantile)")
    print(f"  - scaler.pkl")
    print(f"  - metrics.json")
    print(f"  - feature_importance.json")


if __name__ == "__main__":
    main()
