"""
train_model.py — ShadowMap Model Training Script

Standalone script to generate data, engineer features, train the GBR model
suite, and save all artifacts. Run this before starting the API server.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import generate_synthetic_data, generate_geojson
from model import train_model


def main():
    """Run the full training pipeline."""
    print("=" * 60)
    print("ShadowMap — Model Training Pipeline")
    print("=" * 60)

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

    print(f"\n[STEP 3] Training model suite...")
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
    print(f"\nTop Features:")
    importances = results["metrics"]["feature_importance"]
    for fname, pct in sorted(importances.items(), key=lambda x: -x[1])[:5]:
        print(f"  {fname:35s}: {pct:.2f}%")
    print(f"\nArtifacts saved to: {os.path.join(base_dir, 'model_artifacts')}")
    print(f"  - uhi_model.pkl")
    print(f"  - uhi_model_lower.pkl")
    print(f"  - uhi_model_upper.pkl")
    print(f"  - scaler.pkl")
    print(f"  - metrics.json")
    print(f"  - feature_importance.json")


if __name__ == "__main__":
    main()
