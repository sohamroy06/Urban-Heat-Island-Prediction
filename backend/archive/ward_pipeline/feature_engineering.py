"""
feature_engineering.py — ShadowMap Feature Engineering

Builds the block-level feature matrix with interaction features and scaling.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "building_density",
    "green_cover",
    "road_density",
    "avg_building_height",
    "distance_to_water",
    "impervious_surface_fraction",
]

INTERACTION_FEATURES = [
    "heat_stress_index",
    "cooling_potential",
]

ALL_FEATURES = FEATURE_COLUMNS + INTERACTION_FEATURES

TARGET_COLUMN = "lst"

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts")


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute interaction features from base features.

    heat_stress_index = building_density * (1 - green_cover) * road_density
    cooling_potential = green_cover * distance_to_water_normalized

    Args:
        df: DataFrame with base feature columns.

    Returns:
        DataFrame with added interaction feature columns.
    """
    df = df.copy()

    df["heat_stress_index"] = (
        df["building_density"]
        * (1.0 - df["green_cover"])
        * df["road_density"]
    )

    dtw = df["distance_to_water"]
    dtw_max = dtw.max()
    if dtw_max > 0:
        dtw_normalized = 1.0 - (dtw / dtw_max)
    else:
        dtw_normalized = pd.Series(np.ones(len(df)), index=df.index)

    df["cooling_potential"] = df["green_cover"] * dtw_normalized

    return df


def fit_scaler(df: pd.DataFrame, save: bool = True) -> StandardScaler:
    """
    Fit a StandardScaler on the feature columns and optionally save it.

    Args:
        df: DataFrame containing all feature columns (including interactions).
        save: Whether to save the scaler to model_artifacts.

    Returns:
        Fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(df[ALL_FEATURES])

    if save:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Scaler saved to {scaler_path}")

    return scaler


def load_scaler() -> StandardScaler:
    """
    Load a previously fitted StandardScaler from model_artifacts.

    Returns:
        Fitted StandardScaler instance.

    Raises:
        FileNotFoundError: If no saved scaler is found.
    """
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"No scaler found at {scaler_path}. Run training first."
        )
    return joblib.load(scaler_path)


def scale_features(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply StandardScaler transformation to feature columns.

    Args:
        df: DataFrame with feature columns.
        scaler: Fitted StandardScaler.

    Returns:
        DataFrame with scaled feature columns (other columns preserved).
    """
    df = df.copy()
    scaled_values = scaler.transform(df[ALL_FEATURES])
    df[ALL_FEATURES] = scaled_values
    return df


def prepare_features(
    df: pd.DataFrame,
    fit: bool = False,
    scaler: StandardScaler = None,
) -> tuple:
    """
    Full feature preparation pipeline.

    1. Compute interaction features
    2. Fit or load scaler
    3. Scale all features

    Args:
        df: Raw DataFrame with base features and target.
        fit: Whether to fit a new scaler (True for training, False for inference).
        scaler: Pre-fitted scaler to use (if fit=False and scaler is provided).

    Returns:
        Tuple of (processed_df, scaler, feature_matrix, target_series).
    """
    df = compute_interaction_features(df)

    if fit:
        scaler = fit_scaler(df, save=True)
    elif scaler is None:
        scaler = load_scaler()

    df_scaled = scale_features(df, scaler)

    X = df_scaled[ALL_FEATURES].values
    y = df[TARGET_COLUMN].values if TARGET_COLUMN in df.columns else None

    return df, scaler, X, y


def get_feature_names() -> list:
    """Return the list of all feature names used by the model."""
    return ALL_FEATURES.copy()


def get_raw_feature_vector(row: dict) -> np.ndarray:
    """
    Extract a raw (unscaled) feature vector from a single row dictionary.

    Args:
        row: Dictionary containing feature values.

    Returns:
        numpy array of feature values in the correct order.
    """
    df = pd.DataFrame([row])
    df = compute_interaction_features(df)
    return df[ALL_FEATURES].values[0]


def get_scaled_feature_vector(row: dict, scaler: StandardScaler) -> np.ndarray:
    """
    Extract a scaled feature vector from a single row dictionary.

    Args:
        row: Dictionary containing feature values.
        scaler: Fitted StandardScaler.

    Returns:
        numpy array of scaled feature values.
    """
    raw = get_raw_feature_vector(row)
    return scaler.transform(raw.reshape(1, -1))[0]


if __name__ == "__main__":
    print("=" * 60)
    print("ShadowMap Feature Engineering — Test Run")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data.csv")
    if not os.path.exists(data_path):
        print("[ERROR] sample_data.csv not found. Run data_pipeline.py first.")
    else:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows from sample_data.csv")

        df_processed, scaler, X, y = prepare_features(df, fit=True)

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target range: {y.min():.1f} – {y.max():.1f}")
        print(f"\nFeature statistics (raw):")
        for feat in ALL_FEATURES:
            vals = df_processed[feat]
            print(f"  {feat:35s}: mean={vals.mean():.4f}, std={vals.std():.4f}")
        print(f"\nScaled feature matrix sample (first 3 rows):")
        print(X[:3])
