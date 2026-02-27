"""
model.py — ShadowMap ML Model

GradientBoostingRegressor with spatial cross-validation, quantile regression
for confidence intervals, and what-if simulation logic.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

from feature_engineering import (
    ALL_FEATURES,
    FEATURE_COLUMNS,
    compute_interaction_features,
    prepare_features,
    load_scaler,
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts")

BLOCK_AREA_KM2 = 0.25  # 500m x 500m block
AVG_BUILDING_FOOTPRINT_KM2 = 0.0002  # ~200 sq m per building
AVG_TREE_COVER_KM2 = 0.00005  # ~50 sq m canopy per tree


def spatial_train_test_split(df: pd.DataFrame, lon_threshold: float = 77.15):
    """
    Split data spatially: western Delhi for training, eastern for testing.

    This prevents data leakage between geographically adjacent blocks.

    Args:
        df: DataFrame with 'lon' column.
        lon_threshold: Longitude boundary for the split.

    Returns:
        Tuple of (train_indices, test_indices).
    """
    train_mask = df["lon"] <= lon_threshold
    test_mask = df["lon"] > lon_threshold

    train_idx = df[train_mask].index.tolist()
    test_idx = df[test_mask].index.tolist()

    print(f"[INFO] Spatial split: {len(train_idx)} train (west), {len(test_idx)} test (east)")

    if len(test_idx) < 10:
        print("[WARN] Very few test samples. Adjusting threshold...")
        median_lon = df["lon"].median()
        train_mask = df["lon"] <= median_lon
        test_mask = df["lon"] > median_lon
        train_idx = df[train_mask].index.tolist()
        test_idx = df[test_mask].index.tolist()
        print(f"[INFO] Adjusted split: {len(train_idx)} train, {len(test_idx)} test")

    return train_idx, test_idx


def spatial_cross_validation(X: np.ndarray, y: np.ndarray, df: pd.DataFrame,
                              n_splits: int = 5) -> float:
    """
    Perform spatial cross-validation by sorting blocks by longitude.

    Args:
        X: Feature matrix.
        y: Target values.
        df: DataFrame with 'lon' column for spatial ordering.
        n_splits: Number of CV folds.

    Returns:
        Mean cross-validation R² score.
    """
    lon_order = df["lon"].argsort().values
    X_sorted = X[lon_order]
    y_sorted = y[lon_order]

    kf = KFold(n_splits=n_splits, shuffle=False)
    scores = []

    for train_idx, test_idx in kf.split(X_sorted):
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_split=10, random_state=42
        )
        model.fit(X_sorted[train_idx], y_sorted[train_idx])
        score = model.score(X_sorted[test_idx], y_sorted[test_idx])
        scores.append(score)

    mean_score = float(np.mean(scores))
    print(f"[INFO] Spatial CV R² scores: {[f'{s:.4f}' for s in scores]}")
    print(f"[INFO] Mean Spatial CV R²: {mean_score:.4f}")
    return mean_score


def train_model(df: pd.DataFrame):
    """
    Train the full model suite: mean predictor + quantile regressors.

    Steps:
    1. Prepare features with interaction terms
    2. Spatial train/test split
    3. GridSearchCV for hyperparameter tuning
    4. Train mean model, lower quantile (0.1), upper quantile (0.9)
    5. Compute metrics and feature importances
    6. Save all artifacts

    Args:
        df: Raw DataFrame with features and 'lst' target.

    Returns:
        Dictionary containing models, metrics, and feature importances.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df_processed, scaler, X, y = prepare_features(df, fit=True)

    train_idx, test_idx = spatial_train_test_split(df)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("\n[INFO] Running GridSearchCV for hyperparameter tuning...")
    param_grid = {
        "n_estimators": [150, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "min_samples_split": [5, 10],
    }

    base_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid,
        cv=3, scoring="r2", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"[INFO] Best params: {best_params}")

    mean_model = GradientBoostingRegressor(**best_params, random_state=42)
    mean_model.fit(X_train, y_train)

    print("[INFO] Training lower quantile model (alpha=0.1)...")
    lower_model = GradientBoostingRegressor(
        **{k: v for k, v in best_params.items()},
        loss="quantile", alpha=0.1, random_state=42
    )
    lower_model.fit(X_train, y_train)

    print("[INFO] Training upper quantile model (alpha=0.9)...")
    upper_model = GradientBoostingRegressor(
        **{k: v for k, v in best_params.items()},
        loss="quantile", alpha=0.9, random_state=42
    )
    upper_model.fit(X_train, y_train)

    y_pred = mean_model.predict(X_test)
    y_pred_lower = lower_model.predict(X_test)
    y_pred_upper = upper_model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"\n[INFO] Test Metrics:")
    print(f"  RMSE: {rmse:.4f}°C")
    print(f"  MAE:  {mae:.4f}°C")
    print(f"  R²:   {r2:.4f}")

    spatial_cv_score = spatial_cross_validation(X, y, df)

    importances = mean_model.feature_importances_
    feature_importance = {}
    total_imp = float(importances.sum())
    for fname, imp in zip(ALL_FEATURES, importances):
        feature_importance[fname] = round(float(imp) / total_imp * 100, 2)

    print(f"\n[INFO] Feature Importances:")
    for fname, pct in sorted(feature_importance.items(), key=lambda x: -x[1]):
        print(f"  {fname:35s}: {pct:.2f}%")

    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "spatial_cv_r2": round(spatial_cv_score, 4),
        "best_params": best_params,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_features": len(ALL_FEATURES),
        "feature_importance": feature_importance,
    }

    joblib.dump(mean_model, os.path.join(ARTIFACTS_DIR, "uhi_model.pkl"))
    joblib.dump(lower_model, os.path.join(ARTIFACTS_DIR, "uhi_model_lower.pkl"))
    joblib.dump(upper_model, os.path.join(ARTIFACTS_DIR, "uhi_model_upper.pkl"))

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "feature_importance.json"), "w") as f:
        json.dump(feature_importance, f, indent=2)

    print(f"\n[INFO] All artifacts saved to {ARTIFACTS_DIR}")

    return {
        "mean_model": mean_model,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_importance": feature_importance,
    }


def load_models():
    """
    Load all trained model artifacts from disk.

    Returns:
        Dictionary containing models, scaler, metrics, and feature importances.
    """
    mean_model = joblib.load(os.path.join(ARTIFACTS_DIR, "uhi_model.pkl"))
    lower_model = joblib.load(os.path.join(ARTIFACTS_DIR, "uhi_model_lower.pkl"))
    upper_model = joblib.load(os.path.join(ARTIFACTS_DIR, "uhi_model_upper.pkl"))
    scaler = load_scaler()

    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    importance_path = os.path.join(ARTIFACTS_DIR, "feature_importance.json")
    with open(importance_path, "r") as f:
        feature_importance = json.load(f)

    return {
        "mean_model": mean_model,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_importance": feature_importance,
    }


def predict_block(features_dict: dict, models: dict) -> dict:
    """
    Predict LST for a single block with confidence intervals.

    Args:
        features_dict: Dictionary of raw feature values for the block.
        models: Dictionary from load_models() or train_model().

    Returns:
        Dictionary with predicted_lst, ci_lower, ci_upper, ci_width.
    """
    df = pd.DataFrame([features_dict])
    df = compute_interaction_features(df)

    X_raw = df[ALL_FEATURES].values
    X_scaled = models["scaler"].transform(X_raw)

    pred_mean = float(models["mean_model"].predict(X_scaled)[0])
    pred_lower = float(models["lower_model"].predict(X_scaled)[0])
    pred_upper = float(models["upper_model"].predict(X_scaled)[0])

    if pred_lower > pred_mean:
        pred_lower = pred_mean - abs(pred_upper - pred_mean)
    if pred_upper < pred_mean:
        pred_upper = pred_mean + abs(pred_mean - pred_lower)

    return {
        "predicted_lst": round(pred_mean, 1),
        "ci_lower": round(pred_lower, 1),
        "ci_upper": round(pred_upper, 1),
        "ci_width": round(pred_upper - pred_lower, 1),
    }


def compute_feature_contributions(features_dict: dict, models: dict, city_mean_lst: float) -> list:
    """
    Compute approximate feature contributions for a block (SHAP-style).

    Uses a simple perturbation-based approach: for each feature, predict
    with and without it (set to mean) and compute the difference.

    Args:
        features_dict: Raw feature values for the block.
        models: Dictionary from load_models().
        city_mean_lst: City-wide mean LST for reference.

    Returns:
        List of dicts with feature name, contribution, and direction.
    """
    baseline = predict_block(features_dict, models)["predicted_lst"]

    contributions = []
    for feature in ALL_FEATURES:
        perturbed = features_dict.copy()

        if feature == "heat_stress_index" or feature == "cooling_potential":
            continue

        if feature in perturbed:
            original_val = perturbed[feature]
            mean_val = {
                "building_density": 0.35,
                "green_cover": 0.25,
                "road_density": 10.0,
                "avg_building_height": 12.0,
                "distance_to_water": 8.0,
                "impervious_surface_fraction": 0.45,
            }.get(feature, original_val)

            perturbed[feature] = mean_val
            perturbed_pred = predict_block(perturbed, models)["predicted_lst"]
            contribution = baseline - perturbed_pred

            contributions.append({
                "feature": feature,
                "value": round(original_val, 4),
                "contribution": round(contribution, 2),
                "direction": "heating" if contribution > 0 else "cooling",
            })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions


def simulate_whatif(
    block_features: dict,
    delta_buildings: int,
    delta_trees: int,
    delta_albedo: float,
    models: dict,
) -> dict:
    """
    Simulate a what-if scenario for a block.

    Takes the original feature vector, modifies it based on user-specified
    interventions, and returns the predicted impact.

    Args:
        block_features: Dictionary of original raw feature values.
        delta_buildings: Number of buildings to add (0-50).
        delta_trees: Number of trees to add (0-500).
        delta_albedo: Percentage increase in roof albedo (0-50).
        models: Dictionary from load_models().

    Returns:
        Dictionary with baseline_lst, predicted_lst, delta_temp,
        confidence_interval, and narrative_explanation.
    """
    baseline = predict_block(block_features, models)

    modified = block_features.copy()

    original_bd = modified["building_density"]
    original_gc = modified["green_cover"]
    original_isf = modified["impervious_surface_fraction"]

    if delta_buildings > 0:
        added_building_area = delta_buildings * AVG_BUILDING_FOOTPRINT_KM2
        density_increase = added_building_area / BLOCK_AREA_KM2
        modified["building_density"] = min(0.95, original_bd + density_increase)
        modified["impervious_surface_fraction"] = min(
            0.98, original_isf + density_increase * 0.8
        )

    if delta_trees > 0:
        added_canopy = delta_trees * AVG_TREE_COVER_KM2
        gc_increase = added_canopy / BLOCK_AREA_KM2
        modified["green_cover"] = min(0.90, original_gc + gc_increase)
        modified["impervious_surface_fraction"] = max(
            0.05, modified["impervious_surface_fraction"] - gc_increase * 0.5
        )

    if delta_albedo > 0:
        albedo_factor = delta_albedo / 100.0
        modified["building_density"] = max(
            0.01, modified["building_density"] * (1 - albedo_factor * 0.15)
        )
        modified["impervious_surface_fraction"] = max(
            0.05, modified["impervious_surface_fraction"] * (1 - albedo_factor * 0.10)
        )

    new_prediction = predict_block(modified, models)

    delta_temp = round(new_prediction["predicted_lst"] - baseline["predicted_lst"], 1)

    parts = []
    if delta_trees > 0:
        new_gc_pct = round(modified["green_cover"] * 100, 1)
        old_gc_pct = round(original_gc * 100, 1)
        parts.append(
            f"Adding {delta_trees} trees increases green cover from "
            f"{old_gc_pct}% to {new_gc_pct}%"
        )
    if delta_buildings > 0:
        new_bd_pct = round(modified["building_density"] * 100, 1)
        old_bd_pct = round(original_bd * 100, 1)
        parts.append(
            f"Adding {delta_buildings} buildings increases building density from "
            f"{old_bd_pct}% to {new_bd_pct}%"
        )
    if delta_albedo > 0:
        parts.append(
            f"Increasing roof albedo by {delta_albedo}% reduces effective "
            f"heat absorption"
        )

    if delta_temp < 0:
        direction = "reduce"
        temp_str = f"{abs(delta_temp)}°C"
    elif delta_temp > 0:
        direction = "increase"
        temp_str = f"{abs(delta_temp)}°C"
    else:
        direction = "not significantly change"
        temp_str = "0.0°C"

    intervention_str = ", ".join(parts) if parts else "No changes applied"
    ci_str = (
        f"(80% CI: {new_prediction['ci_lower']}°C – "
        f"{new_prediction['ci_upper']}°C)"
    )

    if delta_temp != 0:
        narrative = (
            f"{intervention_str}. This is predicted to {direction} surface "
            f"temperature by {temp_str} {ci_str}, "
            f"from {baseline['predicted_lst']}°C to "
            f"{new_prediction['predicted_lst']}°C."
        )
    else:
        narrative = (
            f"{intervention_str}. The predicted surface temperature remains "
            f"approximately {baseline['predicted_lst']}°C {ci_str}."
        )

    return {
        "baseline_lst": baseline["predicted_lst"],
        "predicted_lst": new_prediction["predicted_lst"],
        "delta_temp": delta_temp,
        "confidence_interval": {
            "lower": new_prediction["ci_lower"],
            "upper": new_prediction["ci_upper"],
        },
        "baseline_ci": {
            "lower": baseline["ci_lower"],
            "upper": baseline["ci_upper"],
        },
        "modified_features": {
            "building_density": round(modified["building_density"], 4),
            "green_cover": round(modified["green_cover"], 4),
            "impervious_surface_fraction": round(modified["impervious_surface_fraction"], 4),
        },
        "narrative_explanation": narrative,
    }


def generate_intervention_curve(
    block_features: dict,
    intervention_type: str,
    models: dict,
    steps: int = 10,
) -> list:
    """
    Generate an intervention curve: how temperature changes as one
    intervention variable goes from 0 to its maximum.

    Args:
        block_features: Original block features.
        intervention_type: One of 'buildings', 'trees', 'albedo'.
        models: Loaded models dict.
        steps: Number of points on the curve.

    Returns:
        List of {value, predicted_lst, delta_temp} dicts.
    """
    max_vals = {"buildings": 50, "trees": 500, "albedo": 50}
    max_val = max_vals.get(intervention_type, 50)

    baseline = predict_block(block_features, models)["predicted_lst"]
    curve = []

    for i in range(steps + 1):
        val = int(max_val * i / steps)
        kwargs = {"delta_buildings": 0, "delta_trees": 0, "delta_albedo": 0.0}

        if intervention_type == "buildings":
            kwargs["delta_buildings"] = val
        elif intervention_type == "trees":
            kwargs["delta_trees"] = val
        elif intervention_type == "albedo":
            kwargs["delta_albedo"] = float(val)

        result = simulate_whatif(block_features, models=models, **kwargs)
        curve.append({
            "value": val,
            "predicted_lst": result["predicted_lst"],
            "delta_temp": round(result["predicted_lst"] - baseline, 1),
        })

    return curve
