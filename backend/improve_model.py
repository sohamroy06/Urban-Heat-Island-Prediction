import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from feature_engineering import compute_interaction_features

df = pd.read_csv("sample_data.csv")
df = compute_interaction_features(df)

# Full feature set (current)
FULL_FEATURES = [
    "building_density", "green_cover", "road_density", "avg_building_height",
    "distance_to_water", "impervious_surface_fraction",
    "heat_stress_index", "cooling_potential",
]

# Reduced set (dropping weak/noisy avg_building_height)
REDUCED_FEATURES = [
    "building_density", "green_cover", "road_density",
    "distance_to_water", "impervious_surface_fraction",
    "heat_stress_index", "cooling_potential",
]

y = df["lst"].values
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

def evaluate(features, model, name):
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="r2")
    print(f"{name:45s}: R2 = {scores.mean():.4f} (+/- {scores.std():.4f})")

print("=== Repeated 5-fold CV (5x10=50 runs), more stable than single split ===\n")

# 1. Current XGBoost config, full features
xgb_current = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.1,
                                 min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                 reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
evaluate(FULL_FEATURES, xgb_current, "XGBoost (current config, full features)")

# 2. Simplified XGBoost, full features
xgb_simple = xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                                min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                                reg_alpha=1.0, reg_lambda=3.0, random_state=42, verbosity=0)
evaluate(FULL_FEATURES, xgb_simple, "XGBoost (simplified, full features)")

# 3. Simplified XGBoost, reduced features
evaluate(REDUCED_FEATURES, xgb_simple, "XGBoost (simplified, reduced features)")

# 4. Ridge regression, reduced features
ridge = Ridge(alpha=5.0, random_state=42)
evaluate(REDUCED_FEATURES, ridge, "Ridge Regression (reduced features)")

# 5. Random Forest, reduced features
rf = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42)
evaluate(REDUCED_FEATURES, rf, "Random Forest (reduced features)")

# 6. Random Forest, full features
evaluate(FULL_FEATURES, rf, "Random Forest (full features)")