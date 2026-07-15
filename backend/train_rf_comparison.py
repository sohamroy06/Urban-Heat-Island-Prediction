import pandas as pd
import numpy as np
import json
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from feature_engineering import compute_interaction_features, ALL_FEATURES

df = pd.read_csv("sample_data.csv")
df = compute_interaction_features(df)
X = df[ALL_FEATURES].values
y = df["lst"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=3, random_state=42)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="r2")

rf.fit(X_scaled, y)

results = {
    "model_type": "RandomForestRegressor",
    "purpose": "Comparison/reference model, not used in production API",
    "repeated_cv_r2_mean": round(float(scores.mean()), 4),
    "repeated_cv_r2_std": round(float(scores.std()), 4),
    "feature_importances": dict(zip(ALL_FEATURES, [round(float(f), 4) for f in rf.feature_importances_])),
}

with open("model_artifacts/rf_comparison_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nSaved to model_artifacts/rf_comparison_metrics.json")