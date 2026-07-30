"""
feature_engineering_v2.py - ShadowMap grid schema (SCHEMA_VERSION 2)

Replaces the ward-level 8-feature schema in feature_engineering.py.
That file is left intact so the ward pipeline stays reproducible.

Key differences from v1:
  - 4 spectral features instead of 6 urban + 2 interaction features
  - NO StandardScaler. XGBoost trees are scale-invariant; scaling was
    never functional and added a pickle dependency.
  - No interaction features. Tested depth 2-8; signal is near-additive.
  - Explicit feature bounds for extrapolation detection.
"""

import os
import json
import numpy as np
import pandas as pd

SCHEMA_VERSION = 2

FEATURE_COLUMNS = ["albedo", "ndbi", "mndwi", "bsi"]
INTERACTION_FEATURES = []
ALL_FEATURES = FEATURE_COLUMNS + INTERACTION_FEATURES
TARGET_COLUMN = "lst"

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts_grid")
MASTER_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_master.csv")

FEATURE_METADATA = {
    "albedo": {
        "name": "Surface albedo",
        "description": "Broadband shortwave reflectance, fraction of sunlight reflected.",
        "source": "Landsat 8/9 C2 L2 surface reflectance, Liang (2001) coefficients",
        "valid_range": [0.0, 1.0],
        "caveat": "In Delhi, correlates POSITIVELY with LST because dry bare soil is "
                  "both bright and hot. Not usable as a cooling intervention lever "
                  "(tested Steps 22-23, both cross-sectional and within-cell designs).",
    },
    "ndbi": {
        "name": "Normalized Difference Built-up Index",
        "description": "(SWIR1 - NIR) / (SWIR1 + NIR).",
        "source": "Landsat 8/9 C2 L2 surface reflectance",
        "valid_range": [-1.0, 1.0],
        "caveat": "DESPITE THE NAME, this does NOT measure built-up intensity in Delhi. "
                  "Correlation with OSM building density is -0.323 (negative). It tracks "
                  "SURFACE DRYNESS. Describe it as a dryness index, never as urbanisation.",
    },
    "mndwi": {
        "name": "Modified Normalized Difference Water Index",
        "description": "(Green - SWIR1) / (Green + SWIR1).",
        "source": "Landsat 8/9 C2 L2 surface reflectance",
        "valid_range": [-1.0, 1.0],
        "caveat": "Separates water and moist surfaces from built surfaces more reliably "
                  "than NDWI.",
    },
    "bsi": {
        "name": "Bare Soil Index",
        "description": "((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue)).",
        "source": "Landsat 8/9 C2 L2 surface reflectance",
        "valid_range": [-1.0, 1.0],
        "caveat": "Correlated with ndbi at r = 0.989. Both retained: paired-test p < 0.05 "
                  "at 5 km blocks, though the pair is partly redundant.",
    },
}

REMOVED_FEATURES = {
    "ndvi": "Removed Step 21. The stored ndvi came from the older pipeline WITHOUT "
            "per-pixel QA_PIXEL masking, while all other indices are QA-masked. Same "
            "year, same city, values differed 2.1x (0.128 vs 0.270). Removing it cut "
            "extrapolation flags on 2023 data from 79.4% to 8.1% and cost no accuracy "
            "(blocked-CV R2 0.7714 -> 0.7722).",
    "building_density": "No daytime predictive skill. Permutation importance -0.003. "
                        "URBAN-only feature sets scored -0.07 to -0.62 (worse than the mean).",
    "road_density": "Same as building_density; no skill once spectral indices present.",
    "population": "Blocked-CV importance 0.004 despite r = -0.29. Apparent gains in "
                  "earlier work came from leaky random-split CV.",
    "elevation": "Leave-one-out delta +0.011 at 5 km blocks (removing it HELPED).",
    "dist_to_water": "Built from an OSM layer containing 176 wastewater tanks, 567 ponds "
                     "and a fountain. Not significant once spectral indices present.",
    "pct_residential/pct_industrial_commercial/pct_green_farm": "Wilcoxon p = 0.003 in "
                     "favour of dropping them at 10 km blocks. Also carried a "
                     "cross-category polygon overlap bug (93 cells summed above 1.0).",
}


def load_feature_bounds():
    """Load p01/p99 training bounds from model_meta.json."""
    path = os.path.join(ARTIFACTS_DIR, "model_meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError("model_meta.json not found. Run step21_rebuild.py first.")
    with open(path) as f:
        return json.load(f)["feature_bounds"]


def validate_features(df, strict=True):
    """Check a DataFrame carries usable v2 features.

    Returns (ok, list_of_problems). strict=True also enforces valid_range.
    """
    problems = []
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        problems.append("missing columns: %s" % missing)
        return False, problems

    for c in FEATURE_COLUMNS:
        v = df[c].values.astype(float)
        n_null = int(np.sum(~np.isfinite(v)))
        if n_null:
            problems.append("%s: %d non-finite values" % (c, n_null))
        if strict:
            lo, hi = FEATURE_METADATA[c]["valid_range"]
            n_out = int(np.sum((v < lo) | (v > hi)))
            if n_out:
                problems.append("%s: %d values outside physical range [%s, %s]"
                                % (c, n_out, lo, hi))
    return (len(problems) == 0), problems


def flag_extrapolation(df):
    """Boolean array: True where any feature falls outside the p01-p99 training range."""
    bounds = load_feature_bounds()
    flags = np.zeros(len(df), dtype=bool)
    per_feature = {}
    for c in FEATURE_COLUMNS:
        b = bounds[c]
        out = (df[c].values < b["p01"]) | (df[c].values > b["p99"])
        per_feature[c] = int(out.sum())
        flags |= out
    return flags, per_feature


def prepare_features(df, validate=True):
    """v2 pipeline: validate, extract matrix, extract target if present.

    No scaling, no interaction terms. Returns (X, y, info).
    """
    if validate:
        ok, problems = validate_features(df, strict=True)
        if not ok:
            raise ValueError("feature validation failed: %s" % "; ".join(problems))
    X = df[FEATURE_COLUMNS].values.astype(float)
    y = df[TARGET_COLUMN].values.astype(float) if TARGET_COLUMN in df.columns else None
    flags, per_feature = flag_extrapolation(df) if os.path.exists(
        os.path.join(ARTIFACTS_DIR, "model_meta.json")) else (np.zeros(len(df), bool), {})
    return X, y, {"n_rows": len(df), "schema_version": SCHEMA_VERSION,
                  "extrapolation_flags": flags, "out_of_range_per_feature": per_feature}


def load_grid_data():
    """Load the canonical grid master table."""
    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError("grid_master.csv not found.")
    return pd.read_csv(MASTER_CSV)


def get_feature_names():
    return ALL_FEATURES.copy()


if __name__ == "__main__":
    print("=" * 64)
    print("ShadowMap feature_engineering_v2 - self test")
    print("=" * 64)
    print("schema version :", SCHEMA_VERSION)
    print("features       :", FEATURE_COLUMNS)
    print("interactions   :", INTERACTION_FEATURES or "(none - by design)")
    print("scaler         : none (XGBoost is scale-invariant)")

    df = load_grid_data()
    print()
    print("loaded grid_master.csv:", df.shape)

    ok, problems = validate_features(df)
    print("validation     :", "PASS" if ok else "FAIL")
    for p in problems:
        print("   -", p)

    X, y, info = prepare_features(df)
    print()
    print("X shape        :", X.shape)
    print("y range        : %.2f to %.2f C" % (y.min(), y.max()))
    print("extrapolation  : %d rows (%.1f%%)" % (info["extrapolation_flags"].sum(),
          100 * info["extrapolation_flags"].mean()))
    print("per feature    :", info["out_of_range_per_feature"])

    print()
    print("=== FEATURE SUMMARY ===")
    for c in FEATURE_COLUMNS:
        v = df[c]
        print("%-7s min %+.4f  max %+.4f  mean %+.4f" % (c, v.min(), v.max(), v.mean()))

    print()
    print("=== BAD-INPUT TEST (should raise) ===")
    bad = df.head(5).copy()
    bad.loc[bad.index[0], "ndbi"] = 7.5
    try:
        prepare_features(bad)
        print("FAIL - did not raise on out-of-range ndbi")
    except ValueError as e:
        print("PASS - raised:", str(e)[:90])

    print()
    print("=== MISSING-COLUMN TEST (should raise) ===")
    try:
        prepare_features(df.drop(columns=["bsi"]))
        print("FAIL - did not raise on missing column")
    except ValueError as e:
        print("PASS - raised:", str(e)[:90])
