"""
train_grid.py - canonical trainer for the ShadowMap grid model (SCHEMA_VERSION 2)

Single entrypoint. Rebuilds every artifact in artifacts_grid/ from grid_master.csv.
Replaces model.py for the grid pipeline; model.py is kept for the ward pipeline.

    python train_grid.py

Evaluation is 5 km spatially blocked CV with buffer zones. Random-split CV is
also computed but ONLY as a leakage reference - it is not a performance claim.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

import feature_engineering_v2 as fe

ART = fe.ARTIFACTS_DIR

HYPERPARAMS = dict(
    n_estimators=600, max_depth=2, learning_rate=0.05,
    reg_alpha=1.0, reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8,
    objective="reg:squarederror", verbosity=0, n_jobs=4, random_state=42,
)

BLOCK_KM_PRIMARY = 5
BLOCK_KM_CURVE = [1, 2, 5, 10]
N_REPEATS = 4


def blocked_folds(cx, cy, km, reps=N_REPEATS, seed=500):
    """Spatially blocked folds. Training cells within one block-width of any
    test cell are dropped (buffer) to prevent autocorrelation leakage."""
    B = km * 1000.0
    blk = (np.floor((cx - cx.min()) / B).astype(int) * 10000 +
           np.floor((cy - cy.min()) / B).astype(int))
    ub = np.unique(blk)
    XY = np.c_[cx, cy]
    folds = []
    for rep in range(reps):
        perm = np.random.default_rng(seed + rep).permutation(ub)
        for f in np.array_split(perm, 5):
            te = np.isin(blk, f)
            if te.sum() < 25:
                continue
            hit = cKDTree(XY[te]).query_ball_point(XY, r=B, p=np.inf)
            tr = (~te) & ~np.array([len(h) > 0 for h in hit])
            if tr.sum() < 200:
                continue
            folds.append((tr, te))
    return folds


def evaluate(X, y, folds):
    s = [r2_score(y[te], XGBRegressor(**HYPERPARAMS).fit(X[tr], y[tr]).predict(X[te]))
         for tr, te in folds]
    return np.array(s)


def main():
    print("=" * 68)
    print("ShadowMap grid model - canonical training (schema v%d)" % fe.SCHEMA_VERSION)
    print("=" * 68)

    df = fe.load_grid_data()
    ok, problems = fe.validate_features(df)
    if not ok:
        raise SystemExit("feature validation failed: %s" % problems)
    X = df[fe.FEATURE_COLUMNS].values.astype(float)
    y = df[fe.TARGET_COLUMN].values.astype(float)
    cx, cy = df.cx.values, df.cy.values
    print("cells %d | features %s" % (len(df), fe.FEATURE_COLUMNS))
    print("target std %.3f C" % y.std())

    print()
    print("=== R2 vs BLOCK SIZE (leakage diagnostic) ===")
    print("%10s %8s %10s %10s %9s" % ("block_km", "folds", "R2", "sd", "RMSE_C"))
    curve, curve_sd = {}, {}
    for km in BLOCK_KM_CURVE:
        f = blocked_folds(cx, cy, km)
        s = evaluate(X, y, f)
        curve["%dkm" % km] = round(float(s.mean()), 4)
        curve_sd["%dkm" % km] = round(float(s.std()), 4)
        print("%10d %8d %10.4f %10.4f %9.2f"
              % (km, len(f), s.mean(), s.std(), y.std() * np.sqrt(max(0, 1 - s.mean()))))

    rs = []
    for rep in range(4):
        for tr_i, te_i in KFold(5, shuffle=True, random_state=100 + rep).split(X):
            rs.append(r2_score(y[te_i], XGBRegressor(**HYPERPARAMS).fit(X[tr_i], y[tr_i]).predict(X[te_i])))
    random_r2 = float(np.mean(rs))
    print("%10s %8d %10.4f %10s %9s" % ("random", len(rs), random_r2, "-", "LEAKY"))
    print("inflation vs %d km blocks: %+.4f" % (BLOCK_KM_PRIMARY, random_r2 - curve["%dkm" % BLOCK_KM_PRIMARY]))

    print()
    print("=== OUT-OF-FOLD PERFORMANCE (%d km blocked) ===" % BLOCK_KM_PRIMARY)
    folds = blocked_folds(cx, cy, BLOCK_KM_PRIMARY)
    op, olo, ohi, oy = [], [], [], []
    for tr, te in folds:
        m_ = XGBRegressor(**HYPERPARAMS).fit(X[tr], y[tr])
        lo_ = XGBRegressor(**{**HYPERPARAMS, "objective": "reg:quantileerror",
                              "quantile_alpha": 0.10}).fit(X[tr], y[tr])
        hi_ = XGBRegressor(**{**HYPERPARAMS, "objective": "reg:quantileerror",
                              "quantile_alpha": 0.90}).fit(X[tr], y[tr])
        op.append(m_.predict(X[te])); olo.append(lo_.predict(X[te]))
        ohi.append(hi_.predict(X[te])); oy.append(y[te])
    p, lo, hi, yy = map(np.concatenate, (op, olo, ohi, oy))
    R2 = float(r2_score(yy, p))
    RMSE = float(np.sqrt(np.mean((yy - p) ** 2)))
    MAE = float(mean_absolute_error(yy, p))
    BIAS = float(np.mean(p - yy))
    print("R2 %.4f | RMSE %.3f C | MAE %.3f C | bias %+.3f C | n_oof %d" % (R2, RMSE, MAE, BIAS, len(yy)))

    cov = float(np.mean((yy >= lo) & (yy <= hi)))
    k = 1.0
    if cov < 0.78:
        for s_ in np.arange(1.0, 3.05, 0.05):
            mid = (lo + hi) / 2
            if float(np.mean((yy >= mid - (mid - lo) * s_) & (yy <= mid + (hi - mid) * s_))) >= 0.80:
                k = float(round(s_, 2))
                break
    print("PI coverage raw %.1f%% -> multiplier %.2f" % (100 * cov, k))

    print()
    print("=== PERMUTATION IMPORTANCE (blocked, gain importance is unreliable here) ===")
    imp = {c: [] for c in fe.FEATURE_COLUMNS}
    rng = np.random.default_rng(7)
    for tr, te in folds[:10]:
        mo = XGBRegressor(**HYPERPARAMS).fit(X[tr], y[tr])
        b = r2_score(y[te], mo.predict(X[te]))
        for j, c in enumerate(fe.FEATURE_COLUMNS):
            Xp = X[te].copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            imp[c].append(b - r2_score(y[te], mo.predict(Xp)))
    perm = {c: round(float(np.mean(v)), 4) for c, v in imp.items()}
    for c, v in sorted(perm.items(), key=lambda t: -t[1]):
        print("  %-8s %.4f" % (c, v))

    PRIMARY_R2 = curve["%dkm" % BLOCK_KM_PRIMARY]
    PRIMARY_SD = curve_sd["%dkm" % BLOCK_KM_PRIMARY]
    PRIMARY_RMSE = round(float(y.std() * np.sqrt(max(0.0, 1 - PRIMARY_R2))), 2)

    print()
    print("=== TRAINING FINAL MODELS ON ALL DATA ===")
    os.makedirs(ART, exist_ok=True)
    mean_m = XGBRegressor(**HYPERPARAMS).fit(X, y)
    lo_m = XGBRegressor(**{**HYPERPARAMS, "objective": "reg:quantileerror", "quantile_alpha": 0.10}).fit(X, y)
    hi_m = XGBRegressor(**{**HYPERPARAMS, "objective": "reg:quantileerror", "quantile_alpha": 0.90}).fit(X, y)
    mean_m.save_model(os.path.join(ART, "uhi_grid_mean.json"))
    lo_m.save_model(os.path.join(ART, "uhi_grid_p10.json"))
    hi_m.save_model(os.path.join(ART, "uhi_grid_p90.json"))
    print("saved 3 model files")

    meta = {
        "schema_version": fe.SCHEMA_VERSION,
        "features": fe.FEATURE_COLUMNS,
        "target": "landsat_daytime_lst_celsius",
        "n_cells": int(len(df)),
        "grid_cell_size_m": [448.5, 505.7],
        "grid_cell_area_km2": 0.218,
        "source": ("Landsat 8/9 Collection 2 Level-2 median composite, "
                   "2024-04-01 to 2024-06-30, per-pixel QA_PIXEL cloud/shadow masked"),
        "overpass_local_time": "~10:30",
        "hyperparams": {kk: vv for kk, vv in HYPERPARAMS.items() if kk != "n_jobs"},
        "validation": {
            "method": "5 km spatially blocked CV, buffered, %d repeats" % N_REPEATS,
            "headline": ("blocked 5 km CV: R2 = %.3f +/- %.3f, RMSE = %.2f C"
                         % (PRIMARY_R2, PRIMARY_SD, PRIMARY_RMSE)),
            "r2": PRIMARY_R2,
            "r2_primary_mean_of_folds": PRIMARY_R2,
            "r2_primary_sd": PRIMARY_SD,
            "rmse_c_primary": PRIMARY_RMSE,
            "r2_pooled_oof": round(R2, 4),
            "rmse_c": round(RMSE, 3),
            "r2_aggregation_note": (
                "r2_primary_mean_of_folds averages per-fold R2, each scored against its "
                "own test block variance. This is the conservative standard for blocked "
                "CV and is the number to quote. r2_pooled_oof concatenates all fold "
                "predictions and scores once against global variance, which is "
                "systematically higher. Both come from the same 5 km blocked folds."),
            "mae_c": round(MAE, 3),
            "bias_c": round(BIAS, 3),
            "interval_coverage": round(cov, 4),
            "interval_width_multiplier": k,
            "n_oof_predictions": int(len(yy)),
        },
        "r2_vs_block_size_sd": curve_sd,
        "r2_vs_block_size": curve,
        "random_split_r2_LEAKY_DO_NOT_QUOTE": round(random_r2, 4),
        "permutation_importance": perm,
        "temporal_holdout_2023": {
            "note": "2024-trained model applied to an independent 2023 composite",
            "r2_raw": -0.5611, "r2_debiased": 0.5558,
            "spearman_like_corr": 0.8231, "bias_c": 2.63,
            "interpretation": ("Spatial ranking transfers across years. Absolute "
                               "temperature does not; recalibration required per season."),
        },
        "feature_bounds": {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                               "p01": float(df[c].quantile(0.01)),
                               "p99": float(df[c].quantile(0.99))} for c in fe.FEATURE_COLUMNS},
        "whatif_supported": False,
        "known_limitations": [
            "Daytime SURFACE temperature (~10:30 local) only; NOT air temperature",
            "Single season, April-June 2024 pre-monsoon; absolute values do not transfer across years",
            "Delhi NCT only; no evidence of transfer to other cities",
            "ndbi correlates -0.323 with OSM building density: it measures SURFACE DRYNESS, not urbanisation",
            "Landsat L2 emissivity retrieval uses NDVI, giving partial circularity with optical predictors",
            "Predicts the thermal band of the same Landsat composite the predictors come from; "
            "cross-sensor check against MODIS gives R2 0.298 against a sensor-agreement ceiling of 0.472",
            "OSM building/road and WorldPop features show NO daytime predictive skill; excluded",
            "Daytime Delhi shows a surface urban COOL island: dense areas are ~3.5 C cooler at 10:30, "
            "and ~3.2 C hotter at 01:30 (MODIS). Do not use for building-density policy inference",
            "Model compresses extremes: under-predicts hottest cells ~1 C, over-predicts coldest ~0.7 C",
            "Errors ~1.7x larger in cells adjacent to the NCT boundary",
            "Dense-population zones (SE industrial belt) under-predicted; anthropogenic heat is not "
            "observable from optical reflectance",
            "Intervention/what-if modelling unsupported; albedo failed causal testing in two designs",
        ],
    }
    with open(os.path.join(ART, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved model_meta.json")

    pred = mean_m.predict(X)
    p10r, p90r = lo_m.predict(X), hi_m.predict(X)
    mid = (p10r + p90r) / 2
    out = df[["cell_id", "lon", "lat", fe.TARGET_COLUMN]].copy()
    out["pred"] = pred
    out["p10"] = np.minimum(mid - (mid - p10r) * k, pred)
    out["p90"] = np.maximum(mid + (p90r - mid) * k, pred)
    out["residual"] = out["pred"] - out[fe.TARGET_COLUMN]
    flags, _ = fe.flag_extrapolation(df)
    out["extrapolation"] = flags
    out.to_csv("grid_predictions.csv", index=False)
    print("saved grid_predictions.csv (%d rows)" % len(out))

    print()
    print("=" * 68)
    print("HEADLINE (quote this): blocked 5 km R2 %.4f +/- %.4f, RMSE %.2f C"
          % (PRIMARY_R2, PRIMARY_SD, PRIMARY_RMSE))
    print("  secondary: pooled out-of-fold R2 %.4f (higher; scored vs global variance)" % R2)
    print("  NEVER quote the random-split figure of %.4f - it is leakage." % random_r2)
    print()
    print("NEXT: run  python add_oof_columns.py  to restore out-of-fold columns")
    print("      in grid_predictions.csv, which the API serves.")
    print("=" * 68)


if __name__ == "__main__":
    main()
