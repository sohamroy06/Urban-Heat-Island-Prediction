"""
predict_grid.py - production predictor, ShadowMap grid model (SCHEMA_VERSION 2)

Target: Landsat 8/9 daytime land surface temperature, ~10:30 local overpass,
Delhi NCT, April-June 2024 median composite. NOT air temperature.

Validated by 5 km spatially blocked cross-validation with buffer zones.
Random-split CV on this data reads 0.7422 and is LEAKY - do not quote it.
"""

import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import feature_engineering_v2 as fe

ART = fe.ARTIFACTS_DIR
_cache = {}


def load_models():
    """Load mean/p10/p90 models and metadata. Cached after first call."""
    if _cache:
        return _cache
    meta_path = os.path.join(ART, "model_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("model_meta.json missing. Run step21_rebuild.py.")
    with open(meta_path) as f:
        meta = json.load(f)

    if meta["features"] != fe.FEATURE_COLUMNS:
        raise RuntimeError(
            "SCHEMA MISMATCH: model trained on %s but feature_engineering_v2 "
            "declares %s. Retrain or fix the schema." % (meta["features"], fe.FEATURE_COLUMNS))

    models = {}
    for key, fn in [("mean", "uhi_grid_mean.json"),
                    ("p10", "uhi_grid_p10.json"),
                    ("p90", "uhi_grid_p90.json")]:
        path = os.path.join(ART, fn)
        if not os.path.exists(path):
            raise FileNotFoundError("model file missing: %s" % path)
        mo = XGBRegressor()
        mo.load_model(path)
        models[key] = mo

    _cache.update(models=models, meta=meta)
    return _cache


def predict(df, validate=True):
    """Predict LST for rows carrying the v2 feature columns.

    Returns a DataFrame with pred_lst_c, p10_c, p90_c, interval_width_c
    and an extrapolation flag. Intervals are widened by the calibration
    multiplier measured on blocked CV (raw quantile intervals are too narrow).
    """
    c = load_models()
    models, meta = c["models"], c["meta"]

    X, _, info = fe.prepare_features(df, validate=validate)

    mean = models["mean"].predict(X)
    p10_raw = models["p10"].predict(X)
    p90_raw = models["p90"].predict(X)

    k = float(meta["validation"].get("interval_width_multiplier", 1.0))
    mid = (p10_raw + p90_raw) / 2.0
    p10 = mid - (mid - p10_raw) * k
    p90 = mid + (p90_raw - mid) * k

    lo_bad = int(np.sum(p10 > mean))
    hi_bad = int(np.sum(p90 < mean))
    p10 = np.minimum(p10, mean)
    p90 = np.maximum(p90, mean)

    out = pd.DataFrame({
        "pred_lst_c": mean,
        "p10_c": p10,
        "p90_c": p90,
        "interval_width_c": p90 - p10,
        "extrapolation": info["extrapolation_flags"],
    }, index=df.index)
    out.attrs["interval_multiplier"] = k
    out.attrs["out_of_range_per_feature"] = info["out_of_range_per_feature"]
    out.attrs["quantile_crossings_fixed"] = {"p10_above_mean": lo_bad, "p90_below_mean": hi_bad}
    return out


def predict_single(features, validate=True):
    """Predict for one location. features: dict of the v2 feature columns."""
    r = predict(pd.DataFrame([features]), validate=validate)
    row = r.iloc[0]
    return {
        "predicted_lst": round(float(row.pred_lst_c), 2),
        "ci_lower": round(float(row.p10_c), 2),
        "ci_upper": round(float(row.p90_c), 2),
        "ci_width": round(float(row.interval_width_c), 2),
        "extrapolation": bool(row.extrapolation),
        "confidence_note": ("Inputs fall outside the training range; this prediction is "
                            "unreliable." if bool(row.extrapolation) else
                            "Inputs within training range."),
    }


def model_info():
    return load_models()["meta"]


def whatif_status():
    """What-if intervention modelling is DISABLED. Returns the reason."""
    return {
        "available": False,
        "reason": (
            "Intervention modelling is not supported. The four model features are "
            "spectral measurements, not policy levers. Albedo was tested as the one "
            "physically causal candidate and failed twice: cross-sectionally within "
            "built-up cells (r = +0.699) and in a within-cell 2023-2024 change design "
            "(+21.6 C per unit albedo, wrong sign). In Delhi's pre-monsoon dry season, "
            "surface brightness tracks dryness, so brighter surfaces are hotter. "
            "Urban-form features (building density, road density, population) show no "
            "daytime predictive skill at all."),
        "what_would_be_needed": [
            "Night-time LST at sub-kilometre resolution (MODIS 1 km is too coarse)",
            "Multi-year before/after observations of actual interventions",
            "Air temperature ground truth, not just surface temperature",
        ],
    }


if __name__ == "__main__":
    meta = model_info()
    print("=" * 64)
    print("predict_grid.py - self test (schema v%d)" % fe.SCHEMA_VERSION)
    print("=" * 64)
    print("features        :", meta["features"])
    print("blocked 5km R2  :", meta["validation"]["r2"])
    print("RMSE            : %.3f C" % meta["validation"]["rmse_c"])
    print("raw PI coverage : %.1f%%" % (100 * meta["validation"]["interval_coverage"]))
    print("PI multiplier   : %.2f" % meta["validation"]["interval_width_multiplier"])

    df = fe.load_grid_data()
    r = predict(df)
    print()
    print("=== BATCH on %d cells ===" % len(df))
    print("pred  min %.2f  mean %.2f  max %.2f" % (r.pred_lst_c.min(), r.pred_lst_c.mean(), r.pred_lst_c.max()))
    print("true  min %.2f  mean %.2f  max %.2f" % (df.lst.min(), df.lst.mean(), df.lst.max()))
    print("mean interval width %.2f C" % r.interval_width_c.mean())
    print("extrapolation flags %d (%.1f%%)" % (r.extrapolation.sum(), 100 * r.extrapolation.mean()))
    print("quantile crossings fixed:", r.attrs["quantile_crossings_fixed"])

    old = pd.read_csv("grid_predictions.csv")
    diff = float(np.abs(r.pred_lst_c.values - old.pred.values).max())
    print()
    print("reproducibility vs step21: max diff %.2e -> %s" % (diff, "PASS" if diff < 1e-4 else "FAIL"))

    print()
    print("=== SINGLE PREDICTION ===")
    row = df.iloc[0]
    print(json.dumps(predict_single({c: float(row[c]) for c in fe.FEATURE_COLUMNS}), indent=2))

    print()
    print("=== EXTRAPOLATION GUARD ===")
    print(json.dumps(predict_single({c: 0.9 for c in fe.FEATURE_COLUMNS}), indent=2))

    print()
    print("=== INVALID INPUT GUARD (should raise) ===")
    try:
        predict_single({c: 50.0 for c in fe.FEATURE_COLUMNS})
        print("FAIL - did not raise")
    except ValueError as e:
        print("PASS - raised:", str(e)[:80])

    print()
    print("=== WHATIF STATUS ===")
    w = whatif_status()
    print("available:", w["available"])
    print("reason   :", w["reason"][:110] + "...")
