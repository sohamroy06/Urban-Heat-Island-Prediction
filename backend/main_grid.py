"""
main_grid.py - ShadowMap grid API (SCHEMA_VERSION 2)

Separate FastAPI app from main.py. The ward-level app is left untouched.
Run on a different port:

    uvicorn main_grid:app --reload --port 8001

Serves daytime Landsat surface temperature predictions for 6,709 grid cells
over Delhi NCT. NOT air temperature. What-if intervention modelling is
deliberately unsupported - see /api/whatif.
"""

import os
import json
import numpy as np
import pandas as pd

import feature_engineering_v2 as fe
import predict_grid as PG

BASE = os.path.dirname(os.path.abspath(__file__))
GRID_GEOJSON = os.path.join(BASE, "delhi_grid_filtered.geojson")
PREDICTIONS_CSV = os.path.join(BASE, "grid_predictions.csv")

_state = {}

CONFIDENCE_NOTES = {
    "normal": "Features within the training range, away from the NCT boundary, "
              "out-of-fold error under 3 C. Typical absolute error ~0.9 C.",
    "edge_higher_error": "Within 1 km of the NCT boundary. Errors run ~1.3x higher "
                         "(mean ~1.2 C) because the grid cell is clipped and "
                         "surrounding context falls outside the study area.",
    "atypical_features": "One or more features fall outside the p01-p99 training "
                         "range. Mean absolute error ~1.6 C, 95th percentile ~3.6 C.",
    "poorly_predicted": "Out-of-fold absolute error above 3 C (mean ~3.4 C). Usually "
                        "anthropogenic heat sources - industrial zones, landfill - "
                        "that optical reflectance cannot observe.",
}


# ----------------------------------------------------------------- core logic

def load_state():
    """Load predictions, metadata and cell geometry once."""
    if _state:
        return _state

    meta = PG.model_info()
    preds = pd.read_csv(PREDICTIONS_CSV)
    master = fe.load_grid_data()

    df = preds.merge(
        master[["cell_id"] + fe.FEATURE_COLUMNS + ["area_m2"]],
        on="cell_id", how="left", validate="one_to_one")

    lst = df["lst"].values
    city_mean = float(np.mean(lst))
    df["uhi_delta_c"] = df["pred"] - city_mean
    q = np.quantile(lst, [0.2, 0.4, 0.6, 0.8])
    df["heat_rank"] = np.digitize(df["pred"].values, q) + 1

    geo = None
    if os.path.exists(GRID_GEOJSON):
        with open(GRID_GEOJSON) as f:
            geo = json.load(f)

    _state.update(meta=meta, df=df, geo=geo, city_mean=city_mean)
    return _state


def svc_health():
    s = load_state()
    return {
        "status": "ok",
        "schema_version": fe.SCHEMA_VERSION,
        "n_cells": int(len(s["df"])),
        "features": fe.FEATURE_COLUMNS,
        "geometry_loaded": s["geo"] is not None,
        "confidence_classes": sorted(s["df"].confidence_class.unique().tolist()),
        "whatif_supported": False,
    }


def svc_model_info():
    s = load_state()
    m = s["meta"]
    v = m["validation"]
    return {
        "schema_version": m["schema_version"],
        "features": m["features"],
        "target": m["target"],
        "target_description": ("Land surface temperature in Celsius from Landsat 8/9 "
                               "thermal band, daytime overpass ~10:30 local. "
                               "This is a SURFACE temperature, not air temperature."),
        "n_training_cells": m["n_cells"],
        "grid_cell_size_m": m["grid_cell_size_m"],
        "data_source": m["source"],
        "hyperparameters": m["hyperparams"],
        "performance": {
            "headline": v["headline"],
            "r2": v["r2_primary_mean_of_folds"],
            "r2_sd": v["r2_primary_sd"],
            "rmse_c": v["rmse_c_primary"],
            "mae_c": v["mae_c"],
            "bias_c": v["bias_c"],
            "validation_method": v["method"],
            "aggregation_note": v["r2_aggregation_note"],
        },
        "r2_vs_block_size": m["r2_vs_block_size"],
        "leakage_warning": {
            "random_split_r2": m["random_split_r2_LEAKY_DO_NOT_QUOTE"],
            "note": ("Random-split CV is inflated by spatial autocorrelation. LST "
                     "correlates 0.87 between adjacent 450 m cells and the "
                     "semivariogram range is ~14 km, so random folds place a cell's "
                     "own neighbours in training. Do not quote this figure."),
        },
        "permutation_importance": m["permutation_importance"],
        "importance_note": ("Permutation importance on blocked folds. Gain-based "
                            "importance is unreliable under the collinearity here "
                            "(ndbi and bsi correlate 0.989)."),
        "temporal_holdout_2023": m["temporal_holdout_2023"],
        "prediction_interval": {
            "nominal_coverage": 0.80,
            "raw_quantile_coverage": v["interval_coverage"],
            "width_multiplier_applied": v["interval_width_multiplier"],
            "note": ("Raw P10-P90 quantile intervals covered only %.1f%% of held-out "
                     "values, so they are widened by %.2fx to reach the nominal 80%%."
                     % (100 * v["interval_coverage"], v["interval_width_multiplier"])),
        },
    }


def svc_limitations():
    s = load_state()
    return {
        "n_limitations": len(s["meta"]["known_limitations"]),
        "limitations": s["meta"]["known_limitations"],
        "diurnal_finding": {
            "summary": ("Delhi shows a surface urban COOL island by day and a surface "
                        "urban HEAT island by night."),
            "day_overpass": "~10:30 local (Landsat)",
            "night_overpass": "~01:30 local (MODIS MYD11A2, 1 km)",
            "least_built_quartile": {"day_c": 47.09, "night_c": 22.78},
            "most_built_quartile": {"day_c": 43.56, "night_c": 26.01},
            "building_density_corr_day": -0.342,
            "building_density_corr_night": 0.371,
            "caveat": ("Night values are 1 km MODIS resampled onto ~450 m cells, so "
                       "roughly 5 cells share one pixel. Sub-kilometre night variation "
                       "is interpolation, not measurement. A night PREDICTION model was "
                       "attempted and scored R2 0.117 at 5 km blocks - too weak to "
                       "serve, so night is reported as an observation only."),
        },
    }


def svc_cells(min_lon=None, max_lon=None, min_lat=None, max_lat=None, limit=None):
    s = load_state()
    df = s["df"]
    if None not in (min_lon, max_lon, min_lat, max_lat):
        df = df[(df.lon >= min_lon) & (df.lon <= max_lon) &
                (df.lat >= min_lat) & (df.lat <= max_lat)]
    if limit:
        df = df.head(int(limit))
    return {
        "n_cells": int(len(df)),
        "city_mean_lst_c": round(s["city_mean"], 2),
        "cells": [
            {"cell_id": int(r.cell_id), "lon": round(float(r.lon), 5),
             "lat": round(float(r.lat), 5),
             "observed_lst_c": round(float(r.lst), 2),
             "predicted_lst_c": round(float(r.pred), 2),
             "ci_lower_c": round(float(r.p10), 2),
             "ci_upper_c": round(float(r.p90), 2),
             "uhi_delta_c": round(float(r.uhi_delta_c), 2),
             "heat_rank": int(r.heat_rank),
             "confidence_class": str(r.confidence_class)}
            for r in df.itertuples()],
    }


def svc_cell(cell_id):
    s = load_state()
    df = s["df"]
    hit = df[df.cell_id == int(cell_id)]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {
        "cell_id": int(r.cell_id),
        "location": {"lon": round(float(r.lon), 5), "lat": round(float(r.lat), 5)},
        "area_km2": round(float(r.area_m2) / 1e6, 4),
        "observed_lst_c": round(float(r.lst), 2),
        "predicted_lst_c": round(float(r.pred), 2),
        "ci_lower_c": round(float(r.p10), 2),
        "ci_upper_c": round(float(r.p90), 2),
        "residual_c": round(float(r.residual), 2),
        "uhi_delta_c": round(float(r.uhi_delta_c), 2),
        "heat_rank": int(r.heat_rank),
        "heat_rank_meaning": "1 = coolest quintile, 5 = hottest quintile",
        "features": {c: round(float(r[c]), 4) for c in fe.FEATURE_COLUMNS},
        "out_of_fold_predicted_lst_c": (None if pd.isna(r.oof_pred)
                                        else round(float(r.oof_pred), 2)),
        "out_of_fold_residual_c": (None if pd.isna(r.oof_residual)
                                   else round(float(r.oof_residual), 2)),
        "confidence_class": str(r.confidence_class),
        "confidence_note": CONFIDENCE_NOTES.get(str(r.confidence_class), ""),
        "features_outside_training_range": bool(r.out_of_range),
        "near_nct_boundary": bool(r.near_boundary),
        "dist_to_boundary_m": (None if pd.isna(r.dist_to_boundary_m)
                               else round(float(r.dist_to_boundary_m), 1)),
    }


def svc_predict(features):
    return PG.predict_single(features)


def svc_whatif():
    return PG.whatif_status()


def svc_city_stats():
    s = load_state()
    df = s["df"]
    res_ins = df["residual"].values
    res_oof = df["oof_residual"].values
    return {
        "n_cells": int(len(df)),
        "grid_cell_area_km2": 0.218,
        "observed_lst_c": {"min": round(float(df.lst.min()), 2),
                           "mean": round(float(df.lst.mean()), 2),
                           "max": round(float(df.lst.max()), 2),
                           "std": round(float(df.lst.std()), 2)},
        "predicted_lst_c": {"min": round(float(df.pred.min()), 2),
                            "mean": round(float(df.pred.mean()), 2),
                            "max": round(float(df.pred.max()), 2)},
        "prediction_range_note": ("The model compresses extremes: it under-predicts the "
                                  "hottest cells by ~1 C and over-predicts the coldest "
                                  "by ~0.7 C, so predicted range is narrower than observed."),
        "residuals_c": {
            "source": "out-of-fold (5 km blocked); in-sample values are optimistic",
            "mean": round(float(np.nanmean(res_oof)), 3),
            "std": round(float(np.nanstd(res_oof)), 3),
            "abs_gt_3c": int(np.nansum(np.abs(res_oof) > 3)),
            "abs_gt_5c": int(np.nansum(np.abs(res_oof) > 5)),
            "in_sample_std_DO_NOT_QUOTE": round(float(res_ins.std()), 3),
            "rmse_note": ("These out-of-fold residuals average 4 held-out predictions "
                          "per cell, which cancels noise. The honest single-prediction "
                          "RMSE is 1.459 C from model-info, not this std."),
        },
        "confidence_class_counts": {str(k): int(v) for k, v in
                                    df.confidence_class.value_counts().items()},
        "mean_abs_error_by_confidence_class": {
            str(k): round(float(v), 3) for k, v in
            df.assign(ae=np.abs(res_oof)).groupby("confidence_class",
                                                  observed=True).ae.mean().items()},
        "uhi_intensity_c": round(float(df.lst.max() - df.lst.min()), 2),
        "cells_with_atypical_features": int(df.out_of_range.sum()),
        "cells_near_boundary": int(df.near_boundary.sum()),
        "heat_rank_counts": {int(k): int(v) for k, v in
                             df.heat_rank.value_counts().sort_index().items()},
    }


# ----------------------------------------------------------------- FastAPI app

def build_app():
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    app = FastAPI(title="ShadowMap Grid API",
                  description="Daytime surface temperature for Delhi NCT. Not air temperature.",
                  version="2.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    class PredictRequest(BaseModel):
        albedo: float = Field(..., ge=0.0, le=1.0)
        ndbi: float = Field(..., ge=-1.0, le=1.0)
        mndwi: float = Field(..., ge=-1.0, le=1.0)
        bsi: float = Field(..., ge=-1.0, le=1.0)

    @app.get("/api/health")
    def health():
        return svc_health()

    @app.get("/api/model-info")
    def model_info():
        return svc_model_info()

    @app.get("/api/limitations")
    def limitations():
        return svc_limitations()

    @app.get("/api/cells")
    def cells(min_lon: float = Query(None), max_lon: float = Query(None),
              min_lat: float = Query(None), max_lat: float = Query(None),
              limit: int = Query(None)):
        return svc_cells(min_lon, max_lon, min_lat, max_lat, limit)

    @app.get("/api/cell/{cell_id}")
    def cell(cell_id: int):
        r = svc_cell(cell_id)
        if r is None:
            raise HTTPException(status_code=404, detail="cell_id %d not found" % cell_id)
        return r

    @app.post("/api/predict")
    def predict(req: PredictRequest):
        try:
            return svc_predict(req.model_dump())
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.api_route("/api/whatif", methods=["GET", "POST"], status_code=501)
    def whatif():
        return svc_whatif()

    @app.get("/api/city-stats")
    def city_stats():
        return svc_city_stats()

    return app


app = build_app()


if __name__ == "__main__":
    print("=" * 68)
    print("main_grid.py - endpoint self test (no server, direct function calls)")
    print("=" * 68)

    def show(label, obj, keys=None):
        print()
        print("--- %s ---" % label)
        if keys:
            for k in keys:
                print("  %-26s %s" % (k, obj.get(k)))
        else:
            print(json.dumps(obj, indent=2)[:700])

    show("/api/health", svc_health())

    mi = svc_model_info()
    print()
    print("--- /api/model-info ---")
    print("  headline    :", mi["performance"]["headline"])
    print("  r2          :", mi["performance"]["r2"], "+/-", mi["performance"]["r2_sd"])
    print("  leaky r2    :", mi["leakage_warning"]["random_split_r2"])
    print("  importance  :", mi["permutation_importance"])
    print("  PI multiplr :", mi["prediction_interval"]["width_multiplier_applied"])

    lim = svc_limitations()
    print()
    print("--- /api/limitations ---")
    print("  count       :", lim["n_limitations"])
    print("  diurnal     : day least-built %.2f C vs most-built %.2f C"
          % (lim["diurnal_finding"]["least_built_quartile"]["day_c"],
             lim["diurnal_finding"]["most_built_quartile"]["day_c"]))
    print("               night least-built %.2f C vs most-built %.2f C"
          % (lim["diurnal_finding"]["least_built_quartile"]["night_c"],
             lim["diurnal_finding"]["most_built_quartile"]["night_c"]))

    c = svc_cells(limit=3)
    print()
    print("--- /api/cells?limit=3 ---")
    print("  n_cells     :", c["n_cells"], "| city mean", c["city_mean_lst_c"])
    for x in c["cells"]:
        print("   ", x)

    print()
    print("--- /api/cells (all) ---")
    print("  n_cells     :", svc_cells()["n_cells"])

    print()
    print("--- /api/cell/840 (worst residual cell) ---")
    print(json.dumps(svc_cell(840), indent=2))

    print()
    print("--- /api/cell/999999 (should be None -> 404) ---")
    print("  result      :", svc_cell(999999))

    df = fe.load_grid_data()
    row = df.iloc[100]
    print()
    print("--- /api/predict (valid) ---")
    print(json.dumps(svc_predict({c_: float(row[c_]) for c_ in fe.FEATURE_COLUMNS}), indent=2))

    print()
    print("--- /api/predict (out of physical range, should raise) ---")
    try:
        svc_predict({c_: 9.0 for c_ in fe.FEATURE_COLUMNS})
        print("  FAIL - did not raise")
    except ValueError as e:
        print("  PASS - raised:", str(e)[:70])

    print()
    print("--- /api/whatif (501) ---")
    w = svc_whatif()
    print("  available   :", w["available"])
    print("  needed      :", len(w["what_would_be_needed"]), "prerequisites listed")

    print()
    print("--- /api/city-stats ---")
    cs = svc_city_stats()
    print("  observed    :", cs["observed_lst_c"])
    print("  predicted   :", cs["predicted_lst_c"])
    print("  residuals   : std %.3f C (out-of-fold) | in-sample %.3f C (do not quote)"
          % (cs["residuals_c"]["std"], cs["residuals_c"]["in_sample_std_DO_NOT_QUOTE"]))
    print("  |err|>3C    :", cs["residuals_c"]["abs_gt_3c"], "| >5C:", cs["residuals_c"]["abs_gt_5c"])
    print("  conf counts :", cs["confidence_class_counts"])
    print("  err by class:", cs["mean_abs_error_by_confidence_class"])
    print("  heat ranks  :", cs["heat_rank_counts"])
    print("  atypical    :", cs["cells_with_atypical_features"],
          "| near boundary:", cs["cells_near_boundary"])

    print()
    print("=" * 68)
    print("ROUTES:", [r.path for r in app.routes if str(r.path).startswith("/api")])
    print("=" * 68)
