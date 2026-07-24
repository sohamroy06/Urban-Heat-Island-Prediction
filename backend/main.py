"""
main.py — ShadowMap FastAPI Backend (CPU-Optimized Edition)

All API routes for the Urban Heat Island Prediction & What-If Simulator.
Backend uses XGBoost with multi-threaded CPU inference.

Architecture: Multi-threaded, hardware-agnostic inference
optimized for AMD EPYC server-class CPUs.
"""

import json
import os
import sys
import traceback
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import generate_synthetic_data, generate_geojson
from feature_engineering import ALL_FEATURES, FEATURE_COLUMNS, compute_interaction_features
from model import (
    train_model,
    load_models,
    predict_block,
    predict_batch,
    compute_feature_contributions,
    simulate_whatif,
    simulate_whatif_batch,
    generate_intervention_curve,
    get_feature_importance,
    ARTIFACTS_DIR,
)

app_state = {
    "models": None,
    "df": None,
    "geojson": None,
    "predictions": None,
    "city_stats": None,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_cors_origins():
    """Load allowed CORS origins from env, with localhost defaults for dev."""
    configured = os.getenv("CORS_ORIGINS", "").strip()
    if configured:
        origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
        if origins:
            return origins

    return [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
    ]


def build_city_stats(df: pd.DataFrame, predictions: dict) -> dict:
    """Aggregate per-block predictions into city-wide stats (mean/max/min LST,
    top/bottom 5, category breakdown, feature correlations)."""
    city_mean = float(np.mean([p["predicted_lst"] for p in predictions.values()]))
    all_lsts = sorted(predictions.items(), key=lambda x: x[1]["predicted_lst"], reverse=True)

    top5_hot = []
    for bid, pred in all_lsts[:5]:
        row = df[df["block_id"] == bid].iloc[0]
        top5_hot.append({
            "block_id": bid,
            "block_name": row["block_name"],
            "ward": row["ward"],
            "predicted_lst": pred["predicted_lst"],
            "ci_lower": pred["ci_lower"],
            "ci_upper": pred["ci_upper"],
        })

    top5_cool = []
    for bid, pred in all_lsts[-5:]:
        row = df[df["block_id"] == bid].iloc[0]
        top5_cool.append({
            "block_id": bid,
            "block_name": row["block_name"],
            "ward": row["ward"],
            "predicted_lst": pred["predicted_lst"],
            "ci_lower": pred["ci_lower"],
            "ci_upper": pred["ci_upper"],
        })

    all_pred_lsts = [p["predicted_lst"] for p in predictions.values()]
    max_lst = max(all_pred_lsts)
    min_lst = min(all_pred_lsts)

    n_critical = sum(1 for t in all_pred_lsts if t >= 42)
    n_hot = sum(1 for t in all_pred_lsts if 38 <= t < 42)
    n_moderate = sum(1 for t in all_pred_lsts if 34 <= t < 38)
    n_cool = sum(1 for t in all_pred_lsts if t < 34)
    total = len(all_pred_lsts)

    city_stats = {
        "city_mean_lst": round(city_mean, 2),
        "max_lst": round(max_lst, 2),
        "min_lst": round(min_lst, 2),
        "uhi_intensity": round(max_lst - min_lst, 2),
        "top5_hottest": top5_hot,
        "top5_coolest": top5_cool,
        "categories": {
            "critical": {"count": n_critical, "percentage": round(n_critical / total * 100, 1)},
            "hot": {"count": n_hot, "percentage": round(n_hot / total * 100, 1)},
            "moderate": {"count": n_moderate, "percentage": round(n_moderate / total * 100, 1)},
            "cool": {"count": n_cool, "percentage": round(n_cool / total * 100, 1)},
        },
        "total_blocks": total,
        "feature_correlations": {},
    }

    for feat in FEATURE_COLUMNS:
        corr = float(np.corrcoef(df[feat].values, df["lst"].values)[0, 1])
        city_stats["feature_correlations"][feat] = round(corr, 3)

    return city_stats


def load_or_train():
    """Load existing model artifacts or train from scratch."""
    csv_path = os.path.join(BASE_DIR, "sample_data.csv")
    geojson_path = os.path.join(BASE_DIR, "delhi_blocks.geojson")
    model_path = os.path.join(ARTIFACTS_DIR, "uhi_model.json")

    if os.path.exists(csv_path):
        print("[STARTUP] Loading data from sample_data.csv")
        df = pd.read_csv(csv_path)
    else:
        print("[STARTUP] Generating synthetic data...")
        df = generate_synthetic_data(n_blocks=300)
        df.to_csv(csv_path, index=False)
        print(f"[STARTUP] Saved {len(df)} blocks to {csv_path}")

    # Always regenerate from the current df rather than reusing whatever's
    # on disk: delhi_blocks.geojson previously cached a stale 300-block
    # synthetic layout that silently stopped matching sample_data.csv once
    # the real 95-ward dataset replaced it, so every block's geometry,
    # name and ward were wrong despite sharing a block_id. Regeneration is
    # cheap (pure computation from an already-loaded df, no external calls).
    print("[STARTUP] Generating GeoJSON from current data...")
    generate_geojson(df, geojson_path)

    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    if os.path.exists(model_path):
        print("[STARTUP] Loading existing model artifacts...")
        try:
            models = load_models()
            print("[STARTUP] Models loaded successfully.")
        except Exception as e:
            print(f"[STARTUP] Failed to load models: {e}, retraining...")
            models = train_model(df)
    else:
        print("[STARTUP] No trained model found. Training now...")
        models = train_model(df)

    # --- AMD EPYC OPTIMIZED CPU MULTI-THREAD SECTION ---
    print(f"[STARTUP] Computing predictions for all blocks (CPU, n_jobs=-1)...")
    batch_features_df = df[FEATURE_COLUMNS].copy()
    batch_results = predict_batch(batch_features_df, models)

    predictions = {}
    for i, (_, row) in enumerate(df.iterrows()):
        block_id = row["block_id"]
        predictions[block_id] = {
            "predicted_lst": float(batch_results.iloc[i]["predicted_lst"]),
            "ci_lower": float(batch_results.iloc[i]["ci_lower"]),
            "ci_upper": float(batch_results.iloc[i]["ci_upper"]),
            "ci_width": float(batch_results.iloc[i]["ci_width"]),
        }
    # --- END AMD EPYC OPTIMIZED CPU MULTI-THREAD SECTION ---

    for feature in geojson["features"]:
        block_id = feature["properties"]["block_id"]
        if block_id in predictions:
            feature["properties"]["predicted_lst"] = predictions[block_id]["predicted_lst"]
            feature["properties"]["ci_lower"] = predictions[block_id]["ci_lower"]
            feature["properties"]["ci_upper"] = predictions[block_id]["ci_upper"]
            feature["properties"]["ci_width"] = predictions[block_id]["ci_width"]

    city_stats = build_city_stats(df, predictions)

    app_state["models"] = models
    app_state["df"] = df
    app_state["geojson"] = geojson
    app_state["predictions"] = predictions
    app_state["city_stats"] = city_stats

    print(f"[STARTUP] Ready! {len(df)} blocks, Mean LST: {city_stats['city_mean_lst']:.1f}°C")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events for the FastAPI app."""
    print("=" * 60)
    print("ShadowMap API — Starting Up (CPU-Optimized)")
    print("=" * 60)
    try:
        load_or_train()
    except Exception as e:
        print(f"[ERROR] Startup failed: {e}")
        traceback.print_exc()
        raise
    yield
    print("[SHUTDOWN] ShadowMap API shutting down.")


app = FastAPI(
    title="ShadowMap API",
    description="Urban Heat Island Prediction & What-If Simulator for Delhi",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WhatIfRequest(BaseModel):
    """Request body for the what-if simulation endpoint."""
    block_id: str = Field(..., description="ID of the block to simulate")
    delta_buildings: int = Field(0, ge=0, le=50, description="Number of buildings to add")
    delta_trees: int = Field(0, ge=0, le=500, description="Number of trees to add")
    delta_albedo: float = Field(0.0, ge=0.0, le=50.0, description="Roof albedo increase (%)")


class CityWideWhatIfRequest(BaseModel):
    """Request body for the city-wide what-if simulation endpoint."""
    delta_buildings: int = Field(0, ge=0, le=50, description="Number of buildings to add per block")
    delta_trees: int = Field(0, ge=0, le=500, description="Number of trees to add per block")
    delta_albedo: float = Field(0.0, ge=0.0, le=50.0, description="Roof albedo increase (%)")
    scope: str = Field("all", description="'all' blocks, or 'hottest' for the top N hottest")
    top_n: int = Field(20, ge=1, le=95, description="Block count when scope='hottest'")


@app.get("/api/blocks")
async def get_blocks():
    """
    Returns GeoJSON of all Delhi blocks with predicted UHI scores,
    confidence intervals, and feature values for choropleth rendering.
    """
    if app_state["geojson"] is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    return app_state["geojson"]


@app.get("/api/block/{block_id}")
async def get_block(block_id: str):
    """
    Returns detailed info for one block including all features, UHI score,
    confidence interval, feature contributions, and contextual information.
    """
    if app_state["df"] is None or app_state["models"] is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    df = app_state["df"]
    block_row = df[df["block_id"] == block_id]
    if block_row.empty:
        raise HTTPException(status_code=404, detail=f"Block {block_id} not found")

    row = block_row.iloc[0]
    features = {col: float(row[col]) for col in FEATURE_COLUMNS}

    prediction = predict_block(features, app_state["models"])

    contributions = compute_feature_contributions(
        features, app_state["models"], app_state["city_stats"]["city_mean_lst"]
    )

    all_lsts = sorted(
        app_state["predictions"].items(),
        key=lambda x: x[1]["predicted_lst"],
        reverse=True
    )
    rank = next(
        (i + 1 for i, (bid, _) in enumerate(all_lsts) if bid == block_id),
        None
    )

    lst = prediction["predicted_lst"]
    if lst >= 42:
        risk = "Critical"
    elif lst >= 38:
        risk = "Hot"
    elif lst >= 34:
        risk = "Moderate"
    else:
        risk = "Cool"

    city_mean = app_state["city_stats"]["city_mean_lst"]
    diff = round(lst - city_mean, 1)
    if diff > 0:
        context = f"This block is {diff}°C warmer than the Delhi average ({city_mean}°C)."
    elif diff < 0:
        context = f"This block is {abs(diff)}°C cooler than the Delhi average ({city_mean}°C)."
    else:
        context = f"This block is at the Delhi average temperature ({city_mean}°C)."

    return {
        "block_id": block_id,
        "block_name": row["block_name"],
        "ward": row["ward"],
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "features": features,
        "predicted_lst": prediction["predicted_lst"],
        "ci_lower": prediction["ci_lower"],
        "ci_upper": prediction["ci_upper"],
        "ci_width": prediction["ci_width"],
        "feature_contributions": contributions,
        "rank": rank,
        "total_blocks": len(app_state["predictions"]),
        "risk_level": risk,
        "historical_context": context,
    }


@app.post("/api/whatif")
async def whatif(request: WhatIfRequest):
    """
    Simulate a what-if scenario: add buildings, trees, or change albedo
    for a specific block and see the predicted temperature change.
    """
    if app_state["df"] is None or app_state["models"] is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    df = app_state["df"]
    block_row = df[df["block_id"] == request.block_id]
    if block_row.empty:
        raise HTTPException(status_code=404, detail=f"Block {request.block_id} not found")

    row = block_row.iloc[0]
    features = {col: float(row[col]) for col in FEATURE_COLUMNS}

    result = simulate_whatif(
        block_features=features,
        delta_buildings=request.delta_buildings,
        delta_trees=request.delta_trees,
        delta_albedo=request.delta_albedo,
        models=app_state["models"],
    )

    curves = {}
    if request.delta_buildings > 0 or request.delta_trees > 0 or request.delta_albedo > 0:
        for intervention in ["buildings", "trees", "albedo"]:
            curves[intervention] = generate_intervention_curve(
                features, intervention, app_state["models"], steps=8
            )

    result["intervention_curves"] = curves
    return result


@app.post("/api/whatif-citywide")
async def whatif_citywide(request: CityWideWhatIfRequest):
    """
    Simulate a uniform what-if intervention applied to every block in scope
    (all wards, or the top-N hottest), and report the resulting shift in
    city-wide stats plus per-block deltas for recoloring the map.
    """
    if app_state["df"] is None or app_state["models"] is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    df = app_state["df"]

    if request.scope == "hottest":
        target_ids = {
            bid for bid, _ in sorted(
                app_state["predictions"].items(),
                key=lambda x: x[1]["predicted_lst"],
                reverse=True,
            )[: request.top_n]
        }
        target_df = df[df["block_id"].isin(target_ids)]
    else:
        target_df = df

    if target_df.empty:
        raise HTTPException(status_code=400, detail="No blocks in scope")

    sim_results = simulate_whatif_batch(
        blocks_df=target_df,
        delta_buildings=request.delta_buildings,
        delta_trees=request.delta_trees,
        delta_albedo=request.delta_albedo,
        models=app_state["models"],
    )

    new_predictions = dict(app_state["predictions"])
    block_deltas = []
    for idx, row in target_df.iterrows():
        block_id = row["block_id"]
        sim_row = sim_results.loc[idx]
        new_predictions[block_id] = {
            "predicted_lst": float(sim_row["predicted_lst"]),
            "ci_lower": float(sim_row["ci_lower"]),
            "ci_upper": float(sim_row["ci_upper"]),
            "ci_width": float(sim_row["ci_upper"] - sim_row["ci_lower"]),
        }
        block_deltas.append({
            "block_id": block_id,
            "baseline_lst": float(sim_row["baseline_lst"]),
            "predicted_lst": float(sim_row["predicted_lst"]),
            "delta_temp": float(sim_row["delta_temp"]),
        })

    baseline_city_stats = app_state["city_stats"]
    new_city_stats = build_city_stats(df, new_predictions)

    parts = []
    if request.delta_trees > 0:
        parts.append(f"planting {request.delta_trees} trees")
    if request.delta_buildings > 0:
        parts.append(f"adding {request.delta_buildings} buildings")
    if request.delta_albedo > 0:
        parts.append(f"raising roof albedo by {request.delta_albedo}%")
    scope_str = "every ward" if request.scope == "all" else f"the {request.top_n} hottest wards"
    intervention_str = " and ".join(parts) if parts else "no changes"
    delta_mean = round(new_city_stats["city_mean_lst"] - baseline_city_stats["city_mean_lst"], 2)
    narrative = (
        f"Applying {intervention_str} across {scope_str} shifts the city mean from "
        f"{baseline_city_stats['city_mean_lst']}°C to {new_city_stats['city_mean_lst']}°C "
        f"({'+' if delta_mean >= 0 else ''}{delta_mean}°C)."
    )

    return {
        "scope": request.scope,
        "n_blocks_affected": len(target_df),
        "baseline_city_stats": baseline_city_stats,
        "new_city_stats": new_city_stats,
        "delta_city_mean_lst": delta_mean,
        "block_deltas": block_deltas,
        "narrative_explanation": narrative,
    }


@app.get("/api/city-stats")
async def get_city_stats():
    """
    Returns city-wide UHI statistics including top 5 hottest/coolest blocks,
    mean LST, UHI intensity, category distribution, and feature correlations.
    """
    if app_state["city_stats"] is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    return app_state["city_stats"]


@app.get("/api/model-info")
async def get_model_info():
    """
    Returns model performance metrics, feature importances, and training summary.
    """
    if app_state["models"] is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    metrics = app_state["models"]["metrics"]
    df = app_state["df"]

    return {
        "model_type": "XGBRegressor",
        "tree_method": "hist",
        "architecture": "Multi-threaded, hardware-agnostic inference optimized for AMD EPYC server-class CPUs",
        "hardware_agnostic": True,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "spatial_cv_r2": metrics["spatial_cv_r2"],
        "random_split_r2": metrics.get("random_split_r2"),
        "random_split_mae": metrics.get("random_split_mae"),
        "repeated_cv_r2_mean": metrics.get("repeated_cv_r2_mean"),
        "repeated_cv_r2_std": metrics.get("repeated_cv_r2_std"),
        "n_train": metrics["n_train"],
        "n_test": metrics["n_test"],
        "n_features": metrics["n_features"],
        "best_params": metrics["best_params"],
        "feature_importances": metrics["feature_importance"],
        "training_data_summary": {
            "total_blocks": len(df),
            "lst_mean": round(float(df["lst"].mean()), 1),
            "lst_std": round(float(df["lst"].std()), 1),
            "lst_min": round(float(df["lst"].min()), 1),
            "lst_max": round(float(df["lst"].max()), 1),
            "spatial_extent": {
                "lat_range": [round(float(df["lat"].min()), 4),
                             round(float(df["lat"].max()), 4)],
                "lon_range": [round(float(df["lon"].min()), 4),
                             round(float(df["lon"].max()), 4)],
            },
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
