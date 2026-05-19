# ShadowMap — Urban Heat Island Prediction & What-If Simulator

**Predict street-block level Urban Heat Island intensity across Delhi and simulate the impact of urban interventions in real-time.** . 

Built for urban planners, municipal officers, and smart city policymakers  .

----

## Architecture

````
┌──────────────────────────────────────────────────────────────┐
│                     REACT FRONTEND (Vite)                    │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │CityStats │  │  MapView     │  │ HeatPanel/WhatIfPanel│    │
│  │ Donut    │  │  Leaflet     │  │ Sliders, Charts     │    │
│  │ Rankings │  │  Choropleth  │  │ Confidence Intervals│    │
│  └──────────┘  └──────────────┘  └─────────────────────┘    │
│                         │ fetch /api/*                        │
└─────────────────────────┼────────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────┐
│                   FASTAPI BACKEND                            │
│  ┌──────────────────────┼─────────────────────────────┐      │
│  │  /api/blocks   /api/block/{id}   /api/whatif       │      │
│  │  /api/city-stats     /api/model-info               │      │
│  └──────────────────────┼─────────────────────────────┘      │
│                         │                                     │
│  ┌─────────────┐  ┌────┴────────┐  ┌──────────────────┐     │
│  │ data_       │  │  model.py   │  │  feature_         │     │
│  │ pipeline.py │  │  GBR+QR     │  │  engineering.py   │     │
│  │ OSM/Synth   │  │  What-If    │  │  StandardScaler   │     │
│  └─────────────┘  └─────────────┘  └──────────────────┘     │
│                         │                                     │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  model_artifacts/  (uhi_model.pkl, scaler.pkl, etc) │     │
│  │  sample_data.csv   delhi_blocks.geojson             │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Backend Setup

```bash
cd shadowmap/backend

# Install dependencies
pip install -r requirements.txt

# Train the model (generates data + trains GBR + saves artifacts)
python train_model.py

# Start the API server
uvicorn main:app --reload --port 8000
```

> **Note:** If you skip `train_model.py`, the server will auto-train on first startup.

### 2. Frontend Setup

```bash
cd shadowmap/frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Free Deployment (Vercel + Netlify)

This repo is now wired for:
- Backend (FastAPI + ML): **Vercel**
- Frontend (Vite React): **Netlify**

### Why this split?

The frontend is static and fits Netlify well. The backend is exposed as a Vercel Python function that serves the FastAPI app under `/api/*`.

### 1. Deploy Backend on Vercel

In Vercel, create a new project from this repo and set:

- **Root Directory:** `shadowmap/backend`
- **Framework Preset:** `Other`
- **Build Command:** leave empty
- **Output Directory:** leave empty
- **Install Command:** `pip install -r requirements.txt`

Add these environment variables:

- `PYTHON_VERSION=3.11`
- `CORS_ORIGINS=https://YOUR-NETLIFY-SITE.netlify.app`

Important: the backend `requirements.txt` is now trimmed to runtime-only packages. The ONNX/export scripts are still in the repo for local use, but they are not part of the Vercel deploy bundle.

If you already know the frontend URL, add it here now. If not, deploy the frontend first, then come back and update this value.

The backend routes are served from:

- `https://YOUR-VERCEL-BACKEND.vercel.app/api/blocks`
- `https://YOUR-VERCEL-BACKEND.vercel.app/api/block/{block_id}`
- `https://YOUR-VERCEL-BACKEND.vercel.app/api/whatif`
- `https://YOUR-VERCEL-BACKEND.vercel.app/api/city-stats`
- `https://YOUR-VERCEL-BACKEND.vercel.app/api/model-info`

### 2. Deploy Frontend on Netlify

In Netlify, create a new site from this repo and set:

- **Base directory:** `shadowmap/frontend`
- **Build command:** `npm run build`
- **Publish directory:** `dist`

Add this environment variable:

- `VITE_API_BASE_URL=https://YOUR-VERCEL-BACKEND.vercel.app/api`

The frontend code already falls back to `/api` for local development, but in Netlify you should set the full Vercel API URL above.

### 3. Exact Values To Enter

Use these exact fields:

Backend on Vercel:

- Root Directory: `shadowmap/backend`
- Install Command: `pip install -r requirements.txt`
- Build Command: empty
- Output Directory: empty
- Environment Variables:
    - `PYTHON_VERSION=3.11`
    - `CORS_ORIGINS=https://YOUR-NETLIFY-SITE.netlify.app`

Frontend on Netlify:

- Base Directory: `shadowmap/frontend`
- Build Command: `npm run build`
- Publish Directory: `dist`
- Environment Variables:
    - `VITE_API_BASE_URL=https://YOUR-VERCEL-BACKEND.vercel.app/api`

### 4. Final Step

After both deploys finish, make sure the backend `CORS_ORIGINS` contains the final Netlify URL exactly, then redeploy the backend once.

The API is ready when this URL returns JSON:

`https://YOUR-VERCEL-BACKEND.vercel.app/api/model-info`

---

## Data Sources

| Source | Data | URL |
|--------|------|-----|
| OpenStreetMap | Building footprints, road network | https://www.openstreetmap.org |
| Sentinel-3 SLSTR | Land Surface Temperature (LST) | https://scihub.copernicus.eu |
| Landsat 8/9 | NDVI (vegetation index) | https://earthexplorer.usgs.gov |
| osmnx | Automated OSM data download | https://github.com/gboeing/osmnx |

### Downloading Real Data

1. **Sentinel-3 LST:** Register at Copernicus Open Access Hub, search for "SL_2_LST" over Delhi (May-June), download the NetCDF, and place the resampled GeoTIFF in `backend/`.
2. **Landsat NDVI:** Use Google Earth Engine or USGS EarthExplorer to get cloud-free Landsat 8/9 imagery for Delhi. Compute NDVI = (NIR - Red) / (NIR + Red), export as GeoTIFF.
3. **Fallback:** The app works end-to-end with synthetic data that realistically models Delhi's urban landscape (~300 blocks).

---

## How to Retrain the Model

```bash
cd shadowmap/backend

# Option 1: With existing data
python train_model.py

# Option 2: Regenerate synthetic data first
python -c "from data_pipeline import generate_synthetic_data; df = generate_synthetic_data(500); df.to_csv('sample_data.csv', index=False)"
python train_model.py
```

Artifacts are saved to `backend/model_artifacts/`:
- `uhi_model.pkl` — Mean prediction model
- `uhi_model_lower.pkl` — 10th percentile quantile model
- `uhi_model_upper.pkl` — 90th percentile quantile model
- `scaler.pkl` — StandardScaler for feature normalization
- `metrics.json` — Performance metrics
- `feature_importance.json` — Feature importance percentages

---

## Methodology

### Urban Heat Islands (UHI)

Urban Heat Islands are metropolitan areas significantly warmer than surrounding rural areas due to human activities. Factors include:
- **Building density:** Concrete and steel absorb and re-emit heat
- **Vegetation loss:** Reduced evapotranspiration cooling
- **Impervious surfaces:** Roads and parking lots store thermal energy
- **Waste heat:** Air conditioning, vehicles, industrial processes

### Why Gradient Boosting Regressor (GBR)?

GBR was chosen for several reasons:
1. **Non-linear relationships:** UHI relationships are inherently non-linear (e.g., vegetation cooling has diminishing returns)
2. **Feature interactions:** GBR naturally captures interactions between building density, green cover, and road density
3. **Robustness:** Handles missing values and outliers well
4. **Interpretability:** Feature importance scores provide actionable insights for urban planners
5. **Quantile regression:** GBR supports quantile loss for confidence interval estimation

### Spatial Cross-Validation

Standard random cross-validation leaks spatial information between train and test sets because adjacent blocks have correlated features and temperatures. Our approach:
1. **Spatial split:** Western Delhi (training) vs. Eastern Delhi (testing)
2. **Spatial CV:** Blocks sorted by longitude, split into 5 folds without shuffling
3. This ensures the model generalizes to unseen geographic areas

### Quantile Regression for Confidence Intervals

Instead of just a mean prediction, we train three models:
- **Mean model:** `GradientBoostingRegressor(loss='squared_error')` — point estimate
- **Lower bound:** `GradientBoostingRegressor(loss='quantile', alpha=0.1)` — 10th percentile
- **Upper bound:** `GradientBoostingRegressor(loss='quantile', alpha=0.9)` — 90th percentile

This gives an 80% prediction interval, showing uncertainty in each block's temperature prediction.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/blocks` | GeoJSON of all blocks with predictions |
| GET | `/api/block/{block_id}` | Detailed block info with feature contributions |
| POST | `/api/whatif` | What-if simulation with intervention parameters |
| GET | `/api/city-stats` | City-wide statistics and rankings |
| GET | `/api/model-info` | Model performance metrics and feature importances |

---

## Features

- 🗺️ **Interactive Choropleth Map** — Blue-to-red heat scale across 300 Delhi blocks
- 🌡️ **Block-Level Predictions** — Surface temperature with 80% confidence intervals
- 🔬 **What-If Simulator** — Add buildings, plant trees, or change roof albedo
- 📊 **Feature Contributions** — See what drives each block's temperature
- 🏆 **City Rankings** — Top 5 hottest and coolest blocks at a glance
- 📈 **Intervention Curves** — Visualize how temperature changes with each intervention
- 🌙 **Dark Theme** — Premium dark UI optimized for desktop

---

## Screenshots

*Screenshots will be added after deployment.*

---

## Team

*Team information placeholder.*

---

## License

This project is for educational and research purposes.
