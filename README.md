# 🌡️ ShadowMap — Delhi Urban Heat Island Simulator

**Predict. Visualize. Simulate.** ShadowMap is a machine-learning-powered map of Delhi that predicts surface temperature block-by-block, explains *why* each area runs hot, and lets you simulate real interventions — more trees, fewer buildings, cooler roofs — to see their impact before you build anything.

![Status](https://img.shields.io/badge/data-real%20satellite%20%2B%20OSM-brightgreen)
![Model](https://img.shields.io/badge/model-XGBoost%20%2B%20Quantile%20Regression-blue)
![Coverage](https://img.shields.io/badge/coverage-95%20Delhi%20wards-orange)

---

## 🔥 What It Does

Delhi's dense, paved neighborhoods run several degrees hotter than its green ones — that's the Urban Heat Island (UHI) effect. ShadowMap makes that effect visible, predictable, and *actionable*:

- 🗺️ **Interactive heat map** — every ward color-coded by surface temperature, with hottest-blocks leaderboard
- 🎯 **Block-level predictions** — surface temp with 80% confidence intervals, not just a single guess
- 🧠 **Explainable AI** — see exactly which features (green cover, building density, road density...) are driving the heat in any given block
- 🌳 **What-if simulator** — add trees, remove buildings, boost roof albedo, and watch predicted temperature shift in real time
- 📊 **City-wide analytics** — mean/max/min LST, UHI intensity distribution, ward rankings

---

## 🛰️ Real Data, Not Guesswork

Early versions of this project ran on synthetic data. **Not anymore.**

| Feature | Source | Resolution |
|---|---|---|
| Land Surface Temperature | Landsat 8/9 (median composite, Apr–Jun) | 30m |
| NDVI (Green Cover) | Landsat 8/9, same composite | 30m |
| Building Density | OpenStreetMap (osmnx) | vector |
| Road Density | OpenStreetMap (osmnx) | vector |
| Building Height | OSM `building:levels` tags | vector |
| Ward Boundaries | OSM admin boundaries | vector |
| Distance to Water | Computed vs. Yamuna River | — |

**Coverage:** 95 real Delhi wards, pulled live via Google Earth Engine + Overpass API — no fabricated numbers.

---

## 🧪 Model — Honest Metrics, No Hand-Waving

XGBoost regression with quantile models for uncertainty bands (P10/P90). Because the dataset is real and modest in size, we report **multiple validation strategies** instead of cherry-picking the flattering one:

| Metric | Score | What it tells you |
|---|---|---|
| **Repeated 5-Fold CV R²** | **0.45** (±0.16) | Most reliable estimate — real predictive signal |
| Random-split R² | 0.37 | Single-split sanity check |
| Spatial CV R² | -1.01 | Hardest test: generalize to *unseen* geography |
| MAE (spatial test) | 1.8°C | Typical prediction error |

> **Why the spatial score looks worse:** Delhi's west and east sides differ by ~2.2°C on average — a real geographic effect, not a model failure. We report it anyway because burying inconvenient numbers isn't science.

**Top predictive features:** Green Cover (26%) → Building Density (13%) → Impervious Surface (12%) → Distance to Water (11%)

A Random Forest comparison model was also benchmarked (R² = 0.48) — kept as reference, not deployed, since it can't produce the confidence intervals the app relies on.

---

## 🏗️ Tech Stack

**Backend:** FastAPI · XGBoost · scikit-learn · GeoPandas · osmnx
**Frontend:** React 18 · Vite · Leaflet · Recharts · TailwindCSS
**Data Pipeline:** Google Earth Engine (Landsat) · Overpass API (OSM) · GDAL/CRS reprojection

---

## 🚀 Quick Start

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` — backend must be running on port 8000 for the map to load.

---

## 📡 API Endpoints

| Endpoint | Returns |
|---|---|
| `GET /api/blocks` | All ward predictions with features |
| `GET /api/block/{block_id}` | Single block detail + feature contributions |
| `GET /api/city-stats` | City-wide temperature statistics |
| `GET /api/model-info` | Full model metrics, feature importances, training summary |

---

## 🔄 Rebuilding the Data Pipeline

Want to refresh with newer satellite passes or extend coverage?

1. `get_wards.py` — pull ward boundaries from OSM
2. `fetch_lst_v3.py` — single-composite Landsat LST + NDVI (avoids date-mismatch noise)
3. `fetch_density_v3.py` — building/road density via osmnx (rate-limit resilient, resumable)
4. `fetch_heights.py` — building heights from OSM tags where available
5. `merge_final_v2.py` — assembles everything into `sample_data.csv`
6. `train_model.py` — trains mean + quantile models, logs all validation metrics

> **Heads up:** Overpass API rate-limits aggressively. Scripts are built to resume from where they left off — just rerun if interrupted.

---

## 🎯 Roadmap

- [ ] Expand ward coverage beyond the current 95 (address the ~6 wards with no OSM building/road data)
- [ ] Multi-season LST composites (currently single Apr–Jun window)
- [ ] Sentinel-2 NDVI (10m) for finer green-cover resolution
- [ ] Real ward names for the "Unknown" fallback cases

---

## 🤝 Contributing

Built by students, for a real city problem. PRs welcome — especially on data coverage, model robustness, or frontend polish.

---

*ShadowMap doesn't just show you where it's hot — it shows you why, and what you can do about it.*