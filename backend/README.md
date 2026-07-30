# ShadowMap

Surface temperature modelling for Delhi NCT from Landsat satellite imagery.

**Blocked 5 km cross-validation R2 = 0.759 +/- 0.074, RMSE = 1.50 C**

Predicts daytime land surface temperature across 6,709 grid cells (~450 x 505 m,
0.218 km2) covering the full Delhi National Capital Territory, from four
spectral reflectance indices.

> **This is surface temperature, not air temperature.** They are different
> physical quantities. Human heat stress depends on air temperature and
> humidity. Do not use this model for individual health advice.

Full methodology, findings and limitations: **[RESULTS.md](RESULTS.md)**

---

## What it does

| | |
|---|---|
| Target | Landsat 8/9 C2 L2 thermal band, Apr-Jun 2024 median, QA_PIXEL masked |
| Overpass | ~10:30 local time |
| Features | albedo, ndbi, mndwi, bsi |
| Grid | 6,709 cells, EPSG:32643 |
| Model | XGBoost, depth 2, 600 trees, plus P10/P90 quantile models |
| Validation | 5 km spatially blocked CV with buffer zones |

Every prediction ships with a calibrated interval and a confidence class.

## What it does NOT do

- **No what-if / intervention modelling.** `/api/whatif` returns HTTP 501.
  Albedo was tested as the one physically causal lever and failed in two
  independent designs (RESULTS.md 4.2).
- **No policy inference on building density or green cover.** Daytime Delhi
  shows a surface urban *cool* island; the model would recommend paving over
  open land (RESULTS.md 3.1).
- **No transfer to other cities or seasons.** Absolute values do not transfer
  across years; spatial ranking does (RESULTS.md 2.6).

---

## Two headline findings

**1. Delhi runs a surface urban COOL island by day.** All five hottest cells in
the NCT have zero mapped buildings. Dry pre-monsoon fallow soil outheats
concrete at 10:30 local, having no moisture to evaporate and low thermal
inertia.

**2. The pattern reverses at night.**

| building density | day 10:30 | night 01:30 |
|---|---|---|
| least built quartile | 47.09 C | 22.78 C |
| most built quartile | 43.56 C | 26.01 C |

Cool island by day, heat island by night. Reported as an observation only - a
night prediction model scored R2 0.117 and was not deployed, because MODIS
night data is 1 km while the grid is ~450 m.

---

## Honest evaluation

Land surface temperature is strongly spatially autocorrelated: adjacent 450 m
cells correlate at **0.870**, and the semivariogram range is **14.35 km**.
Random k-fold cross-validation therefore places a cell's own neighbours in the
training set and measures interpolation, not prediction.

| block size | R2 | task |
|---|---|---|
| random split | 0.8722 | **leakage - do not quote** |
| 1 km | 0.8527 | gap-filling between measured cells |
| 5 km | **0.7592 +/- 0.0739** | **primary - new neighbourhood** |
| 10 km | 0.5819 +/- 0.2046 | new district; unstable |

Random-split inflation: **+0.113**. There is no plateau - R2 is a property of
the prediction task, not of the model alone.

Raw quantile intervals covered only 60.3% against a nominal 80%, so a 1.60x
width multiplier is applied before serving.

Cross-sensor check against MODIS (an independent satellite) gives R2 0.2982
against a sensor-agreement ceiling of 0.4715, confirming the relationship is
physical rather than a same-image artifact.

## Notable negative results

Recorded because they constrain what can be claimed:

- **OSM and WorldPop features have no daytime predictive skill.** 415,165
  buildings and 701,599 road segments, plus population and land use: urban
  features alone score **-0.07** at 5 km blocks, worse than predicting the
  citywide mean. `building_density` permutation importance is **-0.003**.
- **NDBI does not measure built-up area here.** It correlates **-0.323** with
  OSM building density. It tracks surface *dryness*, since dry bare soil
  reflects SWIR much like concrete. It is the model's dominant feature.
- **Albedo is not a usable cooling lever.** Within-cell 2023-2024 change
  analysis gave **+16.1 C per unit albedo** - the wrong sign - because albedo
  in this dataset largely reads soil moisture.
- **Earlier feature gains were leakage.** Elevation, distance-to-water,
  population and land-use gains measured under random-split CV do not survive
  blocked CV.

---

## Setup
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
Google Earth Engine is required only to regenerate data, not to run the model:
## Run
uvicorn main_grid:app --port 8001
python test_api.py


The legacy 95-ward app is preserved at `archive/ward_pipeline/main.py` and still
runs on port 8000. Its reported R2 of 0.4523 came from repeated random-split CV
and is inflated by spatial autocorrelation.

---

## API

| endpoint | returns |
|---|---|
| `GET /api/health` | status, schema version, cell count |
| `GET /api/model-info` | metrics, block-size curve, leakage warning, importance |
| `GET /api/limitations` | 12 documented limitations plus the diurnal finding |
| `GET /api/cells` | all cells, or bbox-filtered via min_lon/max_lon/min_lat/max_lat |
| `GET /api/cell/{id}` | one cell: prediction, interval, features, confidence class |
| `POST /api/predict` | prediction from arbitrary feature values |
| `GET/POST /api/whatif` | **501** with the reason it is unsupported |
| `GET /api/city-stats` | distributions, out-of-fold residuals, class counts |

Confidence classes, with measured out-of-fold error:

| class | n | mean abs error |
|---|---|---|
| normal | 5,228 | 0.909 C |
| edge_higher_error | 1,012 | 1.174 C |
| atypical_features | 369 | 1.600 C |
| poorly_predicted | 100 | 3.418 C |

## Repository layout
backend/
feature_engineering_v2.py schema v2: features, validation, bounds
build_master.py builds grid_master.csv from reproducible inputs
train_grid.py canonical trainer, writes artifacts_grid/
add_oof_columns.py adds out-of-fold columns to predictions
predict_grid.py production predictor, interval calibration
main_grid.py FastAPI app (port 8001)
test_api.py 17 live HTTP tests
generate_grid.py grid construction
clip_grid.py clip to NCT boundary
filter_slivers.py drop <50% coverage cells
bulk_download_osm.py OSM tiles (night analysis only)
merge_osm_tiles.py
aggregate_density.py building/road density (context only)
fetch_grid_lst.py GEE: LST target
step10_fetch_indices.py GEE: the 4 model features
step13_night_lst.py GEE: MODIS night LST (observation only)
artifacts_grid/ models + model_meta.json
archive/ 45 archived scripts, see MANIFEST.json
RESULTS.md full methodology and findings
## Caveat on reproducing the grid

`clip_grid.py` must query `"National Capital Territory of Delhi, India"`.
Querying `"Delhi, India"` returns only New Delhi district - about 161 km2
instead of the full 1,478 km2.

Three OSM files are gitignored for size: `delhi_all_buildings.geojson`
(4.5 GB), `delhi_all_roads.geojson` (375 MB), `delhi_landuse.geojson` (34 MB).
Regenerate with `bulk_download_osm.py` and `merge_osm_tiles.py`. None is needed
to run or retrain the deployed model.
