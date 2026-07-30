# ShadowMap - Results

Daytime surface temperature modelling for Delhi NCT. Schema version 2.

**Headline: blocked 5 km cross-validation R2 = 0.759 +/- 0.074, RMSE = 1.50 C.**

---

## 1. What this model does

Predicts **land surface temperature** from Landsat 8/9 optical reflectance
indices on a 6,709-cell grid (~450 x 505 m, 0.218 km2 per cell) covering the
full Delhi National Capital Territory.

- Target: Landsat 8/9 Collection 2 Level-2 thermal band (ST_B10), median
  composite, 1 April - 30 June 2024, per-pixel QA_PIXEL cloud/shadow masked
- Overpass: ~10:30 local time
- Features: albedo, ndbi, mndwi, bsi

**This is surface temperature, not air temperature.** Different physical
quantities. Human heat stress depends on air temperature and humidity.

---

## 2. Validation

### 2.1 Why random-split cross-validation is invalid here

LST is strongly spatially autocorrelated. Measured directly:

| lag | distance | corr E-W | corr N-S |
|---|---|---|---|
| 1 | 0.45 km | 0.870 | 0.860 |
| 2 | 0.90 km | 0.764 | 0.746 |
| 4 | 1.79 km | 0.661 | 0.652 |
| 8 | 3.59 km | 0.546 | 0.573 |
| 16 | 7.18 km | 0.334 | 0.432 |
| 32 | 14.35 km | 0.212 | 0.249 |
| 48 | 21.53 km | **-0.002** | 0.291 |
| 64 | 28.70 km | **-0.317** | 0.220 |

Semivariogram reaches 95.5% of sill at **14.35 km**. Effective independent
sample size is roughly 1478 km2 / 200 km2 = 7, not 6,709.

Adjacent cells correlate at 0.870, so random k-fold places a cell's own
neighbours in training. The model interpolates rather than predicts.

East-west decorrelates then goes **negative** while north-south stays
positive: a two-regime structure visible from geometry alone, before any
feature is introduced.

### 2.2 R2 as a function of block size

Spatially blocked CV, buffered (training cells within one block-width of any
test cell dropped), 4 repeats:

| block size | R2 | interpretation |
|---|---|---|
| random split | 0.8722 | **LEAKY - do not quote** |
| 1 km | 0.8527 | gap-filling between measured cells |
| 2 km | 0.8318 | |
| **5 km** | **0.7592 +/- 0.0739** | **primary metric - new neighbourhood** |
| 10 km | 0.5819 +/- 0.2046 | new district; unstable |

Random-split inflation over the 5 km figure: **+0.113**. No plateau. R2 is a
function of the prediction task, not a single property of the model.

### 2.3 Error in interpretable units

| aggregation | RMSE | note |
|---|---|---|
| mean-of-folds, 5 km | **1.50 C** | quote this |
| pooled out-of-fold | 1.459 C | scored against global variance; higher |
| per-cell OOF (4-fold averaged) | 1.313 C | ensemble effect, NOT single-model error |
| in-sample | 1.039 C | **do not quote** |

MAE 1.127 C. Bias +0.001 C. Target standard deviation 3.056 C.
Out-of-fold: 184 cells (2.7%) exceed 3 C absolute error, 5 cells exceed 5 C.

### 2.4 Prediction intervals

Raw XGBoost P10-P90 quantile intervals covered only **60.3%** of held-out
values against a nominal 80%. A **1.60x width multiplier** is applied in
predict_grid.py. Mean served interval width 4.12 C.

### 2.5 Confidence classes

| class | n | mean abs error | p95 |
|---|---|---|---|
| normal | 5,228 | 0.909 C | 2.228 |
| edge_higher_error | 1,012 | 1.174 C | 2.853 |
| atypical_features | 369 | 1.600 C | 3.589 |
| poorly_predicted | 100 | 3.418 C | 4.293 |

Monotonic - the classes predict error, they are not decorative.

### 2.6 Temporal holdout (independent year)

2024-trained model applied to an independent Apr-Jun 2023 composite, same
pipeline both sides:

| metric | value |
|---|---|
| R2 raw | -0.5611 |
| R2 after removing constant offset | 0.5558 |
| corr(prediction, truth) | **0.8231** |
| bias | +2.63 C |

**Spatial ranking transfers across years. Absolute temperature does not.**
2023 was genuinely cooler and greener pre-monsoon (LST -3.90 C, NDVI +0.197).
Any new season requires recalibration.

### 2.7 Cross-sensor check

Landsat optical indices predicting MODIS daytime LST - different satellite,
different overpass, 1 km resolution:

| test | R2 |
|---|---|
| Landsat optical -> Landsat LST (same sensor) | 0.7694 |
| Landsat optical -> MODIS LST (cross-sensor) | 0.2982 |
| MODIS LST -> Landsat LST (agreement ceiling) | **0.4715** |

0.2982 against a ceiling of 0.4715 is ~63% of the achievable maximum. Model
predictions correlate with MODIS at 0.7237 versus 0.785 for the actual Landsat
measurement - 92% as good a stand-in as the real measurement. The relationship
is physical, not a same-image artifact, though some same-image advantage
remains and both figures are reported.

---

## 3. Findings

### 3.1 Delhi has a daytime surface urban COOL island

Every built-form feature correlates **negatively** with daytime LST:

| feature | corr with day LST |
|---|---|
| ndvi | -0.552 |
| building_density | -0.342 |
| population_density | -0.292 |
| pct_residential | -0.282 |
| road_density | -0.210 |

The five hottest cells in the NCT all have building_density = 0.0. The hottest
(55.39 C) is bare ground in the far west.

Mechanism: pre-monsoon dry fallow soil has near-zero moisture, so no
evaporative cooling and low thermal inertia. It outheats concrete at 10:30
local. Dense built-up areas also cast shadows into the sensor view.

**Consequence: this model must not be used for building-density policy
inference.** It would recommend paving over open land.

### 3.2 The pattern reverses at night

MODIS MYD11A2, ~01:30 local overpass, same date window:

| building density quartile | day 10:30 | night 01:30 |
|---|---|---|
| Q1 least built | 47.09 C | 22.78 C |
| Q2 | 46.48 C | 22.96 C |
| Q3 | 44.46 C | 25.38 C |
| Q4 most built | 43.56 C | 26.01 C |

Every feature flips sign:

| feature | day | night | shift |
|---|---|---|---|
| building_density | -0.342 | **+0.371** | +0.71 |
| road_density | -0.217 | **+0.347** | +0.56 |
| population_density | -0.292 | **+0.496** | +0.79 |
| ndbi | +0.864 | -0.375 | -1.24 |
| albedo | +0.748 | -0.499 | -1.25 |

**Surface urban cool island by day, surface urban heat island by night.**

Reported as an **observation, not a model.** A night prediction model was
attempted and scored R2 0.117 at 5 km blocks - too weak to serve. Cause: MODIS
night data is 1 km, so ~5 grid cells share one pixel, giving roughly 1,500
effective measurements rather than 6,518.

### 3.3 NDBI does not measure built-up area in Delhi

corr(ndbi, building_density) = **-0.323**, negative. By OSM building-density
quartile, mean NDBI runs -0.011 -> -0.027 -> -0.042 -> -0.053.

NDBI uses SWIR, which dry bare soil reflects much like concrete. In a
semi-arid city with extensive fallow land, NDBI tracks **surface dryness**.

It is the model's dominant feature (permutation importance 0.833). Describe it
as a dryness index. Never as urbanisation.

### 3.4 Permutation importance

Blocked folds. Gain-based importance is unreliable here (ndbi-bsi correlate
0.989):

| feature | importance |
|---|---|
| ndbi | 0.8326 |
| albedo | 0.1199 |
| bsi | 0.0672 |
| mndwi | 0.0469 |

bsi carries little independent weight despite r = 0.989 with ndbi; retained on
a paired test (p < 0.05 at 5 km), but the pair is partly redundant.

---

## 4. Negative results

Recorded because they constrain what the model can claim.

### 4.1 OSM and WorldPop features have no daytime predictive skill

| feature set | R2 5 km | R2 10 km |
|---|---|---|
| SAT only (4 spectral) | 0.7605 | 0.5896 |
| **URBAN only** (buildings, roads, population, elevation, dist_to_water) | **-0.0704** | **-0.6241** |
| FULL (10 features) | 0.7714 | 0.5853 |

Urban features alone are **worse than predicting the citywide mean**. Adding
them to the spectral set: +0.012 (p = 0.19) and +0.018 (p = 0.41), not
significant. Confirmed independently on the MODIS daytime target: -0.2331.
building_density permutation importance was **-0.003**, negative.

Effort involved: 415,165 buildings and 701,599 road segments bulk-downloaded
across 6 tiles, plus WorldPop 2020 and OSM land use. Measurably zero
contribution to daytime prediction. Retained in the repo for the night-time
analysis (3.2), where they carry real signal.

### 4.2 Albedo is not usable as a cooling intervention lever

Tested as the one physically causal candidate (white roofs). Failed in two
independent designs:

| design | albedo coefficient | verdict |
|---|---|---|
| cross-sectional, built-up cells only | r = +0.699 | wrong sign |
| partial corr, controlling bsi + ndvi | r = +0.495 | wrong sign |
| **within-cell 2023->2024 change**, controlled | **+16.1 C per unit albedo** | wrong sign |
| within-cell change, built-up only | +21.6 C per unit albedo | wrong sign |

Within-cell differencing cancels building geometry, shadows and land use -
everything fixed about a location. The coefficient stayed positive and grew.

Reason: albedo in this dataset is largely a **moisture reading**. Drying makes
ground simultaneously brighter and hotter, and no statistical control separates
them from a single satellite snapshot.

Binned response was non-monotonic (darkened +4.53 C, unchanged +3.90 C,
brightened +6.03 C). Both extremes warmed more: the signature of disturbance,
not a causal direction.

**/api/whatif therefore returns HTTP 501.** No intervention modelling offered.

### 4.3 Seasonal NDVI features rejected

Wet minus dry season NDVI improved random-split R2 by only +0.003 while
**worsening** spatial R2 in both directions (0.2619 -> 0.1888 and -0.0367 ->
-0.0491). Dry-season NDVI duplicated the existing NDVI feature. Rejected.

### 4.4 Earlier feature gains were leakage artifacts

Gains measured under random-split CV that do not survive blocked CV:

| feature | claimed gain | blocked LOFO delta |
|---|---|---|
| elevation | +0.043 | **+0.011 (removing it helped)** |
| dist_to_water | +0.036 | -0.014 |
| population | +0.024 | -0.004 |
| land-use percentages | +0.005 | +0.003 to +0.005 (harmful) |

The full R2-progression table from the earlier development phase should be
treated as invalid evidence.

### 4.5 Model capacity was not the bottleneck

Depth 2 beat depths 3, 4, 6 and 8 monotonically. Tree count mattered: 600 trees
at lr 0.05 beat 100 at lr 0.1 by +0.031. Signal is near-additive.

---

## 5. Known limitations

1. **Daytime surface temperature only** (~10:30 local). Not air temperature.
2. **Single season** (Apr-Jun 2024 pre-monsoon). Absolute values do not
   transfer across years; ranking does (2.6).
3. **Delhi NCT only.** No transfer evidence to other cities.
4. **ndbi measures dryness, not urbanisation** (3.3).
5. **Partial circularity.** Landsat L2 emissivity retrieval uses NDVI-derived
   fractional vegetation cover, so vegetation state enters production of the
   target. Magnitude is order 1-2 C, not 20 C.
6. **Same-composite prediction.** Predictors and target come from the same
   Landsat composite. Cross-sensor validation gives 0.2982 against a 0.4715
   ceiling (2.7).
7. **Extremes compressed.** Under-predicts the hottest cells by ~1 C,
   over-predicts the coldest by ~0.7 C. Predicted range 36.16-53.06 C versus
   observed 33.59-55.39 C. **The tool understates worst-case hotspots.**
8. **Boundary cells worse.** Errors ~1.3x higher within 1 km of the NCT
   boundary (1.174 C vs 0.909 C).
9. **Anthropogenic heat invisible.** Dense-population and industrial zones are
   under-predicted. Worst cell (840, Okhla area, 77.315 E 28.519 N) reads
   54.10 C observed against 44.97 C predicted, a -9.13 C out-of-fold residual,
   while every spectral index sits near the city median.
10. **No what-if / intervention modelling** (4.2).
11. **OSM building coverage is incomplete.** 2,995 cells (44.6%) have zero
    mapped buildings; informal settlements are systematically under-mapped.
    Does not affect the deployed model, which uses no OSM features, but it
    constrains the night-time analysis.
12. **Night data is 1 km.** Sub-kilometre variation in grid_lst_night.csv is
    interpolation, not measurement.

---

## 6. Appropriate and inappropriate use

**Appropriate**

- Spatial screening and triage: ranking which cells are hottest. Ranking
  transfers across years (corr 0.82) and across sensors.
- Gap-filling LST where cloud cover or the 16-day revisit leaves holes.
- Characterising Delhi's daytime surface thermal structure at ~450 m.

**Inappropriate**

- Individual health or heat-stress advice (surface, not air, temperature).
- Policy inference on building density, green cover or albedo (3.1, 4.2).
- Absolute temperature prediction for another season or year without
  recalibration (2.6).
- Transfer to another city (no evidence).

---

## 7. Reproducing

Pipeline order:

    generate_grid.py            ->  delhi_grid.geojson
    clip_grid.py                ->  delhi_grid_clipped.geojson
    filter_slivers.py           ->  delhi_grid_filtered.geojson  (6,709 cells)
    fetch_grid_lst.py       GEE ->  grid_lst_ndvi.csv            (target)
    step10_fetch_indices.py GEE ->  grid_indices.csv             (4 features)
    aggregate_density.py        ->  grid_density.csv             (context only)
    step13_night_lst.py     GEE ->  grid_lst_night.csv           (observation only)
    build_master.py             ->  grid_master.csv
    train_grid.py               ->  artifacts_grid/*, grid_predictions.csv
    add_oof_columns.py          ->  out-of-fold columns
    test_api.py                 ->  17 HTTP tests
    uvicorn main_grid:app --port 8001

clip_grid.py must query "National Capital Territory of Delhi, India".
Querying "Delhi, India" returns only New Delhi district, ~161 km2 instead of
~1,478 km2.

grid_master.csv is rebuilt by build_master.py from four inputs that all have
surviving producer scripts. The earlier grid_merged_v2..v6 chain had no
producer on disk; it fed only features later dropped, so nothing needed was
lost.

Archived development scripts and their provenance: archive/MANIFEST.json.
