# ShadowMap Repository Cleanup Report

**Date:** 2026-08-07
**Scope:** Full-repository conservative refactor. No feature changes, no UI changes, no model or API contract changes. Every change below was verified against the running application before being committed.

---

## 1. Repository Summary

ShadowMap is an Urban Heat Island prediction system for Delhi, with two genuinely independent, both-legitimate backend systems that had become tangled together by a merge:

- **Ward pipeline** (`backend/main.py`, `data_pipeline.py`, `feature_engineering.py`, `model.py`) — 95 Delhi wards, XGBoost mean + quantile models, a What-If intervention simulator. **This is the system actually deployed to production**: `backend/api/index.py` (the Vercel entrypoint per `vercel.json`) imports it directly, and the entire frontend (`frontend/src/App.jsx`, `frontend/src/api/shadowmap.js`, `WhatIfPanel.jsx`, etc.) calls its exact endpoint shapes (`/api/blocks`, `/api/block/{id}`, `/api/whatif`, `/api/whatif-citywide`, `/api/city-stats`, `/api/model-info`).
- **Grid pipeline** (`backend/main_grid.py`, `predict_grid.py`, `feature_engineering_v2.py`, `train_grid.py`, and their supporting scripts) — a separate, newer, and far more rigorously validated system covering 6,709 grid cells with spatially-blocked cross-validation, documented in `backend/README.md` and `backend/RESULTS.md`. It runs standalone on port 8001, by explicit design (`main_grid.py`'s own docstring: "the ward-level app is left untouched"). It is **not** wired to the deployed frontend or to Vercel, and its What-If endpoint deliberately returns HTTP 501 (albedo/building-density interventions were tested and found not to be a valid causal lever — see `RESULTS.md` §4.2).

Both systems are legitimate and both are kept. The cleanup's main job was untangling accidental damage from the merge that combined them, then removing genuinely dead files around the edges.

**The most important finding:** the currently-committed `main` branch's Vercel deployment was **broken**. `backend/api/index.py` did `from main import app`, but `main.py` (and its three dependencies) had been moved into `backend/archive/ward_pipeline/` during the `grid-schema-v2` merge, while the data those files need (`sample_data.csv`, `delhi_blocks.geojson`, `model_artifacts/`) stayed behind at `backend/` root. This was clearly unintentional: the merge diff never touched `api/index.py`, and the frontend was never updated to speak the grid pipeline's API shape. This has been fixed — see §2.

---

## 2. Critical Fix: Restored the Deployed Backend

`backend/main.py`, `data_pipeline.py`, `feature_engineering.py`, and `model.py` were moved back from `backend/archive/ward_pipeline/` to `backend/` root.

**Why this was safe, not a redesign:** every file the restored `main.py` needs (`sample_data.csv`, `delhi_blocks.geojson`, `backend/model_artifacts/*`) was already sitting at `backend/` root, untouched by the merge — only the *code* had been archived. Restoring the code is the minimal change that makes `api/index.py`'s existing, unmodified import resolve, and it exactly matches every endpoint the frontend already calls. Nothing about the API contract, the model, or the frontend changed.

**Verification performed:**
- `main.load_or_train()` loads existing `model_artifacts/` (no retraining) and serves 95 blocks, mean LST 43.91°C.
- Simulated the actual Vercel entrypoint (`api/index.py`) with a `TestClient` and exercised every endpoint the frontend calls: `/api/blocks` (200, 95 features), `/api/block/{id}` (200), `/api/city-stats` (200), `/api/whatif` (200), `/api/whatif-citywide` (200), `/api/model-info` (200).
- `main_grid.py` (the untouched standalone system) still loads 6,709 cells and its own 17-test HTTP suite (`test_api.py`) passes 17/17 against a live server on port 8001.

`main_grid.py` was **not** touched or repurposed — it remains the separate research pipeline exactly as documented.

---

## 3. Files Archived

Nothing was force-deleted except compiled bytecode (§4). Everything else with unclear future value was moved into `backend/archive/`, organized by *why* it's dead weight, with full rationale recorded in `backend/archive/MANIFEST.json`:

| Destination | Files | Why archived, not deleted |
|---|---|---|
| `archive/orphaned_data/` | `grid_elevation.csv`, `grid_landuse.csv`, `grid_merged.csv`, `grid_merged_coords.csv`, `grid_merged_v2/v3/v4/v6.csv`, `grid_ndvi_seasonal.csv`, `grid_population.csv`, `grid_water_dist.csv`, `delhi_water.geojson` | Verified **zero references anywhere** in the repo — no producer, no consumer, in active code or in any archive folder. Likely remnants of an abandoned 9-feature grid schema (elevation/water-distance/landuse/population were tested and dropped — see `RESULTS.md` §4.4) and the pre-`build_master.py` merge chain. Kept because the underlying satellite/OSM/WorldPop data would be costly to reacquire if a future schema revisits these features. |
| `archive/analysis/` | `grid_2023.csv`, `grid_2023_matched.csv`, `grid_2024_matched.csv`, `grid_merged_v5.csv`, `patch_train_headline.py` | The three CSVs are inputs/outputs of the temporal-holdout analysis scripts already archived there (`step19`–`step21`, `step23`); `grid_merged_v5.csv` is what `build_master.py`'s own docstring calls out as the predecessor with "no surviving producer script." `patch_train_headline.py` is a one-time script that already patched `train_grid.py` — confirmed applied by grepping the patched fields into both `train_grid.py` and `artifacts_grid/model_meta.json`. Re-running it is a safe no-op (every patch target string is already gone). |
| `archive/superseded/` | `real_density.csv`, `real_heights.csv`, `real_lst_ndvi_v2.csv`, `real_lst_ndvi_v3.csv`, `delhi_wards_valid.geojson`, `rf_comparison_metrics.json` | Ward-era intermediate data, moved alongside the already-archived scripts that produce/consume them. |
| `archive/ward_pipeline/` | `real_lst_ndvi.csv`, `delhi_wards.geojson`, `uhi_model.onnx` | Moved alongside their producer scripts (`fetch_real_data.py`, `get_wards.py`, `export_onnx.py`) which already lived here. |
| `archive/legacy_artifacts/` (new) | `uhi_model.pkl`, `uhi_model_lower.pkl`, `uhi_model_upper.pkl` | Pickle-format duplicates of the ward model. **Zero scripts anywhere** (active or archived) reference these filenames; `model.py` exclusively loads the `.json` equivalents (its own comment: "no pickle"). Almost certainly relics from before the switch to the portable JSON save format. |

`backend/README.md`'s "Reproducing" pipeline order and `RESULTS.md`'s claims (e.g. that `grid_density.csv`/`grid_lst_night.csv` are context/observation only, never model features) were independently re-verified by grepping `feature_engineering_v2.py`'s `FEATURE_COLUMNS` and `build_master.py`'s merge logic — both confirmed accurate.

---

## 4. Files Deleted

Only one category was hard-deleted, and only because it's compiled, auto-regenerating build output that was never supposed to be tracked:

- **`backend/__pycache__/*.pyc`** (4 files) — Python bytecode cache. `backend/.gitignore` already listed `__pycache__/` and `*.py[cod]`; these files were tracked before that rule existed (or before it was enforced) and were simply never untracked. Removing them from git has no effect on any running system — they regenerate automatically on next `import`.

No source code, dataset, or documentation file was permanently deleted. Everything else is one `git mv` away from being restored (see git history) or already living under `archive/`.

---

## 5. Files Refactored

All refactors were verified to produce **identical output** before and after (same 95-block predictions, same 6,709-cell predictions, same 17/17 test pass) — every change below is subtraction of unreachable code, not behavior change.

| File | Change |
|---|---|
| `backend/main.py` | Removed unused imports `ALL_FEATURES`, `compute_interaction_features`, `get_feature_importance` (imported, never referenced in the file body). |
| `backend/model.py` | Removed unused imports `typing.Optional`, `joblib`. |
| `backend/data_pipeline.py` | Removed the entire dead `try: import geopandas / shapely ... except ImportError: HAS_GEO = False` block — `HAS_GEO` was set but never read anywhere in the codebase. |
| `backend/train_grid.py` | Removed unused `import pandas as pd`. |
| `backend/bulk_download_osm.py` | Removed unused `import geopandas as gpd`. |
| `backend/step10_fetch_indices.py`, `step13_night_lst.py` | Removed unused `numpy as np`. |
| `backend/add_oof_columns.py` | Removed unused `from scipy.spatial import cKDTree`; fixed a stale internal docstring that still called the file `step30_oof_predictions.py` (its name before a rename that predates this cleanup — the old name was also still floating around in `MANIFEST.json`, fixed in the same pass). |
| `frontend/src/components/Icons.jsx` | Removed `TargetIcon` and `CompareIcon` — defined and exported but never imported anywhere in `frontend/src` (verified by repo-wide grep). |
| `backend/DEPLOYMENT_AMD.md` | Pre-existing doc drift, not caused by this cleanup: referenced `train_model.py`/`export_onnx.py`/`benchmark.py` at `backend/` root (all archived) and a `model.XGB_DEVICE` symbol that doesn't exist anywhere in the code (git history shows GPU support was added, then explicitly removed in favor of CPU-only `tree_method="hist"`, and this doc was never updated). Added an accuracy note at the top, corrected the script paths to their real archived locations, and replaced the fabricated verification snippet with one that reflects what `model.py` actually does. |
| `README.md` (root) | "Rebuilding the Data Pipeline" listed archived script names with no path (`get_wards.py`, `train_model.py`, etc.) as if runnable from `backend/` root. Added the correct `archive/...` paths. |
| `backend/README.md` | Still claimed `main.py` was archived and only `main_grid.py` was live — stale as of the fix in §2. Corrected to state that `main.py` is the actually-deployed app and `main_grid.py` is the standalone research system. |
| `backend/run_training.bat` | Pointed at `backend/train_model.py`, which no longer exists at that path (only in `archive/ward_pipeline/`). Fixed. |
| `backend/archive/MANIFEST.json` | Documented every file moved in §3, fixed a stale reference to a script called `step30_oof_predictions.py` that no longer exists under that name (renamed to `add_oof_columns.py` before this cleanup, but the manifest was never updated), and added a note explaining the §2 restoration. |
| `.gitignore` (root) | Added `dist/` — Netlify builds the frontend fresh from source (`netlify.toml`: `npm ci && vite build` → publish `dist`), so it should never be committed. Surfaced as an untracked directory during local build verification. |

---

## 6. Files Merged

**None.** No two implementations were found doing the same job that needed consolidating into one. The ward pipeline and grid pipeline look superficially similar (both predict Delhi surface temperature) but are genuinely different systems serving different purposes — one is the deployed product, the other is a separate, standalone, more rigorously-validated research track that was always designed to run independently ("the ward-level app is left untouched," `main_grid.py`'s own words). Merging them would be a functional change (rewriting the frontend's API contract, or replacing a working deployed system with one whose What-If endpoint is intentionally disabled) — explicitly out of scope per the "no API contract changes" instruction. Where true duplicates existed (`grid_merged_v2..v6.csv`, the `.pkl` model artifacts), they were archived, not merged, since there was nothing to merge — the superseding version already existed and was already in use.

---

## 7. Dependency Map (Final State)

```
backend/
├── main.py, data_pipeline.py, feature_engineering.py, model.py   [ward pipeline — DEPLOYED]
├── sample_data.csv, delhi_blocks.geojson, model_artifacts/*      [ward pipeline data]
├── api/index.py                                                  [Vercel entrypoint -> main.py]
│
├── main_grid.py, predict_grid.py, feature_engineering_v2.py,
│   train_grid.py, build_master.py, generate_grid.py,
│   clip_grid.py, filter_slivers.py, aggregate_density.py,
│   bulk_download_osm.py, merge_osm_tiles.py, fetch_grid_lst.py,
│   step10_fetch_indices.py, step13_night_lst.py,
│   add_oof_columns.py, test_api.py                                [grid pipeline — standalone, port 8001]
├── grid_density.csv, grid_indices.csv, grid_lst_ndvi.csv,
│   grid_lst_night.csv, grid_master.csv, grid_predictions.csv,
│   delhi_grid*.geojson, artifacts_grid/*                          [grid pipeline data]
│
├── requirements.txt, vercel.json, run_training.bat, .python-version, .gitattributes
├── README.md, RESULTS.md, DEPLOYMENT_AMD.md
│
└── archive/                                                       [nothing here runs in production]
    ├── MANIFEST.json          full rationale for every archived file
    ├── analysis/              44 numbered validation/diagnostic scripts + their data
    ├── superseded/            ward-era fetch/merge iterations + their data
    ├── ward_pipeline/         non-runtime ward scripts (train_model.py, export_onnx.py, benchmark.py, get_wards.py, fetch_real_data.py, install_dependencies.py) + their outputs
    ├── orphaned_data/         zero-reference datasets, kept for potential future reuse
    └── legacy_artifacts/      pre-JSON-format pickle model duplicates

frontend/
└── src/
    ├── App.jsx, main.jsx, index.css
    ├── api/shadowmap.js                    [calls ward-pipeline endpoints only]
    └── components/                         [all 8 components in active use, no dead code]
```

---

## 8. Technical Debt Removed

- **Broken production deployment** (the headline issue — see §2).
- **Committed compiled bytecode** that should never have been tracked.
- **11+ orphaned CSV/GeoJSON files** with zero references anywhere in the repository, cluttering `backend/` root and making it unclear which datasets the live model actually depends on.
- **Duplicate model artifacts** (3 pickle files) shadowing the actually-used JSON format.
- **Dead imports and a dead conditional-import block** across 8 backend scripts.
- **Two unused frontend icon components.**
- **Stale documentation** pointing at files that had moved or never existed (`DEPLOYMENT_AMD.md`, root `README.md`, `backend/README.md`, `run_training.bat`, `archive/MANIFEST.json`) — all of it dating from the same merge that broke the deployment, now consistent with the actual repository layout.
- **An uncommittable build artifact** (`frontend/dist/`) that wasn't yet gitignored.

---

## 9. Risks

- **`backend/DEPLOYMENT_AMD.md`'s ROCm/AMD-GPU content** describes a device-selection architecture (`device="cpu"`/`device="cuda"`) that does not exist in the current `model.py` (which hardcodes `tree_method="hist"`, CPU-only). Rather than deleting a document that may still have value as a reference for reintroducing GPU support, I added an accuracy note and fixed the concrete broken paths, but left the aspirational architecture content as-is. If this document should be reduced to only what's true today, that's a content decision I didn't make unilaterally.
- **Archived scripts' internal relative paths were not verified to still resolve if re-run.** Several archived scripts (e.g. in `archive/superseded/`) were originally written assuming they'd be run with the working directory at `backend/` root; moving their input/output CSVs into the same archive subfolder means re-running them with the old working-directory convention would no longer find those files. This does not affect anything that actually runs in production — `MANIFEST.json` already documented (before this cleanup) that nothing in `archive/` is needed to run or retrain the model — but it means the archive is a provenance record, not a guaranteed-runnable pipeline.
- **The `.pkl` legacy model artifacts and the fully-orphaned datasets** were archived rather than deleted out of caution, per the task's conservative-by-default instruction. Both categories have zero references anywhere in the repository (including archived code), so outright deletion would also have been defensible; I erred toward the reversible option.

---

## 10. Manual Review Required

1. **`backend/DEPLOYMENT_AMD.md`'s scope** — decide whether this document should stay as an aspirational GPU-deployment reference (current state, with the accuracy note added) or be trimmed/removed since no GPU code path currently exists.
2. **`archive/orphaned_data/` and `archive/legacy_artifacts/`** — these 12 files have zero references anywhere and are pure historical/provenance value. If disk footprint matters, they're safe to delete outright (nothing depends on them); I left them archived rather than deleting per the task's default-to-caution instruction.
3. **The two-pipeline situation itself** — the grid pipeline (`main_grid.py`) is scientifically more rigorous and honest about its limitations than the deployed ward pipeline, but it isn't wired to the frontend and its What-If endpoint is intentionally disabled. Whether to eventually migrate the frontend to the grid API is a product decision well outside this cleanup's scope (it would change the API contract and UI, both explicitly off-limits here) — flagging it because it's the kind of thing a new engineer would otherwise have to rediscover from scratch.

---

## 11. Verification Performed

- ✅ `main.load_or_train()` — loads existing artifacts, 95 blocks, mean LST 43.91°C, unchanged before/after all edits.
- ✅ `api/index.py` simulated as the real Vercel entrypoint via `TestClient` — every frontend-called endpoint returns 200 (`/api/blocks`, `/api/block/{id}`, `/api/city-stats`, `/api/whatif`, `/api/whatif-citywide`, `/api/model-info`).
- ✅ `main_grid.load_state()` — 6,709 cells, unchanged before/after all edits.
- ✅ `test_api.py` run against a live `main_grid:app` server on port 8001 — **17/17 tests pass**.
- ✅ `npm run build` (Vite) — 1,281 modules transformed, no broken imports, build succeeds.
- ✅ `backend/archive/MANIFEST.json` — valid JSON after edits.
- ✅ Working tree clean; every change is committed in small, individually-reviewable commits (see `git log`).
