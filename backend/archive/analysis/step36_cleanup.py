import os, io, shutil, json

DRY = os.environ.get('DRY', '1') == '1'
print('MODE:', 'DRY RUN (no files moved)' if DRY else 'LIVE (moving files)')
print()

CORE = ['feature_engineering_v2.py', 'predict_grid.py', 'train_grid.py',
        'main_grid.py', 'build_master.py']

PIPELINE = ['generate_grid.py', 'clip_grid.py', 'filter_slivers.py',
            'bulk_download_osm.py', 'merge_osm_tiles.py', 'aggregate_density.py',
            'fetch_grid_lst.py', 'step10_fetch_indices.py', 'step13_night_lst.py',
            'add_oof_columns.py', 'test_api.py']

WARD = ['data_pipeline.py', 'feature_engineering.py', 'model.py', 'main.py',
        'train_model.py', 'get_wards.py', 'fetch_real_data.py', 'benchmark.py',
        'export_onnx.py', 'install_dependencies.py']

ANALYSIS = ['step2_autocorr.py', 'step4_master.py', 'step5_cv.py', 'step6_baselines.py',
            'step7_features.py', 'step8_capacity.py', 'step9_nested.py',
            'step11_test_indices.py', 'step12_ablation.py', 'step14_night_model.py',
            'step15_crosssensor.py', 'step16_final_model.py', 'step17_diagnose.py',
            'step19_temporal.py', 'step20_composite_check.py', 'step21_rebuild.py',
            'step22_albedo_gate.py', 'step23_change_test.py', 'step33_inventory.py',
            'fix_headline.py', 'patch_main_grid.py', 'patch_centroid.py']

SUPERSEDED = {
    'fetch_density.py': 'ward-era Overpass density fetch, superseded by fetch_density_v3 then by aggregate_density.py',
    'fetch_density_v2.py': 'ward-era retry variant of fetch_density.py',
    'fetch_density_v3.py': 'ward-era final density fetch; grid pipeline uses aggregate_density.py',
    'fetch_heights.py': 'ward-era OSM building:levels fetch; height dropped, no predictive skill',
    'fetch_lst_v2.py': 'ward-era per-scene LST fetch; abandoned for date-mismatch noise',
    'fetch_lst_v3.py': 'ward-era LST fetch, superseded by fetch_grid_lst.py',
    'merge_final.py': 'ward-era sample_data.csv builder, v1',
    'merge_final_v2.py': 'ward-era sample_data.csv builder, v2',
    'mop_up.py': 'ward-era gap filler for rate-limited Overpass cells',
    'refetch_failed.py': 'ward-era retry harness for failed Overpass cells',
    'improve_model.py': 'ward-era hyperparameter experiment, superseded by train_grid.py',
    'train_rf_comparison.py': 'ward-era RandomForest benchmark (R2 0.4829); not adopted, no quantile intervals',
    'test_gee.py': 'three-line GEE connectivity check',
}

MOVES = {}
for f in ANALYSIS:
    MOVES[f] = 'archive/analysis'
for f in SUPERSEDED:
    MOVES[f] = 'archive/superseded'
for f in WARD:
    MOVES[f] = 'archive/ward_pipeline'

present = {f for f in os.listdir('.') if f.endswith('.py')}
missing = [f for f in list(CORE) + PIPELINE if f not in present]
if missing:
    raise SystemExit('CORE/PIPELINE file missing from disk: %s' % missing)

unclassified = present - set(CORE) - set(PIPELINE) - set(MOVES) - {'step36_cleanup.py'}
print('=== PLAN ===')
print('keep in place   : %d core + %d pipeline' % (len(CORE), len(PIPELINE)))
for f in CORE:
    print('   CORE      %s' % f)
for f in PIPELINE:
    print('   PIPELINE  %s' % f)
print()
counts = {}
for f, dest in sorted(MOVES.items()):
    if f in present:
        counts[dest] = counts.get(dest, 0) + 1
for dest, n in sorted(counts.items()):
    print('move %-26s %d files' % (dest, n))
if unclassified:
    print()
    print('UNCLASSIFIED (left in place, review manually):')
    for f in sorted(unclassified):
        print('   %s' % f)

if not DRY:
    for dest in set(MOVES.values()):
        os.makedirs(dest, exist_ok=True)
    moved = []
    for f, dest in sorted(MOVES.items()):
        if f in present:
            shutil.move(f, os.path.join(dest, f))
            moved.append((f, dest))
    print()
    print('=== MOVED %d FILES ===' % len(moved))

    manifest = {
        'purpose': ('Archived scripts from ShadowMap development. Nothing here is needed '
                    'to run or retrain the model. Kept for provenance and reproducibility '
                    'of the analysis history.'),
        'active_pipeline_order': [
            'generate_grid.py -> delhi_grid.geojson',
            'clip_grid.py -> delhi_grid_clipped.geojson (query "National Capital Territory of Delhi, India")',
            'filter_slivers.py -> delhi_grid_filtered.geojson (6709 cells, master grid)',
            'bulk_download_osm.py + merge_osm_tiles.py -> delhi_all_{buildings,roads}.geojson (gitignored, 4.5GB/375MB)',
            'aggregate_density.py -> grid_density.csv (not used by the model; kept for context)',
            'fetch_grid_lst.py -> grid_lst_ndvi.csv (GEE; provides the lst target)',
            'step10_fetch_indices.py -> grid_indices.csv (GEE; provides albedo/ndbi/mndwi/bsi)',
            'step13_night_lst.py -> grid_lst_night.csv (GEE; MODIS night, observation only)',
            'build_master.py -> grid_master.csv',
            'train_grid.py -> artifacts_grid/*, grid_predictions.csv',
            'step30_oof_predictions.py -> adds out-of-fold columns to grid_predictions.csv',
            'uvicorn main_grid:app --port 8001',
        ],
        'archive/analysis': ('Validation and diagnostic scripts, numbered by development '
                             'step. These produced the findings in RESULTS.md: spatial '
                             'autocorrelation range, R2-vs-block-size curve, feature '
                             'ablations, temporal holdout, albedo causal tests.'),
        'archive/superseded': SUPERSEDED,
        'archive/ward_pipeline': ('The original 95-ward pipeline and its FastAPI app '
                                  '(main.py, port 8000). Still functional and untouched. '
                                  'Reported R2 0.4523 under repeated random-split CV, which '
                                  'is inflated by spatial autocorrelation.'),
        'note_on_step_numbering': ('Gaps in step numbers (1,3,18,24-29,31,34,35) were '
                                   'inline commands or patches that produced no standalone '
                                   'script file.'),
    }
    io.open('archive/MANIFEST.json', 'w', encoding='utf-8').write(json.dumps(manifest, indent=2))
    print('wrote archive/MANIFEST.json')

    print()
    print('=== REMAINING .py IN backend/ ===')
    for f in sorted(x for x in os.listdir('.') if x.endswith('.py')):
        print('  %s' % f)

    print()
    print('=== IMPORT CHECK AFTER MOVE ===')
    import importlib
    for mod in ['feature_engineering_v2', 'predict_grid', 'train_grid', 'main_grid']:
        try:
            m = importlib.import_module(mod)
            importlib.reload(m)
            print('  %-26s OK' % mod)
        except Exception as e:
            print('  %-26s FAIL %s' % (mod, str(e)[:70]))
