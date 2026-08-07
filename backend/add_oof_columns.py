"""
add_oof_columns.py

Fixes two reporting bugs found in the API self-test:

BUG 1: grid_predictions.csv held IN-SAMPLE predictions, so residual std read
       1.039 C instead of the honest out-of-fold 1.459 C, and large-error
       counts were understated ~20x.
BUG 2: the extrapolation flag fired on 369 training cells (5.5%), mostly at
       the NCT boundary. A cell that was IN training is not extrapolation.

Adds oof_pred / oof_residual columns and a 3-level confidence class.
"""
import numpy as np, pandas as pd, json, os
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

import feature_engineering_v2 as fe
from train_grid import HYPERPARAMS, blocked_folds, BLOCK_KM_PRIMARY

df = fe.load_grid_data()
X = df[fe.FEATURE_COLUMNS].values.astype(float)
y = df[fe.TARGET_COLUMN].values.astype(float)
print('cells %d' % len(df))

folds = blocked_folds(df.cx.values, df.cy.values, BLOCK_KM_PRIMARY)
print('folds %d' % len(folds))

print()
print('=== BUILDING OUT-OF-FOLD PREDICTIONS ===')
sum_p = np.zeros(len(df)); n_p = np.zeros(len(df))
for i, (tr, te) in enumerate(folds):
    mo = XGBRegressor(**HYPERPARAMS).fit(X[tr], y[tr])
    sum_p[te] += mo.predict(X[te]); n_p[te] += 1
print('cells predicted at least once: %d of %d' % (int((n_p > 0).sum()), len(df)))
print('mean times each cell held out: %.1f' % n_p[n_p > 0].mean())

oof = np.full(len(df), np.nan)
ok = n_p > 0
oof[ok] = sum_p[ok] / n_p[ok]
oof_res = oof - y

print()
print('=== IN-SAMPLE vs OUT-OF-FOLD (the bug) ===')
old = pd.read_csv('grid_predictions.csv')
ins_res = old['residual'].values
print('%-24s %10s %10s' % ('metric', 'in-sample', 'out-of-fold'))
print('%-24s %10.3f %10.3f' % ('residual std (C)', ins_res.std(), np.nanstd(oof_res)))
print('%-24s %10.3f %10.3f' % ('RMSE (C)', float(np.sqrt(np.mean(ins_res**2))),
                               float(np.sqrt(np.nanmean(oof_res**2)))))
print('%-24s %10d %10d' % ('cells |err| > 3 C', int(np.sum(np.abs(ins_res) > 3)),
                           int(np.nansum(np.abs(oof_res) > 3))))
print('%-24s %10d %10d' % ('cells |err| > 5 C', int(np.sum(np.abs(ins_res) > 5)),
                           int(np.nansum(np.abs(oof_res) > 5))))
print('%-24s %10.4f %10.4f' % ('R2', r2_score(y, old['pred'].values),
                               r2_score(y[ok], oof[ok])))

print()
print('=== FIXING THE CONFIDENCE FLAG ===')
bounds = fe.load_feature_bounds()
out_of_range = np.zeros(len(df), bool)
for c in fe.FEATURE_COLUMNS:
    b = bounds[c]
    out_of_range |= (df[c].values < b['p01']) | (df[c].values > b['p99'])
print('cells outside p01-p99 : %d (%.1f%%)' % (out_of_range.sum(), 100*out_of_range.mean()))

g = None
try:
    import geopandas as gpd
    gg = gpd.read_file('delhi_grid_filtered.geojson').to_crs(32643)
    edge = gg.union_all().boundary
    dist_edge = gpd.GeoSeries(gpd.points_from_xy(df.cx, df.cy), crs=32643).distance(edge).values
    near_edge = dist_edge < 1000
    print('cells within 1 km of NCT boundary: %d (%.1f%%)' % (near_edge.sum(), 100*near_edge.mean()))
except Exception as e:
    print('boundary distance unavailable (%s); using area proxy' % str(e)[:40])
    dist_edge = np.full(len(df), np.nan)
    near_edge = df.area_m2.values < 0.99*df.area_m2.max()

conf = np.array(['normal'] * len(df), dtype=object)
conf[near_edge] = 'edge_higher_error'
conf[out_of_range] = 'atypical_features'
big = np.abs(oof_res) > 3
conf[big & (conf == 'normal')] = 'poorly_predicted'
print()
print('confidence classes:')
for k, v in pd.Series(conf).value_counts().items():
    print('  %-20s %5d (%.1f%%)' % (k, v, 100*v/len(df)))

print()
print('=== ERROR BY CONFIDENCE CLASS (out-of-fold) ===')
tmp = pd.DataFrame({'conf': conf, 'abs_err': np.abs(oof_res)})
print(tmp.groupby('conf', observed=True).agg(
    n=('abs_err','size'), mean_abs_err=('abs_err','mean'),
    p95_abs_err=('abs_err', lambda v: float(np.nanpercentile(v, 95)))).round(3).to_string())

out = old.copy()
out['oof_pred'] = np.round(oof, 3)
out['oof_residual'] = np.round(oof_res, 3)
out['out_of_range'] = out_of_range
out['near_boundary'] = near_edge
out['dist_to_boundary_m'] = np.round(dist_edge, 1)
out['confidence_class'] = conf
out.to_csv('grid_predictions.csv', index=False)
print()
print('WROTE grid_predictions.csv  shape %s' % (out.shape,))
print('cols:', list(out.columns))

meta_path = os.path.join(fe.ARTIFACTS_DIR, 'model_meta.json')
with open(meta_path) as f:
    meta = json.load(f)
meta['residual_diagnostics_out_of_fold'] = {
    'residual_std_c': round(float(np.nanstd(oof_res)), 3),
    'rmse_c': round(float(np.sqrt(np.nanmean(oof_res**2))), 3),
    'cells_abs_err_gt_3c': int(np.nansum(np.abs(oof_res) > 3)),
    'cells_abs_err_gt_5c': int(np.nansum(np.abs(oof_res) > 5)),
    'note': ('Computed from 5 km blocked out-of-fold predictions. In-sample residuals '
             'read ~1.04 C std and must not be reported as model error.'),
}
meta['confidence_classes'] = {
    'normal': 'features within p01-p99, not near boundary, out-of-fold error <= 3 C',
    'edge_higher_error': 'within 1 km of the NCT boundary; errors ~1.7x larger',
    'atypical_features': 'one or more features outside the p01-p99 training range',
    'poorly_predicted': 'out-of-fold absolute error above 3 C',
}
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)
print('updated model_meta.json with out-of-fold diagnostics')

print()
print('=== WORST 5 OUT-OF-FOLD CELLS ===')
w = out.reindex(pd.Series(np.abs(oof_res)).sort_values(ascending=False).index).head(5)
print(w[['cell_id','lon','lat','lst','oof_pred','oof_residual','confidence_class']].to_string(index=False))
