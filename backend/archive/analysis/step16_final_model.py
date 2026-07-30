import numpy as np, pandas as pd, json, os
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.spatial import cKDTree
from xgboost import XGBRegressor

FEATURES = ['albedo','ndbi','mndwi','bsi','ndvi']
TARGET = 'lst'
os.makedirs('artifacts_grid', exist_ok=True)

m = pd.read_csv('grid_master.csv')
X = m[FEATURES].values; y = m[TARGET].values
cx, cy = m.cx.values, m.cy.values
print('cells %d | features %s' % (len(m), FEATURES))

BEST = dict(n_estimators=600, max_depth=2, learning_rate=0.05, reg_alpha=1.0,
            reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8,
            objective='reg:squarederror', verbosity=0, n_jobs=4, random_state=42)

def blocked(km, reps, seed):
    B = km*1000.0
    blk = (np.floor((cx-cx.min())/B).astype(int)*10000 +
           np.floor((cy-cy.min())/B).astype(int))
    ub = np.unique(blk); out = []
    for rep in range(reps):
        for f in np.array_split(np.random.default_rng(seed+rep).permutation(ub), 5):
            te = np.isin(blk, f)
            if te.sum() < 25: continue
            hit = cKDTree(np.c_[cx[te], cy[te]]).query_ball_point(np.c_[cx, cy], r=B, p=np.inf)
            tr = (~te) & ~np.array([len(h) > 0 for h in hit])
            if tr.sum() < 200: continue
            out.append((tr, te))
    return out

print()
print('=== OUT-OF-FOLD PERFORMANCE (5 km blocked) ===')
F = blocked(5, 4, 500)
oof_p, oof_lo, oof_hi, oof_y = [], [], [], []
for tr, te in F:
    mo = XGBRegressor(**BEST).fit(X[tr], y[tr])
    lo = XGBRegressor(**{**BEST, 'objective':'reg:quantileerror', 'quantile_alpha':0.10}).fit(X[tr], y[tr])
    hi = XGBRegressor(**{**BEST, 'objective':'reg:quantileerror', 'quantile_alpha':0.90}).fit(X[tr], y[tr])
    oof_p.append(mo.predict(X[te])); oof_lo.append(lo.predict(X[te]))
    oof_hi.append(hi.predict(X[te])); oof_y.append(y[te])
p = np.concatenate(oof_p); lo = np.concatenate(oof_lo)
hi = np.concatenate(oof_hi); yy = np.concatenate(oof_y)
r2 = r2_score(yy, p); rmse = float(np.sqrt(np.mean((yy-p)**2)))
print('R2   %.4f' % r2)
print('RMSE %.3f degC' % rmse)
print('MAE  %.3f degC' % mean_absolute_error(yy, p))
print('bias %.3f degC (mean residual)' % float(np.mean(p-yy)))

cov = float(np.mean((yy >= lo) & (yy <= hi)))
print()
print('=== PREDICTION INTERVAL CALIBRATION ===')
print('P10-P90 interval should contain 80%% of true values')
print('actual coverage: %.1f%%  -> %s' % (100*cov,
      'WELL CALIBRATED' if 0.74 <= cov <= 0.86 else ('TOO NARROW (overconfident)' if cov < 0.74 else 'TOO WIDE')))
print('mean interval width %.2f degC' % float(np.mean(hi-lo)))
scale = 1.0
if cov < 0.78:
    for s in np.arange(1.0, 3.05, 0.05):
        mid = (lo+hi)/2
        c = float(np.mean((yy >= mid-(mid-lo)*s) & (yy <= mid+(hi-mid)*s)))
        if c >= 0.80: scale = float(s); break
    print('recommended width multiplier: %.2f  -> corrected coverage ~80%%' % scale)

print()
print('=== RESIDUAL DIAGNOSTICS ===')
res = p - yy
print('residual std %.3f | min %.2f | max %.2f' % (res.std(), res.min(), res.max()))
print('cells with |error| > 3 degC: %d (%.1f%%)' % (int((np.abs(res)>3).sum()), 100*np.mean(np.abs(res)>3)))
print('cells with |error| > 5 degC: %d (%.1f%%)' % (int((np.abs(res)>5).sum()), 100*np.mean(np.abs(res)>5)))
qs = pd.qcut(pd.Series(yy).rank(method='first'), 4, labels=['coldest','Q2','Q3','hottest'])
print()
print('error by temperature quartile (checks over/under-prediction at extremes):')
print(pd.DataFrame({'true':yy,'err':res}).groupby(qs, observed=True).agg(
      mean_true=('true','mean'), mean_err=('err','mean'), rmse=('err', lambda v: float(np.sqrt(np.mean(v**2))))).round(2))

print()
print('=== TRAINING FINAL MODELS ON ALL DATA ===')
mean_m = XGBRegressor(**BEST).fit(X, y)
lo_m = XGBRegressor(**{**BEST, 'objective':'reg:quantileerror', 'quantile_alpha':0.10}).fit(X, y)
hi_m = XGBRegressor(**{**BEST, 'objective':'reg:quantileerror', 'quantile_alpha':0.90}).fit(X, y)
mean_m.save_model('artifacts_grid/uhi_grid_mean.json')
lo_m.save_model('artifacts_grid/uhi_grid_p10.json')
hi_m.save_model('artifacts_grid/uhi_grid_p90.json')
print('saved 3 models to artifacts_grid/')

bounds = {f: {'min': float(m[f].min()), 'max': float(m[f].max()),
              'p01': float(m[f].quantile(0.01)), 'p99': float(m[f].quantile(0.99))} for f in FEATURES}
meta = {
    'features': FEATURES, 'target': 'landsat_daytime_lst_celsius',
    'n_cells': int(len(m)), 'grid_cell_size_m': [448.5, 505.7],
    'source': 'Landsat 8/9 C2 L2 median composite, 2024-04-01 to 2024-06-30, QA_PIXEL masked',
    'overpass_local_time': '~10:30', 'hyperparams': {k: v for k, v in BEST.items() if k != 'n_jobs'},
    'validation': {'method': '5km spatially blocked CV with buffer, nested tuning',
                   'r2': round(r2,4), 'rmse_c': round(rmse,3),
                   'interval_coverage': round(cov,4), 'interval_width_multiplier': scale},
    'reference_numbers': {'random_split_r2_LEAKY_do_not_use': 0.7422,
                          'blocked_5km_r2_HONEST': round(r2,4),
                          'blocked_10km_r2': 0.5754,
                          'cross_sensor_modis_r2': 0.2982,
                          'cross_sensor_ceiling': 0.4715},
    'feature_bounds': bounds,
    'known_limitations': [
        'Daytime surface temperature only (~10:30 local); NOT air temperature',
        'Single season: April-June 2024 pre-monsoon; no other season validated',
        'Delhi NCT only; no evidence of transfer to other cities',
        'Landsat L2 emissivity retrieval uses NDVI, creating partial circularity with optical predictors',
        'Urban form features (OSM buildings/roads, WorldPop) show no daytime predictive skill and are excluded',
        'Daytime shows surface urban COOL island; do not use for policy what-if on building density',
        'Predictions outside feature_bounds are extrapolation and unreliable'
    ]
}
with open('artifacts_grid/model_meta.json','w') as f:
    json.dump(meta, f, indent=2)
print('saved artifacts_grid/model_meta.json')

pred_all = mean_m.predict(X)
out = m[['cell_id','lon','lat',TARGET]].copy()
out['pred'] = pred_all
out['p10'] = lo_m.predict(X); out['p90'] = hi_m.predict(X)
out['residual'] = out['pred'] - out[TARGET]
out.to_csv('grid_predictions.csv', index=False)
print('saved grid_predictions.csv', out.shape)
print()
print('worst 5 in-sample residuals:')
print(out.reindex(out.residual.abs().sort_values(ascending=False).index).head(5)[
      ['cell_id','lon','lat','lst','pred','residual']].round(3).to_string(index=False))
