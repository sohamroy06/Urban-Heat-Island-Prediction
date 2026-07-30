import numpy as np, pandas as pd, json, os
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.spatial import cKDTree
from xgboost import XGBRegressor

FEATURES = ['albedo','ndbi','mndwi','bsi']
os.makedirs('artifacts_grid', exist_ok=True)

m = pd.read_csv('grid_master.csv')
X = m[FEATURES].values; y = m.lst.values
cx, cy = m.cx.values, m.cy.values
print('cells %d | features %s (ndvi REMOVED - pipeline inconsistency)' % (len(m), FEATURES))

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

for km in [5, 10]:
    s = []
    for tr, te in blocked(km, 4, 500):
        s.append(r2_score(y[te], XGBRegressor(**BEST).fit(X[tr], y[tr]).predict(X[te])))
    s = np.array(s)
    print('blocked %2d km: R2 %.4f +/- %.4f  RMSE %.3f C' % (km, s.mean(), s.std(), y.std()*np.sqrt(max(0,1-s.mean()))))

F = blocked(5, 4, 500)
op, olo, ohi, oy = [], [], [], []
for tr, te in F:
    mo = XGBRegressor(**BEST).fit(X[tr], y[tr])
    lo = XGBRegressor(**{**BEST,'objective':'reg:quantileerror','quantile_alpha':0.10}).fit(X[tr], y[tr])
    hi = XGBRegressor(**{**BEST,'objective':'reg:quantileerror','quantile_alpha':0.90}).fit(X[tr], y[tr])
    op.append(mo.predict(X[te])); olo.append(lo.predict(X[te])); ohi.append(hi.predict(X[te])); oy.append(y[te])
p, lo, hi, yy = map(np.concatenate, (op, olo, ohi, oy))
R2 = r2_score(yy, p); RMSE = float(np.sqrt(np.mean((yy-p)**2)))
print()
print('=== OOF (5 km blocked) ===')
print('R2 %.4f | RMSE %.3f C | MAE %.3f C | bias %+.3f C' % (R2, RMSE, mean_absolute_error(yy,p), float(np.mean(p-yy))))

cov = float(np.mean((yy>=lo)&(yy<=hi))); k = 1.0
if cov < 0.78:
    for s_ in np.arange(1.0, 3.05, 0.05):
        mid = (lo+hi)/2
        if float(np.mean((yy >= mid-(mid-lo)*s_) & (yy <= mid+(hi-mid)*s_))) >= 0.80:
            k = float(s_); break
print('raw PI coverage %.1f%% -> multiplier %.2f' % (100*cov, k))

mean_m = XGBRegressor(**BEST).fit(X, y)
lo_m = XGBRegressor(**{**BEST,'objective':'reg:quantileerror','quantile_alpha':0.10}).fit(X, y)
hi_m = XGBRegressor(**{**BEST,'objective':'reg:quantileerror','quantile_alpha':0.90}).fit(X, y)
mean_m.save_model('artifacts_grid/uhi_grid_mean.json')
lo_m.save_model('artifacts_grid/uhi_grid_p10.json')
hi_m.save_model('artifacts_grid/uhi_grid_p90.json')

meta = {
 'features': FEATURES, 'target': 'landsat_daytime_lst_celsius',
 'n_cells': int(len(m)), 'grid_cell_size_m': [448.5, 505.7],
 'source': 'Landsat 8/9 C2 L2 median composite 2024-04-01..2024-06-30, QA_PIXEL per-pixel masked',
 'overpass_local_time': '~10:30',
 'hyperparams': {kk: vv for kk, vv in BEST.items() if kk != 'n_jobs'},
 'validation': {'method': '5km spatially blocked CV with buffer',
                'r2': round(R2,4), 'rmse_c': round(RMSE,3),
                'interval_coverage': round(cov,4), 'interval_width_multiplier': k},
 'feature_bounds': {f: {'min': float(m[f].min()), 'max': float(m[f].max()),
                        'p01': float(m[f].quantile(0.01)), 'p99': float(m[f].quantile(0.99))} for f in FEATURES},
 'known_limitations': [
   'Daytime surface temperature (~10:30 local) only; NOT air temperature',
   'Single season Apr-Jun 2024; absolute values do not transfer across years (see temporal_holdout)',
   'Delhi NCT only; no transfer evidence to other cities',
   'Landsat L2 emissivity retrieval uses NDVI, giving partial circularity with optical predictors',
   'OSM building/road and WorldPop features show no daytime predictive skill and are excluded',
   'Daytime shows a surface urban COOL island; not valid for building-density policy what-if',
   'Model compresses extremes: under-predicts hottest cells by ~1 C, over-predicts coldest by ~0.7 C',
   'Errors ~1.7x larger in cells adjacent to the NCT boundary',
   'Dense-population zones (e.g. SE industrial belt) under-predicted; anthropogenic heat is not observable optically']}
with open('artifacts_grid/model_meta.json','w') as f:
    json.dump(meta, f, indent=2)
print('saved models + meta (4 features)')

import importlib, predict_grid as PG
importlib.reload(PG); PG._cache.clear()

print()
print('=== TEMPORAL TRANSFER, CONSISTENT PIPELINES BOTH SIDES ===')
for tag, path in [('2023 full window','grid_2023.csv'), ('2023 date-matched','grid_2023_matched.csv')]:
    if not os.path.exists(path): continue
    d = pd.read_csv(path).dropna().reset_index(drop=True)
    r = PG.predict(d)
    yt = d.lst.values; pt = r.pred_lst_c.values; b = float(np.mean(pt-yt))
    print('%-20s R2raw %+.4f | R2debiased %.4f | corr %.4f | bias %+.2f C | extrap %.1f%%'
          % (tag, r2_score(yt,pt), r2_score(yt,pt-b), float(np.corrcoef(pt,yt)[0,1]), b, 100*r.extrapolation.mean()))

if os.path.exists('grid_2024_matched.csv'):
    d = pd.read_csv('grid_2024_matched.csv').dropna().reset_index(drop=True)
    r = PG.predict(d)
    yt = d.lst.values; pt = r.pred_lst_c.values
    print('%-20s R2raw %+.4f | corr %.4f | extrap %.1f%%   <- SAME year sanity check'
          % ('2024 date-matched', r2_score(yt,pt), float(np.corrcoef(pt,yt)[0,1]), 100*r.extrapolation.mean()))

out = m[['cell_id','lon','lat','lst']].copy()
out['pred'] = mean_m.predict(X)
mid = (lo_m.predict(X)+hi_m.predict(X))/2
out['p10'] = mid-(mid-lo_m.predict(X))*k; out['p90'] = mid+(hi_m.predict(X)-mid)*k
out['residual'] = out.pred-out.lst
out.to_csv('grid_predictions.csv', index=False)
print()
print('saved grid_predictions.csv')
