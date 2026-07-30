import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree
from scipy import stats
from xgboost import XGBRegressor

ALL9 = ['ndvi','building_density','road_density','elevation','dist_to_water',
        'pct_residential','pct_industrial_commercial','pct_green_farm','population']
LEAN = ['ndvi','building_density','road_density','elevation','dist_to_water','population']

GRID = [
    dict(n_estimators=100, max_depth=2, learning_rate=0.10),
    dict(n_estimators=300, max_depth=2, learning_rate=0.05),
    dict(n_estimators=600, max_depth=2, learning_rate=0.05),
    dict(n_estimators=600, max_depth=3, learning_rate=0.05),
    dict(n_estimators=300, max_depth=4, learning_rate=0.05),
    dict(n_estimators=1200, max_depth=2, learning_rate=0.02),
]
FIXED = dict(reg_alpha=1.0, reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8,
             objective='reg:squarederror', verbosity=0, n_jobs=4, random_state=42)

m = pd.read_csv('grid_master.csv')
y = m.lst.values
cx, cy = m.cx.values, m.cy.values
XY = np.c_[cx, cy]

def blocked_folds(idx, km, reps, seed):
    B = km*1000.0
    sx, sy = cx[idx], cy[idx]
    blk = (np.floor((sx-sx.min())/B).astype(int)*10000 +
           np.floor((sy-sy.min())/B).astype(int))
    ub = np.unique(blk); out = []
    for rep in range(reps):
        perm = np.random.default_rng(seed+rep).permutation(ub)
        for f in np.array_split(perm, 5):
            te_loc = np.isin(blk, f)
            if te_loc.sum() < 25: continue
            hit = cKDTree(np.c_[sx[te_loc], sy[te_loc]]).query_ball_point(np.c_[sx, sy], r=B, p=np.inf)
            tr_loc = (~te_loc) & ~np.array([len(h) > 0 for h in hit])
            if tr_loc.sum() < 200: continue
            out.append((idx[tr_loc], idx[te_loc]))
    return out

def fit_pred(cols, P, tr, te):
    X = m[cols].values
    mo = XGBRegressor(**{**FIXED, **P}).fit(X[tr], y[tr])
    return mo.predict(X[te])

def nested(cols, km, label):
    outer = blocked_folds(np.arange(len(m)), km, 4, 500)
    scores, chosen = [], []
    for tr, te in outer:
        inner = blocked_folds(tr, km, 2, 900)
        if len(inner) < 3:
            best = GRID[2]
        else:
            means = []
            for P in GRID:
                s = [r2_score(y[ite], fit_pred(cols, P, itr, ite)) for itr, ite in inner]
                means.append(np.mean(s))
            best = GRID[int(np.argmax(means))]
        chosen.append('d%d_n%d' % (best['max_depth'], best['n_estimators']))
        scores.append(r2_score(y[te], fit_pred(cols, best, tr, te)))
    scores = np.array(scores)
    print('%-22s %2d km | folds %2d | R2 %.4f +/- %.4f | RMSE %.2f degC'
          % (label, km, len(scores), scores.mean(), scores.std(),
             y.std()*np.sqrt(max(0, 1-scores.mean()))))
    u, c = np.unique(chosen, return_counts=True)
    print('%-22s   configs picked: %s' % ('', dict(zip(u, c.tolist()))))
    return scores

print('=== NESTED SPATIAL CV (tuning inside outer folds - UNBIASED) ===')
n5_all  = nested(ALL9, 5, 'ALL 9 features')
n5_lean = nested(LEAN, 5, 'LEAN 6 features')
print()
n10_all  = nested(ALL9, 10, 'ALL 9 features')
n10_lean = nested(LEAN, 10, 'LEAN 6 features')

print()
print('=== PAIRED TEST: does dropping the 2 land-use pcts + pct_residential matter? ===')
for km, a, b in [(5, n5_all, n5_lean), (10, n10_all, n10_lean)]:
    n = min(len(a), len(b)); d = b[:n] - a[:n]
    t, p = stats.wilcoxon(d) if n >= 6 else (np.nan, np.nan)
    print('%2d km: lean-minus-all mean %+.4f | median %+.4f | wilcoxon p=%.4f -> %s'
          % (km, d.mean(), np.median(d), p,
             'DIFFERENT' if p < 0.05 else 'NO SIGNIFICANT DIFFERENCE'))

print()
print('=== BIAS CHECK vs STEP 8 ===')
print('step8 tuned-on-same-folds 5km : 0.5247')
print('step9 nested unbiased    5km : %.4f   (optimism %+.4f)' % (n5_all.mean(), 0.5247-n5_all.mean()))
print('step8 tuned-on-same-folds 10km: 0.3229')
print('step9 nested unbiased    10km: %.4f   (optimism %+.4f)' % (n10_all.mean(), 0.3229-n10_all.mean()))
