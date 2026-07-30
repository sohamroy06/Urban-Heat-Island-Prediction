import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree
from scipy import stats
from xgboost import XGBRegressor

m = pd.read_csv('grid_master.csv')
y = m.lst.values
cx, cy = m.cx.values, m.cy.values
print('cells %d | cols %d' % (len(m), m.shape[1]))

SAT4   = ['albedo','ndbi','mndwi','bsi']
URBAN  = ['building_density','road_density','elevation','dist_to_water','population']
SETS = {
    'SAT only (4)':          SAT4,
    'SAT + ndvi (5)':        SAT4 + ['ndvi'],
    'URBAN only (5)':        URBAN,
    'FULL (10)':             SAT4 + ['ndvi'] + URBAN,
    'FULL minus ndbi (9)':   [c for c in SAT4 if c != 'ndbi'] + ['ndvi'] + URBAN,
    'FULL minus bsi (9)':    [c for c in SAT4 if c != 'bsi'] + ['ndvi'] + URBAN,
}

GRID = [dict(n_estimators=300, max_depth=2, learning_rate=0.05),
        dict(n_estimators=600, max_depth=2, learning_rate=0.05),
        dict(n_estimators=1200, max_depth=2, learning_rate=0.02),
        dict(n_estimators=600, max_depth=3, learning_rate=0.05),
        dict(n_estimators=300, max_depth=4, learning_rate=0.05)]
FIXED = dict(reg_alpha=1.0, reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8,
             objective='reg:squarederror', verbosity=0, n_jobs=4, random_state=42)

def blocked(idx, km, reps, seed):
    B = km*1000.0
    sx, sy = cx[idx], cy[idx]
    blk = (np.floor((sx-sx.min())/B).astype(int)*10000 +
           np.floor((sy-sy.min())/B).astype(int))
    ub = np.unique(blk); out = []
    for rep in range(reps):
        for f in np.array_split(np.random.default_rng(seed+rep).permutation(ub), 5):
            te = np.isin(blk, f)
            if te.sum() < 25: continue
            hit = cKDTree(np.c_[sx[te], sy[te]]).query_ball_point(np.c_[sx, sy], r=B, p=np.inf)
            tr = (~te) & ~np.array([len(h) > 0 for h in hit])
            if tr.sum() < 200: continue
            out.append((idx[tr], idx[te]))
    return out

def nested(cols, km):
    X = m[cols].values; sc = []
    for tr, te in blocked(np.arange(len(m)), km, 4, 500):
        inner = blocked(tr, km, 2, 900)
        best = GRID[1]
        if len(inner) >= 3:
            mu = [np.mean([r2_score(y[b], XGBRegressor(**{**FIXED,**P}).fit(X[a],y[a]).predict(X[b]))
                           for a, b in inner]) for P in GRID]
            best = GRID[int(np.argmax(mu))]
        sc.append(r2_score(y[te], XGBRegressor(**{**FIXED,**best}).fit(X[tr],y[tr]).predict(X[te])))
    return np.array(sc)

res = {}
for km in [5, 10]:
    print()
    print('=== NESTED SPATIAL CV, %d km ===' % km)
    print('%-24s %9s %9s %9s' % ('feature set','R2','sd','RMSE_C'))
    res[km] = {}
    for k, cols in SETS.items():
        s = nested(cols, km); res[km][k] = s
        print('%-24s %9.4f %9.4f %9.2f' % (k, s.mean(), s.std(), y.std()*np.sqrt(max(0,1-s.mean()))))

print()
print('=== DOES URBAN DATA ADD ANYTHING OVER SATELLITE ALONE? ===')
for km in [5, 10]:
    a = res[km]['SAT + ndvi (5)']; b = res[km]['FULL (10)']
    n = min(len(a), len(b)); d = b[:n] - a[:n]
    try: _, p = stats.wilcoxon(d)
    except Exception: p = np.nan
    v = 'URBAN DATA ADDS VALUE' if (p < 0.05 and d.mean() > 0) else 'OSM/WorldPop WORK CONTRIBUTES NOTHING'
    print('%2d km: FULL minus SAT+ndvi = %+.4f  p=%.4f  -> %s' % (km, d.mean(), p, v))

print()
print('=== NDBI / BSI DUPLICATE TEST (vs FULL) ===')
for km in [5, 10]:
    base = res[km]['FULL (10)']
    for k in ['FULL minus ndbi (9)','FULL minus bsi (9)']:
        s = res[km][k]; n = min(len(base), len(s)); d = s[:n] - base[:n]
        try: _, p = stats.wilcoxon(d)
        except Exception: p = np.nan
        print('%2d km: %-22s delta %+.4f p=%.4f -> %s' % (km, k, d.mean(), p,
              'safe to drop' if p > 0.05 else 'keep both'))

print()
print('=== PERMUTATION IMPORTANCE (5 km blocked, FULL set) ===')
cols = SETS['FULL (10)']
X = m[cols].values
F = blocked(np.arange(len(m)), 5, 2, 500)
base_s, imp = [], {c: [] for c in cols}
rng = np.random.default_rng(7)
for tr, te in F:
    mo = XGBRegressor(**{**FIXED, **GRID[1]}).fit(X[tr], y[tr])
    b = r2_score(y[te], mo.predict(X[te])); base_s.append(b)
    for j, c in enumerate(cols):
        Xp = X[te].copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        imp[c].append(b - r2_score(y[te], mo.predict(Xp)))
print('base R2 %.4f over %d folds' % (np.mean(base_s), len(F)))
print('%-22s %10s %10s' % ('feature','drop_in_R2','sd'))
for c, v in sorted(imp.items(), key=lambda t: -np.mean(t[1])):
    print('%-22s %10.4f %10.4f' % (c, np.mean(v), np.std(v)))
