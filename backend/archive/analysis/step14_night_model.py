import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree
from scipy import stats
from xgboost import XGBRegressor

m = pd.read_csv('grid_master.csv').merge(pd.read_csv('grid_lst_night.csv'), on='cell_id', how='inner')
print('merged %s | night nulls %d' % (m.shape, m.lst_night.isna().sum()))
m = m.dropna(subset=['lst_night']).reset_index(drop=True)
m['day_night_delta'] = m.lst - m.lst_night
m.to_csv('grid_master_night.csv', index=False)
print('kept %d cells -> grid_master_night.csv' % len(m))

SAT   = ['albedo','ndbi','mndwi','bsi','ndvi']
URBAN = ['building_density','road_density','population','elevation','dist_to_water']
SETS = {'SAT only (5)': SAT, 'URBAN only (5)': URBAN, 'FULL (10)': SAT+URBAN}

cx, cy = m.cx.values, m.cy.values
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

def nested(cols, km, target):
    X = m[cols].values; y = m[target].values; sc = []
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
for target in ['lst', 'lst_night', 'day_night_delta']:
    sd = m[target].std()
    print()
    print('=== TARGET: %s  (std %.2f C) ===' % (target, sd))
    print('%-18s %9s %9s %9s | %9s %9s %9s' % ('','R2_5km','sd','RMSE','R2_10km','sd','RMSE'))
    res[target] = {}
    for k, cols in SETS.items():
        s5 = nested(cols, 5, target); s10 = nested(cols, 10, target)
        res[target][k] = (s5, s10)
        print('%-18s %9.4f %9.4f %9.2f | %9.4f %9.4f %9.2f' %
              (k, s5.mean(), s5.std(), sd*np.sqrt(max(0,1-s5.mean())),
                  s10.mean(), s10.std(), sd*np.sqrt(max(0,1-s10.mean()))))

print()
print('=== IS THE OSM/URBAN DATA RESCUED? (URBAN-only skill by target) ===')
for target in ['lst','lst_night','day_night_delta']:
    s5 = res[target]['URBAN only (5)'][0]
    print('  %-16s URBAN-only R2 (5km) = %+.4f  -> %s' % (target, s5.mean(),
          'USEFUL' if s5.mean() > 0.15 else ('marginal' if s5.mean() > 0 else 'no skill')))

print()
print('=== DOES URBAN ADD OVER SATELLITE? (FULL vs SAT-only) ===')
for target in ['lst','lst_night','day_night_delta']:
    for i, km in [(0,5),(1,10)]:
        a = res[target]['SAT only (5)'][i]; b = res[target]['FULL (10)'][i]
        n = min(len(a), len(b)); d = b[:n]-a[:n]
        try: _, p = stats.wilcoxon(d)
        except Exception: p = np.nan
        print('  %-16s %2dkm delta %+.4f p=%.4f -> %s' % (target, km, d.mean(), p,
              'URBAN ADDS VALUE' if (p < 0.05 and d.mean() > 0) else 'no sig. gain'))

print()
print('=== PERMUTATION IMPORTANCE, NIGHT TARGET (5 km blocked) ===')
cols = SETS['FULL (10)']; X = m[cols].values; y = m.lst_night.values
F = blocked(np.arange(len(m)), 5, 2, 500)
base, imp = [], {c: [] for c in cols}
rng = np.random.default_rng(7)
for tr, te in F:
    mo = XGBRegressor(**{**FIXED, **GRID[1]}).fit(X[tr], y[tr])
    b = r2_score(y[te], mo.predict(X[te])); base.append(b)
    for j, c in enumerate(cols):
        Xp = X[te].copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        imp[c].append(b - r2_score(y[te], mo.predict(Xp)))
print('base R2 %.4f over %d folds' % (np.mean(base), len(F)))
print('%-20s %11s %10s' % ('feature','drop_in_R2','sd'))
for c, v in sorted(imp.items(), key=lambda t: -np.mean(t[1])):
    print('%-20s %11.4f %10.4f' % (c, np.mean(v), np.std(v)))
