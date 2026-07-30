import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import cKDTree
from xgboost import XGBRegressor

FEATURES = ['ndvi','building_density','road_density','elevation','dist_to_water',
            'pct_residential','pct_industrial_commercial','pct_green_farm','population']
P = dict(n_estimators=100, max_depth=2, learning_rate=0.1, reg_alpha=1.0,
         reg_lambda=1.0, random_state=42, n_jobs=4,
         objective='reg:squarederror', verbosity=0)

m = pd.read_csv('grid_master.csv')
y = m.lst.values
cx, cy = m.cx.values, m.cy.values
XY = np.c_[cx, cy]
XF = m[FEATURES].values
XFC = np.c_[XF, cx, cy]
print('cells %d | LST std %.3f | LST var %.3f' % (len(m), y.std(), y.var()))

def folds(km, reps=6):
    B = km*1000.0
    blk = (np.floor((cx-cx.min())/B).astype(int)*10000 +
           np.floor((cy-cy.min())/B).astype(int))
    ub = np.unique(blk)
    k = 5 if len(ub) >= 15 else 3
    out = []
    for rep in range(reps):
        perm = np.random.default_rng(300+rep).permutation(ub)
        for f in np.array_split(perm, k):
            te = np.isin(blk, f)
            if te.sum() < 30: continue
            tree = cKDTree(np.c_[cx[te], cy[te]])
            hit = tree.query_ball_point(XY, r=B, p=np.inf)
            near = np.array([len(h) > 0 for h in hit])
            tr = (~te) & (~near)
            if tr.sum() < 200: continue
            out.append((tr, te))
    return out

def run(km):
    F = folds(km)
    res = {k: [] for k in ['mean','coords','knn','feat','feat+xy']}
    for tr, te in F:
        res['mean'].append(r2_score(y[te], np.full(te.sum(), y[tr].mean())))
        res['coords'].append(r2_score(y[te], XGBRegressor(**P).fit(XY[tr], y[tr]).predict(XY[te])))
        kn = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(XY[tr], y[tr])
        res['knn'].append(r2_score(y[te], kn.predict(XY[te])))
        res['feat'].append(r2_score(y[te], XGBRegressor(**P).fit(XF[tr], y[tr]).predict(XF[te])))
        res['feat+xy'].append(r2_score(y[te], XGBRegressor(**P).fit(XFC[tr], y[tr]).predict(XFC[te])))
    return {k: (np.mean(v), np.std(v)) for k, v in res.items()}, len(F)

print()
print('%9s %8s %9s %9s %9s %9s %9s' % ('block_km','folds','mean','coordsXGB','knn5','FEATURES','feat+xy'))
store = {}
for km in [1, 2, 5, 10]:
    r, n = run(km)
    store[km] = r
    print('%9d %8d %9.4f %9.4f %9.4f %9.4f %9.4f' %
          (km, n, r['mean'][0], r['coords'][0], r['knn'][0], r['feat'][0], r['feat+xy'][0]))

print()
print('=== DO FEATURES BEAT PURE LOCATION? ===')
for km in [1, 2, 5, 10]:
    r = store[km]
    best_sp = max(r['coords'][0], r['knn'][0])
    d = r['feat'][0] - best_sp
    v = 'FEATURES WIN' if d > 0.03 else ('tie' if d > -0.03 else 'LOCATION WINS')
    print('%2d km: features %.4f  vs  best-spatial %.4f   delta %+.4f   -> %s' % (km, r['feat'][0], best_sp, d, v))

print()
print('=== RMSE degC (interpretable) ===')
for km in [1, 2, 5, 10]:
    r = store[km]
    print('%2d km: features RMSE %.2f degC   (naive mean RMSE %.2f)' %
          (km, y.std()*np.sqrt(max(0,1-r['feat'][0])), y.std()*np.sqrt(max(0,1-r['mean'][0]))))
