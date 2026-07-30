import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree
from scipy import stats
from xgboost import XGBRegressor

m = pd.read_csv('grid_master.csv')
ix = pd.read_csv('grid_indices.csv')
print('master %s | indices %s' % (m.shape, ix.shape))
assert set(m.cell_id) == set(ix.cell_id), 'cell_id mismatch'

m = m.merge(ix, on='cell_id', how='inner', validate='one_to_one')
print('merged %s' % (m.shape,))

print()
print('=== DUPLICATE CHECK: ndbi vs ndwi ===')
r = np.corrcoef(m.ndbi, m.ndwi)[0,1]
print('corr(ndbi, ndwi) = %+.6f   max|ndbi+ndwi| = %.2e' % (r, np.abs(m.ndbi+m.ndwi).max()))
print('-> ndwi is redundant, dropping' if r < -0.999 else '-> keeping both')
if r < -0.999:
    m = m.drop(columns=['ndwi'])

CAND = [c for c in ['albedo','ndbi','mndwi','bsi'] if c in m.columns]
print()
print('=== COLLINEARITY AMONG PREDICTORS ===')
cols = ['ndvi'] + CAND
C = m[cols].corr()
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        if abs(C.iloc[i,j]) > 0.7:
            print('  %-8s %-8s r = %+.3f  HIGH' % (cols[i], cols[j], C.iloc[i,j]))

m.to_csv('grid_master.csv', index=False)
print()
print('UPDATED grid_master.csv ->', m.shape)

LEAN = ['ndvi','building_density','road_density','elevation','dist_to_water','population']
SETS = {
    'LEAN 6 (step9 winner)': LEAN,
    'LEAN + albedo':         LEAN + ['albedo'],
    'LEAN + ndbi':           LEAN + ['ndbi'],
    'LEAN + bsi':            LEAN + ['bsi'],
    'LEAN + mndwi':          LEAN + ['mndwi'],
    'LEAN + ALL 4':          LEAN + CAND,
    'ALL4 only (no ndvi)':   [c for c in LEAN if c != 'ndvi'] + CAND,
}

y = m.lst.values
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

def nested(cols, km):
    X = m[cols].values
    sc = []
    for tr, te in blocked(np.arange(len(m)), km, 4, 500):
        inner = blocked(tr, km, 2, 900)
        best = GRID[1]
        if len(inner) >= 3:
            mu = []
            for P in GRID:
                mu.append(np.mean([r2_score(y[i2], XGBRegressor(**{**FIXED,**P}).fit(X[i1],y[i1]).predict(X[i2]))
                                   for i1, i2 in inner]))
            best = GRID[int(np.argmax(mu))]
        sc.append(r2_score(y[te], XGBRegressor(**{**FIXED,**best}).fit(X[tr],y[tr]).predict(X[te])))
    return np.array(sc)

print()
print('=== NESTED SPATIAL CV, 5 km ===')
print('%-24s %9s %9s %9s' % ('feature set','R2','sd','RMSE_C'))
res5 = {}
for k, cols in SETS.items():
    s = nested(cols, 5); res5[k] = s
    print('%-24s %9.4f %9.4f %9.2f' % (k, s.mean(), s.std(), y.std()*np.sqrt(max(0,1-s.mean()))))

print()
print('=== NESTED SPATIAL CV, 10 km ===')
print('%-24s %9s %9s %9s' % ('feature set','R2','sd','RMSE_C'))
res10 = {}
for k, cols in SETS.items():
    s = nested(cols, 10); res10[k] = s
    print('%-24s %9.4f %9.4f %9.2f' % (k, s.mean(), s.std(), y.std()*np.sqrt(max(0,1-s.mean()))))

print()
print('=== PAIRED WILCOXON vs LEAN 6 ===')
base_k = 'LEAN 6 (step9 winner)'
for km, res in [(5, res5), (10, res10)]:
    print('-- %d km --' % km)
    b = res[base_k]
    for k, s in res.items():
        if k == base_k: continue
        n = min(len(b), len(s)); d = s[:n] - b[:n]
        try: _, p = stats.wilcoxon(d)
        except Exception: p = np.nan
        v = 'SIGNIFICANT GAIN' if (p < 0.05 and d.mean() > 0) else ('significant LOSS' if (p < 0.05 and d.mean() < 0) else 'no sig. difference')
        print('  %-24s delta %+.4f  p=%.4f  -> %s' % (k, d.mean(), p, v))
