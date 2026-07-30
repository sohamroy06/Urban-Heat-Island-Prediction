import numpy as np, pandas as pd, itertools
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree
from xgboost import XGBRegressor

FEATURES = ['ndvi','building_density','road_density','elevation','dist_to_water',
            'pct_residential','pct_industrial_commercial','pct_green_farm','population']
m = pd.read_csv('grid_master.csv')
y = m.lst.values
X = m[FEATURES].values
cx, cy = m.cx.values, m.cy.values
XY = np.c_[cx, cy]

def folds(km, reps=6):
    B = km*1000.0
    blk = (np.floor((cx-cx.min())/B).astype(int)*10000 +
           np.floor((cy-cy.min())/B).astype(int))
    ub = np.unique(blk); out = []
    for rep in range(reps):
        perm = np.random.default_rng(300+rep).permutation(ub)
        for f in np.array_split(perm, 5):
            te = np.isin(blk, f)
            if te.sum() < 30: continue
            hit = cKDTree(XY[te]).query_ball_point(XY, r=B, p=np.inf)
            tr = (~te) & ~np.array([len(h) > 0 for h in hit])
            if tr.sum() < 200: continue
            out.append((tr, te))
    return out

F5, F10 = folds(5), folds(10)
print('folds 5km=%d 10km=%d' % (len(F5), len(F10)))

def score(P, F):
    s = []
    for tr, te in F:
        mo = XGBRegressor(objective='reg:squarederror', verbosity=0,
                          n_jobs=4, random_state=42, **P).fit(X[tr], y[tr])
        s.append(r2_score(y[te], mo.predict(X[te])))
    return np.mean(s), np.std(s)

BASE = dict(n_estimators=100, max_depth=2, learning_rate=0.1, reg_alpha=1.0, reg_lambda=1.0)
b5, bs5 = score(BASE, F5); b10, bs10 = score(BASE, F10)
print('CURRENT (depth2,100):  5km %.4f (+/-%.4f)  10km %.4f (+/-%.4f)' % (b5, bs5, b10, bs10))

print()
print('=== DEPTH x TREES SWEEP (5 km blocked) ===')
print('%6s %7s %9s %9s' % ('depth','trees','R2_5km','sd'))
best = (b5, BASE)
for d in [2, 3, 4, 6, 8]:
    for n in [100, 300, 600]:
        P = dict(n_estimators=n, max_depth=d, learning_rate=0.05,
                 reg_alpha=1.0, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8)
        r, sd = score(P, F5)
        flag = ' *' if r > best[0] else ''
        print('%6d %7d %9.4f %9.4f%s' % (d, n, r, sd, flag))
        if r > best[0]: best = (r, P)

print()
print('=== REGULARIZATION SWEEP on best depth/trees (5 km) ===')
bd, bn = best[1]['max_depth'], best[1]['n_estimators']
print('tuning around depth=%d trees=%d' % (bd, bn))
print('%8s %8s %8s %9s' % ('lr','alpha','lambda','R2_5km'))
for lr in [0.02, 0.05, 0.1]:
    for a, l in [(0.0, 1.0), (1.0, 1.0), (1.0, 5.0), (5.0, 10.0)]:
        P = dict(n_estimators=bn, max_depth=bd, learning_rate=lr, reg_alpha=a,
                 reg_lambda=l, subsample=0.8, colsample_bytree=0.8)
        r, sd = score(P, F5)
        flag = ' *' if r > best[0] else ''
        print('%8.2f %8.1f %8.1f %9.4f%s' % (lr, a, l, r, flag))
        if r > best[0]: best = (r, P)

print()
print('=== BEST CONFIG ===')
print(best[1])
t5, ts5 = score(best[1], F5); t10, ts10 = score(best[1], F10)
print(' 5km: %.4f (+/-%.4f)   vs current %.4f   delta %+.4f' % (t5, ts5, b5, t5-b5))
print('10km: %.4f (+/-%.4f)   vs current %.4f   delta %+.4f' % (t10, ts10, b10, t10-b10))
print('RMSE 5km: tuned %.2f degC | current %.2f degC' % (y.std()*np.sqrt(max(0,1-t5)), y.std()*np.sqrt(max(0,1-b5))))

print()
print('=== DOES CAPACITY REVIVE THE "USELESS" FEATURES? (best config, 5 km) ===')
def score_cols(cols, P, F):
    Xc = m[cols].values; s = []
    for tr, te in F:
        mo = XGBRegressor(objective='reg:squarederror', verbosity=0,
                          n_jobs=4, random_state=42, **P).fit(Xc[tr], y[tr])
        s.append(r2_score(y[te], mo.predict(Xc[te])))
    return np.mean(s)
full = score_cols(FEATURES, best[1], F5)
print('%-28s %9s %9s' % ('dropped','R2_5km','delta'))
for f in FEATURES:
    r = score_cols([c for c in FEATURES if c != f], best[1], F5)
    print('%-28s %9.4f %+9.4f' % (f, r, r-full))
