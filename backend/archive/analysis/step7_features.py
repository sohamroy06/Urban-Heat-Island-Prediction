import numpy as np, pandas as pd
from sklearn.metrics import r2_score
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
print('folds: 5km=%d  10km=%d' % (len(F5), len(F10)))

def score(cols, F):
    X = m[cols].values; s = []
    for tr, te in F:
        s.append(r2_score(y[te], XGBRegressor(**P).fit(X[tr], y[tr]).predict(X[te])))
    return np.mean(s), np.std(s)

b5, sd5 = score(FEATURES, F5)
b10, sd10 = score(FEATURES, F10)
print('BASELINE all 9:  5km %.4f (+/-%.4f)   10km %.4f (+/-%.4f)' % (b5, sd5, b10, sd10))

print()
print('=== LEAVE-ONE-FEATURE-OUT (delta = drop_score - baseline; positive = feature was HURTING) ===')
print('%-28s %9s %9s %9s %9s' % ('dropped','R2_5km','d_5km','R2_10km','d_10km'))
lofo = []
for f in FEATURES:
    c = [x for x in FEATURES if x != f]
    a5, _ = score(c, F5); a10, _ = score(c, F10)
    lofo.append((f, a5-b5, a10-b10))
    print('%-28s %9.4f %+9.4f %9.4f %+9.4f' % (f, a5, a5-b5, a10, a10-b10))

print()
print('=== EACH FEATURE ALONE ===')
print('%-28s %9s %9s' % ('feature','R2_5km','R2_10km'))
for f in FEATURES:
    a5, _ = score([f], F5); a10, _ = score([f], F10)
    print('%-28s %9.4f %9.4f' % (f, a5, a10))

print()
print('=== VERDICT (5km, blocked) ===')
for f, d5, d10 in sorted(lofo, key=lambda t: -t[1]):
    if d5 > 0.005:   v = 'HARMFUL - remove'
    elif d5 > -0.005: v = 'USELESS - remove'
    elif d5 > -0.02:  v = 'marginal'
    else:             v = 'KEEP'
    print('%-28s drop-delta %+.4f  -> %s' % (f, d5, v))

keep = [f for f, d5, d10 in lofo if d5 <= -0.005]
print()
print('SURVIVING SET (%d): %s' % (len(keep), keep))
if len(keep) < len(FEATURES):
    k5, ks5 = score(keep, F5); k10, ks10 = score(keep, F10)
    print('pruned model:  5km %.4f (+/-%.4f)  vs baseline %.4f' % (k5, ks5, b5))
    print('pruned model: 10km %.4f (+/-%.4f)  vs baseline %.4f' % (k10, ks10, b10))
    print('RMSE 5km: pruned %.2f degC | full %.2f degC' % (y.std()*np.sqrt(max(0,1-k5)), y.std()*np.sqrt(max(0,1-b5))))

mo = XGBRegressor(**P).fit(m[FEATURES].values, y)
print()
print('=== GAIN IMPORTANCE (for contrast - do NOT trust) ===')
for f, g in sorted(zip(FEATURES, mo.feature_importances_), key=lambda t: -t[1]):
    print('%-28s %.4f' % (f, g))
