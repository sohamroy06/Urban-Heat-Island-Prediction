import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

SEED = 42
FEATURES = ['ndvi','building_density','road_density','elevation','dist_to_water',
            'pct_residential','pct_industrial_commercial','pct_green_farm','population']
P = dict(n_estimators=100, max_depth=2, learning_rate=0.1, reg_alpha=1.0,
         reg_lambda=1.0, random_state=42, n_jobs=4,
         objective='reg:squarederror', verbosity=0)

m = pd.read_csv('grid_master.csv')
X = m[FEATURES].values
y = m.lst.values
cx, cy = m.cx.values, m.cy.values
print('cells %d | LST std %.3f' % (len(m), y.std()))

def fit_r2(tr, te):
    if te.sum() < 30 or tr.sum() < 200: return None
    mo = XGBRegressor(**P).fit(X[tr], y[tr])
    return r2_score(y[te], mo.predict(X[te]))

print()
print('=== A. RANDOM 5-FOLD x10 (the leaky number) ===')
s = []
for rep in range(10):
    for tr, te in KFold(5, shuffle=True, random_state=100+rep).split(X):
        a = np.zeros(len(m), bool); a[te] = True
        s.append(fit_r2(~a, a))
s = np.array([v for v in s if v is not None])
RANDOM = s.mean()
print('R2 %.4f +/- %.4f  (min %+.3f max %+.3f)' % (s.mean(), s.std(), s.min(), s.max()))

print()
print('=== B. BLOCKED CV vs BLOCK SIZE, with buffer ===')
print('buffer = cells within 1 block-width of test are dropped from train')
print('%8s %7s %9s %9s %8s %8s' % ('block_km','nblocks','R2_mean','R2_std','R2_min','n_folds'))
rows = []
for km in [1, 2, 5, 10, 15, 20]:
    B = km * 1000.0
    bx = np.floor((cx - cx.min())/B).astype(int)
    by = np.floor((cy - cy.min())/B).astype(int)
    blk = bx*10000 + by
    ub = np.unique(blk)
    k = 5 if len(ub) >= 15 else (3 if len(ub) >= 6 else 2)
    sc = []
    for rep in range(6):
        perm = np.random.default_rng(300+rep).permutation(ub)
        for f in np.array_split(perm, k):
            te = np.isin(blk, f)
            tex, tey = cx[te], cy[te]
            near = np.zeros(len(m), bool)
            for i in np.where(~te)[0]:
                if ((np.abs(tex-cx[i]) < B) & (np.abs(tey-cy[i]) < B)).any():
                    near[i] = True
            tr = (~te) & (~near)
            r = fit_r2(tr, te)
            if r is not None: sc.append(r)
    sc = np.array(sc)
    if len(sc) == 0:
        print('%8d %7d   -- too few blocks --' % (km, len(ub))); continue
    print('%8d %7d %9.4f %9.4f %+8.3f %8d' % (km, len(ub), sc.mean(), sc.std(), sc.min(), len(sc)))
    rows.append((km, sc.mean()))

print()
print('=== C. VERDICT ===')
for km, v in rows:
    print('  %2d km blocks: R2 %.4f   (random-split inflation %+.4f)' % (km, v, RANDOM - v))
if rows:
    hon = [v for km, v in rows if km >= 10]
    print()
    print('LEAKY (random split)   : %.4f  <- currently in your README' % RANDOM)
    if hon:
        print('HONEST (>=10km blocks) : %.4f  <- real generalization' % np.mean(hon))
        print('OVERSTATEMENT         : %.4f R2 points' % (RANDOM - np.mean(hon)))

B = 5000.0
m['block'] = (np.floor((cx-cx.min())/B).astype(int)*10000 + np.floor((cy-cy.min())/B).astype(int))
m[['cell_id','block','cx','cy']].to_csv('grid_blocks.csv', index=False)
print()
print('WROTE grid_blocks.csv (5 km blocks, %d unique)' % m.block.nunique())
