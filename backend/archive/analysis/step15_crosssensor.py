import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from scipy.spatial import cKDTree
from scipy import stats
from xgboost import XGBRegressor

m = pd.read_csv('grid_master_night.csv')
m = m.dropna(subset=['lst_day_modis']).reset_index(drop=True)
print('cells %d' % len(m))
print('Landsat day std %.2f | MODIS day std %.2f' % (m.lst.std(), m.lst_day_modis.std()))

SAT   = ['albedo','ndbi','mndwi','bsi','ndvi']
URBAN = ['building_density','road_density','population','elevation','dist_to_water']
cx, cy = m.cx.values, m.cy.values
GRID = [dict(n_estimators=300, max_depth=2, learning_rate=0.05),
        dict(n_estimators=600, max_depth=2, learning_rate=0.05),
        dict(n_estimators=1200, max_depth=2, learning_rate=0.02),
        dict(n_estimators=600, max_depth=3, learning_rate=0.05)]
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

print()
print('=== SAME-SENSOR vs CROSS-SENSOR (satellite indices as predictors) ===')
print('%-34s %9s %9s %9s' % ('','R2_5km','sd','RMSE_C'))
for target, label in [('lst','Landsat optical -> LANDSAT temp'),
                      ('lst_day_modis','Landsat optical -> MODIS temp')]:
    s = nested(SAT, 5, target)
    print('%-34s %9.4f %9.4f %9.2f' % (label, s.mean(), s.std(),
          m[target].std()*np.sqrt(max(0,1-s.mean()))))

print()
print('=== CONTROL: how well does MODIS temp predict LANDSAT temp directly? ===')
s = nested(['lst_day_modis'], 5, 'lst')
print('MODIS temp alone -> Landsat temp: R2 %.4f (ceiling for cross-sensor agreement)' % s.mean())

print()
print('=== URBAN FEATURES ON MODIS DAY TARGET (independent sensor) ===')
for cols, label in [(SAT,'SAT only'), (URBAN,'URBAN only'), (SAT+URBAN,'FULL')]:
    s = nested(cols, 5, 'lst_day_modis')
    print('%-12s R2 %.4f (sd %.4f)' % (label, s.mean(), s.std()))

print()
print('=== TRANSFER TEST: train on Landsat target, score against MODIS target ===')
X = m[SAT].values; yL = m.lst.values; yM = m.lst_day_modis.values
rL, rM = [], []
for tr, te in blocked(np.arange(len(m)), 5, 4, 500):
    mo = XGBRegressor(**{**FIXED, **GRID[1]}).fit(X[tr], yL[tr])
    p = mo.predict(X[te])
    rL.append(r2_score(yL[te], p))
    rM.append(np.corrcoef(p, yM[te])[0,1])
print('trained on Landsat: R2 vs Landsat %.4f | corr(pred, MODIS) %.4f' % (np.mean(rL), np.mean(rM)))

print()
print('=== VERDICT ===')
sL = nested(SAT, 5, 'lst').mean()
sM = nested(SAT, 5, 'lst_day_modis').mean()
print('same-sensor  %.4f' % sL)
print('cross-sensor %.4f' % sM)
if sM > 0.45:
    print('-> Relationship is PHYSICAL. Holds across independent sensors. Day model is valid.')
elif sM > 0.25:
    print('-> Partially physical. Some same-image advantage. Report both numbers.')
else:
    print('-> Mostly same-image artifact. Day model claim must be weakened substantially.')
