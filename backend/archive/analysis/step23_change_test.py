import numpy as np, pandas as pd
from scipy import stats

a = pd.read_csv('grid_2024_matched.csv')
b = pd.read_csv('grid_2023_matched.csv')
print('2024 %s | 2023 %s' % (a.shape, b.shape))
d = a.merge(b, on='cell_id', suffixes=('_24','_23')).dropna().reset_index(drop=True)
print('paired cells %d' % len(d))

V = ['albedo','ndbi','mndwi','bsi','ndvi','lst']
for v in V:
    d['d_'+v] = d[v+'_24'] - d[v+'_23']

print()
print('=== CHANGE 2023 -> 2024 (per cell) ===')
print('%-8s %10s %10s %10s %10s' % ('var','mean_d','sd_d','min_d','max_d'))
for v in V:
    s = d['d_'+v]
    print('%-8s %10.4f %10.4f %10.4f %10.4f' % (v, s.mean(), s.std(), s.min(), s.max()))

print()
print('=== IS ALBEDO CHANGE REAL SIGNAL OR NOISE? ===')
print('spatial sd of albedo within 2024 : %.4f' % d.albedo_24.std())
print('sd of per-cell albedo change     : %.4f' % d.d_albedo.std())
print('ratio (change / spatial spread)  : %.2f' % (d.d_albedo.std()/d.albedo_24.std()))
print('corr(albedo_23, albedo_24)       : %+.3f  (high = stable, change is small)'
      % d[['albedo_23','albedo_24']].corr().iloc[0,1])

print()
print('=== RAW: change in albedo vs change in LST ===')
r, p = stats.pearsonr(d.d_albedo, d.d_lst)
print('corr(d_albedo, d_lst) = %+.3f  p=%.2e  n=%d' % (r, p, len(d)))

def ols(y, Xc, names):
    X = np.c_[np.ones(len(y)), Xc]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    print('%-14s %12s %10s %10s' % ('term','coef','se','t'))
    for i, nm in enumerate(['intercept'] + names):
        print('%-14s %12.4f %10.4f %10.2f' % (nm, beta[i], se[i], beta[i]/se[i]))
    ss = 1 - float(resid @ resid) / float(((y-y.mean())**2).sum())
    print('R2 %.4f' % ss)
    return beta, se

print()
print('=== CONTROLLED: d_lst ~ d_albedo + d_bsi + d_ndvi + d_mndwi ===')
print('(within-cell differencing already removes fixed geometry/shadow/land-use)')
ctrl = ['d_albedo','d_bsi','d_ndvi','d_mndwi']
beta, se = ols(d.d_lst.values, d[ctrl].values, ctrl)
print()
print('-> albedo coefficient = %+.3f degC per 1.0 albedo' % beta[1])
print('-> per +0.01 albedo   = %+.4f degC' % (beta[1]/100))

print()
print('=== SAME TEST, OSM-CONFIRMED BUILT-UP CELLS ONLY ===')
m = pd.read_csv('grid_master.csv')[['cell_id','building_density']]
d2 = d.merge(m, on='cell_id')
bu = d2[d2.building_density > 0.05]
print('cells with building_density > 0.05: %d' % len(bu))
if len(bu) > 100:
    r2_, p2_ = stats.pearsonr(bu.d_albedo, bu.d_lst)
    print('raw corr(d_albedo, d_lst) = %+.3f  p=%.2e' % (r2_, p2_))
    print()
    b2, s2 = ols(bu.d_lst.values, bu[ctrl].values, ctrl)
    print()
    print('-> built-up albedo coefficient = %+.3f degC per 1.0 albedo' % b2[1])
    print('-> per +0.01 albedo            = %+.4f degC' % (b2[1]/100))
else:
    print('too few cells')

print()
print('=== BINNED: mean d_lst by quintile of d_albedo (all cells) ===')
q = pd.qcut(d.d_albedo, 5, labels=['darkened most','Q2','no change','Q4','brightened most'])
print(d.groupby(q, observed=True).agg(
    n=('d_lst','size'), mean_d_albedo=('d_albedo','mean'),
    mean_d_lst=('d_lst','mean'), mean_d_bsi=('d_bsi','mean'),
    mean_d_ndvi=('d_ndvi','mean')).round(4).to_string())

print()
print('=== VERDICT ===')
c = beta[1]
if c < -2.0:
    print('PASS: brightening a cell is associated with COOLING. Option B viable.')
    print('      slider effect ~ %.3f degC per +0.01 albedo' % (c/100))
elif c < 0:
    print('WEAK: direction correct but effect small. Option B only with wide error bars.')
else:
    print('FAIL: brightening still associated with warming even within the same cell.')
    print('      Albedo is not usable as a causal lever in this dataset. GO TO OPTION A.')
