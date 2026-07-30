import numpy as np, pandas as pd
from scipy import stats

m = pd.read_csv('grid_master.csv')
print('cells %d' % len(m))

print()
print('=== NDBI DISTRIBUTION (built-up proxy) ===')
print(m.ndbi.describe([.1,.25,.5,.75,.9]).round(4).to_string())

print()
print('=== NDBI vs OSM building_density AGREEMENT ===')
has = m.building_density > 0
print('cells with OSM buildings: %d (%.1f%%)' % (has.sum(), 100*has.mean()))
print('corr(ndbi, building_density) all cells   : %+.3f' % m[['ndbi','building_density']].corr().iloc[0,1])
print('corr(ndbi, building_density) OSM-present : %+.3f' % m[has][['ndbi','building_density']].corr().iloc[0,1])
print()
print('mean values by OSM building_density quartile (OSM-present cells):')
q = pd.qcut(m[has].building_density.rank(method='first'), 4, labels=['Q1 low','Q2','Q3','Q4 high'])
print(m[has].groupby(q, observed=True).agg(
    n=('ndbi','size'), mean_ndbi=('ndbi','mean'),
    mean_albedo=('albedo','mean'), mean_lst=('lst','mean')).round(4).to_string())

def classify(r):
    if r['mndwi'] > -0.15: return 'water'
    if r['ndvi']  >  0.22: return 'vegetated'
    if r['ndbi']  >  0.02: return 'built_up'
    if r['bsi']   >  0.10: return 'bare_soil'
    return 'mixed'

m['surface'] = m.apply(classify, axis=1)
print()
print('=== SURFACE CLASSIFICATION ===')
print(m.groupby('surface', observed=True).agg(
    n=('lst','size'),
    pct=('lst', lambda v: round(100.0*len(v)/len(m), 1)),
    mean_lst=('lst','mean'), mean_albedo=('albedo','mean'),
    mean_ndbi=('ndbi','mean'), mean_bsi=('bsi','mean'),
    mean_osm_bd=('building_density','mean')).round(4).to_string())

print()
print('=== THE GATE: corr(albedo, LST) WITHIN each surface type ===')
print('%-12s %7s %10s %12s %14s' % ('surface','n','corr','p_value','verdict'))
for s in sorted(m.surface.unique()):
    d = m[m.surface == s]
    if len(d) < 30:
        print('%-12s %7d   too few cells' % (s, len(d))); continue
    r, p = stats.pearsonr(d.albedo, d.lst)
    v = 'NEGATIVE ok' if r < -0.10 else ('flat' if r < 0.10 else 'POSITIVE bad')
    print('%-12s %7d %10.3f %12.2e %14s' % (s, len(d), r, p, v))

print()
print('=== BUILT-UP ONLY, VARYING NDBI STRICTNESS ===')
print('(water and vegetation already excluded)')
print('%-18s %7s %10s %12s %12s' % ('threshold','n','corr','p_value','mean_albedo'))
for th in [0.00, 0.02, 0.04, 0.06, 0.08]:
    d = m[(m.ndbi > th) & (m.mndwi <= -0.15) & (m.ndvi <= 0.22)]
    if len(d) < 30:
        print('%-18s %7d  too few' % ('ndbi > %.2f' % th, len(d))); continue
    r, p = stats.pearsonr(d.albedo, d.lst)
    print('%-18s %7d %10.3f %12.2e %12.4f' % ('ndbi > %.2f' % th, len(d), r, p, d.albedo.mean()))

print()
print('=== OSM-CONFIRMED BUILT-UP (ndbi>0.02 AND building_density>0.02) ===')
d = m[(m.ndbi > 0.02) & (m.building_density > 0.02) & (m.mndwi <= -0.15) & (m.ndvi <= 0.22)]
if len(d) >= 30:
    r, p = stats.pearsonr(d.albedo, d.lst)
    print('n=%d  corr(albedo,lst) = %+.3f  p=%.2e' % (len(d), r, p))
    print('albedo range %.3f to %.3f (mean %.3f)' % (d.albedo.min(), d.albedo.max(), d.albedo.mean()))
    print('lst    range %.2f to %.2f (mean %.2f)' % (d.lst.min(), d.lst.max(), d.lst.mean()))
else:
    print('only %d cells - too few' % len(d))

print()
print('=== PARTIAL CORRELATION: albedo vs LST, controlling for bsi and ndvi ===')
print('(statistically removes the dry-soil confound, all cells)')
def resid(target, ctrl, df):
    C = np.c_[np.ones(len(df)), df[ctrl].values]
    beta, *_ = np.linalg.lstsq(C, df[target].values, rcond=None)
    return df[target].values - C @ beta
ra = resid('albedo', ['bsi','ndvi'], m)
rl = resid('lst',    ['bsi','ndvi'], m)
pr, pp = stats.pearsonr(ra, rl)
print('raw     corr(albedo, lst) = %+.3f' % m[['albedo','lst']].corr().iloc[0,1])
print('partial corr(albedo, lst) = %+.3f   p=%.2e' % (pr, pp))

print()
print('=== GATE DECISION ===')
d = m[(m.ndbi > 0.02) & (m.mndwi <= -0.15) & (m.ndvi <= 0.22)]
r, p = stats.pearsonr(d.albedo, d.lst)
print('built-up cells: n=%d  corr=%+.3f  p=%.2e' % (len(d), r, p))
if r < -0.10 and p < 0.05 and len(d) > 300:
    print('-> PASS: albedo behaves physically in built-up areas. OPTION B VIABLE.')
elif r < 0.05:
    print('-> MARGINAL: effect weak or flat. Option B possible but with wide uncertainty.')
else:
    print('-> FAIL: albedo still positive in built-up areas. FALL BACK TO OPTION A.')
