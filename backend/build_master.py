"""
build_master.py - canonical builder for grid_master.csv (SCHEMA_VERSION 2)

Rebuilds the training table from ONLY reproducible inputs:

  delhi_grid_filtered.geojson  <- filter_slivers.py <- clip_grid.py <- generate_grid.py
  grid_lst_ndvi.csv            <- fetch_grid_lst.py        (GEE)
  grid_indices.csv             <- step10_fetch_indices.py  (GEE)
  grid_density.csv             <- aggregate_density.py     (local, optional)

Replaces the old chain, which ran through grid_merged_v5.csv - a file with no
surviving producer script. Those merged files fed only DROPPED features
(elevation, population, land-use, dist_to_water), so nothing needed is lost.

Verifies every rebuilt column against the existing grid_master.csv before
overwriting it.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd

GRID = 'delhi_grid_filtered.geojson'
LST = 'grid_lst_ndvi.csv'
IDX = 'grid_indices.csv'
DEN = 'grid_density.csv'
OUT = 'grid_master.csv'

MODEL_FEATURES = ['albedo', 'ndbi', 'mndwi', 'bsi']

for f in (GRID, LST, IDX):
    if not os.path.exists(f):
        raise SystemExit('required input missing: %s' % f)

print('=== INPUTS ===')
g = gpd.read_file(GRID)
print('%-28s %5d rows' % (GRID, len(g)))
lst = pd.read_csv(LST)
print('%-28s %5d rows  cols %s' % (LST, len(lst), list(lst.columns)))
idx = pd.read_csv(IDX)
print('%-28s %5d rows  cols %s' % (IDX, len(idx), list(idx.columns)))
den = None
if os.path.exists(DEN):
    den = pd.read_csv(DEN)
    print('%-28s %5d rows  cols %s' % (DEN, len(den), list(den.columns)))

print()
print('=== GEOMETRY ===')
gp = g.to_crs(32643)
cent_utm = gp.geometry.centroid
cent_wgs = cent_utm.to_crs(4326)
geo = pd.DataFrame({
    'cell_id': g.cell_id.values,
    'lon': cent_wgs.x.values,
    'lat': cent_wgs.y.values,
    'cx': cent_utm.x.values,
    'cy': cent_utm.y.values,
    'area_m2': gp.geometry.area.values,
})
print('area_m2 min %.0f med %.0f max %.0f' % (geo.area_m2.min(), geo.area_m2.median(), geo.area_m2.max()))

print()
print('=== JOIN ===')
m = geo.merge(lst, on='cell_id', how='inner', validate='one_to_one')
print('after lst    : %d rows' % len(m))
m = m.merge(idx.drop(columns=[c for c in ('ndwi',) if c in idx.columns]),
            on='cell_id', how='inner', validate='one_to_one')
print('after indices: %d rows' % len(m))
if den is not None:
    m = m.merge(den, on='cell_id', how='inner', validate='one_to_one')
    print('after density: %d rows' % len(m))

if 'ndvi' in m.columns:
    m = m.rename(columns={'ndvi': 'ndvi_unmasked_legacy'})
    print()
    print('renamed ndvi -> ndvi_unmasked_legacy')
    print('  (built WITHOUT per-pixel QA_PIXEL masking, unlike the 4 model features;')
    print('   same year/city gave 0.128 here vs 0.270 QA-masked. Excluded from the model.)')

m = m.sort_values('cell_id').reset_index(drop=True)

print()
print('=== VERIFY AGAINST EXISTING grid_master.csv ===')
if os.path.exists(OUT):
    old = pd.read_csv(OUT).sort_values('cell_id').reset_index(drop=True)
    old.to_csv('grid_master_legacy.csv', index=False)
    print('backup written: grid_master_legacy.csv (%d cols)' % old.shape[1])
    if not np.array_equal(old.cell_id.values, m.cell_id.values):
        raise SystemExit('FATAL: cell_id sets differ from existing master')
    checks = [('lst', 'lst')] + [(c, c) for c in MODEL_FEATURES] + \
             [('cx', 'cx'), ('cy', 'cy'), ('lon', 'lon'), ('lat', 'lat'), ('area_m2', 'area_m2')]
    if 'ndvi' in old.columns:
        checks.append(('ndvi', 'ndvi_unmasked_legacy'))
    for oc in ('building_density', 'road_density'):
        if oc in old.columns and oc in m.columns:
            checks.append((oc, oc))
    bad = 0
    print('%-24s %14s %s' % ('column', 'max_abs_diff', 'status'))
    for oc, nc in checks:
        d = float(np.max(np.abs(old[oc].values - m[nc].values)))
        tol = 1e-4 if nc in ('lon', 'lat') else 1e-6
        ok = d < tol
        bad += (not ok)
        print('%-24s %14.2e %s' % (nc, d, 'MATCH' if ok else 'MISMATCH'))
    dropped = [c for c in old.columns if c not in m.columns and c != 'ndvi']
    print()
    print('columns dropped (%d): %s' % (len(dropped), dropped))
    print('  reason: these fed features removed in Steps 12/21 and came from')
    print('  grid_merged_v5.csv, which has no surviving producer script.')
    if bad:
        raise SystemExit('FATAL: %d columns mismatched. Not overwriting.' % bad)
    print()
    print('all shared columns match -> safe to overwrite')
else:
    print('no existing master; writing fresh')

print()
print('=== SANITY ===')
print('rows %d | cols %d' % m.shape)
print('nulls:', {k: int(v) for k, v in m.isna().sum().items() if v})
for c in MODEL_FEATURES:
    lo, hi = (0.0, 1.0) if c == 'albedo' else (-1.0, 1.0)
    n_out = int(((m[c] < lo) | (m[c] > hi)).sum())
    print('  %-8s min %+.4f max %+.4f  out-of-physical-range %d' % (c, m[c].min(), m[c].max(), n_out))
print('  %-8s min %.2f max %.2f mean %.2f' % ('lst', m.lst.min(), m.lst.max(), m.lst.mean()))

m.to_csv(OUT, index=False)
print()
print('WROTE %s  shape %s' % (OUT, (m.shape,)))
print('cols:', list(m.columns))

print()
print('=== DOWNSTREAM CHECK ===')
import importlib
import feature_engineering_v2 as fe
importlib.reload(fe)
df = fe.load_grid_data()
ok, problems = fe.validate_features(df)
print('feature_engineering_v2 validation:', 'PASS' if ok else 'FAIL %s' % problems)
X, y, info = fe.prepare_features(df)
print('X shape %s | y range %.2f-%.2f | extrapolation %d' % (X.shape, y.min(), y.max(),
      info['extrapolation_flags'].sum()))
