import geopandas as gpd, pandas as pd, numpy as np, sys

d = pd.read_csv('grid_merged_v5.csv')
g = gpd.read_file('delhi_grid_filtered.geojson')

print('=== JOIN INTEGRITY ===')
print('csv rows %d | geojson rows %d' % (len(d), len(g)))
print('csv cell_id dupes %d | geojson cell_id dupes %d' % (d.cell_id.duplicated().sum(), g.cell_id.duplicated().sum()))
print('set equal:', set(d.cell_id) == set(g.cell_id))
print('same order:', bool((d.cell_id.values == g.cell_id.values).all()))
if set(d.cell_id) != set(g.cell_id):
    print('FATAL: cell_id sets differ'); sys.exit(1)

gp = g.to_crs(32643)
geo = pd.DataFrame({
    'cell_id': g.cell_id.values,
    'area_m2': gp.geometry.area.values,
    'cx': gp.geometry.centroid.x.values,
    'cy': gp.geometry.centroid.y.values,
})

m = d.merge(geo, on='cell_id', how='inner', validate='one_to_one')
print('merged rows %d (expect %d)' % (len(m), len(d)))
assert len(m) == len(d)

print()
print('=== CENTROID CROSS-CHECK (csv lon/lat vs geojson geometry) ===')
chk = gpd.GeoDataFrame(geometry=gpd.points_from_xy(m.lon, m.lat), crs=4326).to_crs(32643)
off = np.hypot(chk.geometry.x.values - m.cx.values, chk.geometry.y.values - m.cy.values)
print('offset m: med %.1f  p99 %.1f  max %.1f  (want max < 5 m)' % (np.median(off), np.percentile(off,99), off.max()))
print('cells offset > 50 m: %d' % (off > 50).sum())

m['population_density'] = m.population / (m.area_m2 / 1e6)
print()
print('=== population -> population_density (per km2) ===')
print('count  : min %.0f  med %.0f  mean %.0f  max %.0f' % (m.population.min(), m.population.median(), m.population.mean(), m.population.max()))
print('density: min %.0f  med %.0f  mean %.0f  max %.0f' % (m.population_density.min(), m.population_density.median(), m.population_density.mean(), m.population_density.max()))
print('Delhi NCT actual ~11,000 /km2  -> our mean %.0f' % m.population_density.mean())
print('corr(area, population)         = %+.3f' % np.corrcoef(m.area_m2, m.population)[0,1])
print('corr(area, population_density) = %+.3f' % np.corrcoef(m.area_m2, m.population_density)[0,1])

m['pct_sum'] = m[['pct_residential','pct_industrial_commercial','pct_green_farm']].sum(axis=1)
print()
print('=== FLAGS CARRIED FORWARD (not fixed yet) ===')
print('pct_sum > 1.001        : %d cells (max %.3f)' % ((m.pct_sum > 1.001).sum(), m.pct_sum.max()))
print('building_density == 0  : %d cells (%.1f%%)' % ((m.building_density==0).sum(), 100*(m.building_density==0).mean()))
print('dist_to_water |r| w lst: %.3f  (noise, slated for removal)' % abs(np.corrcoef(m.dist_to_water, m.lst)[0,1]))

m = m.sort_values('cell_id').reset_index(drop=True)
m.to_csv('grid_master.csv', index=False)
print()
print('WROTE grid_master.csv  shape', m.shape)
print('cols:', list(m.columns))
