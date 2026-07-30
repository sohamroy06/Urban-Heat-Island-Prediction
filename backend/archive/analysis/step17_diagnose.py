import numpy as np, pandas as pd, geopandas as gpd, json

p = pd.read_csv('grid_predictions.csv')
m = pd.read_csv('grid_master.csv')
d = p.merge(m.drop(columns=['lon','lat','lst']), on='cell_id', how='inner')
d['abs_res'] = d.residual.abs()

print('=== SPATIAL CLUSTERING OF LARGE ERRORS ===')
bad = d[d.abs_res > 4]
print('cells with |error| > 4 C: %d' % len(bad))
print('their lon range %.3f-%.3f | lat range %.3f-%.3f' % (bad.lon.min(), bad.lon.max(), bad.lat.min(), bad.lat.max()))
print('city lon range  %.3f-%.3f | lat range %.3f-%.3f' % (d.lon.min(), d.lon.max(), d.lat.min(), d.lat.max()))
print('under-predicted (model too cool): %d | over-predicted: %d' % ((bad.residual<0).sum(), (bad.residual>0).sum()))

print()
print('=== SE CORNER CLUSTER (lon>77.27, lat<28.55) ===')
cl = d[(d.lon > 77.27) & (d.lat < 28.55)]
rest = d[~((d.lon > 77.27) & (d.lat < 28.55))]
print('cluster n=%d | mean |err| %.2f  vs rest %.2f' % (len(cl), cl.abs_res.mean(), rest.abs_res.mean()))
print()
print('%-22s %10s %10s' % ('feature','cluster','rest'))
for c in ['lst','pred','albedo','ndbi','bsi','mndwi','ndvi','building_density','road_density','population_density','area_m2']:
    if c in d.columns:
        print('%-22s %10.3f %10.3f' % (c, cl[c].mean(), rest[c].mean()))

print()
print('=== IS IT A CLIPPING ARTIFACT? ===')
full = d.area_m2.max()
print('cluster cells with partial area (<99%% full): %d of %d (%.1f%%)' % ((cl.area_m2 < 0.99*full).sum(), len(cl), 100*(cl.area_m2 < 0.99*full).mean()))
print('city-wide partial rate: %.1f%%' % (100*(d.area_m2 < 0.99*full).mean()))

print()
print('=== TOP 10 WORST CELLS, FULL PROFILE ===')
w = d.reindex(d.abs_res.sort_values(ascending=False).index).head(10)
print(w[['cell_id','lon','lat','lst','pred','residual','albedo','ndbi','bsi','ndvi','building_density']].round(3).to_string(index=False))

print()
print('=== ERROR vs DISTANCE TO NCT BOUNDARY ===')
g = gpd.read_file('delhi_grid_filtered.geojson').to_crs(32643)
edge = g.union_all().boundary
cent = gpd.GeoSeries(gpd.points_from_xy(d.cx, d.cy), crs=32643)
d['dist_edge'] = cent.distance(edge).values
q = pd.qcut(d.dist_edge, 5, labels=['nearest edge','Q2','Q3','Q4','deep interior'])
print(d.groupby(q, observed=True).agg(n=('abs_res','size'), mean_abs_err=('abs_res','mean'), mean_signed=('residual','mean')).round(3))
