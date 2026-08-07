import ee, geopandas as gpd, pandas as pd, time

ee.Initialize(project='shadowmap-502308')
print('GEE ok')

g = gpd.read_file('delhi_grid_filtered.geojson')
DELHI = ee.Geometry.Rectangle([76.84, 28.40, 77.35, 28.88])

raw = ee.ImageCollection('MODIS/061/MYD11A2').filterDate('2024-04-01','2024-06-30')
print('composites:', raw.size().getInfo())
print('bands:', ee.Image(raw.first()).bandNames().getInfo())

qc = ee.Image(raw.first()).select('QC_Night').bitwiseAnd(3)
hist = qc.reduceRegion(reducer=ee.Reducer.frequencyHistogram(), geometry=DELHI,
                       scale=1000, maxPixels=1e9).getInfo()
print('QC_Night bits0-1 histogram (first composite):', hist)
print('  0=good  1=produced/other quality  2=not produced(cloud)  3=not produced(other)')

def build(img):
    qn = img.select('QC_Night').bitwiseAnd(3)
    qd = img.select('QC_Day').bitwiseAnd(3)
    n = img.select('LST_Night_1km').multiply(0.02).subtract(273.15)
    d = img.select('LST_Day_1km').multiply(0.02).subtract(273.15)
    return (n.updateMask(qn.eq(0)).rename('night_strict')
            .addBands(n.updateMask(qn.lt(2)).rename('night_relaxed'))
            .addBands(n.rename('night_raw'))
            .addBands(d.updateMask(qd.lt(2)).rename('day_relaxed')))

s = raw.map(build).mean()

def to_fc(sub):
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry(geom.__geo_interface__), {'cell_id': int(cid)})
        for cid, geom in zip(sub.cell_id, sub.geometry)])

BANDS = ['night_strict','night_relaxed','night_raw','day_relaxed']
BATCH, rows, t0 = 400, [], time.time()
for i in range(0, len(g), BATCH):
    sub = g.iloc[i:i+BATCH]
    fc = s.reduceRegions(collection=to_fc(sub), reducer=ee.Reducer.mean(), scale=1000)
    for f in fc.getInfo()['features']:
        p = f['properties']
        rows.append({'cell_id': p['cell_id'], **{b: p.get(b) for b in BANDS}})
    print('  %d/%d (%.0fs)' % (min(i+BATCH, len(g)), len(g), time.time()-t0))

df = pd.DataFrame(rows).sort_values('cell_id').reset_index(drop=True)
print()
print('=== NULL COUNTS BY STRATEGY (of %d cells) ===' % len(df))
for b in BANDS:
    print('  %-16s nulls %5d (%.1f%%)' % (b, df[b].isna().sum(), 100*df[b].isna().mean()))

pick = None
for b in ['night_relaxed','night_raw','night_strict']:
    if df[b].isna().mean() < 0.10:
        pick = b; break
print()
if pick is None:
    print('ALL STRATEGIES FAILED - stop here and report')
else:
    print('USING: %s' % pick)
    df['lst_night'] = df[pick]
    out = df[['cell_id','lst_night','day_relaxed']].rename(columns={'day_relaxed':'lst_day_modis'})
    out.to_csv('grid_lst_night.csv', index=False)
    print('WROTE grid_lst_night.csv', out.shape)
    print(out.describe().T[['min','max','mean','std']])
    print('SANITY: Delhi Apr-Jun night surface ~22-32 C | MODIS day ~40-50 C')

    m = pd.read_csv('grid_master.csv').merge(out, on='cell_id', how='inner').dropna(subset=['lst_night'])
    print()
    print('cross-check MODIS day vs Landsat day: r = %+.3f (MODIS %.1f C, Landsat %.1f C)'
          % (m[['lst','lst_day_modis']].corr().iloc[0,1], m.lst_day_modis.mean(), m.lst.mean()))
    m['day_night_delta'] = m.lst - m.lst_night
    print()
    print('=== CORRELATIONS (n=%d) ===' % len(m))
    print('%-20s %12s %12s %12s' % ('feature','DAY_landsat','NIGHT','DAY-NIGHT'))
    for c in ['building_density','road_density','population_density','ndbi','bsi',
              'albedo','mndwi','ndvi','elevation','dist_to_water']:
        if c in m.columns:
            print('%-20s %+12.3f %+12.3f %+12.3f' % (c,
                  m[['lst',c]].corr().iloc[0,1],
                  m[['lst_night',c]].corr().iloc[0,1],
                  m[['day_night_delta',c]].corr().iloc[0,1]))
    print()
    print('=== MEAN LST BY BUILDING DENSITY QUARTILE ===')
    q = pd.qcut(m.building_density.rank(method='first'), 4,
                labels=['Q1 least built','Q2','Q3','Q4 most built'])
    print(m.groupby(q, observed=True)[['lst','lst_night','day_night_delta']].mean().round(2))
    bd_d = m[['lst','building_density']].corr().iloc[0,1]
    bd_n = m[['lst_night','building_density']].corr().iloc[0,1]
    print()
    print('building_density vs DAY %+.3f | vs NIGHT %+.3f | shift %+.3f' % (bd_d, bd_n, bd_n-bd_d))
