import ee, geopandas as gpd, pandas as pd, numpy as np, time, os

ee.Initialize(project='shadowmap-502308')
print('GEE ok')

g = gpd.read_file('delhi_grid_filtered.geojson')
gt = g.geometry.geom_type.value_counts().to_dict()
print('cells %d | geom types %s' % (len(g), gt))

OUT = 'grid_indices.csv'
done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    done = set(prev.cell_id.tolist())
    print('resume: %d cells already fetched' % len(done))

def prep(img):
    sr = img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']) \
            .multiply(0.0000275).add(-0.2)
    qa = img.select('QA_PIXEL')
    clear = qa.bitwiseAnd(1<<1).eq(0).And(qa.bitwiseAnd(1<<2).eq(0)) \
              .And(qa.bitwiseAnd(1<<3).eq(0)).And(qa.bitwiseAnd(1<<4).eq(0))
    return sr.updateMask(clear).copyProperties(img, ['system:time_start'])

col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
       .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
       .filterDate('2024-04-01','2024-06-30')
       .filterBounds(ee.Geometry.Rectangle([76.84,28.40,77.35,28.88]))
       .filter(ee.Filter.lt('CLOUD_COVER', 20))
       .map(prep))
print('scenes', col.size().getInfo())

s = col.median()
B, G, R, N, S1, S2 = [s.select(b) for b in ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']]
albedo = (B.multiply(0.356).add(R.multiply(0.130)).add(N.multiply(0.373))
          .add(S1.multiply(0.085)).add(S2.multiply(0.072)).subtract(0.0018)
          .divide(1.016)).rename('albedo')
ndbi  = S1.subtract(N).divide(S1.add(N)).rename('ndbi')
mndwi = G.subtract(S1).divide(G.add(S1)).rename('mndwi')
ndwi  = N.subtract(S1).divide(N.add(S1)).rename('ndwi')
bsi   = (S1.add(R).subtract(N.add(B))).divide(S1.add(R).add(N).add(B)).rename('bsi')
stack = albedo.addBands([ndbi, mndwi, ndwi, bsi])

todo = g[~g.cell_id.isin(done)].reset_index(drop=True)
print('to fetch: %d cells' % len(todo))

def to_fc(sub):
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry(geom.__geo_interface__), {'cell_id': int(cid)})
        for cid, geom in zip(sub.cell_id, sub.geometry)])

BATCH, rows, t0 = 400, [], time.time()
for i in range(0, len(todo), BATCH):
    sub = todo.iloc[i:i+BATCH]
    fc = stack.reduceRegions(collection=to_fc(sub), reducer=ee.Reducer.mean(), scale=30)
    for f in fc.getInfo()['features']:
        p = f['properties']
        rows.append({'cell_id': p['cell_id'],
                     **{k: p.get(k) for k in ['albedo','ndbi','mndwi','ndwi','bsi']}})
    df = pd.DataFrame(rows)
    if done:
        df = pd.concat([prev, df], ignore_index=True)
    df.drop_duplicates('cell_id').sort_values('cell_id').to_csv(OUT, index=False)
    print('  %d/%d  (%.0fs)' % (min(i+BATCH, len(todo)), len(todo), time.time()-t0))

df = pd.read_csv(OUT).sort_values('cell_id').reset_index(drop=True)
print()
print('WROTE %s  shape %s' % (OUT, df.shape))
print('expected 6709 rows | got %d | missing %d' % (len(df), 6709-len(df)))
print('nulls:', df.isna().sum().to_dict())
print(df.describe().T[['min','max','mean','std']])
print()
print('SANITY:')
for c in ['albedo','ndbi','mndwi','ndwi','bsi']:
    v = df[c].dropna()
    lo, hi = (0.0, 0.6) if c == 'albedo' else (-1.0, 1.0)
    print('  %-7s min %+.3f max %+.3f  out-of-range: %d' % (c, v.min(), v.max(), ((v<lo)|(v>hi)).sum()))

if os.path.exists('grid_master.csv'):
    m = pd.read_csv('grid_master.csv')[['cell_id','lst','ndvi']]
    j = m.merge(df, on='cell_id', how='inner')
    print()
    print('=== CORRELATION WITH LST (n=%d) ===' % len(j))
    for c in ['ndvi','albedo','ndbi','mndwi','ndwi','bsi']:
        print('  %-7s r = %+.3f' % (c, j[['lst',c]].dropna().corr().iloc[0,1]))
