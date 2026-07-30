import ee, geopandas as gpd, pandas as pd, numpy as np, time, os, json
from sklearn.metrics import r2_score
import predict_grid as PG

ee.Initialize(project='shadowmap-502308')
print('GEE ok')
g = gpd.read_file('delhi_grid_filtered.geojson')
BOX = ee.Geometry.Rectangle([76.84, 28.40, 77.35, 28.88])
OUT = 'grid_2023.csv'

def qamask(img):
    qa = img.select('QA_PIXEL')
    return (qa.bitwiseAnd(1<<1).eq(0).And(qa.bitwiseAnd(1<<2).eq(0))
            .And(qa.bitwiseAnd(1<<3).eq(0)).And(qa.bitwiseAnd(1<<4).eq(0)))

def prep(img):
    m = qamask(img)
    sr = img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']).multiply(0.0000275).add(-0.2)
    st = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('lst')
    return sr.addBands(st).updateMask(m)

col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
       .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
       .filterDate('2023-04-01','2023-06-30').filterBounds(BOX)
       .filter(ee.Filter.lt('CLOUD_COVER', 20)).map(prep))
print('2023 scenes:', col.size().getInfo())

s = col.median()
B, G, R, N, S1, S2 = [s.select(b) for b in ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']]
albedo = (B.multiply(0.356).add(R.multiply(0.130)).add(N.multiply(0.373))
          .add(S1.multiply(0.085)).add(S2.multiply(0.072)).subtract(0.0018).divide(1.016)).rename('albedo')
ndbi  = S1.subtract(N).divide(S1.add(N)).rename('ndbi')
mndwi = G.subtract(S1).divide(G.add(S1)).rename('mndwi')
bsi   = (S1.add(R).subtract(N.add(B))).divide(S1.add(R).add(N).add(B)).rename('bsi')
ndvi  = N.subtract(R).divide(N.add(R)).rename('ndvi')
stack = albedo.addBands([ndbi, mndwi, bsi, ndvi, s.select('lst')])

def to_fc(sub):
    return ee.FeatureCollection([ee.Feature(ee.Geometry(geom.__geo_interface__), {'cell_id': int(c)})
                                 for c, geom in zip(sub.cell_id, sub.geometry)])

BANDS = ['albedo','ndbi','mndwi','bsi','ndvi','lst']
rows, t0 = [], time.time()
for i in range(0, len(g), 400):
    sub = g.iloc[i:i+400]
    fc = stack.reduceRegions(collection=to_fc(sub), reducer=ee.Reducer.mean(), scale=30)
    for f in fc.getInfo()['features']:
        p = f['properties']
        rows.append({'cell_id': p['cell_id'], **{b: p.get(b) for b in BANDS}})
    print('  %d/%d (%.0fs)' % (min(i+400, len(g)), len(g), time.time()-t0))

d = pd.DataFrame(rows).sort_values('cell_id').reset_index(drop=True)
d.to_csv(OUT, index=False)
print()
print('WROTE %s %s | nulls %s' % (OUT, d.shape, d.isna().sum().to_dict()))
d = d.dropna().reset_index(drop=True)
print('usable cells %d' % len(d))

m24 = pd.read_csv('grid_master.csv')
print()
print('=== 2023 vs 2024 DISTRIBUTION SHIFT ===')
print('%-10s %12s %12s %10s' % ('feature','2024_mean','2023_mean','delta'))
for c in BANDS:
    a = m24[c].mean(); b = d[c].mean()
    print('%-10s %12.4f %12.4f %+10.4f' % (c, a, b, b-a))

print()
print('=== TEMPORAL HOLDOUT: 2024-trained model -> 2023 truth ===')
r = PG.predict(d)
y = d.lst.values; p = r.pred_lst_c.values
r2 = r2_score(y, p); rmse = float(np.sqrt(np.mean((y-p)**2)))
bias = float(np.mean(p-y))
print('R2   %.4f   (blocked-CV on 2024 was 0.7714)' % r2)
print('RMSE %.3f C (2024 was 1.461 C)' % rmse)
print('bias %+.3f C' % bias)
print('extrapolation flagged: %d (%.1f%%)' % (r.extrapolation.sum(), 100*r.extrapolation.mean()))

print()
print('=== BIAS-CORRECTED (removes year-to-year offset, tests SPATIAL PATTERN skill) ===')
r2c = r2_score(y, p - bias)
print('R2 after removing constant offset: %.4f' % r2c)
print('corr(pred, truth): %.4f' % float(np.corrcoef(p, y)[0,1]))

cov = float(np.mean((y >= r.p10_c.values) & (y <= r.p90_c.values)))
print()
print('calibrated interval coverage on 2023: %.1f%% (target 80%%)' % (100*cov))

print()
print('=== VERDICT ===')
if r2c > 0.6:
    print('STRONG: spatial pattern transfers across years. Model is not year-specific.')
elif r2c > 0.35:
    print('MODERATE: pattern partly transfers. Report as a real limitation.')
else:
    print('WEAK: model is year-specific. Must be stated prominently.')
