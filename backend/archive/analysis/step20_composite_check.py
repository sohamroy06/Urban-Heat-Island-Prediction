import ee, geopandas as gpd, pandas as pd, numpy as np, time
from sklearn.metrics import r2_score
import predict_grid as PG

ee.Initialize(project='shadowmap-502308')
BOX = ee.Geometry.Rectangle([76.84, 28.40, 77.35, 28.88])

def scenes(y0, y1):
    c = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
         .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
         .filterDate(y0, y1).filterBounds(BOX).filter(ee.Filter.lt('CLOUD_COVER', 20)))
    info = c.aggregate_array('system:index').getInfo()
    dates = c.aggregate_array('system:time_start').getInfo()
    cc = c.aggregate_array('CLOUD_COVER').getInfo()
    d = pd.DataFrame({'id': info, 'date': pd.to_datetime(dates, unit='ms'), 'cloud': cc})
    return d.sort_values('date').reset_index(drop=True)

print('=== SCENE INVENTORY ===')
for lab, a, b in [('2024','2024-04-01','2024-06-30'), ('2023','2023-04-01','2023-06-30')]:
    d = scenes(a, b)
    print()
    print('%s: %d scenes | mean cloud %.1f%%' % (lab, len(d), d.cloud.mean()))
    print('  by month:', d.date.dt.month.value_counts().sort_index().to_dict())
    print('  date range: %s to %s' % (d.date.min().date(), d.date.max().date()))
    print('  scenes in June after the 15th: %d' % (((d.date.dt.month==6) & (d.date.dt.day>15)).sum()))

print()
print('=== DATE-MATCHED TEST: April 1 - June 10 only, both years ===')
g = gpd.read_file('delhi_grid_filtered.geojson')

def build(y0, y1):
    def prep(img):
        qa = img.select('QA_PIXEL')
        msk = (qa.bitwiseAnd(1<<1).eq(0).And(qa.bitwiseAnd(1<<2).eq(0))
               .And(qa.bitwiseAnd(1<<3).eq(0)).And(qa.bitwiseAnd(1<<4).eq(0)))
        sr = img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']).multiply(0.0000275).add(-0.2)
        st = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('lst')
        return sr.addBands(st).updateMask(msk)
    c = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
         .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
         .filterDate(y0, y1).filterBounds(BOX).filter(ee.Filter.lt('CLOUD_COVER', 20)).map(prep))
    n = c.size().getInfo()
    s = c.median()
    B, G, R, N, S1, S2 = [s.select(b) for b in ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']]
    alb = (B.multiply(0.356).add(R.multiply(0.130)).add(N.multiply(0.373))
           .add(S1.multiply(0.085)).add(S2.multiply(0.072)).subtract(0.0018).divide(1.016)).rename('albedo')
    return n, alb.addBands([
        S1.subtract(N).divide(S1.add(N)).rename('ndbi'),
        G.subtract(S1).divide(G.add(S1)).rename('mndwi'),
        (S1.add(R).subtract(N.add(B))).divide(S1.add(R).add(N).add(B)).rename('bsi'),
        N.subtract(R).divide(N.add(R)).rename('ndvi'),
        s.select('lst')])

def fetch(stack, tag):
    BANDS = ['albedo','ndbi','mndwi','bsi','ndvi','lst']
    rows, t0 = [], time.time()
    for i in range(0, len(g), 400):
        sub = g.iloc[i:i+400]
        fc = stack.reduceRegions(
            collection=ee.FeatureCollection([
                ee.Feature(ee.Geometry(gm.__geo_interface__), {'cell_id': int(c)})
                for c, gm in zip(sub.cell_id, sub.geometry)]),
            reducer=ee.Reducer.mean(), scale=30)
        for f in fc.getInfo()['features']:
            p = f['properties']
            rows.append({'cell_id': p['cell_id'], **{b: p.get(b) for b in BANDS}})
    print('  %s fetched (%.0fs)' % (tag, time.time()-t0))
    return pd.DataFrame(rows).sort_values('cell_id').reset_index(drop=True)

n24, s24 = build('2024-04-01','2024-06-10')
n23, s23 = build('2023-04-01','2023-06-10')
print('date-matched scene counts: 2024=%d  2023=%d' % (n24, n23))
d24 = fetch(s24, '2024'); d23 = fetch(s23, '2023')
d24.to_csv('grid_2024_matched.csv', index=False); d23.to_csv('grid_2023_matched.csv', index=False)

print()
print('%-10s %12s %12s %10s' % ('feature','2024_matched','2023_matched','delta'))
for c in ['albedo','ndbi','mndwi','bsi','ndvi','lst']:
    print('%-10s %12.4f %12.4f %+10.4f' % (c, d24[c].mean(), d23[c].mean(), d23[c].mean()-d24[c].mean()))

print()
print('=== TRANSFER ON DATE-MATCHED 2023 ===')
d23c = d23.dropna().reset_index(drop=True)
r = PG.predict(d23c)
y = d23c.lst.values; p = r.pred_lst_c.values
bias = float(np.mean(p-y))
print('R2 raw          %.4f' % r2_score(y, p))
print('R2 bias-corrected %.4f' % r2_score(y, p-bias))
print('corr            %.4f' % float(np.corrcoef(p, y)[0,1]))
print('bias            %+.3f C' % bias)
print('RMSE            %.3f C' % float(np.sqrt(np.mean((y-p)**2))))
print('extrapolation flagged %.1f%%' % (100*r.extrapolation.mean()))

print()
print('=== INTERPRETATION ===')
print('If date-matched NDVI gap shrinks a lot vs the +0.197 seen with full Jun 30 window,')
print('the earlier 2023 composite was monsoon-contaminated and R2=0.0142 was a fetch artifact.')
