import ee
import geopandas as gpd
import pandas as pd
import time
import os

ee.Initialize(project='shadowmap-502308')

grid = gpd.read_file("delhi_grid_filtered.geojson")
print(f"Grid cells: {len(grid)}")

delhi_bounds = ee.Geometry.Rectangle([76.84, 28.40, 77.35, 28.88])

collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
              .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
              .filterBounds(delhi_bounds)
              .filterDate('2024-04-01', '2024-06-30')
              .filter(ee.Filter.lt('CLOUD_COVER', 20)))

print("Images in composite:", collection.size().getInfo())
composite = collection.median()

lst = composite.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
ndvi = composite.normalizedDifference(['SR_B5', 'SR_B4'])
combined = lst.rename('lst').addBands(ndvi.rename('ndvi'))

OUT_FILE = "grid_lst_ndvi.csv"
done_ids = set()
if os.path.exists(OUT_FILE):
    prev = pd.read_csv(OUT_FILE)
    done_ids = set(prev["cell_id"].tolist())
    print(f"Resuming, {len(done_ids)} cells already done")

rows = [] if not done_ids else prev.to_dict("records")

t0 = time.time()
for idx, row in grid.iterrows():
    cell_id = row["cell_id"]
    if cell_id in done_ids:
        continue

    geom = ee.Geometry(row.geometry.__geo_interface__)
    stats = combined.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1e9
    ).getInfo()

    rows.append({"cell_id": cell_id, "lst": stats.get('lst'), "ndvi": stats.get('ndvi')})

    if idx % 200 == 0:
        pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
        print(f"  {idx}/{len(grid)} done ({time.time()-t0:.0f}s elapsed)")

pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
print(f"Finished. Saved {OUT_FILE}")