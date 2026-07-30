import ee, geopandas as gpd, pandas as pd
ee.Initialize(project='shadowmap-502308')

wards = gpd.read_file("delhi_wards_valid.geojson")

delhi_bounds = ee.Geometry.Rectangle([76.84, 28.40, 77.35, 28.88])

collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
              .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
              .filterBounds(delhi_bounds)
              .filterDate('2024-04-01', '2024-06-30')
              .filter(ee.Filter.lt('CLOUD_COVER', 20)))

print("Images in composite:", collection.size().getInfo())

# Median composite — one consistent snapshot across all of Delhi
composite = collection.median()

lst = composite.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15)
ndvi = composite.normalizedDifference(['SR_B5', 'SR_B4'])
combined = lst.rename('lst').addBands(ndvi.rename('ndvi'))

rows = []
for idx, row in wards.iterrows():
    geom = ee.Geometry(row.geometry.__geo_interface__)
    stats = combined.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1e9
    ).getInfo()
    rows.append({"ward_id": idx, "lst": stats.get('lst'), "ndvi": stats.get('ndvi')})
    print(f"done {idx+1}/{len(wards)}")

pd.DataFrame(rows).to_csv("real_lst_ndvi_v3.csv", index=False)
print("Saved real_lst_ndvi_v3.csv")