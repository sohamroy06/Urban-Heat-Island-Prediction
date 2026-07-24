import geopandas as gpd
import numpy as np
from shapely.geometry import box

# Delhi bounding box (same as repo's DELHI_BOUNDS)
MIN_LAT, MAX_LAT = 28.40, 28.88
MIN_LON, MAX_LON = 76.84, 77.35

CELL_SIZE_DEG = 0.0045  # ~500m at Delhi's latitude

lats = np.arange(MIN_LAT, MAX_LAT, CELL_SIZE_DEG)
lons = np.arange(MIN_LON, MAX_LON, CELL_SIZE_DEG)

cells = []
cell_id = 0
for lat in lats:
    for lon in lons:
        cells.append({
            "cell_id": cell_id,
            "geometry": box(lon, lat, lon + CELL_SIZE_DEG, lat + CELL_SIZE_DEG)
        })
        cell_id += 1

grid = gpd.GeoDataFrame(cells, crs="EPSG:4326")
print(f"Total grid cells: {len(grid)}")

grid.to_file("delhi_grid.geojson", driver="GeoJSON")
print("Saved delhi_grid.geojson")