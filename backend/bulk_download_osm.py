import osmnx as ox
import numpy as np
import time
import os

MIN_LAT, MAX_LAT = 28.40, 28.88
MIN_LON, MAX_LON = 76.84, 77.35

# 3x2 grid = 6 regional tiles
lat_splits = np.linspace(MIN_LAT, MAX_LAT, 3)
lon_splits = np.linspace(MIN_LON, MAX_LON, 4)

tiles = []
for i in range(len(lat_splits) - 1):
    for j in range(len(lon_splits) - 1):
        tiles.append((lat_splits[i], lat_splits[i+1], lon_splits[j], lon_splits[j+1]))

print(f"Total tiles: {len(tiles)}")

os.makedirs("osm_tiles", exist_ok=True)

for idx, (lat_min, lat_max, lon_min, lon_max) in enumerate(tiles):
    b_path = f"osm_tiles/buildings_tile_{idx}.geojson"
    r_path = f"osm_tiles/roads_tile_{idx}.geojson"

    if os.path.exists(b_path) and os.path.exists(r_path):
        print(f"Tile {idx} already done, skipping")
        continue

    bbox = (lon_min, lat_min, lon_max, lat_max)
    print(f"\nTile {idx}/{len(tiles)}: bbox={bbox}")

    if not os.path.exists(b_path):
        try:
            t0 = time.time()
            buildings = ox.features_from_bbox(bbox, tags={"building": True})
            buildings.to_file(b_path, driver="GeoJSON")
            print(f"  buildings: {len(buildings)} in {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"  buildings FAILED: {e}")

    if not os.path.exists(r_path):
        try:
            t0 = time.time()
            roads = ox.graph_from_bbox(bbox, network_type="drive")
            roads_gdf = ox.graph_to_gdfs(roads, nodes=False, edges=True)
            roads_gdf.to_file(r_path, driver="GeoJSON")
            print(f"  roads: done in {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"  roads FAILED: {e}")

print("\nAll tiles processed.")