import geopandas as gpd
import pandas as pd
import glob

building_files = sorted(glob.glob("osm_tiles/buildings_tile_*.geojson"))
road_files = sorted(glob.glob("osm_tiles/roads_tile_*.geojson"))

print(f"Merging {len(building_files)} building tiles...")
buildings = pd.concat([gpd.read_file(f) for f in building_files], ignore_index=True)
buildings = gpd.GeoDataFrame(buildings, crs="EPSG:4326")
print(f"Total buildings: {len(buildings)}")
buildings.to_file("delhi_all_buildings.geojson", driver="GeoJSON")

print(f"Merging {len(road_files)} road tiles...")
roads = pd.concat([gpd.read_file(f) for f in road_files], ignore_index=True)
roads = gpd.GeoDataFrame(roads, crs="EPSG:4326")
print(f"Total road segments: {len(roads)}")
roads.to_file("delhi_all_roads.geojson", driver="GeoJSON")

print("Saved delhi_all_buildings.geojson and delhi_all_roads.geojson")