import geopandas as gpd
import osmnx as ox

grid = gpd.read_file("delhi_grid.geojson")
print(f"Before clip: {len(grid)} cells")

delhi_boundary = ox.geocode_to_gdf("National Capital Territory of Delhi, India")
delhi_boundary = delhi_boundary.to_crs(grid.crs)

# Keep only cells that intersect Delhi's actual boundary
clipped = gpd.overlay(grid, delhi_boundary[["geometry"]], how="intersection")
clipped["cell_id"] = range(len(clipped))

print(f"After clip: {len(clipped)} cells")
clipped.to_file("delhi_grid_clipped.geojson", driver="GeoJSON")
print("Saved delhi_grid_clipped.geojson")