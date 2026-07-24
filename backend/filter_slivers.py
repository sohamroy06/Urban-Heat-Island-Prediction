import geopandas as gpd

grid = gpd.read_file("delhi_grid_clipped.geojson")
print(f"Before filter: {len(grid)} cells")

# Reproject to compute real area in m²
grid_proj = grid.to_crs(epsg=32643)
grid["area_m2"] = grid_proj.geometry.area

# Full cell at 500m ≈ 250,000 m². Keep cells with at least 50% coverage.
MIN_AREA = 250000 * 0.5
filtered = grid[grid["area_m2"] >= MIN_AREA].reset_index(drop=True)
filtered["cell_id"] = range(len(filtered))

print(f"After filter: {len(filtered)} cells")
filtered.to_file("delhi_grid_filtered.geojson", driver="GeoJSON")
print("Saved delhi_grid_filtered.geojson")