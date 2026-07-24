import geopandas as gpd
import pandas as pd
import time

print("Loading grid...")
grid = gpd.read_file("delhi_grid_filtered.geojson")
print(f"Grid cells: {len(grid)}")

print("Loading buildings...")
buildings = gpd.read_file("delhi_all_buildings.geojson")
print("Loading roads...")
roads = gpd.read_file("delhi_all_roads.geojson")

# Reproject everything to UTM 43N for correct area/length math
grid_proj = grid.to_crs(epsg=32643)
buildings_proj = buildings.to_crs(epsg=32643)
roads_proj = roads.to_crs(epsg=32643)

print("Building spatial index and computing density per cell...")
t0 = time.time()

results = []
for idx, row in grid_proj.iterrows():
    cell_geom = row.geometry
    cell_area = cell_geom.area

    # Buildings intersecting this cell
    b_in_cell = buildings_proj[buildings_proj.geometry.intersects(cell_geom)]
    b_density = b_in_cell.geometry.intersection(cell_geom).area.sum() / cell_area if len(b_in_cell) > 0 else 0

    # Roads intersecting this cell
    r_in_cell = roads_proj[roads_proj.geometry.intersects(cell_geom)]
    r_length = r_in_cell.geometry.intersection(cell_geom).length.sum() if len(r_in_cell) > 0 else 0
    r_density = r_length / cell_area

    results.append({
        "cell_id": row["cell_id"],
        "building_density": b_density,
        "road_density": r_density,
    })

    if idx % 500 == 0:
        print(f"  {idx}/{len(grid_proj)} cells done ({time.time()-t0:.0f}s elapsed)")

df = pd.DataFrame(results)
df.to_csv("grid_density.csv", index=False)
print(f"\nDone in {time.time()-t0:.0f}s. Saved grid_density.csv")