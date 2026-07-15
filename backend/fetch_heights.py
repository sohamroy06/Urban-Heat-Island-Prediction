import osmnx as ox
import geopandas as gpd
import pandas as pd

wards = gpd.read_file("delhi_wards_valid.geojson")
density = pd.read_csv("real_density.csv")
usable_ids = density[~((density["building_density"] == 0) & (density["road_density"] == 0))]["ward_id"].tolist()

rows = []
for idx in usable_ids:
    row = wards.iloc[idx]
    try:
        buildings = ox.features_from_polygon(row.geometry, tags={"building": True})
        heights = []
        if "building:levels" in buildings.columns:
            levels = pd.to_numeric(buildings["building:levels"], errors="coerce").dropna()
            heights = (levels * 3.0).tolist()  # ~3m per floor, standard estimate
        if "height" in buildings.columns:
            direct_h = pd.to_numeric(buildings["height"], errors="coerce").dropna()
            heights.extend(direct_h.tolist())

        avg_height = sum(heights) / len(heights) if heights else None
        rows.append({"ward_id": idx, "avg_building_height": avg_height})
        print(f"ward {idx}: avg_height={avg_height}, n_tagged={len(heights)}")
    except Exception as e:
        print(f"ward {idx} failed: {e}")
        rows.append({"ward_id": idx, "avg_building_height": None})

pd.DataFrame(rows).to_csv("real_heights.csv", index=False)
print("Saved real_heights.csv")