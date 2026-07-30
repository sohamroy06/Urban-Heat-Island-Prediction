import osmnx as ox
import geopandas as gpd
import pandas as pd

wards = gpd.read_file("delhi_wards.geojson")
rows = []

for idx, row in wards.iterrows():
    try:
        buildings = ox.features_from_polygon(row.geometry, tags={"building": True})
        roads = ox.graph_from_polygon(row.geometry, network_type="drive")
        b_density = buildings.geometry.area.sum() / row.geometry.area
        r_length = sum(d.get('length', 0) for u, v, d in roads.edges(data=True))
        r_density = r_length / row.geometry.area
    except Exception as e:
        b_density, r_density = None, None
        print(f"ward {idx} failed: {e}")

    rows.append({"ward_id": idx, "building_density": b_density, "road_density": r_density})
    print(f"done {idx+1}/{len(wards)}")

pd.DataFrame(rows).to_csv("real_density.csv", index=False)
print("Saved real_density.csv")