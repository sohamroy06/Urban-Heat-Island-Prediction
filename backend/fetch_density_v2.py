import osmnx as ox
import geopandas as gpd
import pandas as pd

wards = gpd.read_file("delhi_wards.geojson")
wards = wards[wards.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
print("Usable wards:", len(wards))

rows = []
for idx, row in wards.iterrows():
    b_density, r_density = None, None
    try:
        buildings = ox.features_from_polygon(row.geometry, tags={"building": True})
        b_density = buildings.geometry.area.sum() / row.geometry.area
    except Exception as e:
        print(f"ward {idx} buildings failed: {e}")
        b_density = 0  # no buildings found = legitimate 0, not a failure

    try:
        roads = ox.graph_from_polygon(row.geometry, network_type="drive")
        r_length = sum(d.get('length', 0) for u, v, d in roads.edges(data=True))
        r_density = r_length / row.geometry.area
    except Exception as e:
        print(f"ward {idx} roads failed: {e}")
        r_density = 0

    rows.append({"ward_id": idx, "building_density": b_density, "road_density": r_density})
    print(f"done {idx+1}/{len(wards)}")

pd.DataFrame(rows).to_csv("real_density.csv", index=False)
print("Saved real_density.csv, wards used:", len(wards))