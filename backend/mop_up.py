import osmnx as ox
import geopandas as gpd
import pandas as pd
import time

ox.settings.timeout = 180
ox.settings.overpass_url = "https://overpass-api.de/api"

wards = gpd.read_file("delhi_wards_valid.geojson")
wards_proj = wards.to_crs(epsg=32643)

df = pd.read_csv("real_density.csv")
failed_ids = df[(df["building_density"] == 0) & (df["road_density"] == 0)]["ward_id"].tolist()
print(f"Mopping up {len(failed_ids)} wards, one at a time with cooldown")

for ward_id in failed_ids:
    row = wards.iloc[ward_id]
    ward_area_m2 = wards_proj.geometry.iloc[ward_id].area
    b_density, r_density = 0, 0

    try:
        buildings = ox.features_from_polygon(row.geometry, tags={"building": True})
        if len(buildings) > 0:
            b_proj = buildings.to_crs(epsg=32643)
            b_density = b_proj.geometry.area.sum() / ward_area_m2
        print(f"ward {ward_id} buildings OK")
    except Exception as e:
        print(f"ward {ward_id} buildings still failing: {e}")

    time.sleep(45)  # long cooldown between calls, even within same ward

    try:
        roads = ox.graph_from_polygon(row.geometry, network_type="drive")
        r_length = sum(d.get('length', 0) for u, v, d in roads.edges(data=True))
        r_density = r_length / ward_area_m2
        print(f"ward {ward_id} roads OK")
    except Exception as e:
        print(f"ward {ward_id} roads still failing: {e}")

    df.loc[df["ward_id"] == ward_id, "building_density"] = b_density
    df.loc[df["ward_id"] == ward_id, "road_density"] = r_density
    df.to_csv("real_density.csv", index=False)
    print(f"saved ward {ward_id}, sleeping before next...")

    time.sleep(45)  # long cooldown before next ward

print("Mop-up complete")