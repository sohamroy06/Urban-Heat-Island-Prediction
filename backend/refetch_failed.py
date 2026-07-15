import osmnx as ox
import geopandas as gpd
import pandas as pd
import time

ox.settings.timeout = 180
ox.settings.overpass_rate_limit = True

# Alternate Overpass mirrors to rotate through if main one is blocking us
MIRRORS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.openstreetmap.ru/api",
]

wards = gpd.read_file("delhi_wards_valid.geojson")
wards_proj = wards.to_crs(epsg=32643)

df = pd.read_csv("real_density.csv")
failed_ids = df[(df["building_density"] == 0) & (df["road_density"] == 0)]["ward_id"].tolist()
print(f"Re-fetching {len(failed_ids)} wards: {failed_ids}")

for ward_id in failed_ids:
    row = wards.iloc[ward_id]
    ward_area_m2 = wards_proj.geometry.iloc[ward_id].area
    b_density, r_density = 0, 0
    success = False

    for mirror in MIRRORS:
        ox.settings.overpass_url = mirror
        try:
            buildings = ox.features_from_polygon(row.geometry, tags={"building": True})
            if len(buildings) > 0:
                b_proj = buildings.to_crs(epsg=32643)
                b_density = b_proj.geometry.area.sum() / ward_area_m2
            roads = ox.graph_from_polygon(row.geometry, network_type="drive")
            r_length = sum(d.get('length', 0) for u, v, d in roads.edges(data=True))
            r_density = r_length / ward_area_m2
            success = True
            print(f"ward {ward_id} OK via {mirror}")
            break
        except Exception as e:
            print(f"ward {ward_id} failed on {mirror}: {e}")
            time.sleep(20)

    if not success:
        print(f"ward {ward_id} still failing after all mirrors, leaving as 0 for now")

    df.loc[df["ward_id"] == ward_id, "building_density"] = b_density
    df.loc[df["ward_id"] == ward_id, "road_density"] = r_density
    df.to_csv("real_density.csv", index=False)

print("Re-fetch pass complete")