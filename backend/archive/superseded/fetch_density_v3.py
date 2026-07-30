import osmnx as ox
import geopandas as gpd
import pandas as pd
import time
import os

ox.settings.timeout = 180
ox.settings.overpass_rate_limit = True

wards = gpd.read_file("delhi_wards.geojson")
wards = wards[wards.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
wards.to_file("delhi_wards_valid.geojson", driver="GeoJSON")
print("Usable wards:", len(wards))

# Reproject to UTM zone 43N (meters) for correct area/length math
wards_proj = wards.to_crs(epsg=32643)

OUT_FILE = "real_density.csv"
done_ids = set()
if os.path.exists(OUT_FILE):
    prev = pd.read_csv(OUT_FILE)
    done_ids = set(prev["ward_id"].tolist())
    print(f"Resuming, {len(done_ids)} wards already done")

rows = [] if not done_ids else prev.to_dict("records")

for idx, row in wards.iterrows():
    if idx in done_ids:
        continue

    b_density, r_density = 0, 0
    ward_area_m2 = wards_proj.geometry.iloc[idx].area

    for attempt in range(3):
        try:
            buildings = ox.features_from_polygon(row.geometry, tags={"building": True})
            if len(buildings) > 0:
                b_proj = buildings.to_crs(epsg=32643)
                b_density = b_proj.geometry.area.sum() / ward_area_m2
            break
        except Exception as e:
            print(f"ward {idx} buildings attempt {attempt+1} failed: {e}")
            time.sleep(5)

    for attempt in range(3):
        try:
            roads = ox.graph_from_polygon(row.geometry, network_type="drive")
            r_length = sum(d.get('length', 0) for u, v, d in roads.edges(data=True))
            r_density = r_length / ward_area_m2
            break
        except Exception as e:
            print(f"ward {idx} roads attempt {attempt+1} failed: {e}")
            time.sleep(5)

    rows.append({"ward_id": idx, "building_density": b_density, "road_density": r_density})
    pd.DataFrame(rows).to_csv(OUT_FILE, index=False)  # save after EVERY ward
    print(f"done {idx+1}/{len(wards)}")

print("Finished. Saved real_density.csv")