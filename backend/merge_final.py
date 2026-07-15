import geopandas as gpd
import pandas as pd
import numpy as np
from data_pipeline import _distance_to_river, WARD_NAMES

wards = gpd.read_file("delhi_wards_valid.geojson")
density = pd.read_csv("real_density.csv")
lstndvi = pd.read_csv("real_lst_ndvi_v2.csv")

# Merge on ward_id
df = density.merge(lstndvi, on="ward_id")

# Drop wards where OSM data never resolved (still 0/0) or LST missing
df = df[~((df["building_density"] == 0) & (df["road_density"] == 0))]
df = df.dropna(subset=["lst", "ndvi"])
print(f"Final usable wards: {len(df)}")

# Get centroid lat/lon per ward
wards["lat"] = wards.geometry.centroid.y
wards["lon"] = wards.geometry.centroid.x
wards["ward_id"] = wards.index

# Try to get real OSM ward name if available, else fallback
if "name" in wards.columns:
    wards["ward_name"] = wards["name"].fillna("Unknown")
else:
    wards["ward_name"] = [WARD_NAMES[i % len(WARD_NAMES)] for i in wards.index]

df = df.merge(wards[["ward_id", "lat", "lon", "ward_name"]], on="ward_id")

# Derived features
df["green_cover"] = df["ndvi"].clip(0, 1)
road_width_m = 7.0  # assumed avg road width for area conversion
df["impervious_surface_fraction"] = (df["building_density"] + df["road_density"] * road_width_m).clip(0.02, 0.95)

# Final schema matching data_pipeline.py (minus avg_building_height, confirmed unused)
final = pd.DataFrame({
    "block_id": [f"DEL-{i+1:04d}" for i in range(len(df))],
    "block_name": df["ward_name"] + " Block-" + (df.index + 1).astype(str),
    "ward": df["ward_name"],
    "lat": df["lat"].round(6),
    "lon": df["lon"].round(6),
    "building_density": df["building_density"].round(4),
    "green_cover": df["green_cover"].round(4),
    "road_density": df["road_density"].round(4),
    "distance_to_water": df.apply(lambda r: _distance_to_river(r["lat"], r["lon"]), axis=1).round(2),
    "impervious_surface_fraction": df["impervious_surface_fraction"].round(4),
    "lst": df["lst"].round(1),
})

final.to_csv("sample_data.csv", index=False)
print("Saved sample_data.csv")
print(final.describe())