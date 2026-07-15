import geopandas as gpd
import pandas as pd
import numpy as np
from data_pipeline import _distance_to_river, WARD_NAMES

wards = gpd.read_file("delhi_wards_valid.geojson")
density = pd.read_csv("real_density.csv")
lstndvi = pd.read_csv("real_lst_ndvi_v3.csv")
heights = pd.read_csv("real_heights.csv")

df = density.merge(lstndvi, on="ward_id").merge(heights, on="ward_id", how="left")
df = df[~((df["building_density"] == 0) & (df["road_density"] == 0))]
df = df.dropna(subset=["lst", "ndvi"])
print(f"Final usable wards: {len(df)}")

# Fill missing heights with mean of real tagged heights (not invented, grounded in actual data)
real_height_mean = df["avg_building_height"].mean()
n_filled = df["avg_building_height"].isna().sum()
df["avg_building_height"] = df["avg_building_height"].fillna(real_height_mean)
print(f"Filled {n_filled} missing heights with mean={real_height_mean:.1f}m")

wards["lat"] = wards.geometry.centroid.y
wards["lon"] = wards.geometry.centroid.x
wards["ward_id"] = wards.index
if "name" in wards.columns:
    wards["ward_name"] = wards["name"].fillna("Unknown")
else:
    wards["ward_name"] = [WARD_NAMES[i % len(WARD_NAMES)] for i in wards.index]

df = df.merge(wards[["ward_id", "lat", "lon", "ward_name"]], on="ward_id")

df["green_cover"] = df["ndvi"].clip(0, 1)
road_width_m = 7.0
df["impervious_surface_fraction"] = (df["building_density"] + df["road_density"] * road_width_m).clip(0.02, 0.95)

final = pd.DataFrame({
    "block_id": [f"DEL-{i+1:04d}" for i in range(len(df))],
    "block_name": df["ward_name"] + " Block-" + (df.index + 1).astype(str),
    "ward": df["ward_name"],
    "lat": df["lat"].round(6),
    "lon": df["lon"].round(6),
    "building_density": df["building_density"].round(4),
    "green_cover": df["green_cover"].round(4),
    "road_density": df["road_density"].round(4),
    "avg_building_height": df["avg_building_height"].round(1),
    "distance_to_water": df.apply(lambda r: _distance_to_river(r["lat"], r["lon"]), axis=1).round(2),
    "impervious_surface_fraction": df["impervious_surface_fraction"].round(4),
    "lst": df["lst"].round(1),
})

final.to_csv("sample_data.csv", index=False)
print("Saved sample_data.csv with", len(final), "rows and columns:", list(final.columns))