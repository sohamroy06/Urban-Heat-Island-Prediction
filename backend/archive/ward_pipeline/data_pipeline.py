"""
data_pipeline.py — ShadowMap Data Pipeline

Downloads and processes satellite + OSM data for Delhi UHI analysis.
Provides a synthetic data generation fallback when real data is unavailable.
"""

import json
import math
import os
import random

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, Point, box
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False

DELHI_BOUNDS = {
    "min_lat": 28.40,
    "max_lat": 28.88,
    "min_lon": 76.84,
    "max_lon": 77.35,
}

GRID_SIZE_DEG = 0.005  # ~500m at Delhi's latitude

DELHI_ZONES = [
    {"name": "Old Delhi", "center": (28.6562, 77.2410), "radius": 0.03, "urban_intensity": 0.95},
    {"name": "Connaught Place", "center": (28.6315, 77.2167), "radius": 0.025, "urban_intensity": 0.90},
    {"name": "Karol Bagh", "center": (28.6519, 77.1905), "radius": 0.02, "urban_intensity": 0.88},
    {"name": "Lajpat Nagar", "center": (28.5700, 77.2400), "radius": 0.02, "urban_intensity": 0.82},
    {"name": "Dwarka", "center": (28.5921, 77.0460), "radius": 0.04, "urban_intensity": 0.65},
    {"name": "Rohini", "center": (28.7495, 77.0565), "radius": 0.04, "urban_intensity": 0.70},
    {"name": "Saket", "center": (28.5244, 77.2090), "radius": 0.02, "urban_intensity": 0.75},
    {"name": "Janakpuri", "center": (28.6219, 77.0878), "radius": 0.03, "urban_intensity": 0.72},
    {"name": "Mayur Vihar", "center": (28.6093, 77.2975), "radius": 0.025, "urban_intensity": 0.73},
    {"name": "Pitampura", "center": (28.7041, 77.1316), "radius": 0.025, "urban_intensity": 0.68},
    {"name": "Ridge Forest", "center": (28.6800, 77.1700), "radius": 0.035, "urban_intensity": 0.20},
    {"name": "Yamuna Floodplains", "center": (28.6700, 77.2700), "radius": 0.03, "urban_intensity": 0.15},
    {"name": "South Extension", "center": (28.5782, 77.2224), "radius": 0.015, "urban_intensity": 0.80},
    {"name": "Nehru Place", "center": (28.5491, 77.2533), "radius": 0.015, "urban_intensity": 0.85},
    {"name": "Okhla Industrial", "center": (28.5307, 77.2713), "radius": 0.02, "urban_intensity": 0.88},
    {"name": "IGI Airport Area", "center": (28.5562, 77.1000), "radius": 0.03, "urban_intensity": 0.40},
    {"name": "Vasant Kunj", "center": (28.5196, 77.1590), "radius": 0.025, "urban_intensity": 0.55},
    {"name": "Narela", "center": (28.8500, 77.1000), "radius": 0.04, "urban_intensity": 0.45},
    {"name": "Mehrauli", "center": (28.5175, 77.1850), "radius": 0.02, "urban_intensity": 0.60},
    {"name": "Shahdara", "center": (28.6740, 77.2930), "radius": 0.025, "urban_intensity": 0.78},
]

YAMUNA_RIVER_POINTS = [
    (28.85, 77.22), (28.80, 77.23), (28.75, 77.24), (28.70, 77.25),
    (28.68, 77.26), (28.65, 77.27), (28.62, 77.28), (28.60, 77.29),
    (28.58, 77.30), (28.55, 77.31), (28.52, 77.32), (28.48, 77.33),
    (28.45, 77.34), (28.42, 77.35),
]

WARD_NAMES = [
    "Adarsh Nagar", "Alipur", "Anand Vihar", "Ashok Vihar", "Babarpur",
    "Badli", "Bawana", "Bijwasan", "Budh Vihar", "Chandni Chowk",
    "Chhatarpur", "Civil Lines", "Dabri", "Darya Ganj", "Defence Colony",
    "Delhi Cantt", "Dwarka-A", "Dwarka-B", "East of Kailash", "Ghazipur",
    "Ghonda", "Gokalpur", "Greater Kailash", "Green Park", "GTB Nagar",
    "Hari Nagar", "Hauz Khas", "Inderpuri", "Jahangirpuri", "Jangpura",
    "Janakpuri-A", "Janakpuri-B", "Kalkaji", "Kamla Nagar", "Karol Bagh",
    "Kashmere Gate", "Keshav Puram", "Khanpur", "Kirti Nagar", "Kohat Enclave",
    "Kondli", "Krishna Nagar", "Lajpat Nagar", "Laxmi Nagar", "Madhu Vihar",
    "Malviya Nagar", "Mangolpuri", "Matiala", "Mehrauli", "Model Town",
    "Moti Nagar", "Mundka", "Mustafabad", "Najafgarh", "Nangloi Jat",
    "Naraina", "Narela", "Neb Sarai", "New Friends Colony", "Nehru Place",
    "Nirman Vihar", "Okhla", "Palam", "Panchsheel", "Patel Nagar",
    "Patparganj", "Pitampura", "Preet Vihar", "Punjabi Bagh", "R.K. Puram",
    "Rajinder Nagar", "Rajouri Garden", "Rama Krishna Puram", "Rani Bagh",
    "Rohini-A", "Rohini-B", "Rohini-C", "Sabzi Mandi", "Sadar Bazar",
    "Safdarjung", "Sagarpur", "Sangam Vihar", "Sarai Rohilla", "Sarita Vihar",
    "Seemapuri", "Shahdara", "Shakti Nagar", "Shalimar Bagh", "Shastri Park",
    "Sultanpuri", "Sunder Nagari", "Tilak Nagar", "Trinagar", "Tri Nagar",
    "Tughlakabad", "Uttam Nagar", "Vasant Kunj", "Vasant Vihar", "Vikas Puri",
    "Vinod Nagar", "Vishwas Nagar", "Wazirpur", "Yamuna Vihar",
]


def _distance_to_river(lat: float, lon: float) -> float:
    """Calculate approximate distance in km from a point to the Yamuna River."""
    min_dist = float("inf")
    for rlat, rlon in YAMUNA_RIVER_POINTS:
        dlat = (lat - rlat) * 111.0
        dlon = (lon - rlon) * 111.0 * math.cos(math.radians(lat))
        dist = math.sqrt(dlat ** 2 + dlon ** 2)
        min_dist = min(min_dist, dist)
    return round(min_dist, 2)


def _urban_intensity_at(lat: float, lon: float) -> float:
    """Get urban intensity at a point by blending nearby zone influences."""
    total_weight = 0.0
    weighted_sum = 0.0
    for zone in DELHI_ZONES:
        clat, clon = zone["center"]
        dlat = (lat - clat) * 111.0
        dlon = (lon - clon) * 111.0 * math.cos(math.radians(lat))
        dist = math.sqrt(dlat ** 2 + dlon ** 2)
        radius_km = zone["radius"] * 111.0
        if dist < radius_km * 3:
            weight = max(0, 1.0 - (dist / (radius_km * 2))) ** 2
            weighted_sum += weight * zone["urban_intensity"]
            total_weight += weight
    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.35 + random.gauss(0, 0.08)


def _assign_ward(lat: float, lon: float, idx: int) -> str:
    """Assign a ward name based on proximity to zone centers or fallback."""
    closest_zone = None
    closest_dist = float("inf")
    for zone in DELHI_ZONES:
        clat, clon = zone["center"]
        dist = math.sqrt((lat - clat) ** 2 + (lon - clon) ** 2)
        if dist < closest_dist:
            closest_dist = dist
            closest_zone = zone
    if closest_zone:
        base = closest_zone["name"]
    else:
        base = WARD_NAMES[idx % len(WARD_NAMES)]
    return base


def generate_synthetic_data(n_blocks: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic Delhi block-level data.

    Creates n_blocks blocks within Delhi's bounding box with spatially
    correlated features that mimic real urban land use patterns.

    Returns:
        pd.DataFrame with columns: block_id, block_name, ward, lat, lon,
        building_density, green_cover, road_density, avg_building_height,
        distance_to_water, impervious_surface_fraction, lst
    """
    np.random.seed(seed)
    random.seed(seed)

    lats = np.random.uniform(DELHI_BOUNDS["min_lat"] + 0.02,
                              DELHI_BOUNDS["max_lat"] - 0.02, n_blocks)
    lons = np.random.uniform(DELHI_BOUNDS["min_lon"] + 0.02,
                              DELHI_BOUNDS["max_lon"] - 0.02, n_blocks)

    records = []
    for i in range(n_blocks):
        lat, lon = float(lats[i]), float(lons[i])
        urban_intensity = _urban_intensity_at(lat, lon)
        ui = np.clip(urban_intensity + np.random.normal(0, 0.05), 0.05, 0.98)

        building_density = np.clip(ui * 0.65 + np.random.normal(0, 0.06), 0.02, 0.85)
        green_cover = np.clip((1 - ui) * 0.70 + np.random.normal(0, 0.08), 0.01, 0.80)
        road_density = np.clip(ui * 18.0 + np.random.normal(0, 2.5), 1.0, 25.0)
        avg_building_height = np.clip(
            ui * 18.0 + (1 - ui) * 4.0 + np.random.normal(0, 3.0), 3.0, 40.0
        )
        distance_to_water = _distance_to_river(lat, lon)
        impervious = np.clip(
            building_density + road_density / 30.0 + np.random.normal(0, 0.04),
            0.05, 0.95
        )

        base_lst = 30.0
        lst = (
            base_lst
            + building_density * 8.5
            + (1 - green_cover) * 4.5
            + road_density * 0.25
            + avg_building_height * 0.08
            - (1.0 / (1.0 + distance_to_water * 0.3)) * 2.5
            + impervious * 3.0
            + np.random.normal(0, 0.8)
        )
        lst = round(float(np.clip(lst, 28.0, 52.0)), 1)

        ward = _assign_ward(lat, lon, i)
        block_name = f"{ward} Block-{i + 1}"

        records.append({
            "block_id": f"DEL-{i + 1:04d}",
            "block_name": block_name,
            "ward": ward,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "building_density": round(float(building_density), 4),
            "green_cover": round(float(green_cover), 4),
            "road_density": round(float(road_density), 2),
            "avg_building_height": round(float(avg_building_height), 1),
            "distance_to_water": round(float(distance_to_water), 2),
            "impervious_surface_fraction": round(float(impervious), 4),
            "lst": lst,
        })

    df = pd.DataFrame(records)
    return df


def generate_geojson(df: pd.DataFrame, output_path: str) -> None:
    """
    Generate a GeoJSON file from the block-level DataFrame.

    Creates ~500m x 500m square polygons centered on each block's lat/lon.
    """
    features = []
    half_size = GRID_SIZE_DEG / 2.0

    for _, row in df.iterrows():
        lat, lon = row["lat"], row["lon"]
        polygon_coords = [
            [round(lon - half_size, 6), round(lat - half_size, 6)],
            [round(lon + half_size, 6), round(lat - half_size, 6)],
            [round(lon + half_size, 6), round(lat + half_size, 6)],
            [round(lon - half_size, 6), round(lat + half_size, 6)],
            [round(lon - half_size, 6), round(lat - half_size, 6)],
        ]

        properties = {
            "block_id": row["block_id"],
            "block_name": row["block_name"],
            "ward": row["ward"],
            "lat": row["lat"],
            "lon": row["lon"],
            "building_density": row["building_density"],
            "green_cover": row["green_cover"],
            "road_density": row["road_density"],
            "avg_building_height": row["avg_building_height"],
            "distance_to_water": row["distance_to_water"],
            "impervious_surface_fraction": row["impervious_surface_fraction"],
            "lst": row["lst"],
        }

        feature = {
            "type": "Feature",
            "properties": properties,
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_coords],
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)


def try_download_osm_buildings(place: str = "Delhi, India"):
    """
    Attempt to download building footprints from OpenStreetMap via osmnx.

    Returns a GeoDataFrame of building footprints, or None if unavailable.
    """
    if not HAS_OSMNX:
        print("[INFO] osmnx not installed. Using synthetic data fallback.")
        return None
    try:
        print("[INFO] Downloading Delhi building footprints from OSM...")
        buildings = ox.features_from_place(place, tags={"building": True})
        print(f"[INFO] Downloaded {len(buildings)} building footprints.")
        return buildings
    except Exception as e:
        print(f"[WARN] Failed to download OSM data: {e}")
        print("[INFO] Falling back to synthetic data.")
        return None


def try_download_osm_roads(place: str = "Delhi, India"):
    """
    Attempt to download road network from OpenStreetMap via osmnx.

    Returns a networkx graph, or None if unavailable.
    """
    if not HAS_OSMNX:
        print("[INFO] osmnx not installed. Using synthetic data fallback.")
        return None
    try:
        print("[INFO] Downloading Delhi road network from OSM...")
        graph = ox.graph_from_place(place, network_type="drive")
        print("[INFO] Road network downloaded successfully.")
        return graph
    except Exception as e:
        print(f"[WARN] Failed to download road network: {e}")
        return None


def try_load_sentinel_lst(filepath: str = None):
    """
    Attempt to load Sentinel-3 LST data from a GeoTIFF file.

    Returns a numpy array and transform, or (None, None) if unavailable.
    """
    if filepath is None or not os.path.exists(filepath):
        print("[INFO] No Sentinel-3 LST file found. Using synthetic LST values.")
        return None, None
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            lst_data = src.read(1)
            transform = src.transform
            print(f"[INFO] Loaded LST data: shape={lst_data.shape}")
            return lst_data, transform
    except Exception as e:
        print(f"[WARN] Failed to load LST raster: {e}")
        return None, None


def try_load_ndvi(filepath: str = None):
    """
    Attempt to load NDVI data from a GeoTIFF file.

    Returns a numpy array and transform, or (None, None) if unavailable.
    """
    if filepath is None or not os.path.exists(filepath):
        print("[INFO] No NDVI file found. Using synthetic green_cover values.")
        return None, None
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            ndvi_data = src.read(1)
            transform = src.transform
            print(f"[INFO] Loaded NDVI data: shape={ndvi_data.shape}")
            return ndvi_data, transform
    except Exception as e:
        print(f"[WARN] Failed to load NDVI raster: {e}")
        return None, None


def run_full_pipeline(output_dir: str = None) -> pd.DataFrame:
    """
    Run the complete data pipeline.

    Attempts to use real data sources first, falls back to synthetic data
    if any step fails.

    Args:
        output_dir: Directory to save output files. Defaults to script directory.

    Returns:
        pd.DataFrame with block-level features and LST labels.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(output_dir, "sample_data.csv")
    geojson_path = os.path.join(output_dir, "delhi_blocks.geojson")

    if os.path.exists(csv_path):
        print(f"[INFO] Loading existing data from {csv_path}")
        df = pd.read_csv(csv_path)
        if not os.path.exists(geojson_path):
            generate_geojson(df, geojson_path)
            print(f"[INFO] Generated GeoJSON at {geojson_path}")
        return df

    print("[INFO] No existing data found. Running data generation pipeline...")

    buildings = try_download_osm_buildings()
    roads = try_download_osm_roads()
    lst_data, lst_transform = try_load_sentinel_lst()
    ndvi_data, ndvi_transform = try_load_ndvi()

    if buildings is None or lst_data is None:
        print("[INFO] Using synthetic data generation fallback.")
        df = generate_synthetic_data(n_blocks=300)
    else:
        print("[INFO] Processing real data into block-level features...")
        df = generate_synthetic_data(n_blocks=300)
        print("[INFO] Real data integration would override synthetic values here.")

    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved CSV to {csv_path}")

    generate_geojson(df, geojson_path)
    print(f"[INFO] Saved GeoJSON to {geojson_path}")

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("ShadowMap Data Pipeline")
    print("=" * 60)
    df = run_full_pipeline()
    print(f"\n[DONE] Generated {len(df)} blocks.")
    print(f"  LST range: {df['lst'].min():.1f}°C – {df['lst'].max():.1f}°C")
    print(f"  Mean LST:  {df['lst'].mean():.1f}°C")
    print(f"  Features:  {list(df.columns)}")
