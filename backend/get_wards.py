import osmnx as ox
wards = ox.features_from_place("Delhi, India", tags={"admin_level": "10"})
wards.to_file("delhi_wards.geojson", driver="GeoJSON")
print("Ward count:", len(wards))