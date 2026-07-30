import io, sys

path = 'build_master.py'
src = io.open(path, encoding='utf-8').read()

OLD = """gp = g.to_crs(32643)
cent4326 = g.geometry.centroid
geo = pd.DataFrame({
    'cell_id': g.cell_id.values,
    'lon': cent4326.x.values,
    'lat': cent4326.y.values,
    'cx': gp.geometry.centroid.x.values,
    'cy': gp.geometry.centroid.y.values,
    'area_m2': gp.geometry.area.values,
})"""

NEW = """gp = g.to_crs(32643)
cent_utm = gp.geometry.centroid
cent_wgs = cent_utm.to_crs(4326)
geo = pd.DataFrame({
    'cell_id': g.cell_id.values,
    'lon': cent_wgs.x.values,
    'lat': cent_wgs.y.values,
    'cx': cent_utm.x.values,
    'cy': cent_utm.y.values,
    'area_m2': gp.geometry.area.values,
})"""

if src.count(OLD) != 1:
    print('FAIL: found %d matches, not patching' % src.count(OLD))
    sys.exit(1)

src = src.replace(OLD, NEW)

OLDCHK = "        d = float(np.max(np.abs(old[oc].values - m[nc].values)))\n        ok = d < 1e-6"
NEWCHK = ("        d = float(np.max(np.abs(old[oc].values - m[nc].values)))\n"
          "        tol = 1e-4 if nc in ('lon', 'lat') else 1e-6\n"
          "        ok = d < tol")
if src.count(OLDCHK) != 1:
    print('FAIL: tolerance block not found uniquely')
    sys.exit(1)
src = src.replace(OLDCHK, NEWCHK)

io.open(path, 'w', encoding='utf-8').write(src)
print('patched build_master.py')
print('  centroids now computed in EPSG:32643 then reprojected to EPSG:4326')
print('  lon/lat tolerance relaxed to 1e-4 deg (~11 m) since values will shift slightly')
