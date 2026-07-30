import os, re, io, json
from datetime import datetime

BASE = '.'
py = sorted([f for f in os.listdir(BASE) if f.endswith('.py')])
csvs = sorted([f for f in os.listdir(BASE) if f.endswith('.csv')])
geo = sorted([f for f in os.listdir(BASE) if f.endswith('.geojson')])

print('=== COUNTS ===')
print('python %d | csv %d | geojson %d' % (len(py), len(csvs), len(geo)))

def info(f):
    try:
        src = io.open(f, encoding='utf-8', errors='replace').read()
    except Exception:
        return None
    writes = set(re.findall(r"to_csv\(\s*['\"]([^'\"]+)", src))
    writes |= set(re.findall(r"to_file\(\s*['\"]([^'\"]+)", src))
    writes |= set(re.findall(r"save_model\([^)]*['\"]([^'\"]+\.json)", src))
    reads = set(re.findall(r"read_csv\(\s*['\"]([^'\"]+)", src))
    reads |= set(re.findall(r"read_file\(\s*['\"]([^'\"]+)", src))
    imports = set(re.findall(r"^\s*(?:from|import)\s+([a-zA-Z_][\w]*)", src, re.M))
    local = imports & {p[:-3] for p in py}
    return dict(size=os.path.getsize(f),
                mtime=datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d'),
                lines=src.count('\n') + 1,
                writes=sorted(writes), reads=sorted(reads),
                imports_local=sorted(local),
                uses_ee=('import ee' in src or 'ee.Initialize' in src),
                has_main=('__main__' in src))

meta = {}
for f in py:
    d = info(f)
    if d:
        meta[f] = d

importers = {}
for f, d in meta.items():
    for dep in d['imports_local']:
        importers.setdefault(dep + '.py', []).append(f)

CORE = ['feature_engineering_v2.py', 'predict_grid.py', 'train_grid.py', 'main_grid.py']
PIPELINE = ['generate_grid.py', 'clip_grid.py', 'filter_slivers.py', 'bulk_download_osm.py',
            'merge_osm_tiles.py', 'aggregate_density.py', 'fetch_grid_lst.py',
            'step10_fetch_indices.py', 'step13_night_lst.py']
WARD = ['data_pipeline.py', 'feature_engineering.py', 'model.py', 'main.py',
        'train_model.py', 'get_wards.py', 'fetch_real_data.py', 'benchmark.py',
        'export_onnx.py', 'install_dependencies.py']

def bucket(f):
    if f in CORE: return 'CORE_v2'
    if f in PIPELINE: return 'PIPELINE_data'
    if f in WARD: return 'WARD_legacy'
    if re.match(r'step\d+_', f) or f in ('fix_headline.py', 'patch_main_grid.py'): return 'ANALYSIS_scratch'
    return 'DEAD_candidate'

print()
print('=== CLASSIFICATION ===')
buckets = {}
for f in sorted(meta):
    buckets.setdefault(bucket(f), []).append(f)
for b in ['CORE_v2', 'PIPELINE_data', 'ANALYSIS_scratch', 'WARD_legacy', 'DEAD_candidate']:
    fs = buckets.get(b, [])
    print()
    print('--- %s (%d) ---' % (b, len(fs)))
    for f in fs:
        d = meta[f]
        tag = []
        if d['uses_ee']: tag.append('GEE')
        if importers.get(f): tag.append('imported by %s' % ','.join(importers[f]))
        print('  %-32s %5d lines  %s  %s' % (f, d['lines'], d['mtime'], ' | '.join(tag)))
        if d['writes']:
            print('      writes: %s' % ', '.join(d['writes'][:5]))

print()
print('=== WHO PRODUCED EACH CSV / ARTIFACT? ===')
producers = {}
for f, d in meta.items():
    for w in d['writes']:
        producers.setdefault(os.path.basename(w), []).append(f)
orphans = []
for c in csvs + geo:
    p = producers.get(c, [])
    if p:
        print('  %-30s <- %s' % (c, ', '.join(p)))
    else:
        orphans.append(c)
print()
print('--- NO KNOWN PRODUCER (%d) ---' % len(orphans))
for c in orphans:
    print('  %-30s %8.1f KB  %s' % (c, os.path.getsize(c)/1024,
          datetime.fromtimestamp(os.path.getmtime(c)).strftime('%Y-%m-%d')))

print()
print('=== FILES REQUIRED AT RUNTIME BY CORE ===')
need = set()
for f in CORE:
    if f in meta:
        need |= set(meta[f]['reads'])
need |= {'artifacts_grid/model_meta.json', 'artifacts_grid/uhi_grid_mean.json',
         'artifacts_grid/uhi_grid_p10.json', 'artifacts_grid/uhi_grid_p90.json',
         'delhi_grid_filtered.geojson'}
for n in sorted(need):
    print('  %-40s %s' % (n, 'OK' if os.path.exists(n) else 'MISSING'))

print()
print('=== LARGE FILES (git risk) ===')
for f in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, f)
    if os.path.isfile(p) and os.path.getsize(p) > 5*1024*1024:
        print('  %-34s %8.1f MB' % (f, os.path.getsize(p)/1024/1024))
print('  (.gitignore present: %s)' % os.path.exists('.gitignore'))
if os.path.exists('.gitignore'):
    print('  contents:')
    for line in io.open('.gitignore', encoding='utf-8').read().splitlines():
        if line.strip():
            print('     ', line)

json.dump({f: meta[f] for f in meta}, io.open('file_inventory.json', 'w', encoding='utf-8'), indent=1)
print()
print('wrote file_inventory.json')
