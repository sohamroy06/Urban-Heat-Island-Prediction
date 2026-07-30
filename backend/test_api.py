import json, sys, urllib.request, urllib.error

BASE = 'http://127.0.0.1:8001'
passed, failed = [], []

def call(path, method='GET', body=None, expect=200):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def check(name, path, method='GET', body=None, expect=200, verify=None):
    try:
        status, payload = call(path, method, body)
    except Exception as e:
        failed.append((name, 'connection error: %s' % e)); print('FAIL %-34s %s' % (name, e)); return None
    if status != expect:
        failed.append((name, 'status %d expected %d' % (status, expect)))
        print('FAIL %-34s status %d (expected %d)' % (name, status, expect)); return payload
    if verify:
        ok, msg = verify(payload)
        if not ok:
            failed.append((name, msg)); print('FAIL %-34s %s' % (name, msg)); return payload
    passed.append(name); print('PASS %-34s status %d' % (name, status)); return payload

print('=' * 70)
print('LIVE HTTP SMOKE TEST -> %s' % BASE)
print('=' * 70)

check('health', '/api/health', verify=lambda p: (
    p.get('schema_version') == 2 and p.get('n_cells') == 6709 and p.get('whatif_supported') is False,
    'unexpected health payload: %s' % p))

mi = check('model-info', '/api/model-info', verify=lambda p: (
    abs(p['performance']['r2'] - 0.7592) < 1e-6 and p['leakage_warning']['random_split_r2'] == 0.8722,
    'metrics drifted'))

check('limitations', '/api/limitations', verify=lambda p: (
    p['n_limitations'] == 12 and 'diurnal_finding' in p, 'limitations payload wrong'))

check('cells limit=5', '/api/cells?limit=5', verify=lambda p: (
    p['n_cells'] == 5 and all('confidence_class' in c for c in p['cells']),
    'cells payload wrong'))

check('cells bbox', '/api/cells?min_lon=77.1&max_lon=77.2&min_lat=28.6&max_lat=28.7',
      verify=lambda p: (0 < p['n_cells'] < 6709, 'bbox filter returned %s' % p['n_cells']))

check('cells all', '/api/cells', verify=lambda p: (
    p['n_cells'] == 6709, 'expected 6709, got %s' % p['n_cells']))

check('cell 840', '/api/cell/840', verify=lambda p: (
    p['cell_id'] == 840 and p['confidence_class'] == 'atypical_features'
    and p['out_of_fold_residual_c'] == -9.13, 'cell 840 payload wrong'))

check('cell 0', '/api/cell/0', verify=lambda p: (p['cell_id'] == 0, 'wrong cell'))

check('cell 404', '/api/cell/999999', expect=404)

check('predict valid', '/api/predict', 'POST',
      {'albedo': 0.17, 'ndbi': -0.01, 'mndwi': -0.31, 'bsi': 0.06},
      verify=lambda p: (30 < p['predicted_lst'] < 60 and p['ci_lower'] <= p['predicted_lst'] <= p['ci_upper'],
                        'prediction or interval invalid: %s' % p))

check('predict pydantic reject', '/api/predict', 'POST',
      {'albedo': 5.0, 'ndbi': -0.01, 'mndwi': -0.31, 'bsi': 0.06}, expect=422)

check('predict missing field', '/api/predict', 'POST',
      {'albedo': 0.17, 'ndbi': -0.01}, expect=422)

check('predict extrapolation flag', '/api/predict', 'POST',
      {'albedo': 0.09, 'ndbi': 0.13, 'mndwi': -0.46, 'bsi': 0.17},
      verify=lambda p: (p['extrapolation'] is True, 'guard did not fire: %s' % p))

check('whatif GET 501', '/api/whatif', expect=501,
      verify=lambda p: (p['available'] is False, 'whatif claims available'))

check('whatif POST 501', '/api/whatif', 'POST', {}, expect=501)

check('city-stats', '/api/city-stats', verify=lambda p: (
    p['residuals_c']['source'].startswith('out-of-fold')
    and p['confidence_class_counts']['normal'] == 5228,
    'city-stats wrong'))

check('openapi schema', '/openapi.json', verify=lambda p: (
    len(p['paths']) >= 8, 'only %d paths' % len(p.get('paths', {}))))

print()
print('=' * 70)
print('PASSED %d | FAILED %d' % (len(passed), len(failed)))
if failed:
    print()
    for n, m in failed:
        print('  FAIL %-30s %s' % (n, m))
print('=' * 70)
sys.exit(1 if failed else 0)
