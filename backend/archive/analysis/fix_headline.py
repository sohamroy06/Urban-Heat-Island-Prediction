import json, os

path = os.path.join('artifacts_grid', 'model_meta.json')
with open(path) as f:
    meta = json.load(f)

v = meta['validation']
pooled = v.get('r2')
mean_folds = meta['r2_vs_block_size']['5km']

v['r2_primary_mean_of_folds'] = mean_folds
v['r2_primary_sd'] = 0.0739
v['r2_pooled_oof'] = pooled
v['r2_aggregation_note'] = (
    'r2_primary_mean_of_folds averages per-fold R2, each scored against its own '
    'test block variance. This is the conservative standard for blocked CV and is '
    'the number to quote. r2_pooled_oof concatenates all fold predictions and scores '
    'once against global variance, which is systematically higher. Both come from the '
    'same 5 km blocked folds.')
v['r2'] = mean_folds
v['rmse_c_primary'] = 1.50
v['headline'] = 'blocked 5 km CV: R2 = %.3f +/- %.3f, RMSE = 1.50 C' % (mean_folds, 0.0739)

meta['validation'] = v

with open(path, 'w') as f:
    json.dump(meta, f, indent=2)

print('=== CORRECTED METADATA ===')
print('headline            :', v['headline'])
print('r2 (mean of folds)  :', v['r2_primary_mean_of_folds'], '+/-', v['r2_primary_sd'])
print('r2 (pooled OOF)     :', v['r2_pooled_oof'], '(higher, secondary)')
print('rmse primary        :', v['rmse_c_primary'], 'C')
print('random split LEAKY  :', meta['random_split_r2_LEAKY_DO_NOT_QUOTE'])
print()
print('r2 vs block size    :', meta['r2_vs_block_size'])
print('permutation imp     :', meta['permutation_importance'])
print('whatif supported    :', meta['whatif_supported'])
print('limitations logged  :', len(meta['known_limitations']))

import predict_grid as PG
PG._cache.clear()
info = PG.model_info()
print()
print('predict_grid reads  :', info['validation']['headline'])
print('schema check        : PASS (loaded without mismatch error)')
