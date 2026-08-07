import io, sys

p = 'train_grid.py'
src = io.open(p, encoding='utf-8').read()
io.open('train_grid.py.bak', 'w', encoding='utf-8').write(src)

PATCHES = []

PATCHES.append((
'''    curve = {}
    for km in BLOCK_KM_CURVE:
        f = blocked_folds(cx, cy, km)
        s = evaluate(X, y, f)
        curve["%dkm" % km] = round(float(s.mean()), 4)''',
'''    curve, curve_sd = {}, {}
    for km in BLOCK_KM_CURVE:
        f = blocked_folds(cx, cy, km)
        s = evaluate(X, y, f)
        curve["%dkm" % km] = round(float(s.mean()), 4)
        curve_sd["%dkm" % km] = round(float(s.std()), 4)'''))

PATCHES.append((
'''        "validation": {
            "method": "5 km spatially blocked CV, buffered, %d repeats" % N_REPEATS,
            "r2": round(R2, 4), "rmse_c": round(RMSE, 3), "mae_c": round(MAE, 3),
            "bias_c": round(BIAS, 3),
            "interval_coverage": round(cov, 4),
            "interval_width_multiplier": k,
            "n_oof_predictions": int(len(yy)),
        },''',
'''        "validation": {
            "method": "5 km spatially blocked CV, buffered, %d repeats" % N_REPEATS,
            "headline": ("blocked 5 km CV: R2 = %.3f +/- %.3f, RMSE = %.2f C"
                         % (PRIMARY_R2, PRIMARY_SD, PRIMARY_RMSE)),
            "r2": PRIMARY_R2,
            "r2_primary_mean_of_folds": PRIMARY_R2,
            "r2_primary_sd": PRIMARY_SD,
            "rmse_c_primary": PRIMARY_RMSE,
            "r2_pooled_oof": round(R2, 4),
            "rmse_c": round(RMSE, 3),
            "r2_aggregation_note": (
                "r2_primary_mean_of_folds averages per-fold R2, each scored against its "
                "own test block variance. This is the conservative standard for blocked "
                "CV and is the number to quote. r2_pooled_oof concatenates all fold "
                "predictions and scores once against global variance, which is "
                "systematically higher. Both come from the same 5 km blocked folds."),
            "mae_c": round(MAE, 3),
            "bias_c": round(BIAS, 3),
            "interval_coverage": round(cov, 4),
            "interval_width_multiplier": k,
            "n_oof_predictions": int(len(yy)),
        },
        "r2_vs_block_size_sd": curve_sd,'''))

PATCHES.append((
'''    print()
    print("=== TRAINING FINAL MODELS ON ALL DATA ===")''',
'''    PRIMARY_R2 = curve["%dkm" % BLOCK_KM_PRIMARY]
    PRIMARY_SD = curve_sd["%dkm" % BLOCK_KM_PRIMARY]
    PRIMARY_RMSE = round(float(y.std() * np.sqrt(max(0.0, 1 - PRIMARY_R2))), 2)

    print()
    print("=== TRAINING FINAL MODELS ON ALL DATA ===")'''))

PATCHES.append((
'''    print("HEADLINE (quote this): blocked 5 km R2 %.4f, RMSE %.2f C" % (R2, RMSE))
    print("NEVER quote the random-split figure of %.4f - it is leakage." % random_r2)''',
'''    print("HEADLINE (quote this): blocked 5 km R2 %.4f +/- %.4f, RMSE %.2f C"
          % (PRIMARY_R2, PRIMARY_SD, PRIMARY_RMSE))
    print("  secondary: pooled out-of-fold R2 %.4f (higher; scored vs global variance)" % R2)
    print("  NEVER quote the random-split figure of %.4f - it is leakage." % random_r2)
    print()
    print("NEXT: run  python add_oof_columns.py  to restore out-of-fold columns")
    print("      in grid_predictions.csv, which the API serves.")'''))

fails = []
for i, (old, new) in enumerate(PATCHES, 1):
    if src.count(old) != 1:
        fails.append((i, src.count(old)))
        continue
    src = src.replace(old, new)
    print('patch %d applied' % i)

if fails:
    print()
    for i, n in fails:
        print('FAIL patch %d: %d matches' % (i, n))
    print('file NOT written')
    sys.exit(1)

io.open(p, 'w', encoding='utf-8').write(src)
print()
print('WROTE train_grid.py (%d bytes)' % len(src))
print('backup: train_grid.py.bak (gitignored via *.bak)')
