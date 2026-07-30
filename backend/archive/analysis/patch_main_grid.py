import io, os, sys

path = 'main_grid.py'
src = io.open(path, encoding='utf-8').read()
io.open('main_grid.py.bak', 'w', encoding='utf-8').write(src)
print('backup written: main_grid.py.bak')

REPL = []

REPL.append((
'    res = df["residual"].values',
'    res_ins = df["residual"].values\n    res_oof = df["oof_residual"].values'))

REPL.append((
'''        "residuals_c": {"mean": round(float(res.mean()), 3),
                        "std": round(float(res.std()), 3),
                        "abs_gt_3c": int(np.sum(np.abs(res) > 3)),
                        "abs_gt_5c": int(np.sum(np.abs(res) > 5))},''',
'''        "residuals_c": {
            "source": "out-of-fold (5 km blocked); in-sample values are optimistic",
            "mean": round(float(np.nanmean(res_oof)), 3),
            "std": round(float(np.nanstd(res_oof)), 3),
            "abs_gt_3c": int(np.nansum(np.abs(res_oof) > 3)),
            "abs_gt_5c": int(np.nansum(np.abs(res_oof) > 5)),
            "in_sample_std_DO_NOT_QUOTE": round(float(res_ins.std()), 3),
            "rmse_note": ("These out-of-fold residuals average 4 held-out predictions "
                          "per cell, which cancels noise. The honest single-prediction "
                          "RMSE is 1.459 C from model-info, not this std."),
        },
        "confidence_class_counts": {str(k): int(v) for k, v in
                                    df.confidence_class.value_counts().items()},
        "mean_abs_error_by_confidence_class": {
            str(k): round(float(v), 3) for k, v in
            df.assign(ae=np.abs(res_oof)).groupby("confidence_class",
                                                  observed=True).ae.mean().items()},'''))

REPL.append((
'''        "extrapolation": bool(r.extrapolation),
        "confidence_note": ("Features fall outside the p01-p99 training range; treat "
                            "with caution." if bool(r.extrapolation)
                            else "Features within training range."),''',
'''        "out_of_fold_predicted_lst_c": (None if pd.isna(r.oof_pred)
                                        else round(float(r.oof_pred), 2)),
        "out_of_fold_residual_c": (None if pd.isna(r.oof_residual)
                                   else round(float(r.oof_residual), 2)),
        "confidence_class": str(r.confidence_class),
        "confidence_note": CONFIDENCE_NOTES.get(str(r.confidence_class), ""),
        "features_outside_training_range": bool(r.out_of_range),
        "near_nct_boundary": bool(r.near_boundary),
        "dist_to_boundary_m": (None if pd.isna(r.dist_to_boundary_m)
                               else round(float(r.dist_to_boundary_m), 1)),'''))

REPL.append((
'             "extrapolation": bool(r.extrapolation)}',
'             "confidence_class": str(r.confidence_class)}'))

REPL.append((
'''        "extrapolation_cells": int(df.extrapolation.sum()),''',
'''        "cells_with_atypical_features": int(df.out_of_range.sum()),
        "cells_near_boundary": int(df.near_boundary.sum()),'''))

REPL.append((
'''        "geometry_loaded": s["geo"] is not None,''',
'''        "geometry_loaded": s["geo"] is not None,
        "confidence_classes": sorted(s["df"].confidence_class.unique().tolist()),'''))

REPL.append((
'_state = {}',
'''_state = {}

CONFIDENCE_NOTES = {
    "normal": "Features within the training range, away from the NCT boundary, "
              "out-of-fold error under 3 C. Typical absolute error ~0.9 C.",
    "edge_higher_error": "Within 1 km of the NCT boundary. Errors run ~1.3x higher "
                         "(mean ~1.2 C) because the grid cell is clipped and "
                         "surrounding context falls outside the study area.",
    "atypical_features": "One or more features fall outside the p01-p99 training "
                         "range. Mean absolute error ~1.6 C, 95th percentile ~3.6 C.",
    "poorly_predicted": "Out-of-fold absolute error above 3 C (mean ~3.4 C). Usually "
                        "anthropogenic heat sources - industrial zones, landfill - "
                        "that optical reflectance cannot observe.",
}'''))

REPL.append((
'''    print("  residuals   :", cs["residuals_c"])''',
'''    print("  residuals   : std %.3f C (out-of-fold) | in-sample %.3f C (do not quote)"
          % (cs["residuals_c"]["std"], cs["residuals_c"]["in_sample_std_DO_NOT_QUOTE"]))
    print("  |err|>3C    :", cs["residuals_c"]["abs_gt_3c"], "| >5C:", cs["residuals_c"]["abs_gt_5c"])
    print("  conf counts :", cs["confidence_class_counts"])
    print("  err by class:", cs["mean_abs_error_by_confidence_class"])'''))

REPL.append((
'''    print("  extrapolate :", cs["extrapolation_cells"])''',
'''    print("  atypical    :", cs["cells_with_atypical_features"],
          "| near boundary:", cs["cells_near_boundary"])'''))

fails = []
for i, (old, new) in enumerate(REPL, 1):
    n = src.count(old)
    if n != 1:
        fails.append((i, n, old.strip().splitlines()[0][:60]))
        continue
    src = src.replace(old, new)
    print('patch %d applied' % i)

if fails:
    print()
    print('=== PATCH FAILURES (file NOT written) ===')
    for i, n, snip in fails:
        print('  patch %d: found %d matches for %r' % (i, n, snip))
    sys.exit(1)

io.open(path, 'w', encoding='utf-8').write(src)
print()
print('WROTE main_grid.py (%d patches, %d bytes)' % (len(REPL), len(src)))

import importlib
import main_grid
importlib.reload(main_grid)
print('module reloads cleanly')
