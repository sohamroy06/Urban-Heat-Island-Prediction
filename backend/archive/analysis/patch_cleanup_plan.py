import io, os, sys, shutil

src = io.open('step36_cleanup.py', encoding='utf-8').read()

# step30 and step32 are ACTIVE, not scratch
for name in ("'step30_oof_predictions.py', \n            ", "'step30_oof_predictions.py',\n            "):
    if name in src:
        src = src.replace(name, '')
        break
else:
    src = src.replace("'step30_oof_predictions.py', ", '')

src = src.replace("'step32_smoke.py', ", '')

OLD = "            'fetch_grid_lst.py', 'step10_fetch_indices.py', 'step13_night_lst.py']"
NEW = ("            'fetch_grid_lst.py', 'step10_fetch_indices.py', 'step13_night_lst.py',\n"
       "            'add_oof_columns.py', 'test_api.py']")
if src.count(OLD) != 1:
    print('FAIL: pipeline list not found uniquely'); sys.exit(1)
src = src.replace(OLD, NEW)

io.open('step36_cleanup.py', 'w', encoding='utf-8').write(src)
print('patched step36_cleanup.py')

RENAMES = [('step30_oof_predictions.py', 'add_oof_columns.py'),
           ('step32_smoke.py', 'test_api.py')]
for old, new in RENAMES:
    if os.path.exists(old):
        shutil.move(old, new)
        print('renamed %s -> %s' % (old, new))
    elif os.path.exists(new):
        print('already renamed: %s' % new)
    else:
        print('MISSING: %s' % old)

# add_oof_columns.py imports train_grid, which stays in place - verify
s2 = io.open('add_oof_columns.py', encoding='utf-8').read()
print()
print('add_oof_columns.py imports train_grid:', 'from train_grid import' in s2)
print('train_grid.py present in backend/:', os.path.exists('train_grid.py'))
