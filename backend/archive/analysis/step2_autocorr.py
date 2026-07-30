import numpy as np, pandas as pd

df = pd.read_csv('grid_merged_v5.csv')
STEP = 0.0045
col = np.rint((df.lon.values - df.lon.min()) / STEP).astype(int)
row = np.rint((df.lat.values - df.lat.min()) / STEP).astype(int)

err_x = np.abs((df.lon.values - df.lon.min()) / STEP - col).max()
err_y = np.abs((df.lat.values - df.lat.min()) / STEP - row).max()
print('grid regularity check: max snap error x=%.4f y=%.4f (want < 0.01)' % (err_x, err_y))
print('unique (col,row) pairs %d of %d rows' % (len(set(zip(col, row))), len(df)))
print('cell pitch: x = %.1f m, y = %.1f m' % (STEP*111320*np.cos(np.deg2rad(28.64)), STEP*110570))

lst = df.lst.values
lut = {}
for c, r, v in zip(col, row, lst):
    lut[(c, r)] = v

print()
print('=== LST LAG CORRELATION ===')
print('%5s %8s %10s %10s %10s' % ('lag', 'dist_km', 'r_EW', 'r_NS', 'n_EW'))
for lag in [1,2,3,4,6,8,12,16,24,32,48,64,96]:
    px = [(v, lut[(c+lag, r)]) for c, r, v in zip(col, row, lst) if (c+lag, r) in lut]
    py = [(v, lut[(c, r+lag)]) for c, r, v in zip(col, row, lst) if (c, r+lag) in lut]
    if len(px) < 100 or len(py) < 100: continue
    ax, ay = np.array(px), np.array(py)
    rx = np.corrcoef(ax[:,0], ax[:,1])[0,1]
    ry = np.corrcoef(ay[:,0], ay[:,1])[0,1]
    print('%5d %8.2f %10.3f %10.3f %10d' % (lag, lag*0.4485, rx, ry, len(px)))

print()
print('=== SEMIVARIOGRAM (mean squared diff, degC^2) ===')
sill = lst.var()
print('sill (total variance) = %.3f' % sill)
for lag in [1,2,4,8,16,32,64,96]:
    d = [ (v - lut[(c+lag, r)])**2 for c, r, v in zip(col, row, lst) if (c+lag, r) in lut ]
    if len(d) < 100: continue
    g = 0.5*np.mean(d)
    print('lag %3d (%5.2f km): gamma = %6.3f   = %5.1f%% of sill' % (lag, lag*0.4485, g, 100*g/sill))
