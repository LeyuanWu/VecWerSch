import numpy as np
from scipy.io import loadmat
import pyvista as pv;
import pandas as pd;
import time;
from gravity_forward import *;
# from gravity_forward_numba import *;
# %%
# ! # Load EROS <Geometry>
eros_mat = loadmat('EROS.mat');
eros_vf  = eros_mat['eros11272_22540'];  # shape: (nVert + nFace, 3)
nF       = 22540;
nVpF     = eros_vf.shape[0];
nV       = nVpF - nF;
Vert     = eros_vf[:nV, :].astype(np.float64);
Faces    = eros_vf[nV:, :].astype(np.int64);
X1, X2   = Vert[:, 0].min(), Vert[:, 0].max();
Y1, Y2   = Vert[:, 1].min(), Vert[:, 1].max();
Z1, Z2   = Vert[:, 2].min(), Vert[:, 2].max();
print(f'Number of <Vertex> : {nV}');
print(f'Number of <Faces>  : {nF}');
print(f'Bounding box (km)  :\n'
      f'X in [{X1:8.4f}, {X2:8.4f}]\n'
      f'Y in [{Y1:8.4f}, {Y2:8.4f}]\n'
      f'Z in [{Z2:8.4f}, {Z2:8.4f}]');
# %%
# ! # Load EROS <Gravity> computed in <Matlab>
eros_grav = loadmat('EROS_Grefs.mat');
V   = eros_grav['dV'];
gx  = eros_grav['gx'];   gy = eros_grav['gy'];   gz = eros_grav['gz'];
Txx = eros_grav['Txx']; Txy = eros_grav['Txy']; Txz = eros_grav['Txz'];
Tyy = eros_grav['Tyy']; Tyz = eros_grav['Tyz']; Tzz = eros_grav['Tzz'];
tc_matlab = eros_grav['time_cost'].item();
print(f'Computation size: {V.shape}');
print(f'tc_matlab: {tc_matlab:.2f} sec');
# %%
# ! # 
rho = 2670.;
xgv = np.linspace(-20., 20., 101);
ygv = np.linspace(-10., 10., 101);
[X2d, Y2d] = np.meshgrid(xgv, ygv);
z0 = -6.3;
Z2d = z0 * np.ones(X2d.shape);
P = np.column_stack((X2d.flatten(), Y2d.flatten(), Z2d.flatten()));
######## * Polyhedron
t1 = time.time();
V_cal, gx_cal, gy_cal, gz_cal, \
Txx_cal, Tyy_cal, Tzz_cal, Txy_cal, Txz_cal, Tyz_cal \
    = VecWerSch(P, Vert, Faces, rho);
# V_cal = VecWerSch_numba(P, Vert, Faces, rho);
t2 = time.time();
tc_np = t2 - t1;
print(f'Computation size: {X2d.shape}');
print(f'tc_numpy: {tc_np:.2f} sec');
# %%
######## * Stats
V, gx, gy, gz, Txx, Tyy, Tzz, Txy, Txz, Tyz \
    = (arr[::2,::4] for arr in (V, gx, gy, gz, Txx, Tyy, Tzz, Txy, Txz, Tyz));
Grefs = {'V': V, 'gx': gx, 'gy': gy, 'gz': gz,
         'Txx': Txx, 'Tyy': Tyy, 'Tzz': Tzz,
         'Txy': Txy, 'Txz': Txz, 'Tyz': Tyz};
stats = {
    name: {'Min': arr.min(), 'Max': arr.max(),
           'Mean': arr.mean(), 'Std': arr.std()}
    for name, arr in Grefs.items()};
df1 = pd.DataFrame(stats).T;
df1.index.name = 'Reference';
df1.style.format('{:.12e}').set_properties(**{'text-align': 'center'})
# %%
######## * Stats
Gcals = {'V': V_cal, 'gx': gx_cal, 'gy': gy_cal, 'gz': gz_cal,
         'Txx': Txx_cal, 'Tyy': Tyy_cal, 'Tzz': Tzz_cal,
         'Txy': Txy_cal, 'Txz': Txz_cal, 'Tyz': Tyz_cal};
stats = {
    name: {'Min': arr.min(), 'Max': arr.max(),
           'Mean': arr.mean(), 'Std': arr.std()}
    for name, arr in Gcals.items()};
df2 = pd.DataFrame(stats).T;
df2.index.name = 'Polyhedron';
df2.style.format('{:.12e}').set_properties(**{'text-align': 'center'})