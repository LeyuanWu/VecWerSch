import numpy as np
from scipy.io import loadmat
import pandas as pd;
import time;
import pyvista as pv;
from gravity_forward_numba import VecWerSch_numba;
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
# ! #  Numba code computation
rho = 2670.;
xgv = np.linspace(-20., 20., 401);
ygv = np.linspace(-10., 10., 201);
[X2d, Y2d] = np.meshgrid(xgv, ygv);
z0 = -6.3;
Z2d = z0 * np.ones(X2d.shape);
P = np.column_stack((X2d.flatten(), Y2d.flatten(), Z2d.flatten()));
######## * Polyhedron
t1 = time.time();
V_cal, gx_cal, gy_cal, gz_cal, \
Txx_cal, Tyy_cal, Tzz_cal, Txy_cal, Txz_cal, Tyz_cal \
    = VecWerSch_numba(P, Vert, Faces, rho);
tc_numba = time.time() - t1;
print(f'Computation size: {X2d.shape}');
print(f'tc_numba: {tc_numba:.2f} sec');
# %%
# ! #  Pandas disp stats 
fields = ['V', 'gx', 'gy', 'gz', 'Txx', 'Tyy', 'Tzz', 'Txy', 'Txz', 'Tyz']

df_ref = pd.DataFrame({
    name: {'Min': arr.min(), 'Max': arr.max(), 'Mean': arr.mean(), 'Std': arr.std()}
    for name, arr in zip(fields, [V, gx, gy, gz, Txx, Tyy, Tzz, Txy, Txz, Tyz])
}).T
df_cal = pd.DataFrame({
    name: {'Min': arr.min(), 'Max': arr.max(), 'Mean': arr.mean(), 'Std': arr.std()}
    for name, arr in zip(fields, [V_cal, gx_cal, gy_cal, gz_cal, 
                                  Txx_cal, Tyy_cal, Tzz_cal, Txy_cal, Txz_cal, Tyz_cal])
}).T
df_diff = df_cal - df_ref

def print_table(df, title):
    print(f"\n{title}")
    print("=" * len(title))
    print(df.to_string(float_format="{:12.6f}".format))
print_table(df_ref, "Reference")
print_table(df_cal, "Polyhedron")
print_table(df_diff, "Difference")

# %%
df_ref
# %%
df_cal
# %%
df_diff