##################################################################
# # ! Icosphere
##################################################################
# %%
# # ! Setup
import pyvista as pv;
import numpy as np;
import pandas as pd;
import time;
from gravity_forward_numpy import gsphere;
from gravity_forward_numba import VecWerSch_numba;
# %%
# # ! Icosphere
xc = 0.1; yc = 0.0; zc = 2.0;
a = 1.5;
rho   = 1000.;
xgv = np.linspace(-10., 10., 101);
ygv = np.linspace(-10., 10., 101);
z0 = 0.;
[X2d, Y2d] = np.meshgrid(xgv, ygv);
Z2d = z0 * np.ones(X2d.shape);
XPs, YPs, ZPs = map(lambda x: x.flatten(), [X2d, Y2d, Z2d]);
P = np.column_stack((XPs, YPs, ZPs));

vol = 4*np.pi*a**3/3; area = 4*np.pi*a**2;
print(f'Sphere volume: {vol:12.6f}', flush=True);
print(f'Sphere area: {area:12.6f}', flush=True);
Ns = np.arange(9);
for N in Ns:
    icosph = pv.Icosphere(radius=a, center=(xc, yc, zc), nsub=N);
    Vert  = icosph.points;
    Faces = icosph.regular_faces;
    print(f'N : {N}');
    print(f'Number of <Vertex> : {Vert.shape[0]}');
    print(f'Number of <Faces>  : {Faces.shape[0]}');
    re_vol = np.abs(icosph.volume-vol)/vol * 100;
    re_area = np.abs(icosph.area-area)/area * 100;
    print(f'Icosphere volume: {icosph.volume:12.6f}, e={re_vol:.4f}%', flush=True);
    print(f'Icosphere area: {icosph.area:12.6f}, e={re_area:.4f}%', flush=True);
######## * Sphere
t1 = time.time();
V, gx, gy, gz, Txx, Tyy, Tzz, Txy, Txz, Tyz \
    = gsphere(P[:,0], P[:,1], P[:,2], xc, yc, zc, a, rho);
tc_sphere = time.time() - t1;
print(f'<Sphere> time cost: {tc_sphere:.2f} sec');
######## * Polyhedron
t1 = time.time();
V_cal, gx_cal, gy_cal, gz_cal, \
Txx_cal, Tyy_cal, Tzz_cal, Txy_cal, Txz_cal, Tyz_cal \
    = VecWerSch_numba(P, Vert, Faces, rho);
tc_numba = time.time() - t1;
print(f'<Polyhedron> time cost: {tc_numba:.2f} sec');
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