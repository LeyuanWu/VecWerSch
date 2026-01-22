##################################################################
# # ! Icosphere
##################################################################
# %%
# # ! Setup
import pyvista as pv;
import numpy as np;
import pandas as pd;
import time;
from gravity_forward_numpy import *;
from gravity_forward_numba import *;
# %%
# # ! Icosphere <Geometry>
xc, yc, zc = 0.1, 0.0, 2.0;
a = 1.5;
vol = 4*np.pi*a**3/3; area = 4*np.pi*a**2;
nsub_max = 10;
Nsubs = np.arange(nsub_max);
NVs = np.zeros(nsub_max, dtype=int);
NFs = np.zeros(nsub_max, dtype=int);
Es_vol = np.zeros(nsub_max);
Es_area = np.zeros(nsub_max);
MINs_psi = np.zeros(nsub_max);
MAXs_psi = np.zeros(nsub_max);
for insub, nsub in enumerate(Nsubs):
    icosph = pv.Icosphere(radius=a, center=(xc, yc, zc), nsub=nsub);
    Vert  = icosph.points;
    Faces = icosph.regular_faces;
    NVs[insub] = Vert.shape[0];
    NFs[insub] = Faces.shape[0];
    Es_vol[insub] = 1. - icosph.volume/vol;
    Es_area[insub] = 1. - icosph.area/area;
    vert_unit = (Vert - np.array([xc, yc, zc]))/a;
    min_psi, max_psi = spherical_edge_length_range(vert_unit, Faces);
    MINs_psi[insub] = 60. * np.rad2deg(min_psi); # arc-min
    MAXs_psi[insub] = 60. * np.rad2deg(max_psi);
print(f"{'nsub':>4} {'N_vertices':>10} {'N_faces':>10} "
      f"{'Vol_err':>10} {'Area_err':>10} {'Min_psi':>10} {'Max_psi':>10}")
print("-" * 63)
for i in range(len(Nsubs)):
    print(f"{Nsubs[i]:>4} {int(NVs[i]):>10} {int(NFs[i]):>10} "
          f"{Es_vol[i]:>10.2e} {Es_area[i]:>10.2e} "
          f"{MINs_psi[i]:>10.2f} {MAXs_psi[i]:>10.2f}")
# %%
# # ! Icosphere <Gravity>
rho = 1000.;
xgv = np.linspace(-10., 10., 51);
ygv = np.linspace(-10., 10., 51);
z0 = 0.;
[X2d, Y2d] = np.meshgrid(xgv, ygv);
Z2d = z0 * np.ones(X2d.shape);
XPs, YPs, ZPs = map(lambda x: x.flatten(), [X2d, Y2d, Z2d]);
P = np.column_stack((XPs, YPs, ZPs));
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