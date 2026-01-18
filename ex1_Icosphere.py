##################################################################
# # ! Icosphere
##################################################################
# %%
# # ! Setup
import pyvista as pv;
import numpy as np;
import pandas as pd;
import time;
from gravity_forward import *;
# %%
# # ! Icosphere
N = 4;
icosphere = pv.Icosphere(radius=1.5, center=(0, 0, 2.), nsub=N);
Vert  = icosphere.points;
Faces = icosphere.regular_faces;
print(f'Nf: {Faces.shape[0]}');
rho   = 1000.;
xgv = np.linspace(-10., 10., 101);
ygv = np.linspace(-10., 10., 101);
[X2d, Y2d] = np.meshgrid(xgv, ygv);
z0 = 0.;
Z2d = z0 * np.ones(X2d.shape);
P = np.column_stack((X2d.flatten(), Y2d.flatten(), Z2d.flatten()));
######## * Sphere
start_time = time.time();
V, gx, gy, gz, Txx, Tyy, Tzz, Txy, Txz, Tyz = gsphere(P[:,0], P[:,1], P[:,2], 0., 0., 2., 1.5, rho);
end_time = time.time();
elapsed = end_time - start_time;
print(f'<Sphere> time cost: {elapsed:.6f} sec');
######## * Polyhedron
start_time = time.time();
Vcal = VecWerSch(P, Vert, Faces, rho);
end_time = time.time();
elapsed = end_time - start_time;
print(f'<Polyhedron> time cost: {elapsed:.6f} sec');
# %%
######## * Stats
Grefs = {'V': V, 'gx': gx, 'gy': gy, 'gz': gz,
         'Txx': Txx, 'Tyy': Tyy, 'Tzz': Tzz,
         'Txy': Txy, 'Txz': Txz, 'Tyz': Tyz};
stats = {
    name: {'Min': arr.min(), 'Max': arr.max(),
           'Mean': arr.mean(), 'Std': np.sqrt(np.mean(arr**2))}
    for name, arr in Grefs.items()};
df1 = pd.DataFrame(stats).T;
df1.index.name = 'Reference';
df1.style.format("{:.4e}").set_properties(**{'text-align': 'center'})
# %%
######## * Stats
Gcals = {'V': Vcal};
stats = {
    name: {'Min': arr.min(), 'Max': arr.max(),
           'Mean': arr.mean(), 'Std': np.sqrt(np.mean(arr**2))}
    for name, arr in Gcals.items()};
df2 = pd.DataFrame(stats).T;
df2.index.name = 'Polyhedron';
df2.style.format("{:.4e}").set_properties(**{'text-align': 'center'})