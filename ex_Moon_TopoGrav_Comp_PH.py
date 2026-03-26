# %% 
# # ! Setup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyshtools as pysh
import pygmt
import pyvista as pv
import time
from gravity_forward_numba import *
# %% 
# # ! Constants
myG = 6.67430e-11
G = pysh.constants.G.value
gm_moon = pysh.constants.Moon.gm.value
r_calc = 1748000.0 / 1.e3
# %% 
# # ! Shape of Moon
#### * Shape
lmax_shp = 359
clm_shp_moon = pysh.SHCoeffs.from_file(f'Moon_shape_719.sh', 
                                       lmax=lmax_shp, 
                                       name='LOLA_shape (Moon)',
                                       units='m', format='bshc')
r_itfc = clm_shp_moon.coeffs[0,0,0] / 1.e3
grd_shp_moon = clm_shp_moon.expand()
# %% 
# # ! Computation of topographic potential: Spatial-domain 
rho0 = 2560.0 # kg/m^3
#### * Polyhedron geometry
mesh_shp_moon = pv.Sphere(radius=1.0, center=(0.0, 0.0, 0.0),
                          theta_resolution=grd_shp_moon.nlon-1,
                          phi_resolution=grd_shp_moon.nlat)
[LON, LAT] = np.meshgrid(grd_shp_moon.lons(), grd_shp_moon.lats())
R_SHP = grd_shp_moon.data/1.e3
X = R_SHP * np.cos(np.deg2rad(LAT)) * np.cos(np.deg2rad(LON))
Y = R_SHP * np.cos(np.deg2rad(LAT)) * np.sin(np.deg2rad(LON))
Z = R_SHP * np.sin(np.deg2rad(LAT))
# north pole, south pole, and regular points
pts_x = np.hstack([X[0,0], X[-1,0], X[1:-1,:-1].flatten(order='F')]) 
pts_y = np.hstack([Y[0,0], Y[-1,0], Y[1:-1,:-1].flatten(order='F')])
pts_z = np.hstack([Z[0,0], Z[-1,0], Z[1:-1,:-1].flatten(order='F')])
Verts = np.column_stack((pts_x, pts_y, pts_z))
mesh_shp_moon.points = Verts
Faces = mesh_shp_moon.regular_faces
#### * Calculation points
lon_p = np.arange(0, 361, 5)
lat_p = np.arange(90, -91, -5)
[LON_P, LAT_P] = np.meshgrid(lon_p, lat_p)
X_calc = r_calc * np.cos(np.deg2rad(LAT_P)) * np.cos(np.deg2rad(LON_P))
Y_calc = r_calc * np.cos(np.deg2rad(LAT_P)) * np.sin(np.deg2rad(LON_P))
Z_calc = r_calc * np.sin(np.deg2rad(LAT_P))
calc_lon_rad = np.deg2rad(LON_P.flatten())
calc_lat_rad = np.deg2rad(LAT_P.flatten())
calc_x = X_calc.flatten()
calc_y = Y_calc.flatten()
calc_z = Z_calc.flatten()
P = np.column_stack((calc_x, calc_y, calc_z))
t0 = time.time()
V, gx, gy, gz, Txx, Txy, Txz, Tyy, Tyz, Tzz = WerSch_numba(P, Verts, Faces, rho0)
t1 = time.time() - t0
print(f"Time cost : {t1:8.3f} sec \n"
      f"Number of points: {P.shape[0]} \n"
      f"Number of faces: {Faces.shape[0]} \n")
V0, gx0, gy0, gz0, Txx0, Txy0, Txz0, Tyy0, Tyz0, Tzz0 = \
    gsphere(calc_x, calc_y, calc_z, 0, 0, 0, r_itfc, rho0)
V = V - V0
gx = gx - gx0
gy = gy - gy0
gz = gz - gz0
Txx = Txx - Txx0
Txy = Txy - Txy0
Txz = Txz - Txz0
Tyy = Tyy - Tyy0
Tyz = Tyz - Tyz0
Tzz = Tzz - Tzz0
gN, gE, gD, TNN, TNE, TND, TEE, TED, TDD = \
    rotate_vec_ten_ecef2ned(calc_lon_rad, calc_lat_rad,
                            gx, gy, gz,
                            Txx, Txy, Txz,
                            Tyy, Tyz, Tzz)
print(f'gD (mGal) --> min: {gD.min():.3e}; max: {gD.max():.3e}; mean: {gD.mean():.3e}; std: {gD.std():.3e}')

gD_topoG_moon = pysh.SHGrid.from_array(gD.reshape(LON_P.shape))
# %% 
# # ! Topographic potential
fig = pygmt.Figure()
gD_topoG_moon.plotgmt(fig=fig,
                      projection='mollweide',
                      central_longitude=-90.,
                      grid=[30, 30],
                      tick_interval=None,
                      cmap='haxby',
                      cmap_limits=[-800, 800],
                      colorbar='bottom',
                      cb_triangles='both',
                      cb_label='Downward gravity component (mGal)',
                      axes_labelsize=12,
                      cb_tick_interval=200,
                      cb_minor_tick_interval=100)
fig.show(width=800)
            