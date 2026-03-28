# %% 
# # ! Setup
import numpy as np
import pyshtools as pysh
import pyvista as pv
import time
import xarray as xr
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
Faces = mesh_shp_moon.regular_faces
del mesh_shp_moon

#### * Calculation points
res_deg = 5.0
lon_p = np.arange(0, 360+res_deg, res_deg)
lat_p = np.arange(90, -90-res_deg, -res_deg)
[LON_P, LAT_P] = np.meshgrid(lon_p, lat_p)
XP = r_calc * np.cos(np.deg2rad(LAT_P)) * np.cos(np.deg2rad(LON_P))
YP = r_calc * np.cos(np.deg2rad(LAT_P)) * np.sin(np.deg2rad(LON_P))
ZP = r_calc * np.sin(np.deg2rad(LAT_P))
lon_calc_rad = np.deg2rad(LON_P.flatten())
lat_calc_rad = np.deg2rad(LAT_P.flatten())
xps = XP.flatten()
yps = YP.flatten()
zps = ZP.flatten()
P = np.column_stack((xps, yps, zps))

#### * Computation of topographic potential
t0 = time.time()
vgt_PH = WerSch_numba(P, Verts, Faces, rho0)
t1 = time.time() - t0
print("Computation info: \n"
      f"Geographic grid: dlat = dlon = {res_deg} deg = {int(60*res_deg)} arc-min\n"
      f"Number of computation points: {len(lat_p)} x {len(lon_p)} = {P.shape[0]} \n"
      "Polyhedron geometry: \n"
      f"Number of faces: {Faces.shape[0]} \n"
      f"Number of vertices: {Verts.shape[0]} \n"
      f"Time cost : {t1:8.3f} sec = {t1/60:.3f} min = {t1/(60*60):.3f} hr\n")
vgt_SP = gsphere(xps, yps, zps, 0, 0, 0, r_itfc, rho0)
vgt_TP = tuple(g_PH - g_SP for g_PH, g_SP in zip(vgt_PH, vgt_SP))
V, gx, gy, gz, Txx, Txy, Txz, Tyy, Tyz, Tzz = vgt_TP
gN, gE, gD, TNN, TNE, TND, TEE, TED, TDD = \
    rotate_vec_ten_ecef2ned(lon_calc_rad, lat_calc_rad,
                            gx, gy, gz,
                            Txx, Txy, Txz,
                            Tyy, Tyz, Tzz)
print(f'-----------------------------------------------------------')
print(f'Gravity |   min    |   max    |   mean   |   std   ')
print('GP in m^2/s^2, gravity in mGal, and gradients in Eotvos')
print(f'-----------------------------------------------------------')
print(f'V       | {V.min():8.3f} | {V.max():8.3f} | {V.mean():8.3f} | {V.std():8.3f}')
print(f'gN      | {gN.min():8.3f} | {gN.max():8.3f} | {gN.mean():8.3f} | {gN.std():8.3f}')
print(f'gE      | {gE.min():8.3f} | {gE.max():8.3f} | {gE.mean():8.3f} | {gE.std():8.3f}')
print(f'gD      | {gD.min():8.3f} | {gD.max():8.3f} | {gD.mean():8.3f} | {gD.std():8.3f}')
print(f'TNN     | {TNN.min():8.3f} | {TNN.max():8.3f} | {TNN.mean():8.3f} | {TNN.std():8.3f}')
print(f'TNE     | {TNE.min():8.3f} | {TNE.max():8.3f} | {TNE.mean():8.3f} | {TNE.std():8.3f}')
print(f'TND     | {TND.min():8.3f} | {TND.max():8.3f} | {TND.mean():8.3f} | {TND.std():8.3f}')
print(f'TEE     | {TEE.min():8.3f} | {TEE.max():8.3f} | {TEE.mean():8.3f} | {TEE.std():8.3f}')
print(f'TED     | {TED.min():8.3f} | {TED.max():8.3f} | {TED.mean():8.3f} | {TED.std():8.3f}')
print(f'TDD     | {TDD.min():8.3f} | {TDD.max():8.3f} | {TDD.mean():8.3f} | {TDD.std():8.3f}')
# %%
# # ! Saving results to NetCDF
shape_2d = (len(lat_p), len(lon_p))
data_vars = {
    'V':   (('latitude', 'longitude'), V.reshape(shape_2d)),
    'gN':  (('latitude', 'longitude'), gN.reshape(shape_2d)),
    'gE':  (('latitude', 'longitude'), gE.reshape(shape_2d)),
    'gD':  (('latitude', 'longitude'), gD.reshape(shape_2d)),
    'TNN': (('latitude', 'longitude'), TNN.reshape(shape_2d)),
    'TNE': (('latitude', 'longitude'), TNE.reshape(shape_2d)),
    'TND': (('latitude', 'longitude'), TND.reshape(shape_2d)),
    'TEE': (('latitude', 'longitude'), TEE.reshape(shape_2d)),
    'TED': (('latitude', 'longitude'), TED.reshape(shape_2d)),
    'TDD': (('latitude', 'longitude'), TDD.reshape(shape_2d)),
}
ds = xr.Dataset(
    data_vars=data_vars,
    coords={
        'longitude': lon_p,
        'latitude': lat_p,
    },
    attrs={
        'description': 'Topographic gravity and gradient components of the Moon',
        'r_calc_km': r_calc,
        'rho_kg_m3': rho0,
        'ref_radius_km': r_itfc,
        'resolution_deg': res_deg,
    }
)
nc_file = f'moon_topo_gravity_res{int(res_deg*60)}arcmin.nc'
ds.to_netcdf(nc_file)
print(f"Saved to '{nc_file}'")

            