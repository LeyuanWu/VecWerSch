#########################################################################
# # ! Moon Topographic-Gravitational-Potential Computation: Polyhedron
#########################################################################
# %% 
# # ! Setup
import numpy as np
import pyshtools as pysh
import pyvista as pv
import xarray as xr
import time
from datetime import datetime
from gravity_forward_numba import *
# %%
# # ! Functions
def Shgrid2Mesh(shgrid):
    mesh = pv.Sphere(radius=1.0, center=(0.0, 0.0, 0.0),
                     theta_resolution=shgrid.nlon-1,
                     phi_resolution=shgrid.nlat)
    [LON, LAT] = np.meshgrid(shgrid.lons(), shgrid.lats())
    R = shgrid.data
    X = R * np.cos(np.deg2rad(LAT)) * np.cos(np.deg2rad(LON))
    Y = R * np.cos(np.deg2rad(LAT)) * np.sin(np.deg2rad(LON))
    Z = R * np.sin(np.deg2rad(LAT))
    # north pole, south pole, and regular points
    pts_x = np.hstack([X[0,0], X[-1,0], X[1:-1,:-1].flatten(order='F')]) 
    pts_y = np.hstack([Y[0,0], Y[-1,0], Y[1:-1,:-1].flatten(order='F')])
    pts_z = np.hstack([Z[0,0], Z[-1,0], Z[1:-1,:-1].flatten(order='F')])
    pts_r = np.sqrt(pts_x**2 + pts_y**2 + pts_z**2)
    Verts = np.column_stack((pts_x, pts_y, pts_z))
    mesh.points = Verts
    return mesh, pts_r
# %% 
# # ! Start time
print("=" * 80)
print(f"Start time: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
print("=" * 80)
# %% 
# # ! Constants
myG = 6.67430e-11
G = pysh.constants.G.value
gm_moon = pysh.constants.Moon.gm.value
r_calc_km = 1748000.0 / 1.e3
# %% 
# # ! Shape of Moon
#### * Shape
lmax_shp = 359
clm_shp_moon = \
    pysh.SHCoeffs.from_file(f'input/Moon_shape_719.sh', 
                            lmax=lmax_shp, 
                            name='LOLA_shape (Moon)',
                            units='m', format='bshc')
r_itfc_km = clm_shp_moon.coeffs[0,0,0] / 1.e3
in_res = 6.0/60.0 # degree
lmax_grid = int(90.0/in_res - 1)
grd_shp_moon = clm_shp_moon.expand(lmax=lmax_grid, 
                                   lmax_calc=lmax_shp)
# %% 
# # ! Computation of TGP: Polyhedron
rho0 = 2560.0 # kg/m^3
#### * Polyhedron geometry
mesh_shp_moon, _ = Shgrid2Mesh(grd_shp_moon)
Verts = mesh_shp_moon.points / 1.e3 # km
Faces = mesh_shp_moon.regular_faces
del mesh_shp_moon
#### * Calculation points
out_res = 15.0/60.0 # degree
lon_1D = np.arange(0, 360+out_res, out_res)
lat_1D = np.arange(90, -90-out_res, -out_res)
[LON_2D, LAT_2D] = np.meshgrid(np.deg2rad(lon_1D), np.deg2rad(lat_1D))
XP_2D = r_calc_km * np.cos(LAT_2D) * np.cos(LON_2D)
YP_2D = r_calc_km * np.cos(LAT_2D) * np.sin(LON_2D)
ZP_2D = r_calc_km * np.sin(LAT_2D)
lon_calc_rad = LON_2D.flatten()
lat_calc_rad = LAT_2D.flatten()
xps = XP_2D.flatten()
yps = YP_2D.flatten()
zps = ZP_2D.flatten()
P = np.column_stack((xps, yps, zps))
#### * Computation of TGP
t0 = time.time()
vgt_PH = WerSch_numba_v1(P, Verts, Faces, rho0)
tc = time.time() - t0
print("Computation info: \n"
      f"Geographic grid: dlat = dlon = {out_res} deg = {int(60*out_res)} arc-min\n"
      f"Number of computation points: {len(lat_1D)} x {len(lon_1D)} = {P.shape[0]} \n"
      "Polyhedron geometry: \n"
      f"Number of faces: {Faces.shape[0]} \n"
      f"Number of vertices: {Verts.shape[0]} \n"
      f"Time cost : {tc:8.3f} sec = {tc/60:.3f} min = {tc/(60*60):.3f} hr\n")
#### * TGP = Shape model gravity - Sphere (r=r_itfc) gravity
vgt_SP = gsphere(xps, yps, zps, 0, 0, 0, r_itfc_km, rho0)
vgt_TP = tuple(g_PH - g_SP for g_PH, g_SP in zip(vgt_PH, vgt_SP))
V, gx, gy, gz, Txx, Txy, Txz, Tyy, Tyz, Tzz = vgt_TP
gN, gE, gD, TNN, TNE, TND, TEE, TED, TDD = \
    rotate_vec_ten_ecef2ned(lon_calc_rad, lat_calc_rad,
                            gx, gy, gz,
                            Txx, Txy, Txz,
                            Tyy, Tyz, Tzz)
print(f'-----------------------------------------------------------')
print('GP in m^2/s^2, GV in mGal, and GGT in Eotvos')
print(f'Gravity |    min    |    max    |    mean   |     std   ')
print(f'-----------------------------------------------------------')
print(f'V       | {V.min():9.3f} | {V.max():9.3f} | {V.mean():9.3f} | {V.std():9.3f}')
print(f'gN      | {gN.min():9.3f} | {gN.max():9.3f} | {gN.mean():9.3f} | {gN.std():9.3f}')
print(f'gE      | {gE.min():9.3f} | {gE.max():9.3f} | {gE.mean():9.3f} | {gE.std():9.3f}')
print(f'gD      | {gD.min():9.3f} | {gD.max():9.3f} | {gD.mean():9.3f} | {gD.std():9.3f}')
print(f'TNN     | {TNN.min():9.3f} | {TNN.max():9.3f} | {TNN.mean():9.3f} | {TNN.std():9.3f}')
print(f'TNE     | {TNE.min():9.3f} | {TNE.max():9.3f} | {TNE.mean():9.3f} | {TNE.std():9.3f}')
print(f'TND     | {TND.min():9.3f} | {TND.max():9.3f} | {TND.mean():9.3f} | {TND.std():9.3f}')
print(f'TEE     | {TEE.min():9.3f} | {TEE.max():9.3f} | {TEE.mean():9.3f} | {TEE.std():9.3f}')
print(f'TED     | {TED.min():9.3f} | {TED.max():9.3f} | {TED.mean():9.3f} | {TED.std():9.3f}')
print(f'TDD     | {TDD.min():9.3f} | {TDD.max():9.3f} | {TDD.mean():9.3f} | {TDD.std():9.3f}')
# %%
# # ! Saving results to NetCDF
shape_2D = (len(lat_1D), len(lon_1D))
data_vars = {
    'V':   (('latitude', 'longitude'), V.reshape(shape_2D)),
    'gN':  (('latitude', 'longitude'), gN.reshape(shape_2D)),
    'gE':  (('latitude', 'longitude'), gE.reshape(shape_2D)),
    'gD':  (('latitude', 'longitude'), gD.reshape(shape_2D)),
    'TNN': (('latitude', 'longitude'), TNN.reshape(shape_2D)),
    'TNE': (('latitude', 'longitude'), TNE.reshape(shape_2D)),
    'TND': (('latitude', 'longitude'), TND.reshape(shape_2D)),
    'TEE': (('latitude', 'longitude'), TEE.reshape(shape_2D)),
    'TED': (('latitude', 'longitude'), TED.reshape(shape_2D)),
    'TDD': (('latitude', 'longitude'), TDD.reshape(shape_2D)),
}
ds = xr.Dataset(
    data_vars=data_vars,
    coords={'longitude': lon_1D, 'latitude': lat_1D},
    attrs={'description': 'Topographic gravity and gradient components of the Moon',
           'r_calc_km': r_calc_km,
           'ref_radius_km': r_itfc_km,
           'rho_kg_m3': rho0,
           'resolution_deg': out_res})
nc_file = (f"moon_topo_gravity_in{int(in_res*60)}arcmin"
           f"_out{int(out_res*60)}arcmin.nc")
ds.to_netcdf(nc_file)
print(f"Saved to '{nc_file}'")
# %% 
# # ! End time
print("=" * 80)
print(f"End time: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
print("=" * 80)

            