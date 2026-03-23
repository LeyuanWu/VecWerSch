# %% 
# # ! Setup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyshtools as pysh
import pyshtools.constants.Moon as cstMoon
import pygmt
import pyvista as pv
import time
# %% 
# # ! Gravity of Moon
######## * SHGravCoeffs
lmax_grav = 1500
clm_grav_moon = pysh.SHGravCoeffs.from_file('jggrx_1500e_sha.tab', 
                                            lmax=lmax_grav, 
                                            header_units='km', 
                                            errors=True,
                                            omega=cstMoon.angular_velocity.value, 
                                            name='GL1500E (Moon)',
                                            encoding='utf-8')
print('*'*32+'\n' + 'Class SHGravCoeffs: ')
print(clm_grav_moon)
######## * SHGravGrid
r_calc = 1748000.0
#### * Expand
res_grav = 0.1 # degree
grd_grav_moon = clm_grav_moon.expand(a=r_calc,
                                     f=0.0,
                                     lmax=int(90/res_grav)-1, 
                                     lmax_calc=450)
print('*'*32+'\n' + 'Class SHGravGrid')
print(grd_grav_moon)
gz = -1.e5 * grd_grav_moon.rad
dg =  1.e5 * grd_grav_moon.total
# %% 
# # ! Shape of Moon
######## * Shape
clm_shp_moon = pysh.SHCoeffs.from_file('Moon_shape_1439.sh', 
                                       lmax=1439, 
                                       name='LOLA_shape (Moon)',
                                       units='m', format='bshc')
res_topo = 0.05
lmax_pk = int(90/res_topo)-1
grd_shp_moon = clm_shp_moon.expand(lmax=int(90/res_topo)-1,
                                   lmax_calc=lmax_pk)
grd_topo_moon = (grd_shp_moon - clm_grav_moon.r0) / 1.e3 # km
fig = pygmt.Figure()
grd_topo_moon.plotgmt(fig=fig,
                      projection='mollweide',
                      central_longitude=-90.,
                      grid=[30, 30],
                      tick_interval=None,
                      cmap='rainbow',
                      cmap_limits=[-8, 8],
                      colorbar='bottom',
                      cb_triangles='both',
                      cb_label='Topography (km)',
                      cb_tick_interval=1,
                      cb_minor_tick_interval=0.5,
                      shading=grd_topo_moon)
fig.show(width=800)
# %% 
# # ! Computation of topographic potential: Spectral-domain 
print('nmax | time_sec |        min |        max |       mean |        std')
print('-'*70)
results = []
for nmax_val in range(1, 21):
    nthreads = 1
    t0 = time.time()
    clm_topograv_moon = pysh.SHGravCoeffs.from_shape(shape=clm_shp_moon,
                                                     rho=2560.0,
                                                     gm=clm_grav_moon.gm,
                                                     nmax=nmax_val,
                                                     lmax=lmax_pk,
                                                     lmax_grid=lmax_pk,
                                                     lmax_calc=lmax_pk,
                                                     name=f'LOLA_Topo_Grav (Moon) nmax={nmax_val}',
                                                     backend='ducc',
                                                     nthreads=nthreads)
    tc = time.time() - t0
    res_topograv = res_topo # degree
    grd_grav_moon_tmp = clm_topograv_moon.expand(a=r_calc,
                                                 f=0.0,
                                                 lmax=int(90/res_topograv)-1, 
                                                 lmax_calc=lmax_pk)
    gz_topo_tmp = -1.e5 * grd_grav_moon_tmp.rad
    min_val = np.min(gz_topo_tmp.data)
    max_val = np.max(gz_topo_tmp.data)
    mean_val = np.mean(gz_topo_tmp.data)
    std_val = np.std(gz_topo_tmp.data)
    results.append((nmax_val, tc, min_val, max_val, mean_val, std_val))
    print(f"{nmax_val:4d} | {tc:8.3f} | {min_val:12.4f} | {max_val:12.4f} | {mean_val:12.4f} | {std_val:12.4f}")
grd_grav_moon = grd_grav_moon_tmp
gz_topo = gz_topo_tmp

# fig = pygmt.Figure()
# gz_topo.plotgmt(fig=fig,
#                 projection='mollweide',
#                 central_longitude=-90.,
#                 grid=[30, 30],
#                 tick_interval=None,
#                 cmap='vik',
#                 cmap_limits=[-800, 800],
#                 colorbar='bottom',
#                 cb_triangles='both',
#                 cb_label='Topographic radial gravity (mGal)',
#                 cb_tick_interval=200,
#                 cb_minor_tick_interval=100,
#                 shading=grd_topo_moon)
# fig.show(width=800)
# %% 
# # ! Computation of topographic potential: Spatial-domain 
mesh_shp_moon = pv.Sphere(radius=1.0, center=(0.0, 0.0, 0.0),
                          theta_resolution=grd_shp_moon.nlon-1,
                          phi_resolution=grd_shp_moon.nlat)
[LON, LAT] = np.meshgrid(grd_shp_moon.lons(), grd_shp_moon.lats())
RTOPO = grd_shp_moon.data
X = RTOPO * np.cos(np.deg2rad(LAT)) * np.cos(np.deg2rad(LON))
Y = RTOPO * np.cos(np.deg2rad(LAT)) * np.sin(np.deg2rad(LON))
Z = RTOPO * np.sin(np.deg2rad(LAT))
pts_x = X[1:-1,:-1].flatten(order='F')
pts_y = Y[1:-1,:-1].flatten(order='F')
pts_z = Z[1:-1,:-1].flatten(order='F')
pts_topo = np.column_stack((pts_x, pts_y, pts_z))
north_pole = np.array([X[0,0], Y[0,0], Z[0,0]]);
south_pole = np.array([X[-1,0], Y[-1,0], Z[-1,0]]);
Verts = np.insert(pts_topo, 0, np.vstack([north_pole, south_pole]), axis=0)
mesh_shp_moon.points = Verts
Faces = mesh_shp_moon.regular_faces
pts_r = np.sqrt(np.sum(Verts**2, axis=1))




# t0 = time.time()
# V_g, gx_g, gy_g, gz_g, Txx_g, Txy_g, Txz_g, Tyy_g, Tyz_g, Tzz_g = \
#     WerSch_numba(P, Verts, Faces, rho)
# t_geo = time.time() - t0
# time_geo.append(t_geo        
# # Transform numerical results to NED
# gN_geo, gE_geo, gD_geo, TNN_geo, TNE_geo, TND_geo, TEE_geo, TED_geo, TDD_geo = \
#     rotate_vec_ten_ecef2ned(
#         lon_rad, lat_rad,
#         gx_g, gy_g, gz_g,
#         Txx_g, Txy_g, Txz_g,
#         Tyy_g, Tyz_g, Tzz_g
            
# cal_geo = [V_g, gN_geo, gE_geo, gD_geo,
#             TNN_geo, TNE_geo, TND_geo, TEE_geo, TED_geo, TDD_geo]