# %% 
# # ! Setup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyshtools as pysh
import pygmt
import pyvista as pv
import time
# %% 
# # ! Constants
G = pysh.constants.G.value
gm_moon = pysh.constants.Moon.gm.value
r_calc = 1748000.0
r_ref = 1738000.0
# %% 
# # ! Shape/Topo of Moon
#### * Shape
clm_shp_moon = pysh.SHCoeffs.from_file('Moon_shape_2879.sh', 
                                       lmax=2879, 
                                       name='LOLA_shape (Moon)',
                                       units='m', format='bshc')
res_topo = 0.25
lmax_pk = int(90/res_topo)-1
grd_shp_moon = clm_shp_moon.expand(lmax=int(90/res_topo)-1,
                                   lmax_calc=lmax_pk)
grd_topo_moon = (grd_shp_moon - r_ref) / 1.e3
# %% 
# # ! Computation of topographic potential: Spectral-domain 
nthreads = 1
NMAX = 21
print('nmax | time_sec |         min |         max |        mean |         std')
print('-'*70)
for nmax in range(1, NMAX):
    t0 = time.time()
    clm_topograv_moon = \
        pysh.SHGravCoeffs.from_shape(shape=clm_shp_moon,
                                     rho=2560.0,
                                     gm=gm_moon,
                                     nmax=nmax,
                                     lmax=lmax_pk,
                                     lmax_grid=lmax_pk,
                                     lmax_calc=lmax_pk,
                                     name=f'LOLA_Topo_Grav (Moon) nmax={nmax}',
                                     backend='ducc',
                                     nthreads=nthreads)
    tc = time.time() - t0
    res_topograv = res_topo
    grd_grav_moon = clm_topograv_moon.expand(a=r_calc,
                                             f=0.0,
                                             lmax=int(90/res_topograv)-1, 
                                             lmax_calc=lmax_pk)
    gz_topo = -1.e5 * grd_grav_moon.rad
    min_val = np.min(gz_topo.data)
    max_val = np.max(gz_topo.data)
    mean_val = np.mean(gz_topo.data)
    std_val = np.std(gz_topo.data)
    print(f"{nmax:4d} | {tc:8.3f} | {min_val:12.4f} | {max_val:12.4f} | {mean_val:12.4f} | {std_val:12.4f}")
#### * Mapping
fig = pygmt.Figure()
gz_topo.plotgmt(fig=fig,
                projection='mollweide',
                central_longitude=-90.,
                grid=[30, 30],
                tick_interval=None,
                cmap='vik',
                cmap_limits=[-800, 800],
                colorbar='bottom',
                cb_triangles='both',
                cb_label='Topographic radial gravity (mGal)',
                cb_tick_interval=200,
                cb_minor_tick_interval=100,
                shading=grd_topo_moon)
fig.show(width=800)
######## * Stats
Gs = {'Topographic gravity: radial (mGal)': gz_topo.to_array()}
stats = {
    name: {'Min': arr.min(), 'Max': arr.max(),
           'Mean': arr.mean(), 'Std': arr.std()}
    for name, arr in Gs.items()}
df = pd.DataFrame(stats).T
df.style.format("{:.4f}").set_properties(**{'text-align': 'center'})
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

mesh_shp_moon['Topography'] = (pts_r - r_ref) / 1.e3
sargs = dict(
    title='Topography (km)',
    title_font_size=20,
    label_font_size=16,
    n_labels=9,
    italic=True,
    fmt='%.1f',
    font_family='arial',
    width=0.8,
    height=0.1,
    position_x=0.1,
    position_y=0.1,
    vertical=False)
pl = pv.Plotter()
mesh_front = mesh_shp_moon.translate([0, 2.e6, 0])
mesh_back = mesh_shp_moon.rotate_z(180)
mesh_back.translate([0, -2.e6, 0], inplace=True)
pl.add_mesh(
    mesh_front,
    cmap='seismic',
    clim=[-8, 8],
    scalars='Topography',
    show_scalar_bar=False)
pl.add_mesh(
    mesh_back,
    cmap='seismic',
    clim=[-8, 8],
    scalars='Topography',
    show_scalar_bar=False)
pl.add_scalar_bar(**sargs)
pl.view_yz()
pl.camera.Zoom(1.5)
pl.show()
pl.screenshot("Moon_Topo_3D.png");


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