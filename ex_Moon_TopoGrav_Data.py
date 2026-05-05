##################################################################
# # ! Moon Topography and Gravity Data
##################################################################
# %% 
# # ! Setup
import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyshtools as pysh
import pyshtools.constants.Moon as cstMoon
import pygmt
import pyvista as pv
pv.set_jupyter_backend('static')
import time
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
# # ! Gravity Data of Moon
######## * SHGravCoeffs
lmax = 1500
clm_grav_moon = \
    pysh.SHGravCoeffs.from_file('input/jggrx_1500e_sha.tab', 
                                lmax=lmax, 
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
res_grav = 1.0/8 # degree
grd_grav_moon = clm_grav_moon.expand(a=r_calc,
                                     f=0.0,
                                     lmax=int(90/res_grav)-1, 
                                     lmax_calc=450)
print('*'*32+'\n' + 'Class SHGravGrid')
print(grd_grav_moon)
gx = -1.e5 * grd_grav_moon.theta
gy =  1.e5 * grd_grav_moon.phi
gz = -1.e5 * grd_grav_moon.rad
dg_shtools =  1.e5 * grd_grav_moon.total
gh = gx.copy()
gh.data = np.sqrt(gx.data**2 + gy.data**2)
#### * Compute manually the gravity disturbance
gz_gamma =   1.e5 * clm_grav_moon.gm/r_calc**2
gz_omega = - 1.e5 * clm_grav_moon.omega**2 \
    * r_calc * np.cos(np.deg2rad(gz.lats()))**2
g_normal = gz_gamma + gz_omega[:,np.newaxis]
g_vecnorm = np.sqrt(gx.data**2 + gy.data**2 + gz.data**2)
dgz  = pysh.SHGrid.from_array(gz.data - g_normal)
dgvecnorm = pysh.SHGrid.from_array(g_vecnorm - g_normal)
dif_dg1 = dgz - dg_shtools
dif_dg2 = dgvecnorm - dg_shtools
#### * xarray
xr_grd_grav_moon = grd_grav_moon.to_xarray()
print('*'*32+'\n' + 'Class xarray.Dataset')
print(xr_grd_grav_moon)
# %%
# # ! Mapping & Stats Gravity of Moon
#### * Mapping with Matplotlib
plots = [
    (gh, r'$g_h$ (mGal)', '(a)'),
    (dg_shtools, r'$\delta g_{\mathrm{shtools}}$ (mGal)', '(b)'),
    (dif_dg1, r'$\delta g_z - \delta g_{\mathrm{shtools}}$ (mGal)', '(c)'),
    (dif_dg2, r'$\delta |\mathbf{g}| - \delta g_{\mathrm{shtools}}$ (mGal)', '(d)')
]
fig, axs = plt.subplot_mosaic([['(a)', '(b)'], ['(c)', '(d)']], 
                              figsize=(12, 9))
for data, label, key in plots:
    data.plot(ax=axs[key], colorbar='bottom', cb_label=label,
              cmap='seismic', tick_interval=[60, 30])
# plt.savefig(f'Moon_gh_dgz_dgvecnorm_r{r_calc/1e3:.0f}km.png',
#             dpi=300, bbox_inches='tight')
plt.show()
#### * Mapping with PyGMT
fig = pygmt.Figure()
dg_shtools.plotgmt(fig=fig,
           projection='mollweide',
           central_longitude=-90.,
           grid=[30, 30],
           tick_interval=None,
           cmap='vik',
           cmap_limits=[-500, 500],
           colorbar='bottom',
           cb_triangles='both',
           cb_label='Free-air gravity anomaly (mGal)',
           cb_tick_interval=100,
           cb_minor_tick_interval=50)
# fig.savefig(f'Moon_FAG_r{r_calc/1e3:.0f}km_Mollweide.png', dpi=400)
fig.show(width=800)
######## * Stats
# Gravity fields (mGal)
Gs = {'gx': gx.to_array(), 'gy': gy.to_array(), 
      'gh': gh.to_array(), 'gz': gz.to_array()}
print(f"\nGravity fields (mGal) at r = {r_calc/1e3:.0f} km")
print(f"{'':<12} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
print("-" * 58)
for name, arr in Gs.items():
    print(f"{name:<12} {arr.min():12.3f} {arr.max():12.3f} "
          f"{arr.mean():12.3f} {arr.std():12.3f}")
# Gravity disturbance (mGal)
DGs = {
    'shtools': dg_shtools.to_array(),
    'delta gz': dgz.to_array(),
    'delta |g|': dgvecnorm.to_array(),
    'delta gz - shtools': dif_dg1.to_array(),
    'delta |g| - shtools': dif_dg2.to_array()
}
print(f"\nGravity disturbance (mGal) at r = {r_calc/1e3:.0f} km")
print(f"{'':<22} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
print("-" * 70)
for name, arr in DGs.items():
    print(f"{name:<22} {arr.min():12.6f} {arr.max():12.6f} "
          f"{arr.mean():12.6f} {arr.std():12.6f}")
# %% 
# # ! Shape/Topo of Moon
######## * Shape
r_ref = 1738000.0
res_topo = np.array([1.0/8, 1.0/16, 1.0/32, 1.0/64]) # degree
LMAX_shp = np.asanyarray(90/res_topo-1, dtype=int)
print("Topographic Statistics of the Moon at Increasing Spherical Harmonic Resolutions")
print("(Values in km relative to reference radius = 1738 km)")
print('    lmax | time_sec |      min |      max |     mean |      std')
print('-'*64)
topo_old = pysh.SHGrid.from_zeros(lmax=5759)
for i, lmax_shp in enumerate(LMAX_shp):
    t0 = time.time()
    clm_shp_moon = \
        pysh.SHCoeffs.from_file(f'input/Moon_shape_{lmax_shp}.sh', 
                                lmax=lmax_shp, 
                                name='LOLA_shape (Moon)',
                                units='m', format='bshc')
    grd_shp_moon = clm_shp_moon.expand(lmax=5759, 
                                       lmax_calc=lmax_shp)
    tc = time.time() - t0
    topo_new = (grd_shp_moon - r_ref) / 1.e3
    min_val  = np.min(topo_new.data)
    max_val  = np.max(topo_new.data)
    mean_val = np.mean(topo_new.data)
    std_val  = np.std(topo_new.data)
    print(f"{lmax_shp:8d} | {tc:8.3f} | {min_val:8.4f} | {max_val:8.4f} | {mean_val:8.4f} | {std_val:8.4f}")
    if i>=1:
        dif_topo = topo_new - topo_old
        min_val  = np.min(dif_topo.data)
        max_val  = np.max(dif_topo.data)
        mean_val = np.mean(dif_topo.data)
        std_val  = np.std(dif_topo.data)      
        print(f"nmax: ({LMAX_shp[i]:4d} - {LMAX_shp[i-1]:4d}) | {min_val:8.4f} | {max_val:8.4f} | {mean_val:8.4f} | {std_val:8.4f}")
    topo_old = topo_new.copy()         
# %% 
# # ! PyVista 3D Visualization
#### * Shape/Topo
grd_shp_moon = clm_shp_moon.expand(lmax=719, 
                                   lmax_calc=359)
mesh_shp_moon, pts_r = Shgrid2Mesh(grd_shp_moon)
pts_topo = (pts_r - r_ref) / 1.e3
var_topo = f'Topo_Ref{r_ref/1e3:.0f}km'
mesh_shp_moon[var_topo] = pts_topo
#### * Gravity disturbance
pts_dg = np.hstack([dg_shtools.data[0,0], dg_shtools.data[-1,0],
                    dg_shtools.data[1:-1,:-1].flatten(order='F')]) 
var_FAG = f'FAG_r{r_calc/1e3:.0f}km'
mesh_shp_moon[var_FAG] = pts_dg

Titles = [f'Topography (km)', 
          f'Free-air gravity anomaly (mGal)']
Scalars = [var_topo, var_FAG]
Clims = [[-8, 8], [-500, 500]]
N_lab = [9, 11]
for title, scalar, clim, n_lab in zip(Titles, Scalars, Clims, N_lab):
    bar_x, bar_y = 0.1, 0.1
    bar_w, bar_h = 0.8, 0.1
    sargs = dict(
        title='', 
        label_font_size=18, n_labels=n_lab, fmt='%.0f',
        width=bar_w, height=bar_h,
        position_x=bar_x, position_y=bar_y,
        vertical=False)
    pl = pv.Plotter(image_scale=3)
    mesh_front = mesh_shp_moon.translate([0, 2.e6, 0])
    mesh_back = mesh_shp_moon.rotate_z(180).translate([0, -2.e6, 0])
    pl.add_mesh(mesh_front, cmap='seismic', clim=clim,
                scalars=scalar, show_scalar_bar=False)
    pl.add_mesh(mesh_back, cmap='seismic', clim=clim,
                scalars=scalar, show_scalar_bar=False)
    pl.add_scalar_bar(**sargs)
    text_y = bar_y + bar_h / 2 + 0.03
    pl.add_text(title, position=(0.5-len(title)/2*0.012, text_y), 
                viewport=True, font_size=12)
    # Set camera
    camera_config = dict(azimuth = 0, elevation = 0, distance = 9.e6,
                        pan_x = 0, pan_y = 0.0, pan_z = -5.e5)
    az = np.deg2rad(camera_config['azimuth'])
    el = np.deg2rad(camera_config['elevation'])
    dist = camera_config['distance']
    scene_center = np.array(pl.center)
    focal_point = scene_center + np.array([
        camera_config['pan_x'],
        camera_config['pan_y'],
        camera_config['pan_z']])
    camera_position = focal_point + dist * np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el)])
    pl.camera.position = camera_position
    pl.camera.focal_point = focal_point
    pl.camera.up = (0, 0, 1)

    pl.window_size = (1000, 600)
    # pl.screenshot(f"Moon_{scalar}_3D.png")
    pl.show()