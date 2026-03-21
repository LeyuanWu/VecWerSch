# %% 
# # ! Setup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyshtools as pysh
import pyshtools.constants.Moon as cstMoon
import pygmt
# %% 
# # ! Gravity of Moon
######## * SHGravCoeffs
lmax = 1500
clm_grav_moon = pysh.SHGravCoeffs.from_file('jggrx_1500e_sha.tab', 
                                            lmax=lmax, 
                                            header_units='km', 
                                            errors=True,
                                            omega=cstMoon.angular_velocity.value, 
                                            name='GL1500E (Moon)',
                                            encoding='utf-8')
print('*'*32+'\n' + 'Class SHGravCoeffs: ')
print(clm_grav_moon)
######## * SHGravGrid
r_eval = 1748000.0
#### * Expand
res = 0.1 # degree
grd_grav_moon = clm_grav_moon.expand(a=r_eval,
                                     f=0.0,
                                     lmax=int(90/res)-1, 
                                     lmax_calc=450)
print('*'*32+'\n' + 'Class SHGravGrid')
print(grd_grav_moon)
gx = -1.e5 * grd_grav_moon.theta
gy =  1.e5 * grd_grav_moon.phi
gz = -1.e5 * grd_grav_moon.rad
dg =  1.e5 * grd_grav_moon.total
#### * Compute manually the gravity disturbance
gz_normal = 1.e5 * clm_grav_moon.gm/r_eval**2
gz_omega = - 1.e5 * clm_grav_moon.omega**2 \
    * r_eval * np.cos(np.deg2rad(gz.lats()))**2
g_mag = np.sqrt(gx.data**2 + gy.data**2 + gz.data**2)
dg_rad = pysh.SHGrid.from_array(gz.data - gz_normal - gz_omega[:,np.newaxis])
dg_mag = pysh.SHGrid.from_array(g_mag - gz_normal - gz_omega[:,np.newaxis])
dif_dg1 = dg_rad - dg
dif_dg2 = dg_mag - dg
#### * xarray
xr_grd_grav_moon = grd_grav_moon.to_xarray()
print('*'*32+'\n' + 'Class xarray.Dataset')
print(xr_grd_grav_moon)
# %% 
# # ! Mapping
fig, axs = plt.subplot_mosaic([['(a)'],['(b)']], 
                               figsize=(6, 8), 
                               height_ratios=[0.45, 0.45])
dg.plot(ax=axs['(a)'], colorbar='bottom', 
        cb_label=r'$\delta g$ (mGal)',
        cmap='RdBu_r', tick_interval=[60, 30])
dif_dg1.plot(ax=axs['(b)'], colorbar='bottom', 
             cb_label=r'(Radial - Mag) $\delta g$ (mGal)',
             cmap='RdBu_r', tick_interval=[60, 30])
plt.savefig('moon_dg_rad_mag', dpi=300, bbox_inches='tight')
plt.show()
######## * Stats
Gs = {'Gravity disturbance: magnitude': dg.to_array(), 
      'Gravity disturbance: radial': dg_rad.to_array(),
      'Gravity disturbance: dif 1': dif_dg1.to_array(),
      'Gravity disturbance: dif 2': dif_dg2.to_array()}
stats = {
    name: {'Min': arr.min(), 'Max': arr.max(),
           'Mean': arr.mean(), 'Std': arr.std()}
    for name, arr in Gs.items()}
df = pd.DataFrame(stats).T
df.index.name = 'Gravity disturbance (mGal)'
df.style.format("{:.6f}").set_properties(**{'text-align': 'center'})
# %% 
# # ! Shape of Moon
######## * Shape
clm_shp_moon = pysh.SHCoeffs.from_file('Moon_shape_1439.sh', 
                                       lmax=1439, 
                                       name='LOLA_shape (Moon)',
                                       units='m', format='bshc')
grd_shp_moon = clm_shp_moon.expand(lmax=int(90/res)-1)
grd_topo_moon = grd_shp_moon/1.e3 - clm_grav_moon.r0/1.e3
# %% 
# # ! Topography
fig = pygmt.Figure()
grd_topo_moon.plotgmt(fig=fig,
                      projection='mollweide',
                      central_longitude=-90.,
                      grid=[30, 30],
                      tick_interval=None,
                      cmap='haxby',
                      cmap_limits=[-6, 7],
                      colorbar='bottom',
                      cb_triangles='both',
                      cb_label='Topography (km)',
                      cb_tick_interval=1,
                      cb_minor_tick_interval=0.5,
                      shading=grd_topo_moon)
fig.show(width=800)
fig.savefig('moon_topo.png', dpi=300)
# %% 
# # ! Total gravity anomaly
fig = pygmt.Figure()
dg.plotgmt(fig=fig,
           projection='mollweide',
           central_longitude=-90.,
           grid=[30, 30],
           tick_interval=None,
           cmap='vik',
           cmap_limits=[-500, 500],
           colorbar='bottom',
           cb_triangles='both',
           cb_label='Total gravity anomaly (mGal)',
           cb_tick_interval=100,
           cb_minor_tick_interval=50,
           shading=grd_topo_moon)
fig.show(width=800)
fig.savefig('moon_dg.png', dpi=300)
