# %% 
# # ! Setup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyshtools as pysh
import pyshtools.constants.Moon as cstMoon
import pygmt
import time
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
res_grav = 0.1 # degree
grd_grav_moon = clm_grav_moon.expand(a=r_eval,
                                     f=0.0,
                                     lmax=int(90/res_grav)-1, 
                                     lmax_calc=450)
print('*'*32+'\n' + 'Class SHGravGrid')
print(grd_grav_moon)
gz = -1.e5 * grd_grav_moon.rad
dg =  1.e5 * grd_grav_moon.total
#### * xarray
xr_grd_grav_moon = grd_grav_moon.to_xarray()
print('*'*32+'\n' + 'Class xarray.Dataset')
print(xr_grd_grav_moon)
# %% 
# # ! Shape of Moon
######## * Shape
res_topo = 0.05
clm_shp_moon = pysh.SHCoeffs.from_file('Moon_shape_1439.sh', 
                                       lmax=1439, 
                                       name='LOLA_shape (Moon)',
                                       units='m', format='bshc')
grd_shp_moon = clm_shp_moon.expand(lmax=int(90/res_topo)-1,
                                   lmax_calc=1439)
grd_topo_moon = grd_shp_moon/1.e3 - clm_grav_moon.r0/1.e3
# %% 
# # ! Computation of topographic potential: Spectral-domain 
nmax = 7
nthreads = 1
t0 = time.time()
clm_topograv_moon = pysh.SHGravCoeffs.from_shape(shape=clm_shp_moon,
                                                 rho=2560.0,
                                                 gm=clm_grav_moon.gm,
                                                 nmax=nmax,
                                                 lmax=1439,
                                                 lmax_grid=1439,
                                                 lmax_calc=1439,
                                                 name='LOLA_Topo_Grav (Moon)',
                                                 backend='ducc',
                                                 nthreads=nthreads)
tc = time.time() - t0
print(f"When nthreads={nthreads}, nmax={nmax} --> time cost {tc:.2f} seconds")
# %% 
# # ! Expand & Mapping
res_topograv = 0.05 # degree
grd_grav_moon = clm_topograv_moon.expand(a=r_eval,
                                         f=0.0,
                                         lmax=int(90/res_topograv)-1, 
                                         lmax_calc=1439)
print('*'*32+'\n' + 'Class SHGravGrid')
print(grd_grav_moon)
gz = -1.e5 * grd_grav_moon.rad

fig = pygmt.Figure()
gz.plotgmt(fig=fig,
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