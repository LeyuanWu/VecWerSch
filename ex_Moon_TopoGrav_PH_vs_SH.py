# %% 
# # ! Setup
import numpy as np
import pyshtools as pysh
import pygmt
import time
import xarray as xr
# %% 
# # ! Constants
gm_moon = pysh.constants.Moon.gm.value
r_calc = 1748000.0
# %% 
# # ! Shape of Moon
#### * Shape
lmax_shp = 359
clm_shp_moon = pysh.SHCoeffs.from_file(f'Moon_shape_719.sh', 
                                       lmax=lmax_shp, 
                                       name='LOLA_shape (Moon)',
                                       units='m', format='bshc')
r_itfc = clm_shp_moon.coeffs[0,0,0]
grd_shp_moon = clm_shp_moon.expand()
grd_topo_moon = (grd_shp_moon - r_itfc) / 1.e3
# %% 
# # ! Computation of topographic potential: Spectral-domain 
rho0 = 2560.0 # kg/m^3
nmax = 3
lmax_grid = (lmax_shp+1)*nmax-1
lmax_grav = (lmax_shp+1)*2-1
t0 = time.time()
clm_topoG_moon = \
    pysh.SHGravCoeffs.from_shape(shape=clm_shp_moon,
                                 rho=rho0,
                                 gm=gm_moon,
                                 nmax=nmax,
                                 lmax=lmax_grav,
                                 lmax_grid=lmax_grid,
                                 lmax_calc=lmax_shp,
                                 name=f'LOLA_Topo_Grav (Moon) nmax={nmax}')
tc = time.time() - t0
print(f"Shape --> CLM: \n"
      f"nmax = {nmax:2d} \n"
      f"lmax_grid = {lmax_grid:4d}; lmax_grav = {lmax_grav:4d}; time cost: {tc:8.3f}")
t0 = time.time()
grd_topoG_moon = clm_topoG_moon.expand(a=r_calc, 
                                       f=0.0,
                                       lmax=lmax_grav,
                                       lmax_calc=lmax_grav)
tc = time.time() - t0
print(f"CLM --> Grid: \n"
      f"lmax = {lmax_grav:4d}; lmax_calc = {lmax_grav:4d}; time cost: {tc:8.3f}")
gn_SH = -1.e5 * grd_topoG_moon.theta
ge_SH =  1.e5 * grd_topoG_moon.phi
gd_SH = -1.e5 * grd_topoG_moon.rad
# %% 
# # ! Computation of topographic potential: Spatial-domain
res_deg = 0.25
nc_file = f'./output/moon_topo_gravity_in{int(res_deg*60)}arcmin.nc'
ds = xr.open_dataset(nc_file)
gn_PH = pysh.SHGrid.from_xarray(ds['gN'])
ge_PH = pysh.SHGrid.from_xarray(ds['gE'])
gd_PH = pysh.SHGrid.from_xarray(ds['gD'])
d_gn = gn_PH.data - gn_SH.data[::2,::2]
d_ge = ge_PH.data - ge_SH.data[::2,::2]
d_gd = gd_PH.data - gd_SH.data[::2,::2]
d_gn[[0, -1], :] = 0.0
d_ge[[0, -1], :] = 0.0
print(f'-----------------------------------------------------------')
print(f'Dif|   min    |   max    |   mean   |   std   ')
print('GP in m^2/s^2, GV in mGal, and GGT in Eotvos')
print(f'-----------------------------------------------------------')
print(f'gN | {d_gn.min():9.4f} | {d_gn.max():9.4f} | {d_gn.mean():9.4f} | {d_gn.std():9.4f}')
print(f'gE | {d_ge.min():9.4f} | {d_ge.max():9.4f} | {d_ge.mean():9.4f} | {d_ge.std():9.4f}')
print(f'gD | {d_gd.min():9.4f} | {d_gd.max():9.4f} | {d_gd.mean():9.4f} | {d_gd.std():9.4f}')
# %% 
# # ! Mapping: SH vs PH
grd_d_gd = pysh.SHGrid.from_array(np.abs(d_gd), 
                                  name='Dif gd (PH - SH)')
fig = pygmt.Figure()
grd_d_gd.plotgmt(fig=fig,
                 projection='mollweide',
                 central_longitude=-90.,
                 grid=[30, 30],
                 tick_interval=None,
                 cmap='haxby',
                 cmap_limits=[1.e-1, 1.e1],
                 cmap_scale='log',
                 colorbar='bottom',
                 cb_triangles='both',
                 cb_label=r'@[ \delta{g_z} @[ (mGal)',
                 axes_labelsize=12)
fig.show(width=800)