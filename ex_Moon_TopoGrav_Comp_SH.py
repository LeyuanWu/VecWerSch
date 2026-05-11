##################################################################
# # ! Moon Topographic-Potential Computation: Spherical Harmonics
##################################################################
# %% 
# # ! Setup
import numpy as np
import pyshtools as pysh
import pygmt
import xarray as xr
import time
from datetime import datetime
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
r_calc = 1748000.0
# %% 
# # ! Shape of Moon
#### * Shape
lmax_shp = 359
clm_shp_moon = pysh.SHCoeffs.from_file(f'input/Moon_shape_719.sh', 
                                       lmax=lmax_shp, 
                                       name='LOLA_shape (Moon)',
                                       units='m', format='bshc')
r_itfc = clm_shp_moon.coeffs[0,0,0]
# %% 
# # ! Computation of topographic potential: Spectral-domain 
rho0 = 2560.0 # kg/m^3
max_nmax = 6
NMAXs = np.arange(1, max_nmax+1)
clms = []
print('*'*32+'\n' + 'Computation of Topographic Potential: Spectral-domain\n' + '*'*32)
for nmax in NMAXs:
    t0 = time.time()
    lmax = (lmax_shp+1)*nmax-1
    clm_tgp_moon = \
        pysh.SHGravCoeffs.from_shape(shape=clm_shp_moon,
                                     rho=rho0,
                                     gm=gm_moon,
                                     nmax=nmax,
                                     lmax=lmax,
                                     lmax_grid=lmax,
                                     lmax_calc=lmax_shp,
                                     name=f'LOLA_Topo_Grav (Moon) nmax={nmax}')
    tc = time.time() - t0
    clms.append(clm_tgp_moon)
    print(f"nmax = {nmax:2d}; lmax = {lmax:4d}; time cost: {tc:8.2f} sec")
# %% 
# # ! Single contribution of each nmax in a band range
print('*'*32+'\n' + 'Single Contribution of Each nmax in a Band Range\n' + '*'*32)
print(f"nmax | l-beg | l-end | t-expand |"
      f"       min_gz |       max_gz |       mean_gz|       std_gz |")
for i_clm in np.arange(len(clms)):
    nmax = NMAXs[i_clm]
    if i_clm == 0:
        dif_clm = clms[i_clm]
    else:
        lmax = (lmax_shp+1)*(nmax-1)-1
        clm_1 = clms[i_clm-1]
        clm_2 = clms[i_clm]
        dif_clm = clm_2.copy()
        dif_clm.coeffs[:,:lmax+1,:lmax+1] = \
            clm_2.coeffs[:,:lmax+1,:lmax+1] - clm_1.coeffs
    for n_cut in np.arange(nmax):
        lmax1 = (lmax_shp+1)*n_cut-1
        lmax2 = (lmax_shp+1)*(n_cut+1)-1
        dif_clm.coeffs[:, :lmax1+1, :lmax1+1] = 0.0
        dif_clm.coeffs[:, lmax2+1:, lmax2+1:] = 0.0
        t0 = time.time()
        grd_sc = dif_clm.expand(a=r_calc, 
                                f=0.0,
                                lmax_calc=lmax2)
        t1 = time.time() - t0
        gz = -1.e5 * grd_sc.rad
        min_gz  = np.min(gz.data)
        max_gz  = np.max(gz.data)
        mean_gz = np.mean(gz.data)
        std_gz  = np.std(gz.data)
        fmt = ("{:4d} | {:5d} | {:5d} | {:8.2f} | "
               "{:12.2e} | {:12.2e} | {:12.2e} | {:12.2e} |")
        print(fmt.format(nmax, lmax1+1, lmax2, t1,
                         min_gz, max_gz, mean_gz, std_gz))
# %% 
# # ! Topography
grd_shp_moon = clm_shp_moon.expand()
grd_topo_moon = (grd_shp_moon - r_itfc) / 1.e3
fig = pygmt.Figure()
grd_topo_moon.plotgmt(fig=fig,
                      projection='mollweide',
                      central_longitude=-90.,
                      grid=[30, 30],
                      tick_interval=None,
                      cmap='haxby',
                      cmap_limits=[-8, 8],
                      colorbar='bottom',
                      cb_triangles='both',
                      cb_label="Topography (km)",
                      cb_tick_interval=1,
                      cb_minor_tick_interval=0.5,
                      shading=False)
fig.savefig('Moon_Topo_GMT.png', dpi=400)
# fig.show(width=800)
# %% 
# # ! Topographic potential
clm_tgp_moon = clms[-1]
grd_tgp_moon = clm_tgp_moon.expand(a=r_calc, f=0.0)
gx_tgp_moon = -1.e5 * grd_tgp_moon.theta
gy_tgp_moon =  1.e5 * grd_tgp_moon.phi
gz_tgp_moon = -1.e5 * grd_tgp_moon.rad
arr_gx, arr_gy, arr_gz = gx_tgp_moon.data, gy_tgp_moon.data, gz_tgp_moon.data
vars = ['gx', 'gy', 'gz']
for var, arr in zip(vars, [arr_gx, arr_gy, arr_gz]):
    print(f"Topographic Gravitational Potential: {var}")
    print(f" min: {np.min(arr):12.4f} mGal\n max: {np.max(arr):12.4f} mGal \n"
          f"mean: {np.mean(arr):12.4f} mGal\n std: {np.std(arr):12.4f} mGal")
fig = pygmt.Figure()
gz_tgp_moon.plotgmt(fig=fig,
                    projection='mollweide',
                    central_longitude=-90.,
                    grid=[30, 30],
                    tick_interval=None,
                    cmap='haxby',
                    cmap_limits=[-800, 800],
                    colorbar='bottom',
                    cb_triangles='both',
                    cb_label=r"@[ g_z @[ (mGal)",
                    axes_labelsize=12,
                    cb_tick_interval=200,
                    cb_minor_tick_interval=100,
                    shading=False)
fig.savefig('Moon_TopoGz_GMT.png', dpi=400)
# fig.show(width=800)
# %% 
# # ! Save gx, gy, gz to NetCDF
nc_file = f'moon_topo_gravity_Lshp{lmax_shp}_nmax{max_nmax}.nc'
lat = gz_tgp_moon.lats()
lon = gz_tgp_moon.lons()
res = int(180 * 60 / (len(lat) - 1))
xr.Dataset(
    {'gx': (('latitude', 'longitude'), arr_gx),
     'gy': (('latitude', 'longitude'), arr_gy),
     'gz': (('latitude', 'longitude'), arr_gz)},
    coords={'longitude': lon, 'latitude': lat},
    attrs={'units': 'mGal', 'reference_radius_m': r_calc, 'density_kg_m3': rho0}
).to_netcdf(nc_file)
print(f"Saved to '{nc_file}'")  
# %% 
# # ! End time
print("=" * 80)
print(f"End time: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
print("=" * 80)