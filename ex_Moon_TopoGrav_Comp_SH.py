# %% 
# # ! Setup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyshtools as pysh
import pygmt
import pyvista as pv
import time
from gravity_forward_numba import WerSch_numba
# %% 
# # ! Constants
myG = 6.67430e-11
G = pysh.constants.G.value
gm_moon = pysh.constants.Moon.gm.value
r_calc = 1748000.0
# %% 
# # ! Shape of Moon
#### * Shape
lmax_shp = 719
clm_shp_moon = pysh.SHCoeffs.from_file(f'Moon_shape_{lmax_shp}.sh', 
                                       lmax=lmax_shp, 
                                       name='LOLA_shape (Moon)',
                                       units='m', format='bshc')
r_itfc = clm_shp_moon.coeffs[0,0,0]
# %% 
# # ! Computation of topographic potential: Spectral-domain 
rho0 = 2560.0 # kg/m^3
nmaxs = np.arange(1, 5)
clms = []
for nmax in nmaxs:
    t0 = time.time()
    clm_topograv_moon = \
        pysh.SHGravCoeffs.from_shape(shape=clm_shp_moon,
                                     rho=rho0,
                                     gm=gm_moon,
                                     nmax=nmax,
                                     lmax=(lmax_shp+1)*nmax-1,
                                     lmax_grid=(lmax_shp+1)*nmax-1,
                                     lmax_calc=lmax_shp,
                                     name=f'LOLA_Topo_Grav (Moon) nmax={nmax}')
    tc = time.time() - t0
    clms.append(clm_topograv_moon)
    print(f"nmax: {nmax:2d}; time cost (sec): {tc:8.3f}")
# %% 
# # ! Single contribution of each nmax/degree range
for iclm in np.arange(len(clms)):
    if iclm == 0:
        dif_clm = clms[iclm]
    else:
        clm_temp = clms[iclm-1].change_ref(r0=clms[iclm].r0,
                                           lmax=clms[iclm].lmax)
        dif_clm = clms[iclm] - clm_temp
    for n_cut in np.arange(0, iclm):
        cut1 = (lmax_shp+1)*n_cut-1
        cut2 = (lmax_shp+1)*(n_cut+1)-1
        dif_clm.coeffs[:, :cut1+1, :cut1+1] = 0.0
        dif_clm.coeffs[:, cut2+1:, cut2+1:] = 0.0
        grd_sc = dif_clm.expand(a=r_calc, f=0.0)
        gz_topo = -1.e5 * grd_sc.rad
        min_gz = np.min(gz_topo.data)
        max_gz = np.max(gz_topo.data)
        mean_gz = np.mean(gz_topo.data)
        std_gz = np.std(gz_topo.data)
        print(f"{iclm:4d} | {n_cut:4d} | {min_gz:12.4f} | {max_gz:12.4f} | {mean_gz:12.4f} | {std_gz:12.4f}")
#### * Mapping
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
#                 cb_minor_tick_interval=100)
# fig.show(width=800)