# %% 
# # ! Setup
import numpy as np
import pyshtools as pysh
import pygmt
import time
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
clm_shp_moon = pysh.SHCoeffs.from_file(f'Moon_shape_719.sh', 
                                       lmax=lmax_shp, 
                                       name='LOLA_shape (Moon)',
                                       units='m', format='bshc')
r_itfc = clm_shp_moon.coeffs[0,0,0]
# %% 
# # ! Computation of topographic potential: Spectral-domain 
rho0 = 2560.0 # kg/m^3
nmaxs = np.arange(1, 7)
clms = []
for nmax in nmaxs:
    t0 = time.time()
    lmax = (lmax_shp+1)*nmax-1
    clm_topoG_moon = \
        pysh.SHGravCoeffs.from_shape(shape=clm_shp_moon,
                                     rho=rho0,
                                     gm=gm_moon,
                                     nmax=nmax,
                                     lmax=lmax,
                                     lmax_grid=lmax,
                                     lmax_calc=lmax_shp,
                                     name=f'LOLA_Topo_Grav (Moon) nmax={nmax}')
    tc = time.time() - t0
    clms.append(clm_topoG_moon)
    print(f"nmax = {nmax:2d}; lmax = {lmax:2d}; time cost: {tc:8.3f}")
# %% 
# # ! Single contribution of each nmax in a band range
print("nmax | l-beg | l-end | t-expand |  min_gz  |  max_gz  |  mean_gz |  std_gz")
for i_clm in np.arange(len(clms)):
    nmax = nmaxs[i_clm]
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
        fmt = ("{:4d} | {:5d} | {:5d} | {:8.4f} | "
               "{:8.3f} | {:8.3f} | {:8.3f} | {:8.3f}")
        print(fmt.format(nmax, lmax1+1, lmax2, t1,
                         min_gz, max_gz, mean_gz, std_gz))
# %% 
# # ! Topography
#### * 
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
                      cb_label='Topography (km)',
                      cb_tick_interval=1,
                      cb_minor_tick_interval=0.5,
                      shading=grd_topo_moon)
fig.show(width=800)
fig.savefig('Moon_Topo.png', dpi=300)
# %% 
# # ! Topographic potential
clm_topoG_moon = clms[0]
grd_topoG_moon = clm_topoG_moon.expand(a=r_calc, f=0.0)
gx_topoG_moon = -1.e5 * grd_topoG_moon.theta
gy_topoG_moon =  1.e5 * grd_topoG_moon.phi
gz_topoG_moon = -1.e5 * grd_topoG_moon.rad
fig = pygmt.Figure()
gz_topoG_moon.plotgmt(fig=fig,
                      projection='mollweide',
                      central_longitude=-90.,
                      grid=[30, 30],
                      tick_interval=None,
                      cmap='haxby',
                      cmap_limits=[-800, 800],
                      colorbar='bottom',
                      cb_triangles='both',
                      cb_label='Downward gravity component (mGal)',
                      axes_labelsize=12,
                      cb_tick_interval=200,
                      cb_minor_tick_interval=100,
                      shading=grd_topo_moon)
fig.show(width=800)
fig.savefig('Moon_Topo_Gz.png', dpi=300)
