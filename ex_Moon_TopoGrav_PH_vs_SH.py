#########################################################################
# # ! Moon Topographic-Gravitational-Potential Computation: PH vs SH
#########################################################################
# %% 
# # ! Setup
import numpy as np
import pyshtools as pysh
import pygmt
import xarray as xr
import time
from datetime import datetime
from gravity_forward_numba import *
# %% 
# # ! Start time
print("=" * 80)
print(f"Start time: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
print("=" * 80)
# %% 
# # ! Constants
myG = 6.67430e-11
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
in_res = 15.0/60.0 # degree
lmax_grid = int(90.0/in_res - 1)
grd_shp_moon = clm_shp_moon.expand(lmax=lmax_grid, 
                                   lmax_calc=lmax_shp)
grd_topo_moon = grd_shp_moon/ 1.e3 - r_itfc_km
# %% 
# # ! Load TGP data: SH vs PH
#### * SH
max_nmax = 7
nc_file_SH = f'output/moon_topo_gravity_Lshp{lmax_shp}_nmax{max_nmax}.nc'
data_SH = xr.open_dataset(nc_file_SH)
gn_SH = data_SH['gx'].data[::max_nmax, ::max_nmax]
ge_SH = data_SH['gy'].data[::max_nmax, ::max_nmax]
gd_SH = data_SH['gz'].data[::max_nmax, ::max_nmax]
#### * PH
out_res = 15.0/60.0 # degree
nc_file_PH = (f"output/moon_topo_gravity_in{int(in_res*60)}arcmin"
              f"_out{int(out_res*60)}arcmin.nc")
data_PH = xr.open_dataset(nc_file_PH)
gn_PH = data_PH['gN'].data
ge_PH = data_PH['gE'].data
gd_PH = data_PH['gD'].data
#### * Diff = PH - SH
d_gn, d_ge, d_gd = gn_PH - gn_SH, ge_PH - ge_SH, gd_PH - gd_SH
d_gn[[0, -1], :] = 0.0
d_ge[[0, -1], :] = 0.0
#### * Summary statistics
g_SH = {'N': gn_SH, 'E': ge_SH, 'D': gd_SH}
g_PH = {'N': gn_PH, 'E': ge_PH, 'D': gd_PH}
d_g  = {'N': d_gn,  'E': d_ge,  'D': d_gd}
print('-' * 80)
print('Gravity Vector components in mGal')
print(f"{'Type':4s} | {'Comp':4s} | {'min':>9s} | {'max':>9s} | {'mean':>9s} | {'std':>9s}")
print('-' * 80)
datasets = {'SH': g_SH, 'PH': g_PH, 'Diff': d_g}
for typ in ['SH', 'PH', 'Diff']:
    for comp in ['N', 'E', 'D']:
        data = datasets[typ][comp]
        print(f"{typ:4s} | {f'g{comp}':4s} | {data.min():9.4f} | {data.max():9.4f} | "
              f"{data.mean():9.4f} | {data.std():9.4f}")
    print('=' * 60)
# %% 
# # ! Top-N largest |gD| errors (PH - SH) with >500 km separation
n_largest = 10
min_dist_km = 500.0
R_moon = 1737.4  # km
abs_err = np.abs(d_gd).flatten()
idx_sorted = np.argsort(-abs_err)
lats_1d = data_PH['latitude'].data
lons_1d = data_PH['longitude'].data
nlat, nlon = d_gd.shape
pk_flat_idx = []
for idx in idx_sorted:
    if len(pk_flat_idx) >= n_largest:
        break
    i_lat, j_lon = np.unravel_index(idx, (nlat, nlon))
    lat_new, lon_new = np.radians([lats_1d[i_lat], lons_1d[j_lon]])
    keep = True
    for idx_sel in pk_flat_idx:
        ii, jj = np.unravel_index(idx_sel, (nlat, nlon))
        lat_old, lon_old = np.radians([lats_1d[ii], lons_1d[jj]])
        dlat = lat_new - lat_old
        dlon = lon_new - lon_old
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat_new)*np.cos(lat_old)*np.sin(dlon/2)**2
        dist_km = R_moon * 2 * np.arcsin(np.sqrt(a))
        if dist_km < min_dist_km:
            keep = False
            break
    if keep:
        pk_flat_idx.append(idx)
lat_indices, lon_indices = np.unravel_index(pk_flat_idx, (nlat, nlon))
print("=" * 104)
print(f"Top {n_largest} largest |gD| errors (PH - SH), >{min_dist_km} km apart")
print("=" * 104)
print(f"{'Rank':>4} | {'Lat (°)':>8} | {'Lon (°)':>8} | {'Topo (m)':>9} | "
      f"{'gD_SH':>10} | {'gD_PH':>10} | {'Error':>9} | {'|Error|':>9}")
print("-" * 104)
for idx in range(len(pk_flat_idx)):
    i_lat, j_lon = lat_indices[idx], lon_indices[idx]
    topo_m = grd_topo_moon.data[i_lat, j_lon] * 1.e3
    gd_sh_val = gd_SH[i_lat, j_lon]
    gd_ph_val = gd_PH[i_lat, j_lon]
    err = d_gd[i_lat, j_lon]
    print(f"{idx+1:4d} | {lats_1d[i_lat]:8.3f} | {lons_1d[j_lon]:8.3f} | {topo_m:9.1f} | "
          f"{gd_sh_val:10.4f} | {gd_ph_val:10.4f} | {err:9.4f} | {np.abs(err):9.4f}")
print("=" * 104)
# %% 
# # ! Mapping: SH vs PH
err_txt = ['A','B','C','D','E','F','G','H','I','J']
xr_gd_PH = pysh.SHGrid.from_array(gd_PH).to_xarray()
xr_d_gd = pysh.SHGrid.from_array(d_gd).to_xarray()
fig = pygmt.Figure()
with fig.subplot(nrows=2, ncols=1, figsize=('14c', '16c'), margins="0.5c"):
    with fig.set_panel(panel=0): 
        fig.grdimage(grid=xr_gd_PH, projection="W-90/14c", cmap="haxby", frame="g30")
        fig.plot(x=lons_1d[lon_indices], y=lats_1d[lat_indices], 
                 style="t0.25c", fill="white", pen="0.25p,black")
        fig.text(x=lons_1d[lon_indices], y=lats_1d[lat_indices], text=err_txt, 
                 justify="BL", offset="0.06c/0.0c", font="7p,Helvetica-Bold,black")
        fig.colorbar(position="JBC+o0/0.2i+w10c/0.3h", 
                     frame=["a200f100", "x+l@[ g_z @[", "y+lmGal"])    
    with fig.set_panel(panel=1): 
        fig.grdimage(grid=xr_d_gd, projection="W-90/14c", cmap="haxby", frame="g30")
        fig.colorbar(position="JBC+o0/0.2i+w10c/0.3h", 
                     frame=["a5f5", "x+l@[ \\delta(g_z) @[", "y+lmGal"])    
fig.savefig('Moon_TopoGz_PH_vs_SH_GMT.png', dpi=400)
fig.show(width=800)
# %% 
# # ! End time
print("=" * 80)
print(f"End time: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
print("=" * 80)