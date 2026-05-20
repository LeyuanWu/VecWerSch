#########################################################################
# # ! Moon Topographic-Gravitational-Potential Computation: PH vs SH
#########################################################################
# %% 
# # ! Setup
import numpy as np
import pandas as pd
import pyshtools as pysh
import pygmt
import xarray as xr
from datetime import datetime
from gravity_forward_numba import *
# %% 
# # ! Start time
print("=" * 80)
print(f"Start time: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
print("=" * 80)
# %% 
# # ! Load Topo data
#### * Topography
lmax_shp = 359
in_res = 15.0/60.0 # degree
nc_topo = f'output/moon_topo_Lshp{lmax_shp}_{int(60*in_res)}arcmin.nc'
xr_topo = xr.open_dataset(nc_topo)
arr_topo = xr_topo['topo'].data
# %% 
# # ! Load TGP data: SH vs PH
#### * SH
max_nmax = 7
nc_SH = f'output/moon_topo_gravity_Lshp{lmax_shp}_nmax{max_nmax}.nc'
xr_SH = xr.open_dataset(nc_SH)
gn_SH = xr_SH['gx'].data[::max_nmax, ::max_nmax]
ge_SH = xr_SH['gy'].data[::max_nmax, ::max_nmax]
gd_SH = xr_SH['gz'].data[::max_nmax, ::max_nmax]
#### * PH
out_res = 15.0/60.0 # degree
nc_PH = (f"output/moon_topo_gravity_in{int(in_res*60)}arcmin"
         f"_out{int(out_res*60)}arcmin.nc")
xr_PH = xr.open_dataset(nc_PH)
gn_PH = xr_PH['gN'].data
ge_PH = xr_PH['gE'].data
gd_PH = xr_PH['gD'].data
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
# # ! Top-N largest |gD| errors (PH - SH) with >1000 km separation
n_largest = 10
min_dist = 1000.0
R_moon = 1737.4  # km
abs_err = np.abs(d_gd).flatten()
idx_sorted = np.argsort(-abs_err)
lats_1d = xr_PH['latitude'].data
lons_1d = xr_PH['longitude'].data
nlat, nlon = d_gd.shape
pk_flat_idx = []
for idx in idx_sorted:
    if len(pk_flat_idx) >= n_largest:
        break
    i, j = np.unravel_index(idx, (nlat, nlon))
    lat_new, lon_new = np.radians([lats_1d[i], lons_1d[j]])
    keep = True
    for idx_sel in pk_flat_idx:
        ii, jj = np.unravel_index(idx_sel, (nlat, nlon))
        lat_old, lon_old = np.radians([lats_1d[ii], lons_1d[jj]])
        dlat = lat_new - lat_old
        dlon = lon_new - lon_old
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat_new)*np.cos(lat_old)*np.sin(dlon/2)**2
        dist_km = R_moon * 2 * np.arcsin(np.sqrt(a))
        if dist_km < min_dist:
            keep = False
            break
    if keep:
        pk_flat_idx.append(idx)
lat_indices, lon_indices = np.unravel_index(pk_flat_idx, (nlat, nlon))
print("=" * 100)
print(f"Top {n_largest} largest |gD| errors (PH - SH), >{min_dist} km apart")
print("Topography in m, Gravity vector components in mGal")
print("=" * 100)
print(f"{'Lat (°)':>8} | {'Lon (°)':>8} | {'Topo':>6} | "
      f"{'gN_SH':>9} | {'err_gN':>9} | "
      f"{'gE_SH':>9} | {'err_gE':>9} | "
      f"{'gD_SH':>9} | {'err_gD':>9}")
print("-" * 100)
records = []
for idx in range(len(pk_flat_idx)):
    i, j = lat_indices[idx], lon_indices[idx]
    topo_m = arr_topo[i, j] * 1e3
    gn_sh, ge_sh, gd_sh = gn_SH[i, j], ge_SH[i, j], gd_SH[i, j]
    gn_ph, ge_ph, gd_ph = gn_PH[i, j], ge_PH[i, j], gd_PH[i, j]
    err_gn, err_ge, err_gd = gn_ph - gn_sh, ge_ph - ge_sh, gd_ph - gd_sh
    print(f"{lats_1d[i]:8.3f} | {lons_1d[j]:8.3f} | {topo_m:6.1f} | "
          f"{gn_sh:9.4f} | {err_gn:9.4f} | "
          f"{ge_sh:9.4f} | {err_ge:9.4f} | "
          f"{gd_sh:9.4f} | {err_gd:9.4f}")
    records.append({"Lat": lats_1d[i], "Lon": lons_1d[j], "Topo_m": topo_m,
                    "gN_SH": gn_sh, "err_gN": err_gn,
                    "gE_SH": ge_sh, "err_gE": err_ge,
                    "gD_SH": gd_sh, "err_gD": err_gd})
print("=" * 100)
df_errors = pd.DataFrame(records)
csv_file_err = "Moon_gNED_errors.csv"
df_errors.to_csv(csv_file_err, index=False, float_format="%.4f")
print(f"\nTable data saved to: {csv_file_err}")
# %% 
# # ! Mapping: SH vs PH
err_txt = ['A','B','C','D','E','F','G','H','I','J']
xr_gd_SH = pysh.SHGrid.from_array(gd_SH).to_xarray()
xr_d_gd = pysh.SHGrid.from_array(d_gd).to_xarray()
fig = pygmt.Figure()
with fig.subplot(nrows=2, ncols=1, figsize=('14c', '16.5c'), margins="0.5c"):
    with fig.set_panel(panel=0): 
        fig.grdimage(grid=xr_gd_SH, projection="W-90/14c", 
                     cmap="haxby", frame="g30")
        fig.plot(x=lons_1d[lon_indices], y=lats_1d[lat_indices], 
                 projection="W-90/14c", transparency=25,
                 style="t0.25c", fill="white", pen="0.25p,black")
        fig.text(x=lons_1d[lon_indices], y=lats_1d[lat_indices], text=err_txt, 
                 projection="W-90/14c", justify="BL", 
                 offset="0.06c/0.0c", font="7p,Helvetica-Bold,black")
        fig.colorbar(position="JBC+o0/0.15i+w10c/0.3h", 
                     frame=["a200f100", "x+l@[ g_z^{sh} @[", "y+lmGal"])    
    with fig.set_panel(panel=1): 
        fig.grdimage(grid=xr_d_gd, projection="W-90/14c", 
                     cmap="haxby", frame="g30")
        xlabel = r"@[ g_z^{ph}-g_z^{sh}: \Delta \lambda = \Delta \theta = 15^{\prime}@["
        fig.colorbar(position="JBC+o0/0.15i+w10c/0.3h", 
                     frame=["a5f5", f"x+l{xlabel}", "y+lmGal"])    
# fig.savefig('Moon_TopoGz_PH_vs_SH_1.png', dpi=400)
fig.show(width=800)
# %% 
# # ! End time
print("=" * 80)
print(f"End time: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
print("=" * 80)