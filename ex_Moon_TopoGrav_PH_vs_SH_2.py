#########################################################################
# # ! Moon Topographic-Gravitational-Potential Computation: PH vs SH
#########################################################################
# %% 
# # ! Setup
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
res_topo = 15.0/60.0 # degree
nc_topo = f'output/moon_topo_Lshp{lmax_shp}_{int(60*res_topo)}arcmin.nc'
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
in_res_1 = 15.0/60.0 # degree
in_res_2 = 6.0/60.0 # degree
nc_PH_1 = (f"output/moon_topo_gravity_in{int(in_res_1*60)}arcmin"
           f"_out{int(out_res*60)}arcmin.nc")
nc_PH_2 = (f"output/moon_topo_gravity_in{int(in_res_2*60)}arcmin"
           f"_out{int(out_res*60)}arcmin.nc")
xr_PH_1 = xr.open_dataset(nc_PH_1)
xr_PH_2 = xr.open_dataset(nc_PH_2)
gn_PH_1 = xr_PH_1['gN'].data
ge_PH_1 = xr_PH_1['gE'].data
gd_PH_1 = xr_PH_1['gD'].data
gn_PH_2 = xr_PH_2['gN'].data
ge_PH_2 = xr_PH_2['gE'].data
gd_PH_2 = xr_PH_2['gD'].data
#### * Diff = PH - SH
d_gn_1, d_ge_1, d_gd_1 = gn_PH_1 - gn_SH, ge_PH_1 - ge_SH, gd_PH_1 - gd_SH
d_gn_2, d_ge_2, d_gd_2 = gn_PH_2 - gn_SH, ge_PH_2 - ge_SH, gd_PH_2 - gd_SH
d_gn_1[[0, -1], :] = 0.0
d_ge_1[[0, -1], :] = 0.0
d_gd_1[[0, -1], :] = 0.0
d_gn_2[[0, -1], :] = 0.0
d_ge_2[[0, -1], :] = 0.0
d_gd_2[[0, -1], :] = 0.0
#### * Summary statistics
g_SH = {'N': gn_SH, 'E': ge_SH, 'D': gd_SH}
g_PH_1 = {'N': gn_PH_1, 'E': ge_PH_1, 'D': gd_PH_1}
g_PH_2 = {'N': gn_PH_2, 'E': ge_PH_2, 'D': gd_PH_2}
d_g_1 = {'N': d_gn_1, 'E': d_ge_1, 'D': d_gd_1}
d_g_2 = {'N': d_gn_2, 'E': d_ge_2, 'D': d_gd_2}
print('-' * 80)
print('Gravity Vector components in mGal')
print(f"{'Type':18s} | {'Comp':4s} | {'min':>9s} | {'max':>9s} |"
      f" {'mean':>9s} | {'std':>9s}")
print('-' * 80)
datasets = {'SH': g_SH, 'PH (15 arcmin)': g_PH_1, 'PH (6 arcmin)': g_PH_2, 
            'Diff (15 arcmin)': d_g_1, 'Diff (6 arcmin)': d_g_2}
for typ in ['SH', 'PH (15 arcmin)', 'PH (6 arcmin)', 
            'Diff (15 arcmin)', 'Diff (6 arcmin)']:
    for comp in ['N', 'E', 'D']:
        data = datasets[typ][comp]
        print(f"{typ:18s} | {f'g{comp}':4s} |"
              f" {data.min():9.4f} | {data.max():9.4f} |"
              f" {data.mean():9.4f} | {data.std():9.4f}")
    print('=' * 80)
# %% 
# # ! Load Top-N largest |gD| errors
df_errors = pd.read_csv('output/Moon_gNED_errors.csv')
err_lat = df_errors['Lat'].values
err_lon = df_errors['Lon'].values
# %% 
# # ! Mapping: SH vs PH
err_txt = ['A','B','C','D','E','F','G','H','I','J']
xr_Topo = pysh.SHGrid.from_array(arr_topo).to_xarray()
xr_gd_SH = pysh.SHGrid.from_array(gd_SH).to_xarray()
xr_d_gd_1 = pysh.SHGrid.from_array(d_gd_1).to_xarray()
xr_d_gd_2 = pysh.SHGrid.from_array(d_gd_2).to_xarray()
fig = pygmt.Figure()
with fig.subplot(nrows=2, ncols=2, figsize=('14c', '8.8c'), margins="0.5c"):
    with fig.set_panel(panel=0): 
        fig.grdimage(grid=xr_Topo, projection="W-90/7c", cmap="haxby", frame="g30")
        fig.plot(x=err_lon, y=err_lat, 
                 projection="W-90/7c", transparency=25,
                 style="t0.15c", fill="white", pen="0.25p,black")
        fig.text(x=err_lon, y=err_lat, text=err_txt, 
                 projection="W-90/7c", justify="BL", 
                 offset="0.05c/0.0c", font="5p,Helvetica-Bold,black")
        fig.colorbar(position="JBC+o0.3/0.2i+w5c/0.3h", 
                     frame=["a2f1", "x+lTopography", "y+lkm"])    
    with fig.set_panel(panel=1): 
        fig.grdimage(grid=xr_gd_SH, projection="W-90/7c", cmap="haxby", frame="g30")
        fig.colorbar(position="JBC+o0.3/0.2i+w5c/0.3h", 
                     frame=["a200f100", "x+l@[ g_z^{sh} @[", "y+lmGal"])    
    with fig.set_panel(panel=2): 
        fig.grdimage(grid=xr_d_gd_1, projection="W-90/7c", cmap="haxby", frame="g30")
        xlabel = r"@[ g_z^{ph}-g_z^{sh}: \Delta \lambda = \Delta \theta = 15^{\prime}@["
        fig.colorbar(position="JBC+o0.3/0.2i+w5c/0.3h", 
                     frame=["a5f5", f"x+l{xlabel}", "y+lmGal"])    
    with fig.set_panel(panel=3): 
        fig.grdimage(grid=xr_d_gd_2, projection="W-90/7c", cmap="haxby", frame="g30")
        xlabel = r"@[ g_z^{ph}-g_z^{sh}: \Delta \lambda = \Delta \theta = 6^{\prime}@["
        fig.colorbar(position="JBC+o0.3/0.2i+w5c/0.3h", 
                     frame=["a1f0.5", f"x+l{xlabel}", "y+lmGal"])    
fig.savefig('Moon_TopoGz_PH_vs_SH_2.png', dpi=400)
# fig.show(width=800)
# %% 
# # ! End time
print("=" * 80)
print(f"End time: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
print("=" * 80)