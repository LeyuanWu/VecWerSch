# %%
# # ! Setup
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv
import time
from datetime import datetime
from gravity_forward_numba import *

# %%
# # ! Functions
def fibonacci_sphere_points(N, radius=1.0):
    """Generate N approximately uniformly distributed points on a sphere."""
    i = np.arange(N, dtype=np.float64)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / N)
    theta = np.pi * (1.0 + 5.0**0.5) * (i + 0.5)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack((x, y, z))

# %% 
# # ! Start time
print("=" * 80)
print(f"Start time: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
print("=" * 80)

# %%
# # ! Earth model (km, kg/m³)
r_e = 6371.393      # mean Earth radius [km]
rho_e = 5520.0      # mean Earth density [kg/m³]

# Homogeneous sphere
xc, yc, zc = 0.0, 0.0, 0.0
a = r_e
rho = rho_e

# Observation altitudes (km)
alts = [0.001, 0.1, 1.0, 10.0]
r_obs_list = [r_e + h for h in alts]

# Use Fibonacci sampling instead of lat/lon grid
nobs = 5000  # number of observation points per altitude
print(f"\nUsing Fibonacci sphere sampling with {nobs} points per altitude\n")

# Store observation points and reference solutions
P_list = []
ref_list = []

for r in r_obs_list:
    P = fibonacci_sphere_points(nobs, radius=r)
    P_list.append(P)
    
    x, y, z = P[:,0], P[:,1], P[:,2]
    r_vec = np.sqrt(x**2 + y**2 + z**2)
    lat_rad = np.arcsin(z / r_vec)
    lon_rad = np.arctan2(y, x)
    
    # Analytical solution
    V_ref, gx_ref, gy_ref, gz_ref, Txx_ref, Txy_ref, Txz_ref, Tyy_ref, Tyz_ref, Tzz_ref = \
        gsphere(x, y, z, xc, yc, zc, a, rho)
    
    # Rotate to NED
    gN_ref, gE_ref, gD_ref, TNN_ref, TNE_ref, TND_ref, TEE_ref, TED_ref, TDD_ref = \
        rotate_vec_ten_ecef2ned(
            lon_rad, lat_rad,
            gx_ref, gy_ref, gz_ref,
            Txx_ref, Txy_ref, Txz_ref,
            Tyy_ref, Tyz_ref, Tzz_ref
        )
    
    ref_list.append([V_ref, gN_ref, gE_ref, gD_ref,
                     TNN_ref, TNE_ref, TND_ref, TEE_ref, TED_ref, TDD_ref])

fields = ['V', 'gN', 'gE', 'gD', 'TNN', 'TNE', 'TND', 'TEE', 'TED', 'TDD']

# Mesh refinement
nsub_max = 12
NSUBs = np.arange(nsub_max + 1)
Lon_Res = 5 * (2 ** NSUBs)
Lat_Res = (2 ** (NSUBs + 1)) + 2

# Results containers
err_ico_all = []
err_geo_all = []
time_ico_all = []
time_geo_all = []
ref_vals_all = []

# %%
# # ! Loop over altitudes
for i, (P, refs) in enumerate(zip(P_list, ref_list)):
    h = alts[i]
    print(f"\n=== Altitude: {h:g} km ===")
    
    err_ico = {f: [] for f in fields}
    err_geo = {f: [] for f in fields}
    t_ico = []
    t_geo = []
    
    x, y, z = P[:,0], P[:,1], P[:,2]
    r_obs = np.sqrt(x**2 + y**2 + z**2)
    lat_rad = np.arcsin(z / r_obs)
    lon_rad = np.arctan2(y, x)
    
    for j, nsub in enumerate(NSUBs):
        print(f"  Processing nsub = {nsub}... [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        
        # --- Icosphere ---
        mesh = pv.Icosphere(radius=a, center=(xc, yc, zc), nsub=nsub)
        verts = mesh.points
        faces = mesh.regular_faces
        
        t0 = time.time()
        V, gx, gy, gz, Txx, Txy, Txz, Tyy, Tyz, Tzz = WerSch_numba_v2(P, verts, faces, rho)
        t_ico.append(time.time() - t0)
        
        gN, gE, gD, TNN, TNE, TND, TEE, TED, TDD = rotate_vec_ten_ecef2ned(
            lon_rad, lat_rad, gx, gy, gz, Txx, Txy, Txz, Tyy, Tyz, Tzz)
        cal_ico = [V, gN, gE, gD, TNN, TNE, TND, TEE, TED, TDD]
        
        for f, ref, cal in zip(fields, refs, cal_ico):
            rms_ref = np.linalg.norm(ref) / np.sqrt(ref.size)
            rms_err = np.linalg.norm(cal - ref) / np.sqrt(cal.size)
            err = rms_err if rms_ref < 1e-8 else rms_err / rms_ref
            err_ico[f].append(err)

        # --- Geographic grid ---
        mesh = pv.Sphere(radius=a, center=(xc, yc, zc),
                         theta_resolution=Lon_Res[j],
                         phi_resolution=Lat_Res[j])
        verts = mesh.points
        faces = mesh.regular_faces

        t0 = time.time()
        V, gx, gy, gz, Txx, Txy, Txz, Tyy, Tyz, Tzz = WerSch_numba_v2(P, verts, faces, rho)
        t_geo.append(time.time() - t0)
        
        gN, gE, gD, TNN, TNE, TND, TEE, TED, TDD = rotate_vec_ten_ecef2ned(
            lon_rad, lat_rad, gx, gy, gz, Txx, Txy, Txz, Tyy, Tyz, Tzz)
        cal_geo = [V, gN, gE, gD, TNN, TNE, TND, TEE, TED, TDD]
        
        for f, ref, cal in zip(fields, refs, cal_geo):
            rms_ref = np.linalg.norm(ref) / np.sqrt(ref.size)
            rms_err = np.linalg.norm(cal - ref) / np.sqrt(cal.size)
            err = rms_err if rms_ref < 1e-8 else rms_err / rms_ref
            err_geo[f].append(err)
    
    err_ico_all.append(err_ico)
    err_geo_all.append(err_geo)
    time_ico_all.append(t_ico)
    time_geo_all.append(t_geo)
    ref_vals_all.append({f: np.mean(r) for f, r in zip(fields, refs)})

# %%
# # ! Print results tables
for i, h in enumerate(alts):
    df_ico = pd.DataFrame(err_ico_all[i], index=NSUBs)
    df_ico['Time (s)'] = time_ico_all[i]

    df_geo = pd.DataFrame(err_geo_all[i], index=NSUBs)
    df_geo['Time (s)'] = time_geo_all[i]

    print("\n" + "=" * 80)
    print(f"Altitude = {h:g} km — Icosphere vs Analytical "
          f"({nobs} obs pts)")
    print("=" * 80)
    print(df_ico.to_string(float_format="{:.2e}".format))

    print("\n" + "=" * 80)
    print(f"Altitude = {h:g} km — Geographic Grid vs Analytical "
          f"({nobs} obs pts)")
    print("=" * 80)
    print(df_geo.to_string(float_format="{:.2e}".format))

print("\n" + "=" * 80)
print("Reference Values at Different Altitudes")
print("GP in m²/s², GV in mGal, GGT in Eotvos")
print("=" * 80)
df_refs = pd.DataFrame(ref_vals_all, index=alts)
print(df_refs.to_string(float_format="{:8.2f}".format))

# %%
# # ! Save & Load Data
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# UNCOMMENT THIS BLOCK TO SKIP COMPUTATION AND RELOAD FOR PLOTTING ONLY
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Load data
# data = np.load(f'output/IcoGeoSphere_Global_nsub{nsub_max}_nobs{nobs}.npz', 
#                allow_pickle=True)
# nsub_max = int(data['nsub_max'])
# nobs = int(data['nobs'])
# alts = data['alts']
# NSUBs = data['NSUBs']
# err_ico_all = [dict(item) for item in data['err_ico_all']]
# err_geo_all = [dict(item) for item in data['err_geo_all']]
# time_ico_all = data['time_ico_all']
# time_geo_all = data['time_geo_all']
# ref_vals_all = [dict(item) for item in data['ref_vals_all']]
# fields = data['fields'].tolist()


# Save data
np.savez_compressed(
    f'output/IcoGeoSphere_Global_nsub{nsub_max}_nobs{nobs}.npz',
    nsub_max=nsub_max,
    nobs=nobs,
    alts=alts,
    NSUBs=NSUBs,
    err_ico_all=err_ico_all,
    err_geo_all=err_geo_all,
    time_ico_all=time_ico_all,
    time_geo_all=time_geo_all,
    ref_vals_all=ref_vals_all,
    fields=fields
)

# %%
# # ! Plot representative fields for each altitude
rep_fields = ['V', 'gD', 'TNN', 'TDD']
labels = {'V': r'$V$', 'gD': r'$g_D$', 'TNN': r'$T_{NN}$', 'TDD': r'$T_{DD}$'}
colors = {'V': 'tab:red', 'gD': 'tab:blue', 'TNN': 'tab:orange', 'TDD': 'tab:green'}

fig, axs = plt.subplots(2, 2, figsize=(10, 9))
axs = axs.flatten()

NTs = 20 * (4 ** NSUBs)

for i, h in enumerate(alts):
    ax = axs[i]
    for field in rep_fields:
        c = colors[field]
        lbl = labels[field]
        ax.loglog(NTs, err_ico_all[i][field], '-', marker='o', fillstyle='none',
                  color=c, linewidth=1, markersize=6, label=f'Icosphere - {lbl}')
        ax.loglog(NTs, err_geo_all[i][field], '--', marker='s', fillstyle='none',
                  color=c, linewidth=2, markersize=6, label=f'Geosphere - {lbl}')
    
    # ax.set_title(f'{h:g} km above surface', fontsize=13, pad=10)
    ax.set_xlabel('Number of triangular faces ($N_T$)')
    ax.set_ylabel('Relative $L_2$ error')
    ax.grid(True, which="major", linestyle=':', linewidth=1.0, alpha=1.0)
    ax.legend(fontsize=9, loc='lower left', frameon=True, ncol=2)

[ax.text(-0.15, 1.08, label, transform=ax.transAxes, 
         fontsize=14, fontweight='bold', va='top') 
         for ax, label in zip(axs, ['(a)', '(b)', '(c)', '(d)'])]

plt.tight_layout(pad=2.5)
plt.savefig(f"IcoGeoSphere_Global_nsub{nsub_max}_nobs{nobs}.png", 
            dpi=300, bbox_inches='tight')
# plt.show()

# %% 
# # ! End time
print("=" * 80)
print(f"End time: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
print("=" * 80)