# %%
# Setup
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv
import time
from datetime import datetime
from gravity_forward_numba_v3 import gsphere, WerSch_numba
# %%
# Sphere parameters
xc, yc, zc = 0., 0., 2.
a = 2.0
rho = 1000.0

# Observation heights (z levels) in km
z_levels = [0.001, 0.01, 0.1, 1.0]

# Observation grid (fixed horizontal extent)
xgv = np.linspace(-10., 10., 101)
ygv = np.linspace(-10., 10., 101)
nx_obs = len(xgv)
ny_obs = len(ygv)
N_obs = nx_obs * ny_obs
print(f"\nObservation grid: {nx_obs} x {ny_obs} = {N_obs} points per z level\n")

# Refinement settings
nsub_max = 8
NSUBs = np.arange(nsub_max)

# Geographic resolution rules (used for pv.Sphere)
Lon_Res = 5 * (2 ** NSUBs)
Lat_Res = (2 ** (NSUBs + 1)) + 2

# Containers to store results for each z level
all_err_ico = []
all_err_geo = []
all_time_ico = []
all_time_geo = []

fields = ['V', 'gx', 'gy', 'gz', 'Txx', 'Txy', 'Txz', 'Tyy', 'Tyz', 'Tzz']

# Loop over each observation height (z level)
for z0 in z_levels:
    print(f"\n=== Processing z level: {z0} km ===")
    
    # Build observation grid at height z0
    X2d, Y2d = np.meshgrid(xgv, ygv)
    Z2d = z0 * np.ones_like(X2d)
    P = np.column_stack((X2d.ravel(), Y2d.ravel(), Z2d.ravel()))

    # Analytical solution
    V_ref, gx_ref, gy_ref, gz_ref, Txx_ref, Txy_ref, Txz_ref, Tyy_ref, Tyz_ref, Tzz_ref = \
        gsphere(P[:,0], P[:,1], P[:,2], xc, yc, zc, a, rho)
    ref_arrays = [V_ref, gx_ref, gy_ref, gz_ref, Txx_ref, Txy_ref, Txz_ref, Tyy_ref, Tyz_ref, Tzz_ref]

    # Initialize error and timing lists for this z level
    err_ico = {f: [] for f in fields}
    err_geo = {f: [] for f in fields}
    time_ico = []
    time_geo = []

    for idx, nsub in enumerate(NSUBs):
        print(f"  Processing nsub = {nsub}... [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")

        # --- Icosphere ---
        mesh_ico = pv.Icosphere(radius=a, center=(xc, yc, zc), nsub=nsub)
        verts_ico = mesh_ico.points  # shape (N, 3)
        faces_ico = mesh_ico.regular_faces

        t0 = time.time()
        V_i, gx_i, gy_i, gz_i, Txx_i, Txy_i, Txz_i, Tyy_i, Tyz_i, Tzz_i = \
            WerSch_numba(P, verts_ico, faces_ico, rho)
        t_ico = time.time() - t0
        time_ico.append(t_ico)

        cal_ico = [V_i, gx_i, gy_i, gz_i, Txx_i, Txy_i, Txz_i, Tyy_i, Tyz_i, Tzz_i]
        for field, ref, cal in zip(fields, ref_arrays, cal_ico):
            rms_ref = np.linalg.norm(ref) / np.sqrt(ref.size)
            rms_dif = np.linalg.norm(cal - ref) / np.sqrt(cal.size)
            if rms_ref < 1e-8:
                err = rms_dif
            else:
                err = rms_dif / rms_ref
            err_ico[field].append(err)

        # --- Geographic Grid ---
        theta_res = Lon_Res[idx]
        phi_res   = Lat_Res[idx]
        mesh_geo = pv.Sphere(
            radius=a,
            center=(xc, yc, zc),
            theta_resolution=theta_res,
            phi_resolution=phi_res
        )
        verts_geo = mesh_geo.points
        faces_geo = mesh_geo.regular_faces

        t0 = time.time()
        V_g, gx_g, gy_g, gz_g, Txx_g, Txy_g, Txz_g, Tyy_g, Tyz_g, Tzz_g = \
            WerSch_numba(P, verts_geo, faces_geo, rho)
        t_geo = time.time() - t0
        time_geo.append(t_geo)

        cal_geo = [V_g, gx_g, gy_g, gz_g, Txx_g, Txy_g, Txz_g, Tyy_g, Tyz_g, Tzz_g]
        for field, ref, cal in zip(fields, ref_arrays, cal_geo):
            rms_ref = np.linalg.norm(ref) / np.sqrt(ref.size)
            rms_dif = np.linalg.norm(cal - ref) / np.sqrt(cal.size)
            if rms_ref < 1e-8:
                err = rms_dif
            else:
                err = rms_dif / rms_ref
            err_geo[field].append(err)

    # Store results for this z level
    all_err_ico.append(err_ico)
    all_err_geo.append(err_geo)
    all_time_ico.append(time_ico)
    all_time_geo.append(time_geo)
# %%
# Generate DataFrames and print tables
for i, z0 in enumerate(z_levels):
    df_ico = pd.DataFrame(all_err_ico[i], index=NSUBs)
    df_ico['Time (s)'] = all_time_ico[i]

    df_geo = pd.DataFrame(all_err_geo[i], index=NSUBs)
    df_geo['Time (s)'] = all_time_geo[i]

    print("\n" + "="*80)
    print(f"Z Level = {z0:g} km — Icosphere vs Analytical Sphere "
          f"({nx_obs} x {ny_obs} = {N_obs} obs pts)")
    print("="*80)
    print(df_ico.to_string(float_format="{:.2e}".format))

    print("\n" + "="*80)
    print(f"Z Level = {z0:g} km — Geographic Grid vs Analytical Sphere "
          f"({nx_obs} x {ny_obs} = {N_obs} obs pts)")
    print("="*80)
    print(df_geo.to_string(float_format="{:.2e}".format))
# %%
# Plot representative fields for all z levels
rep_fields = ['V', 'gz', 'Txx', 'Tzz']
latex_labels = {'V': r'$V$', 'gz': r'$g_z$', 'Txx': r'$T_{xx}$', 'Tzz': r'$T_{zz}$'}
colors = {'V': 'tab:red', 'gz': 'tab:blue', 'Txx': 'tab:orange', 'Tzz': 'tab:green'}

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten() 

N_faces = 20 * (4 ** NSUBs)
for i, z0 in enumerate(z_levels):
    ax = axs[i]

    # Plot each field
    for field in rep_fields:
        label_latex = latex_labels[field]
        color = colors[field]

        # Icosphere
        ax.loglog(
            N_faces, all_err_ico[i][field],
            linestyle='-', marker='o', fillstyle='none',
            color=color, linewidth=1, markersize=6,
            label=f'Icosphere - {label_latex}'
        )
        # Geographic Grid
        ax.loglog(
            N_faces, all_err_geo[i][field],
            linestyle='--', marker='s', fillstyle='none',
            color=color, linewidth=2, markersize=6,
            label=f'Geosphere - {label_latex}'
        )

    ax.set_title(f'{z0:g} km above surface', fontsize=13, pad=10)
    ax.set_xlabel('Number of faces', fontsize=11)
    ax.set_ylabel('Relative $L_2$ error', fontsize=11)
    ax.grid(True, which="both", linestyle=':', linewidth=0.7, alpha=0.8)
    ax.legend(fontsize=9, loc='lower left', 
              frameon=True, fancybox=True, shadow=False, ncol=2)

plt.tight_layout(pad=2.5)
plt.savefig(f"IcoGeoSphere_Zs_nsub{nsub_max-1}.png", dpi=300, bbox_inches='tight')
# plt.show()