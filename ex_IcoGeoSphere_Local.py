# %%
# Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
import time
from gravity_forward_numba import gsphere, VecWerSch_numba
# %%
# Sphere parameters
xc, yc, zc = 0, 0, 2
a = 1.999
rho = 1000.0

# Observation grid (same as your original)
xgv = np.linspace(-10., 10., 51)
ygv = np.linspace(-10., 10., 51)
z0 = 0.0
X2d, Y2d = np.meshgrid(xgv, ygv)
Z2d = z0 * np.ones(X2d.shape)
P = np.column_stack((X2d.ravel(), Y2d.ravel(), Z2d.ravel()))

# Analytical reference solution
V_ref, gx_ref, gy_ref, gz_ref, Txx_ref, Txy_ref, Txz_ref, Tyy_ref, Tyz_ref, Tzz_ref = \
    gsphere(P[:,0], P[:,1], P[:,2], xc, yc, zc, a, rho)
fields = ['V', 'gx', 'gy', 'gz', 'Txx', 'Txy', 'Txz', 'Tyy', 'Tyz', 'Tzz']
ref_arrays = [V_ref, gx_ref, gy_ref, gz_ref, Txx_ref, Txy_ref, Txz_ref, Tyy_ref, Tyz_ref, Tzz_ref]

# Refinement settings
nsub_max = 8
NSUBs = np.arange(nsub_max)

# Geographic resolution rules
Lon_Res = 5 * (2 ** NSUBs)
Lat_Res = (2 ** (NSUBs + 1)) + 2

# Initialize error dictionaries and time arrays
err_ico = {f: [] for f in fields}
err_geo = {f: [] for f in fields}
time_ico = []
time_geo = []
# %%
# L2 norm errors and time costs for both methods
for idx, nsub in enumerate(NSUBs):
    print(f"Processing nsub = {nsub}...")

    # --- Icosphere ---
    mesh_ico = pv.Icosphere(radius=a, center=(xc, yc, zc), nsub=nsub)
    verts_ico = mesh_ico.points
    faces_ico = mesh_ico.regular_faces

    t0 = time.time()
    V_i, gx_i, gy_i, gz_i, Txx_i, Txy_i, Txz_i, Tyy_i, Tyz_i, Tzz_i = \
        VecWerSch_numba(P, verts_ico, faces_ico, rho)
    t_ico = time.time() - t0
    time_ico.append(t_ico)
    
    cal_ico = [V_i, gx_i, gy_i, gz_i, Txx_i, Txy_i, Txz_i, Tyy_i, Tyz_i, Tzz_i]
    for field, ref, cal in zip(fields, ref_arrays, cal_ico):
        norm_ref = np.linalg.norm(ref)
        err = np.linalg.norm(cal - ref) / norm_ref
        err_ico[field].append(err)

    # --- Geographic Grid ---
    theta_res = Lon_Res[idx]
    phi_res   = Lat_Res[idx]
    mesh_geo = pv.Sphere(radius=a, center=(xc, yc, zc),
                         theta_resolution=theta_res,
                         phi_resolution=phi_res)
    verts_geo = mesh_geo.points
    faces_geo = mesh_geo.regular_faces

    t0 = time.time()
    V_g, gx_g, gy_g, gz_g, Txx_g, Txy_g, Txz_g, Tyy_g, Tyz_g, Tzz_g = \
        VecWerSch_numba(P, verts_geo, faces_geo, rho)
    t_geo = time.time() - t0
    time_geo.append(t_geo)
    
    cal_geo = [V_g, gx_g, gy_g, gz_g, Txx_g, Txy_g, Txz_g, Tyy_g, Tyz_g, Tzz_g]
    for field, ref, cal in zip(fields, ref_arrays, cal_geo):
        norm_ref = np.linalg.norm(ref)
        err = np.linalg.norm(cal - ref) / norm_ref 
        err_geo[field].append(err)
# %%
# Create DataFrames including computation time
df_ico = pd.DataFrame(err_ico, index=NSUBs)
df_ico['Time (s)'] = time_ico

df_geo = pd.DataFrame(err_geo, index=NSUBs)
df_geo['Time (s)'] = time_geo

print("\n" + "="*80)
print("Relative L2 Errors + Computation Time: Icosphere vs Analytical Sphere")
print("GP in m^2/s^2, GV in mGal, GGT in Eotvos")
print("="*80)
print(df_ico.to_string(float_format="{:.2e}".format))

print("\n" + "="*80)
print("Relative L2 Errors + Computation Time: Geographic Grid vs Analytical Sphere")
print("GP in m^2/s^2, GV in mGal, GGT in Eotvos")
print("="*80)
print(df_geo.to_string(float_format="{:.2e}".format))
# %%
# Plot representative components in three subplots
rep_fields = ['V', 'gz', 'Tzz']
latex_labels = {'V': r'$V$ ', 'gz': r'$g_z$ ', 'Tzz': r'$T_{zz}$ '}
colors = {'V': 'tab:red', 'gz': 'tab:blue', 'Tzz': 'tab:green'}

fig, ax = plt.subplots(figsize=(8, 6))

for idx, field in enumerate(rep_fields):
    label_latex = latex_labels[field]
    color = colors[field]

    # Icosphere
    ax.loglog(
        20*4**NSUBs, err_ico[field],
        linestyle='-', marker='o', fillstyle='none',
        color=color, linewidth=1, markersize=6,
        label=f'Icosphere - {label_latex}'
    )
    # Geographic Grid
    ax.loglog(
        20*4**NSUBs, err_geo[field],
        linestyle='--', marker='s', fillstyle='none',
        color=color, linewidth=2, markersize=6,
        label=f'Geosphere - {label_latex}'
    )

ax.set_ylabel(r'Relative  $L_2$ norm error', fontsize=11)
ax.grid(True, which="major")
ax.legend(
    fontsize=10,
    handlelength=4
)
ax.set_xlabel('Number of faces', fontsize=11)

plt.tight_layout()
plt.savefig("IcoGeoSphere_Local.png", dpi=300, bbox_inches='tight')
plt.show()