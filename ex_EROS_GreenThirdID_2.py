##################################################################
# # ! Validation of Green's 3rd Identity using EROS shape model
##################################################################
# %%
# # ! Setup
import numpy as np
import matplotlib.pyplot as plt
import time
import pyvista as pv
pv.set_jupyter_backend('static')
from scipy.io import loadmat
from gravity_forward_numba import *
# %% 
# # ! Parameters
radii = [17.8, 18, 20]              # Evaluation sphere radii (km)
icosph_level = 5                  # 10 * 4^n + 2
sub_lvls = list(range(6))         # Mesh refinement levels
rho = 2670.0                      # Density (kg/m^3)
# %%
# # ! Load EROS shape model from MAT file
eros_mat = loadmat('input/EROS.mat')
eros_vf = eros_mat['eros856_1708']  # First nV rows: vertices; last nF rows: faces
nF = 1708
nVpF = eros_vf.shape[0]
nV = nVpF - nF

Verts = eros_vf[:nV, :].astype(np.float64)
Faces = eros_vf[nV:, :].astype(np.int64)

pv_faces = np.column_stack([np.full(nF, 3, dtype=np.int32), Faces])
eros_base = pv.PolyData(Verts, pv_faces)
# %%
# # ! Generate evaluation points on concentric spheres (one per radius)
eval_pts_Rs = []
for r in radii:
    sph_unit = pv.Icosphere(radius=1.0, nsub=icosph_level)
    pts = sph_unit.points * r
    eval_pts_Rs.append(pts)
# %%
# # ! Compute reference (exact) gravity potential on each evaluation sphere
print("\nComputing exact gravitational potential on evaluation spheres...")
V_exact_Rs = []
t0_total = time.time()

for i, (r, pts) in enumerate(zip(radii, eval_pts_Rs)):
    t0 = time.time()
    V_exact, *_ = WerSch_numba_v2(pts, Verts, Faces, rho)
    t_elapsed = time.time() - t0
    V_exact_Rs.append(V_exact)
    print(f" -> Radius {r} km: {len(pts)} points, time = {t_elapsed:.3f} s")

t_total = time.time() - t0_total
print(f"Total time for exact potential computation: {t_total:.3f} s")
# %%
# # ! Convergence study: loop over mesh refinement levels
converg_summary = []  # Store error metrics per refinement level
for level in sub_lvls:
    print(f"\nProcessing subdivision level: {level}")
    
    # Refine mesh
    if level == 0:
        eros_sub = eros_base.copy()
    else:
        eros_sub = eros_base.subdivide(level, subfilter='linear')
    
    n_faces = eros_sub.n_cells
    print(f" -> Level {level} EROS refined mesh has {n_faces} faces")

    # Extract face geometry
    cfs = eros_sub.cell_centers().points
    nfs = eros_sub.cell_normals
    area_fs = eros_sub.compute_cell_sizes()["Area"]

    # Compute exact field on face centers (for Green's identity)
    t0 = time.time()
    V_face, gx_face, gy_face, gz_face, *_ = WerSch_numba_v1(cfs, Verts, Faces, rho)
    g_face = np.column_stack((gx_face, gy_face, gz_face))
    t_face = time.time() - t0
    print(f" -> Time (Face center field): {t_face:.3f} s")

    # Evaluate Green's third identity potential on each sphere
    V_green_Rs = []
    t_green_total = 0.0
    for pts in eval_pts_Rs:
        t0 = time.time()
        V_green = green_third_identity_potential_numba(
            cfs, nfs, area_fs, V_face, g_face, pts
        )
        t_green_total += time.time() - t0
        V_green_Rs.append(V_green)
    print(f" -> Time (Green's Third ID):  {t_green_total:.3f} s")

    # Compute relative errors
    rel_rms_errors = []
    rel_max_errors = []
    for i in range(len(radii)):
        V_exact = V_exact_Rs[i]
        V_green = V_green_Rs[i]
        diff = V_green - V_exact

        rms_diff = np.sqrt(np.mean(diff**2))
        rms_exact = np.sqrt(np.mean(V_exact**2))
        rel_rms = rms_diff / rms_exact
        rel_max = np.max(np.abs(diff/V_exact))

        rel_rms_errors.append(rel_rms)
        rel_max_errors.append(rel_max)

    converg_summary.append({
        'level': level,
        'n_faces': n_faces,
        'rel_rms': rel_rms_errors,
        'rel_max': rel_max_errors
    })
# %%
# # ! Print convergence table
print("\n" + "="*92)
print("Convergence of Green's Third Identity Approximation")
print("Relative RMS and Maximum Errors vs. Eros Mesh Refinement")
print("="*92)

header1 = f"{'Level':>6} {'Faces':>8}" + "".join([f" {'RMS':>12} {'Max':>12}" for _ in radii])
header2 = " " * 15 + "".join([(" " * 10 + f"{r:.1f} km".center(16)) for r in radii])
print(header1)
print(header2)
print("-"*92)

for lvl_summary in converg_summary:
    level = lvl_summary['level']
    n_faces = lvl_summary['n_faces']
    row = f"{level:6d} {n_faces:8d}"
    for i in range(len(radii)):
        rms = lvl_summary['rel_rms'][i]
        rmx = lvl_summary['rel_max'][i]
        row += f" {rms:12.2e} {rmx:12.2e}"
    print(row)

print("-"*92)
print("Note: All errors are relative (dimensionless).")
# %%
# # ! Plot convergence
n_faces_list = [lvl_summary['n_faces'] for lvl_summary in converg_summary]
rel_rms_matrix = np.array([lvl_summary['rel_rms'] for lvl_summary in converg_summary])
rel_max_matrix = np.array([lvl_summary['rel_max'] for lvl_summary in converg_summary])

plt.figure(figsize=(8, 6), dpi=300)
colors = ['tab:blue', 'tab:orange', 'tab:green']

for i, r in enumerate(radii):
    rms_vals = rel_rms_matrix[:, i]
    max_vals = rel_max_matrix[:, i]
    color = colors[i % len(colors)]
    
    plt.loglog(n_faces_list, rms_vals,
               marker='o', linestyle='-', color=color,
               label=f'$\epsilon_{{rms}}$ ($r$ = {r:.1f} km)', markersize=5)
    plt.loglog(n_faces_list, max_vals,
               marker='s', linestyle='--', color=color,
               label=f'$\epsilon_{{max}}$ ($r$ = {r:.1f} km)', markersize=5)

plt.xlabel('Number of Faces')
plt.ylabel('Relative Error')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('EROS_GreenThirdID_2_1.png', 
            dpi=300, bbox_inches='tight')
plt.show()
# %%
# # ! Visualize pointwise relative error across refinement levels
plot_levels = [0, 1, 3, 5]  # Choose 4 levels for 2x2 grid

r_target = radii[0]
pts_pkr = eval_pts_Rs[0]
V_exact_pkr = V_exact_Rs[0]

target_sphere = pv.Icosphere(radius=r_target, nsub=icosph_level)

pl = pv.Plotter(shape=(2, 2), border=False, image_scale=3)
pl.set_background('white')

for plot_idx, level in enumerate(plot_levels):
    print(f"Preparing error plot for subdivision level {level}...")
    
    # Reconstruct refined mesh
    eros_sub = eros_base.subdivide(level, subfilter='linear')
    
    # Extract face data
    cfs = eros_sub.cell_centers().points
    nfs = eros_sub.face_normals
    area_fs = eros_sub.compute_cell_sizes()["Area"]
    
    # Compute exact field on face centers
    V_face, gx_face, gy_face, gz_face, *_ = WerSch_numba_v1(cfs, Verts, Faces, rho)
    g_face = np.column_stack((gx_face, gy_face, gz_face))
    
    # Evaluate Green's potential
    V_green_pkr = green_third_identity_potential_numba(
        cfs, nfs, area_fs, V_face, g_face, pts_pkr
    )
    
    # Compute pointwise relative error
    rel_error = np.abs(V_green_pkr - V_exact_pkr) / np.abs(V_exact_pkr)
    target_sphere[f'level{level}'] = rel_error
    
    # Plot Eros shape model
    pl.subplot(plot_idx // 2, plot_idx % 2)
    pl.add_mesh(eros_base, color=None, show_edges=True, line_width=2)
    pl.add_mesh(target_sphere.copy(deep=False), scalars=f'level{level}', cmap='turbo',
                style='surface', opacity=0.75, log_scale=True, show_edges=False, show_scalar_bar=False)
    pl.add_scalar_bar(title=f'Relative error: level {level}', 
                      title_font_size=14, label_font_size=12, 
                      n_labels=3, position_x=0.3, fmt='%.1e')
    
    pl.camera_position = [(-80, -40, 40), (0, 0, -2), (0, 0, 1)]
    pl.add_axes(label_size=(0.1/3, 0.1/3))

pl.link_views()
pl.window_size = (800, 800)
pl.show()
