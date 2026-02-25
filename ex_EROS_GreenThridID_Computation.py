# %%
# =============================================================================
# 1. IMPORTS AND CONFIGURATION
# =============================================================================
import numpy as np
import pyvista as pv
from scipy.io import loadmat
from gravity_forward_numba import VecWerSch_numba, green_third_identity_potential_numba

# Parameters
radii = [18, 20, 25, 30]  # km
icosphere_level = 4        # ~2562 points per sphere
rho = 2670.0               # kg/m³

print("Imports and config loaded.")

# %%
# =============================================================================
# 2. LOAD EROS GEOMETRY
# =============================================================================
print("=== Loading EROS Geometry ===")
eros_mat = loadmat('EROS.mat')
eros_vf = eros_mat['eros856_1708']  # shape: (nVert + nFace, 3)

nF = 1708
nVpF = eros_vf.shape[0]
nV = nVpF - nF

Verts = eros_vf[:nV, :].astype(np.float64)
Faces = eros_vf[nV:, :].astype(np.int64)

# Build PyVista mesh
pv_faces = np.column_stack([np.full(nF, 3, dtype=np.int32), Faces])
eros_ori = pv.PolyData(Verts, pv_faces)
eros_sub = eros_ori.subdivide(3, subfilter='linear')   # 1 level

# Compute face properties
print("Computing face centers, normals, areas...")
face_centers = eros_sub.cell_centers().points          # (1708, 3) km
face_normals = eros_sub.face_normals                   # (1708, 3)
face_areas = eros_sub.compute_cell_sizes()["Area"]     # (1708,) km^2

# %%
# =============================================================================
# 3. COMPUTE REFERENCE FIELD ON FACE CENTERS
# =============================================================================
print("Computing gravity field on face centers...")
V_face, gx_face, gy_face, gz_face, *_ = VecWerSch_numba(face_centers, Verts, Faces, rho)
g_face = np.column_stack((gx_face, gy_face, gz_face))  # (1708, 3) in mGal

# %%
# =============================================================================
# 4. GENERATE EVALUATION POINTS USING ICOSPHERE (LEVEL 4)
# =============================================================================
print("Generating evaluation points using PyVista Icosphere (level=4)...")
all_eval_points = []
radius_per_point = []

for r in radii:
    # Create unit icosphere
    sphere_unit = pv.Icosphere(radius=1.0, nsub=icosphere_level)
    pts_unit = sphere_unit.points  # (N, 3), on unit sphere
    
    # Scale to actual radius (in km)
    pts_actual = pts_unit * r  # now in km
    
    all_eval_points.append(pts_actual)
    radius_per_point.extend([r] * pts_actual.shape[0])

eval_points = np.vstack(all_eval_points)  # (M, 3) in km
radius_per_point = np.array(radius_per_point)
print(f"Total evaluation points: {eval_points.shape[0]} (≈{len(sphere_unit.points)} per sphere)")

# Store unit-sphere points for plotting later
unit_sphere_points = sphere_unit.points.copy()  # (N, 3), same for all spheres

# %%
# =============================================================================
# 5. COMPUTE EXACT AND GREEN'S POTENTIALS
# =============================================================================
print("Computing exact potential (polyhedral forward model)...")
V_exact, *_ = VecWerSch_numba(eval_points, Verts, Faces, rho)

print("Computing Green's identity potential...")
V_green = green_third_identity_potential_numba(
    face_centers, face_normals, face_areas, V_face, g_face, eval_points
)

# %%
# =============================================================================
# 6. ERROR ANALYSIS AND TABLED OUTPUT
# =============================================================================
diff = V_green - V_exact
abs_err = np.abs(diff)
rel_err = abs_err / np.abs(V_exact)

print("\n" + "="*120)
print("VALIDATION RESULTS: Green's Third Identity vs Polyhedral Forward Model")
print("="*120)
print(f"{'Radius':>8} | {'Points':>7} | {'V_exact [m²/s²]':>36} | {'Abs Error [m²/s²]':>36} | {'Rel Error':>12}")
print(f"{'(km)':>8} | {'':>7} | {'min / mean / max / RMS':>36} | {'min / mean / max / RMS':>36} | {'mean / RMS':>12}")
print("-"*120)

plot_data = []  # will store (r, diff_values)

for r in radii:
    mask = (radius_per_point == r)
    V_ref = V_exact[mask]
    diff_r = diff[mask]
    abs_err_r = abs_err[mask]
    rel_err_r = rel_err[mask]
    
    # Reference stats
    V_min = np.min(V_ref)
    V_max = np.max(V_ref)
    V_mean = np.mean(V_ref)
    V_rms = np.sqrt(np.mean(V_ref**2))
    
    # Error stats
    abs_min = np.min(abs_err_r)
    abs_max = np.max(abs_err_r)
    abs_mean = np.mean(abs_err_r)
    abs_rms = np.sqrt(np.mean(diff_r**2))
    
    rel_mean = np.mean(rel_err_r)
    rel_rms = np.sqrt(np.mean((diff_r / V_ref)**2))
    
    print(
        f"{r:8.1f} | {mask.sum():7d} | "
        f"{V_min:8.3e} / {V_mean:8.3e} / {V_max:8.3e} / {V_rms:8.3e} | "
        f"{abs_min:8.3e} / {abs_mean:8.3e} / {abs_max:8.3e} / {abs_rms:8.3e} | "
        f"{rel_mean:8.3e} / {rel_rms:8.3e}"
    )
    
    plot_data.append((r, diff_r))

print("-"*120)

# %%
# =============================================================================
# 7. PLOT DIFFERENCE MAPS ON UNIT SPHERES (WITH INDIVIDUAL COLORBARS)
# =============================================================================
print("\nRendering difference maps on unit spheres...")

# Use the same icosphere mesh for all plots (unit radius)
base_mesh = pv.Icosphere(radius=1.0, nsub=icosphere_level)

p = pv.Plotter(shape=(2, 2), window_size=[800, 800])
p.set_background("white")

for idx, (r, diff_vals) in enumerate(plot_data):
    # Create a copy of the base unit sphere
    mesh_plot = base_mesh.copy(deep=True)
    mesh_plot[f"r={r}"] = diff_vals
    
    row = idx // 2
    col = idx % 2
    p.subplot(row, col)
    
    p.add_mesh(
        mesh_plot,
        scalars=f"r={r}",
        cmap="coolwarm",
        show_edges=False,
        scalar_bar_args={
            "title": "ΔV [m²/s²]",
            "fmt": "%.1e",
            "vertical": True,
            "position_x": 0.82 + col * 0.18,
            "position_y": 0.05,
            "width": 0.05,
            "height": 0.9,
        }
    )
    p.add_text(f"r = {r} km", font_size=12, color="black")
    p.show_axes()

p.link_views()
p.show()

print("\n✅ Validation complete.")