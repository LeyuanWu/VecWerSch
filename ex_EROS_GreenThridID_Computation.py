# validate_green_third_identity.py
import numpy as np
import pyvista as pv
from scipy.io import loadmat
from gravity_forward_numba import VecWerSch_numba  # assumed available
import time

# ----------------------------
# 1. Load EROS high-res mesh
# ----------------------------
print("=== Loading High-Res EROS Mesh ===")
eros_mat = loadmat('EROS.mat')
# Use the same high-res mesh as in ex_EROS_2DPlane.py
eros_vf = eros_mat['eros11272_22540']  # shape: (nVert + nFace, 3)
nF = 22540
nVpF = eros_vf.shape[0]
nV = nVpF - nF

Verts = eros_vf[:nV, :].astype(np.float64)      # (nV, 3)
Faces = eros_vf[nV:, :].astype(np.int64) - 1    # Convert to 0-based

# Create PyVista PolyData
pv_faces = np.hstack((np.full((nF, 1), 3, dtype=np.int32), Faces))
mesh = pv.PolyData(Verts, pv_faces)

# ----------------------------
# 2. Compute face centers, normals, areas
# ----------------------------
print("Computing face properties...")
face_centers = mesh.cell_centers().points          # (nF, 3)
face_normals = mesh.face_normals                   # (nF, 3), already normalized
face_areas = mesh.compute_cell_sizes()["Area"]     # (nF,)

# ----------------------------
# 3. Evaluate V and g on face centers
# ----------------------------
print("Computing gravity on face centers...")
rho = 2670.0  # kg/m³
P_face = face_centers  # in km → convert to meters internally in VecWerSch_numba?
# ASSUMPTION: VecWerSch_numba expects input in **km**, outputs in standard units
# If not, adjust accordingly. Here we assume consistency with ex_EROS_2DPlane.py.

t0 = time.time()
V_face, gx_face, gy_face, gz_face, *_ = VecWerSch_numba(P_face, Verts, Faces, rho)
tc = time.time() - t0
print(f"Forward computation time: {tc:.2f} sec")

g_face = np.column_stack((gx_face, gy_face, gz_face))  # (nF, 3) in mGal → convert to m/s²?
# IMPORTANT: Unit consistency!
# In ex_EROS_2DPlane.py, gx, gy, gz are in **mGal** (1 mGal = 1e-5 m/s²)
g_face_SI = g_face * 1e-5  # Convert to m/s²

# But note: potential V is in m²/s² — consistent.

# However, geometry is in **km**, so we must convert to **meters** for physics!
# Let's be explicit:
scale_km_to_m = 1000.0
face_centers_m = face_centers * scale_km_to_m
face_normals = face_normals  # unitless
face_areas_m2 = face_areas * (scale_km_to_m ** 2)  # km² → m²

# Also convert V and g consistently:
# V is already in m²/s² (fine)
# g_face_SI is in m/s² (fine)

# ----------------------------
# 4. Define evaluation points (on spheres)
# ----------------------------
radii_km = [18, 20, 25, 30]
# For simplicity, use points along +X axis (you can extend to full sphere later)
eval_points_km = np.array([[r, 0, 0] for r in radii_km])  # (4, 3)
eval_points_m = eval_points_km * scale_km_to_m

# Compute "exact" potential at these points using polyhedral formula
print("Computing exact potential at evaluation points...")
V_exact, *_ = VecWerSch_numba(eval_points_km, Verts, Faces, rho)  # still in m²/s²

# ----------------------------
# 5. Numerical surface integral (Green's third identity)
# ----------------------------
print("\n=== Evaluating Green's Third Identity ===")
G = 6.67430e-11  # m³/kg/s² — though cancels out if forward model is consistent

# Precompute constants
inv_4pi = 1.0 / (4 * np.pi)

V_green = []

for i, r0 in enumerate(eval_points_m):
    r0_km = eval_points_km[i]
    print(f"\nEvaluating at r = {np.linalg.norm(r0_km):.1f} km")

    # Vector from face center to field point
    R_vec = r0 - face_centers_m          # (nF, 3)
    R_norm = np.linalg.norm(R_vec, axis=1)  # (nF,)
    
    # Avoid division by zero (shouldn't happen since r0 outside)
    assert np.all(R_norm > 0), "Evaluation point too close to surface!"

    # Term 1: V * d(1/R)/dn = V * (R_vec · n) / R^3
    d_invR_dn = np.einsum('ij,ij->i', R_vec, face_normals) / (R_norm ** 3)  # (nF,)
    term1 = V_face * d_invR_dn  # V in m²/s², d_invR_dn in 1/m² → term1 in 1/s²

    # Term 2: (1/R) * (∂V/∂n) = (1/R) * (-g · n) → but Green's has - (1/R)(∂V/∂n)
    # So: - (1/R) * (∂V/∂n) = + (1/R) * (g · n)
    g_dot_n = np.einsum('ij,ij->i', g_face_SI, face_normals)  # (nF,) in m/s²
    term2 = (1.0 / R_norm) * g_dot_n  # in 1/s²

    # Integrand = term1 + term2
    integrand = term1 + term2  # (nF,)

    # Surface integral ≈ sum(integrand * area)
    integral = np.sum(integrand * face_areas_m2)  # (1/s²) * m² = m²/s²

    V_g = inv_4pi * integral
    V_green.append(V_g)

    # Compare
    err = V_g - V_exact[i]
    rel_err = err / np.abs(V_exact[i])
    print(f"  V_exact = {V_exact[i]:.6e} m²/s²")
    print(f"  V_green = {V_g:.6e} m²/s²")
    print(f"  Absolute error = {err:.3e}")
    print(f"  Relative error = {rel_err:.3e}")

V_green = np.array(V_green)

# ----------------------------
# 6. Optional: Full spherical sampling (commented for speed)
# ----------------------------
"""
# To sample full spheres:
from itertools import product
theta_vals = np.linspace(0, np.pi, 10)
phi_vals = np.linspace(0, 2*np.pi, 20)
errors = []
for r in radii_km:
    for theta, phi in product(theta_vals, phi_vals):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        r0_km = np.array([x, y, z])
        # ... repeat integral ...
"""

print("\n✅ Validation complete.")