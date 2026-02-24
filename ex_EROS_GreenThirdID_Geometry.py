import pyvista as pv
import numpy as np
from scipy.io import loadmat

# %%
# Load EROS geometry from MATLAB file
print("=== Loading EROS Geometry ===")
eros_mat = loadmat('EROS.mat')
eros_vf = eros_mat['eros856_1708']  # shape: (nVert + nFace, 3)

nF = 1708
nVpF = eros_vf.shape[0]
nV = nVpF - nF

Verts = eros_vf[:nV, :].astype(np.float64)
Faces = eros_vf[nV:, :].astype(np.int64)

# Bounding box (in km)
X1, X2 = Verts[:, 0].min(), Verts[:, 0].max()
Y1, Y2 = Verts[:, 1].min(), Verts[:, 1].max()
Z1, Z2 = Verts[:, 2].min(), Verts[:, 2].max()

print(f"Number of vertices : {nV}")
print(f"Number of faces    : {nF}")
print(f"Bounding box (km)  :")
print(f"  X ∈ [{X1:8.4f}, {X2:8.4f}]")
print(f"  Y ∈ [{Y1:8.4f}, {Y2:8.4f}]")
print(f"  Z ∈ [{Z1:8.4f}, {Z2:8.4f}]")

# %%
# Linear refinement

pv_faces = np.hstack((np.full((nF, 1), 3, dtype=np.int32), Faces))
eros_ori = pv.PolyData(Verts, pv_faces)

# Perform subdivisions
eros_sub1 = eros_ori.subdivide(1, subfilter='linear')   # 1 level
eros_sub2 = eros_ori.subdivide(2, subfilter='linear')   # 2 levels

# Compute areas
area_ori  = eros_ori.area
area_sub1 = eros_sub1.area
area_sub2 = eros_sub2.area

# Print comparison
print("\n=== Surface Area Comparison (km²) ===")
print(f"Original mesh      : {area_ori:.10f}")
print(f"Subdivision level 1: {area_sub1:.10f} (error = {area_sub1 - area_ori:+.2e})")
print(f"Subdivision level 2: {area_sub2:.10f} (error = {area_sub2 - area_ori:+.2e})")

# Optional: relative changes
rel_change1 = (area_sub1 - area_ori) / area_ori
rel_change2 = (area_sub2 - area_ori) / area_ori
print(f"Relative change L1 : {rel_change1:+.2e}")
print(f"Relative change L2 : {rel_change2:+.2e}")

# %%
pl = pv.Plotter(image_scale=3)
pl.set_background('white')

# Prepare face centers and normals for both meshes
def add_face_visuals(pl, mesh, color_center='red', color_normal='blue'):
    # Compute face centers and normals
    centers = mesh.cell_centers()
    centers['Normals'] = mesh.cell_normals
    
    # Add face centers
    pl.add_mesh(
        centers,
        color=color_center,
        point_size=8,
        render_points_as_spheres=True
    )
    
    # Add normals as arrows using glyph
    arrows = centers.glyph(
        orient='Normals',
        scale=False,  # Use fixed length
        factor=0.75    # Arrow length 
    )
    pl.add_mesh(arrows, color=color_normal)

# Create subplot
pl = pv.Plotter(shape=(2, 1), border=True,
                image_scale=3, window_size=[800, 900])
pl.set_background('white')

# --- Upper subplot: Level 0 ---
pl.subplot(0, 0)
pl.add_mesh(
    eros_ori,
    color='white',
    show_edges=True,
    edge_color='black',
    line_width=0.8
)
add_face_visuals(pl, eros_ori)

# --- Right subplot: Level 1 ---
pl.subplot(1, 0)
pl.add_mesh(
    eros_sub1,
    color='white',
    show_edges=True,
    edge_color='black',
    line_width=0.5
)
add_face_visuals(pl, eros_sub1)

# Final view settings
pl.link_views()  # Optional: synchronize camera between subplots
pl.camera.zoom(2.0)

# Show interactively (optional) and save
pl.show()  # This opens the interactive window
pl.screenshot("EROS_subdivision_L0_vs_L1_with_normals.png");  # Save high-res image