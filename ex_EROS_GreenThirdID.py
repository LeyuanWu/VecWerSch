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

# >>> CRITICAL: Convert from MATLAB 1-based to Python 0-based indexing <<<
if Faces.min() == 1:
    print("Converting face indices from 1-based (MATLAB) to 0-based (Python)")
    Faces = Faces - 1

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
print(f"Subdivision level 1: {area_sub1:.10f} (Δ = {area_sub1 - area_ori:+.2e})")
print(f"Subdivision level 2: {area_sub2:.10f} (Δ = {area_sub2 - area_ori:+.2e})")

# Optional: relative changes
rel_change1 = (area_sub1 - area_ori) / area_ori
rel_change2 = (area_sub2 - area_ori) / area_ori
print(f"Relative change L1 : {rel_change1:+.2e}")
print(f"Relative change L2 : {rel_change2:+.2e}")

# %%
# Visualization (only shows level-1 subdivision points)
n_ori = eros_ori.n_points
ori_pts = eros_sub1.points[:n_ori]
new_pts = eros_sub1.points[n_ori:]

ori_cloud = pv.PolyData(ori_pts)
new_cloud = pv.PolyData(new_pts)

pl = pv.Plotter(image_scale=3)
pl.set_background('white')

pl.add_mesh(
    eros_ori,
    color='lightgray',
    show_edges=True,
    edge_color='black',
    line_width=1,
)

pl.add_mesh(
    ori_cloud,
    color='red',
    point_size=16,
    render_points_as_spheres=True,
    label='Original Vertices'
)

pl.add_mesh(
    new_cloud,
    color='blue',
    point_size=12,
    render_points_as_spheres=True,
    label='New (Mid-Edge) Vertices'
)

pl.add_legend()
pl.camera.zoom(1.25)

pl.show()
pl.screenshot("EROS_LinearRefinement.png");