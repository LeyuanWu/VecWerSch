# %%
# ! # Setup
import numpy as np;
import pyvista as pv;
from gravity_forward import *;
# %%
# ! # Rectangular prism
hx = 10. / 2.; hy = 10. / 2.; hz = 10. / 2.;
Verts = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz]], dtype=np.float64);
Faces = np.array([
    [0, 2, 1], [0, 3, 2],  # bottom
    [4, 5, 6], [4, 6, 7],  # top
    [3, 6, 2], [3, 7, 6],  # front
    [0, 1, 5], [0, 5, 4],  # back
    [0, 4, 7], [0, 7, 3],  # left
    [1, 2, 6], [1, 6, 5]   # right
], dtype=np.int32);
Edges = Faces2Edges(Faces);

faces_flat = np.column_stack([np.full(len(Faces), 3), Faces]).flatten()

mesh = pv.PolyData(Verts, faces=faces_flat)

plotter = pv.Plotter()
plotter.add_mesh(mesh, color='#cccc00', line_width=2,
                 show_edges=True, edge_color='blue', opacity=0.5)

vertex_labels = [str(i) for i in range(len(Verts))]
plotter.add_point_labels(
    Verts, 
    vertex_labels,
    italic=True,
    font_size=25,
    text_color='red',
    point_color='red',
    point_size=15,
    render_points_as_spheres=True,
    always_visible=True,
    shadow=True,
    fill_shape=False  # no background box
)

face_centers = []
for face in Faces:
    pts = Verts[face]
    center = pts.mean(axis=0)
    face_centers.append(center)
face_centers = np.array(face_centers)

face_labels = [str(i) for i in range(len(Faces))]
plotter.add_point_labels(
    face_centers,
    face_labels,
    font_size=20,
    text_color='green',
    point_color='green',
    point_size=15,
    always_visible=True,
    shadow=True,
    fill_shape=False  # no background box
)

plotter.camera_position = 'xy'
plotter.add_axes()
plotter.show()