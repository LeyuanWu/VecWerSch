# %%
import numpy as np
import pyvista as pv


mesh = pv.Sphere(
    theta_resolution=10,
    phi_resolution=5
)
old_pts = mesh.points
new_pts = old_pts
new_pts[1,2] = new_pts[1,2] - 0.5
mesh.points = new_pts
point_labels = [str(i) for i in range(mesh.n_points)]

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True)
pl.add_point_labels(
    mesh.points,
    point_labels,
    font_size=20,
    fill_shape=False,
    always_visible=True,
    bold=False,
    shadow=False,
)
pl.view_isometric()
pl.show()