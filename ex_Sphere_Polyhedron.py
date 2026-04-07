##################################################################
# # ! Two methods to approximate a sphere with a polyhedron:
# (1) Recursive subdivision of an icosahedron
# (2) Triangulated regular geographic grid
##################################################################
# %%
# # ! Setup
import pyvista as pv
# pv.set_jupyter_backend('static')
import numpy as np
from gravity_forward_numba_v3 import spherical_edge_length_range
# %%
# # ! Recursive subdivision of an icosahedron

# True volume and surface area of unit sphere
VOL_TRUE = 4 * np.pi / 3
AREA_TRUE = 4 * np.pi

nsub_max = 11
NSUBs = np.arange(nsub_max)

# ------------------------------------------------------------------
# (1) Recursive subdivision of an icosahedron
# ------------------------------------------------------------------
print("\n" + "="*68)
print("Recursive subdivision of an icosahedron")
print(f"{'N':>4} {'Nv':>8} {'Nf':>8} {'Vol_err':>10} {'Area_err':>10} {'Min_psi':>10} {'Max_psi':>10}")
print("-" * 68)

for nsub in NSUBs:
    mesh = pv.Icosphere(radius=1.0, nsub=nsub)
    verts = mesh.points
    faces = mesh.regular_faces

    nv = verts.shape[0]
    nf = faces.shape[0]
    vol_err = 1.0 - mesh.volume / VOL_TRUE
    area_err = 1.0 - mesh.area / AREA_TRUE

    min_psi, max_psi = spherical_edge_length_range(verts, faces)
    min_psi_arcmin = 60.0 * np.rad2deg(min_psi)
    max_psi_arcmin = 60.0 * np.rad2deg(max_psi)

    print(f"{nsub:>4} {nv:>8} {nf:>8} "
          f"{vol_err:>10.2e} {area_err:>10.2e} "
          f"{min_psi_arcmin:>10.3f} {max_psi_arcmin:>10.3f}")

# %%
# # ! Triangulated regular geographic grid

# ------------------------------------------------------------------
# (2) Triangulated regular geographic grid
# ------------------------------------------------------------------
print("\n" + "="*68)
print("Triangulated Regular Geographic Grid")
print(f"{'N':>4} {'Nv':>8} {'Nf':>8} {'Vol_err':>10} {'Area_err':>10} {'Min_psi':>10} {'Max_psi':>10}")
print("-" * 68)

Lon_Res = 5 * (2 ** NSUBs)
Lat_Res = (2 ** (NSUBs + 1)) + 2

for i, nsub in enumerate(NSUBs):
    theta_res = Lon_Res[i]
    phi_res   = Lat_Res[i]
    mesh = pv.Sphere(radius=1.0, theta_resolution=theta_res, phi_resolution=phi_res)
    verts = mesh.points
    faces = mesh.regular_faces

    nv = verts.shape[0]
    nf = faces.shape[0]
    vol_err = 1.0 - mesh.volume / VOL_TRUE
    area_err = 1.0 - mesh.area / AREA_TRUE

    min_psi, max_psi = spherical_edge_length_range(verts, faces)
    min_psi_arcmin = 60.0 * np.rad2deg(min_psi)
    max_psi_arcmin = 60.0 * np.rad2deg(max_psi)

    print(f"{nsub:>4} {nv:>8} {nf:>8} "
          f"{vol_err:>10.2e} {area_err:>10.2e} "
          f"{min_psi_arcmin:>10.3f} {max_psi_arcmin:>10.3f}")

# %%
# # ! Plot

pl = pv.Plotter(shape=(2, 3), image_scale=3)
pickNs = [2, 3, 4]

# Top row: Icosphere
for col, nsub in enumerate(pickNs):
    mesh = pv.Icosphere(radius=1.0, nsub=nsub)
    mesh = mesh.compute_cell_sizes()
    areas = mesh['Area']
    mesh[f'Area (%) nsub={nsub}'] = 100 * areas / areas.sum()
    pl.subplot(0, col)
    pl.add_mesh(mesh, scalars=f'Area (%) nsub={nsub}', cmap='viridis',
                scalar_bar_args=dict(title_font_size=14, label_font_size=12,
                                     n_labels=3, position_y=0.05, fmt='%.3f'))
    pl.camera.zoom(1.25)

# Bottom row: Geographic grid
for col, nsub in enumerate(pickNs):
    mesh = pv.Sphere(radius=1.0,
                     theta_resolution=Lon_Res[nsub],
                     phi_resolution=Lat_Res[nsub])
    mesh = mesh.compute_cell_sizes()
    areas = mesh['Area']
    mesh[f'Area (%) level={nsub}'] = 100 * areas / areas.sum()
    pl.subplot(1, col)
    pl.add_mesh(mesh, scalars=f'Area (%) level={nsub}', cmap='viridis',
                scalar_bar_args=dict(title_font_size=14, label_font_size=12,
                                     n_labels=3, position_y=0.05, fmt='%.3f'))
    pl.camera.zoom(1.25)

pl.show()
pl.screenshot("icosphere_geosphere.png");