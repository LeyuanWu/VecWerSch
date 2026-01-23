##################################################################
# # ! Two methods to approximate a sphere with a polyhedron:
# (1) Recursive subdivision of an icosahedron
# (2) Triangulated regular geographic grid
##################################################################
# %%
# # ! Setup
import pyvista as pv
import numpy as np
from gravity_forward_numba import *
from gravity_forward_numpy import *
# %%
# # ! Recursive subdivision of an icosahedron
nsub_max = 9;
Nsubs = np.arange(nsub_max);
NVs = np.zeros(nsub_max, dtype=int);
NFs = np.zeros(nsub_max, dtype=int);
Es_vol   = np.zeros(nsub_max);
Es_area  = np.zeros(nsub_max);
MINs_psi = np.zeros(nsub_max);
MAXs_psi = np.zeros(nsub_max);
vol = 4*np.pi/3; area = 4*np.pi;
for i, nsub in enumerate(Nsubs):
    icosph = pv.Icosphere(nsub=nsub);
    Verts = icosph.points;
    Faces = icosph.regular_faces;
    NVs[i] = Verts.shape[0];
    NFs[i] = Faces.shape[0];
    Es_vol[i]  = 1. - icosph.volume/vol;
    Es_area[i] = 1. - icosph.area/area;
    min_psi, max_psi = spherical_edge_length_range(Verts, Faces);
    MINs_psi[i] = 60. * np.rad2deg(min_psi); # in arc-min
    MAXs_psi[i] = 60. * np.rad2deg(max_psi);
print(f"{'N':>4} {'Nv':>8} {'Nf':>8} "
      f"{'Vol_err':>10} {'Area_err':>10} {'Min_psi':>10} {'Max_psi':>10}");
print("-" * 68);
for i in range(len(Nsubs)):
    print(f"{Nsubs[i]:>4} {int(NVs[i]):>8} {int(NFs[i]):>8}"
          f"{Es_vol[i]:>10.2e} {Es_area[i]:>10.2e}"
          f"{MINs_psi[i]:>10.3f} {MAXs_psi[i]:>10.3f}");
# %%
# # ! Triangulated regular geographic grid
Lon_Res = 6*2**np.arange(nsub_max)+1;
Lat_Res = 3*2**np.arange(nsub_max)+1;
for i, (lon_res, lat_res) in enumerate(zip(Lon_Res, Lat_Res)):
    regsph = pv.Sphere(radius=1.0, 
                       theta_resolution=lat_res, phi_resolution=lon_res);
    Verts = regsph.points;
    Faces = regsph.regular_faces;
    NVs[i] = Verts.shape[0];
    NFs[i] = Faces.shape[0];
    Es_vol[i]  = 1. - regsph.volume/vol;
    Es_area[i] = 1. - regsph.area/area;
    min_psi, max_psi = spherical_edge_length_range(Verts, Faces);
    MINs_psi[i] = 60. * np.rad2deg(min_psi); # in arc-min
    MAXs_psi[i] = 60. * np.rad2deg(max_psi);
print(f"{'N':>4} {'Nv':>8} {'Nf':>8} "
      f"{'Vol_err':>10} {'Area_err':>10} {'Min_psi':>10} {'Max_psi':>10}");
print("-" * 68);
for i in range(len(Nsubs)):
    print(f"{Nsubs[i]:>4} {int(NVs[i]):>8} {int(NFs[i]):>8}"
          f"{Es_vol[i]:>10.2e} {Es_area[i]:>10.2e}"
          f"{MINs_psi[i]:>10.3f} {MAXs_psi[i]:>10.3f}");
# %%
# # ! Plot
pl = pv.Plotter(shape=(2, 3), image_scale=3);
for i, nsub in enumerate([1, 2, 3]):
    icosph = pv.Icosphere(nsub=nsub);
    icosph_with_area = icosph.compute_cell_sizes();
    areas = icosph_with_area['Area'];
    area_percent = 100 * areas / areas.sum();
    scalar_name = f'Area (%), n={nsub}';
    icosph_with_area[scalar_name] = area_percent;
    pl.subplot(0, i);
    sargs = dict(title=scalar_name, title_font_size=14, label_font_size=12, 
                 n_labels=3, position_y=0.05, fmt='%.2f');
    pl.add_mesh(icosph_with_area, scalars=scalar_name, 
                scalar_bar_args=sargs, cmap='viridis'); # viridis, plasma, magma, turbo
    pl.camera.Zoom(1.25);
pl.show();
pl.screenshot("icosphere_geographic");