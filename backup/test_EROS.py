# %% # ! Setup
import vtk
import vtkmodules.vtkRenderingFreeType
import vtkmodules.vtkRenderingMatplotlib
import scipy.io
import numpy as np
import pyvista as pv
# %% # ! Reading EROS shape .mat file
mat_data = scipy.io.loadmat('../EROS.mat')
# variable_names = [key for key in mat_data.keys() if not key.startswith('__')]
# ['eros856_1708','eros3897_7790','eros5078_10152',
# 'eros11272_22540','eros44701_89398','eros100352_200700']
eros_res = {
    'f': ['eros100352_200700', 100352, 200700],
    'h': ['eros44701_89398',4470189398],
    'i': ['eros11272_22540', 11272, 22540],
    'l': ['eros5078_10152', 5078, 10152],
    'c': ['eros856_1708', 856, 1708]
}
res = 'f'
mesh_name, n_points, n_faces = eros_res[res];
mesh_data = mat_data[mesh_name];
Points = mesh_data[:n_points,:];
Faces = mesh_data[n_points:,:].astype(np.int32);
nvpf = np.full((n_faces, 1), 3, dtype=np.int32);
Faces = np.hstack((nvpf, Faces));
mesh = pv.PolyData(Points, Faces);
######## * Assigning scalars
mesh.point_data['gx'] = mesh.points[:,0]
mesh.point_data['gy'] = mesh.points[:,1]
mesh.point_data['gz'] = mesh.points[:,2]
mesh.point_data['gnorm'] = np.sqrt(np.sum(mesh.points**2, axis=1));
# %% # ! Accessing the Wrapped Data Object
print(f'Number of cells: {mesh.n_cells}', flush=True);
print(f'Number of points: {mesh.n_points}', flush=True);
print(f'Number of scalar arrays: {mesh.n_arrays}', flush=True);
print(f'Bounds of the mesh: {mesh.bounds}', flush=True);
print(f'Center of the mesh: {mesh.center}', flush=True);
# %% # ! Plot
cmap = 'jet'
SCALARs = ['gx', 'gy', 'gz', 'gnorm']
TITLEs = ['$g_x$ (mGal)', '$g_y$ (mGal)', 
          '$g_z$ (mGal)', r'$\mathbf{g}$ (mGal)'];
pl = pv.Plotter(shape=(2, 2), border=True, image_scale=4)
for i, (scalar, title) in enumerate(zip(SCALARs, TITLEs)):
    pl.subplot(i//2, i%2)
    pl.add_mesh(mesh.copy(deep=False), style='surface', 
                cmap=cmap, show_edges=False,
                scalars=scalar, show_scalar_bar=False);
    pl.add_scalar_bar(title=title, title_font_size=14, 
                      label_font_size=10, italic=True);
pl.show()

