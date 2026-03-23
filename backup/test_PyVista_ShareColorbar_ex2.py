import pyvista as pv
import numpy as np

shape = (2, 3)
row_weights = [0.2, 1,]
col_weights = [1, 1, 1]
groups = [
    (0, np.s_[:]),  # First group spans over all columns of the first row (0)
    (1, 0),  # Second group spans over row 1-3 of the first column (0)
]


pl = pv.Plotter(shape=shape, row_weights=row_weights, col_weights=col_weights, groups=groups)

pl.subplot(1, 0)
mesh = pv.Cone()
mesh['data'] = np.random.rand(mesh.n_cells)
pl.add_mesh(mesh, show_scalar_bar=False)

pl.subplot(1, 1)
mesh = pv.Sphere()
mesh['data'] = np.random.rand(mesh.n_cells)
pl.add_mesh(mesh, show_scalar_bar=False)

pl.subplot(1, 2)
mesh = pv.ParametricBoy()
mesh['data'] = np.random.rand(mesh.n_cells)

actor = pl.add_mesh(mesh, show_scalar_bar=False)



pl.subplot(0, 0)
mesh = pv.PolyData(np.zeros((2, 3)))
mesh['data'] = (0, 1)

actor = pl.add_mesh(mesh, scalars=None, show_scalar_bar=False)
actor.visibility = False

scalar_bar_kwargs = {
    'color': 'k',
    'title': actor.mapper.lookup_table._lookup_type + '\n',

    'outline': False,
    'title_font_size': 40,
}
label_level = 0
if actor.mapper.lookup_table.below_range_color:
    scalar_bar_kwargs['below_label'] = 'below'
    label_level = 1
if actor.mapper.lookup_table.above_range_color:
    scalar_bar_kwargs['above_label'] = 'above'
    label_level = 1

label_level += actor.mapper.lookup_table._nan_color_set
scalar_bar = pl.add_scalar_bar(**scalar_bar_kwargs)
# scalar_bar.SetLookupTable(actor.mapper.lookup_table)
# scalar_bar.SetMaximumNumberOfColors(actor.mapper.lookup_table.n_values)
# scalar_bar.SetPosition(0.03, 0.1 + label_level * 0.1)
# scalar_bar.SetPosition2(0.95, 0.9 - label_level * 0.1)
# # scalar_bar.SetTextPad(-10)
# if actor.mapper.lookup_table._nan_color_set and actor.mapper.lookup_table.nan_opacity > 0:
#     scalar_bar.SetDrawNanAnnotation(actor.mapper.lookup_table._nan_color_set)

pl.show()