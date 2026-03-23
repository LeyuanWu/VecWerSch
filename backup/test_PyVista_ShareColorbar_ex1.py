import pyvista as pv
import numpy as np

# Create sample meshes with random scalar data
sphere1 = pv.Sphere(center=(0, 0, 0), radius=1)
sphere1['scalars'] = np.random.random(sphere1.n_points)

sphere2 = pv.Sphere(center=(0, 0, 0), radius=1)
sphere2['scalars'] = np.random.random(sphere2.n_points)
sphere2.translate([0, 3, 0], inplace=True)  # Increase distance for clearer side-by-side view

# Calculate shared color limits
all_scalars = np.concatenate([sphere1['scalars'], sphere2['scalars']])
clim = (all_scalars.min(), all_scalars.max())

# Create a single plotter
plotter = pv.Plotter()

# Add both meshes to the same plotter
plotter.add_mesh(sphere1, scalars='scalars', clim=clim, show_scalar_bar=False)
plotter.add_mesh(sphere2, scalars='scalars', clim=clim, show_scalar_bar=False)

# Add a shared horizontal colorbar spanning the entire window
plotter.add_scalar_bar(vertical=False, position_x=0.1, position_y=0.05, width=0.8, height=0.1)

# Set isometric view to ensure both spheres are visible
plotter.view_yz()

# Display the plot
plotter.show()