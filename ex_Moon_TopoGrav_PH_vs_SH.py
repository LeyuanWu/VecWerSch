# %% 
# # ! Setup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyshtools as pysh
import pygmt
import pyvista as pv
import time
import xarray as xr
# %% 
# # ! Topographic potential
ds = xr.open_dataset('moon_topo_gravity.nc')
print(ds)
grd_gD = pysh.SHGrid.from_xarray(ds['gD'])
fig = pygmt.Figure()
grd_gD.plotgmt(fig=fig,
               projection='mollweide',
               central_longitude=-90.,
               grid=[30, 30],
               tick_interval=None,
               cmap='haxby',
               cmap_limits=[-800, 800],
               colorbar='bottom',
               cb_triangles='both',
               cb_label='Downward gravity component (mGal)',
               axes_labelsize=12,
               cb_tick_interval=200,
               cb_minor_tick_interval=100)
fig.show(width=800)