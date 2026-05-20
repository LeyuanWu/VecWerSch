#########################################################################
# # ! Moon Topographic-Gravitational-Potential Computation: Polyhedron
#########################################################################
# %% 
# # ! Setup
import pyshtools as pysh
# %% 
# # ! Shape of Moon
#### * Shape
lmax_shp = 359
clm_shp_moon = \
    pysh.SHCoeffs.from_file(f'input/Moon_shape_719.sh', 
                            lmax=lmax_shp, 
                            name='LOLA_shape (Moon)',
                            units='m', format='bshc')
r_itfc_km = clm_shp_moon.coeffs[0,0,0] / 1.e3
in_res = 10.0/60.0 # degree
lmax_grid = int(90.0/in_res - 1)
grd_shp_moon = clm_shp_moon.expand(lmax=lmax_grid, 
                                   lmax_calc=lmax_shp)
grd_topo_moon = grd_shp_moon/1.e3 - r_itfc_km
nc_file_topo = f'moon_topo_Lshp{lmax_shp}_{int(60*in_res)}arcmin.nc'
grd_topo_moon.to_netcdf(filename = nc_file_topo,
                        title = f'Moon topography (shape model up to degree {lmax_shp})',
                        name='topo', units='km')
print(f"Saved to '{nc_file_topo}'")