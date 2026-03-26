import pyshtools as pysh
import numpy as np

l_test = 3000
clm = pysh.SHGravCoeffs.from_zeros(lmax=l_test, gm=1.0, r0=1.0)
clm.coeffs[0, l_test, 0] = 1e-20

try:
    grid = clm.expand()
    print("Max:", np.nanmax(grid.rad.data))
    print("Has NaN?", np.isnan(grid.rad.data).any())
except Exception as e:
    print("Error:", e)