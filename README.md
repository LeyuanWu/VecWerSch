### *Python code for vectorized efficient computation of polyhedral gravitational fields*

**Target Journal:** [Geophysics]

**Structure:**  
- Theory & Method: [Vectorized code; numpy & numba]
- Numerical examples: 
  1. Sphere approximated by polyhedral using two approaches: (1) recursive subdivision of an icosahedron, and (2) an equal longitude-latitude grid with increasing resolution. > 🖼️ **Fig: PyVista plot showing these two approximations** [level 0, 1, 2]
   
  2. Draw plots to show multiple parameter changes with respect to the level of subdivision $n$ or the resolution $\Delta \lambda$, choose $\Delta \lambda = 60/2^n$ degrees so that the two have similar resolution; > 🖼️ **Fig: plot showing the increase of $N_V$, $N_F$, convergence of $area$, $volume$ compared to the unit sphere, and the change of min/max spherical distance $\psi$, max/min edge ratio $\alpha$ and area ratio $\beta$ for triangles...**
   
  3. DEM models using (1) Planar approximation; (2) Spherical approximation; (3) Ellipsoidal; > 🖼️ **Fig: PyVista plot showing these 3 models, including field points** [Himalaya? Alps?]
   
  4. DEM model comparison: > 🖼️ **Fig: difference between the three models** [How large? $2^{\circ} \times 2^{\circ}$ or $5^{\circ} \times 5^{\circ}$ patches?]

  5. EROS > 🖼️ **Fig: EROS model including field points on the surface and on spheres** [ $18$ or $20$ km?]
   
  6. EROS model computation validating the Green's third identity (see equation~A1)


1. Theory
   - [x] Polyhedral gravity derivation [Jupyter Notebook](https://github.com/LeyuanWu/VecWerSch/blob/main/README.ipynb)
   - [x] Vectorized code design [Jupyter Notebook](https://github.com/LeyuanWu/VecWerSch/blob/main/VecWerSch.ipynb)
2. Numerical Examples
   - [ ] Icosphere [Jupyter Notebook](https://github.com/LeyuanWu/VecWerSch/blob/main/ex1_Icosphere.py)
   - [x] EROS [Jupyter Notebook](https://github.com/LeyuanWu/VecWerSch/blob/main/ex2_EROS.py)


