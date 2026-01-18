### *Python code for vectorized efficient computation of polyhedral gravitational fields*

**Target Journal:** [Geophysics]

**Structure:**  
- Theory & Method: [Vectorized code; numpy & numba]
- Numerical examples: 
  1. Sphere approximated by polyhedral using two approaches: (1) recursive subdivision of an icosahedron, and (2) an equal longitude-latitude grid with increasing resolution. > 🖼️ **Fig: PyVista plot showing these two approximations** [level 0, 1, 2]
   
  2. Draw plots to show multiple parameter changes with respect to the level of subdivision $n$ or the resolution $\Delta \lambda$, choose $\Delta \lambda = 60/2^n$ degrees so that the two have similar resolution; > 🖼️ **Fig: plot showing the increase of $N_V$, $N_F$, convergence of $area$, $volume$ compared to the unit sphere, and the change of min/max spherical distance $\psi$, max/min edge ratio $\alpha$ and area ratio $\beta$ for triangles...**
   
  3. DEM models using (1) Planar approximation; (2) Spherical approximation; (3) Ellipsoidal; > 🖼️ **Fig: PyVista plot showing these 3 models, including field points** [Himalaya? Alps?]
   
  4. DEM model comparison: > 🖼️ **Fig: difference between the three models** [How large? $2^{\circ} \times 2^{\circ}$ or $5^{\circ} \times 5^{\circ}$ patches?]

  1. EROS > 🖼️ **Fig: EROS model including field points on the surface and on spheres** [$18$ km? $20$ km?]
   
  2. EROS model computation validating the Green's third identity (see equation~A1)


1. Theory
   - [ ] Polyhedral gravity derivation [Jupyter Notebook]
         (https://github.com/LeyuanWu/VecWerSch/blob/main/README.ipynb)
1. Numerical Examples
   - [ ] Icosphere [Jupyter Notebook]
         (https://github.com/LeyuanWu/VecWerSch/blob/main/ex1_Icosphere.py)
   - [ ] EROS [Jupyter Notebook]
         (https://github.com/LeyuanWu/VecWerSch/blob/main/ex2_EROS.py)


**Appendix**

*Green's third identity*

$$
V(\mathbf{r}) = \frac{1}{4\pi} \oint_S 
\left[ 
V(\mathbf{r}') \frac{\partial}{\partial n'} \left( \frac{1}{|\mathbf{r} - \mathbf{r}'|} \right)
    - \frac{1}{|\mathbf{r} - \mathbf{r}'|} \frac{\partial V(\mathbf{r}')}{\partial n'}
\right] dS' \tag{A1}
$$