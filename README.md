# VecWerSch: Vectorized/Parallel Polyhedral Gravitational Field Computation

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Efficient Python implementation for computing gravitational fields of polyhedral bodies using Vectorized/Parallel algorithms. This repository provides tools for geophysical modelling, planetary science, and asteroid gravity analysis.

## 🌟 Features

- **Parallel Computation**: High-performance gravity field calculations using Numba
- **Polyhedral Models**: Support for complex 3D geometries approximated by polyhedrons
- **Multiple Coordinate Systems**: Local Cartesian and Global Spherical coordinates
- **Validation Tools**: Green's third identity verification for accuracy assessment
- **Real-world Examples**: Applications to asteroid EROS and Moon topography

## 📋 Table of Contents

- [Installation](#installation)
- [Benchmark](#-benchmark)
- [Numerical Examples](#-numerical-examples)

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/LeyuanWu/VecWerSch.git
cd VecWerSch
conda install numpy numba matplotlib pyvista pygmt pyshtools
```

## 📊 Benchmark

<!-- ![Benchmark comparison of two algorithms](Figs/Benchmark_WerSch_v1v2_nsub8.png) -->
<div align="center">
  <img src="Figs/Benchmark_WerSch_v1v2_nsub8.png" width="500"/>
  <p><em>Benchmark comparison of two algorithms</em></p>
</div>

## 🔬 Numerical Examples

### Sphere Approximation Methods
- [x] **Icosphere & Geosphere**: Two approaches to approximate spheres with polyhedrons
  - Recursive icosahedron subdivision
  - Triangulated geographic grid

<div align="center">
  <img src="Figs/Sphere_Polyhedron.png" width="600"/>
  <p><em>Icosphere vs. geosphere polyhedral approximations of an ideal sphere</em></p>
</div>

[View Sphere Polyhedron Example](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_Sphere_Polyhedron.ipynb)

### Synthetic sphere modelling

[View Sphere gravity computation Example](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_IcoGeoSphere_Global.ipynb)

### Asteroid EROS
- [x] **2D Plane Gravity Computation**: Gravitational anomalies on a 2D plane
- [x] **Green's Third Identity Validation**: Accuracy verification for polyhedral methods

<div align="center">
  <img src="Figs/EROS_2DPlane_GPV.png" width="600"/>
  <p><em>EROS gravity potential and vector on a 2D plane</em></p>
</div>

[View EROS: 2D plane computation](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_EROS_2DPlane.ipynb)

<div align="center">
  <img src="Figs/EROS_Geometry.png" width="600"/>
  <p><em>EROS geometry refinement with surface gravity</em></p>
</div>

[View EROS: mesh refinement and surface gravity](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_EROS_GreenThirdID_1.ipynb)

<div align="center">
  <img src="Figs/EROS_ErrorMaps.png" width="600"/>
  <p><em>Numerical validation of Green's third ID across refinement levels</em></p>
</div>

[View EROS: Green's Third ID convergence](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_EROS_GreenThirdID_2.ipynb)

### Moon Topography
- [x] **Moon Gravity Fields**: Comparison of polyhedral analytical vs. spherical harmonics methods

<div align="center">
  <img src="Figs/Moon_Topo_Ref1738km_3D.png" width="600"/>
  <p><em>Moon topography 3D view </em></p>
</div>

<div align="center">
  <img src="Figs/Moon_FAG_r1748km_3D.png" width="600"/>
  <p><em>3D view of Moon's Free-air gravity anomaly at r=1748 km </em></p>
</div>

[Moon topography & gravity overview](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_Moon_TopoGrav_Data.ipynb)

[Moon topographic gravitational potential computation: SH](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_Moon_TopoGrav_Comp_SH.ipynb)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Leyuan Wu** - [GitHub](https://github.com/LeyuanWu)

---

*If you find this repository useful, please consider starring it on GitHub!* ⭐


