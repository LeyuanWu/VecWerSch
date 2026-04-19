# VecWerSch: Vectorized/Parallel Polyhedral Gravitational Field Computation

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Efficient Python implementation for computing gravitational fields of polyhedral bodies using Vectorized/Parallel algorithms. This repository provides tools for geophysical modelling, planetary science, and asteroid gravity analysis.

## 🌟 Features

- **Vectorized/Parallel Computation**: High-performance gravity field calculations using NumPy and Numba
- **Polyhedral Models**: Support for complex 3D geometries approximated by polyhedrons
- **Multiple Coordinate Systems**: Local Cartesian and Global Spherical coordinates
- **Validation Tools**: Green's third identity verification for accuracy assessment
- **Real-world Examples**: Applications to asteroid EROS and Moon topography

## 📋 Table of Contents

- [Installation](#installation)
- [Theory & Method](#theory-method)
- [Numerical Examples](#numerical-examples)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/LeyuanWu/VecWerSch.git
cd VecWerSch
conda install numpy numba matplotlib pyvista pygmt pyshtools
```

## 📖 Theory & Method

- [x] **Polyhedral Gravitation Theory**: Comprehensive explanation of gravitational potential computation for polyhedral bodies
- [x] **Vectorized Algorithm Design**: Efficient implementation using vector operations

[View Theory & Method Notebook](https://github.com/LeyuanWu/VecWerSch/blob/main/Theory_Method.ipynb)

## 🔬 Numerical Examples

### Sphere Approximation Methods
- [x] **Icosphere & Geosphere**: Two approaches to approximate spheres with polyhedrons
  - Recursive icosahedron subdivision
  - Triangulated geographic grid

![Icosphere vs Geosphere](Figs/icosphere_geosphere.png)

[View Sphere Polyhedron Example](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_Sphere_Polyhedron.ipynb)

### Gravity Forward Modelling
- [x] **Local Cartesian Coordinates**: Icosphere/geosphere gravity modelling
- [x] **Global Spherical Coordinates**: Large-scale planetary gravity fields

[View Local Coordinates Example](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_IcoGeoSphere_Local.ipynb) | [View Global Coordinates Example](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_IcoGeoSphere_Global.ipynb)

### Asteroid EROS Analysis
- [x] **2D Plane Gravity Computation**: Gravitational anomalies on planar surfaces
- [x] **Green's Third Identity Validation**: Accuracy verification for polyhedral methods

![Green's Third Identity Convergence (r=18km)](Figs/greenthirdid_convergence_r18km.png)

[View EROS 2D Example](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_EROS_2DPlane.ipynb) | [View Green's Identity Example](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_EROS_GreenThridID_Comp.ipynb)

### Lunar Topography
- [x] **Moon Gravity Fields**: Comparison of polyhedral analytical vs. spherical harmonics methods

[View Moon Topography Example](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_Moon_TopoGrav_Data.ipynb)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Leyuan Wu** - [GitHub](https://github.com/LeyuanWu)

---

*If you find this repository useful, please consider starring it on GitHub!* ⭐


