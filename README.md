# VecWerSch: Vectorized Polyhedral Gravitational Field Computation

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Efficient Python implementation for computing gravitational fields of polyhedral bodies using vectorized algorithms. This repository provides tools for geophysical modeling, planetary science, and asteroid gravity analysis.

## 🌟 Features

- **Vectorized Computation**: High-performance gravity field calculations using NumPy and Numba
- **Polyhedral Models**: Support for complex 3D geometries approximated by polyhedrons
- **Multiple Coordinate Systems**: Local Cartesian and Global Spherical coordinates
- **Validation Tools**: Green's third identity verification for accuracy assessment
- **Real-world Examples**: Applications to asteroid EROS and Moon topography

## 📋 Table of Contents

- [Installation](#installation)
- [Theory & Method](#theory--method)
- [Numerical Examples](#numerical-examples)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/LeyuanWu/VecWerSch.git
cd VecWerSch
pip install numpy numba matplotlib pyvista
```

## 📖 Theory & Method

- [x] **Polyhedral Gravitation Theory**: Comprehensive explanation of gravitational potential computation for polyhedral bodies
- [x] **Vectorized Algorithm Design**: Efficient implementation using vector operations

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeyuanWu/VecWerSch/blob/main/Theory_Method.ipynb)

[View Theory & Method Notebook](Theory_Method.ipynb) | [View on GitHub](https://github.com/LeyuanWu/VecWerSch/blob/main/Theory_Method.ipynb)

## 🔬 Numerical Examples

### Sphere Approximation Methods
- [x] **Icosphere & Geosphere**: Two approaches to approximate spheres with polyhedrons
  - Recursive icosahedron subdivision
  - Triangulated geographic grid

![Icosphere vs Geosphere](Figs/icosphere_geosphere.png)

[View Sphere Polyhedron Example](ex_Sphere_Polyhedron.ipynb) | [View on GitHub](https://github.com/LeyuanWu/VecWerSch/blob/main/ex_Sphere_Polyhedron.ipynb)

### Gravity Forward Modeling
- [x] **Local Cartesian Coordinates**: Icosphere/geosphere gravity modeling
- [x] **Global Spherical Coordinates**: Large-scale planetary gravity fields

[View Local Coordinates Example](ex_IcoGeoSphere_Local.ipynb) | [View Global Coordinates Example](ex_IcoGeoSphere_Global.ipynb)

### Asteroid EROS Analysis
- [x] **2D Plane Gravity Computation**: Gravitational anomalies on planar surfaces
- [x] **Green's Third Identity Validation**: Accuracy verification for polyhedral methods

![EROS Gravity Potential](Figs/EROS_2DPlane_GPV.png)

[View EROS 2D Example](ex_EROS_2DPlane.ipynb) | [View Green's Identity Example](ex_EROS_GreenThridID_Comp.ipynb)

### Lunar Topography
- [x] **Moon Gravity Fields**: Comparison of polyhedral analytical vs. spherical harmonics methods

[View Moon Topography Example](ex_Moon_TopoGrav_Comp.ipynb)

## 💡 Usage

Import the core modules:

```python
import numpy as np
from gravity_forward_numba import polyhedral_gravity

# Compute gravitational potential
vertices, faces = load_polyhedron_data()
points = np.array([[x, y, z]])  # Observation points
density = 2670  # kg/m³

potential = polyhedral_gravity(vertices, faces, points, density)
```

See the example notebooks for detailed usage patterns.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Leyuan Wu** - [GitHub](https://github.com/LeyuanWu)

---

*If you find this repository useful, please consider starring it on GitHub!* ⭐


