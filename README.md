# VecWerSch: Vectorized/Parallel Polyhedral Gravitational Field Computation

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fast, exact polyhedral gravity modelling in Python. A vectorized and parallelized Numba implementation for geophysical, planetary, and asteroid applications.

## 🌟 Features

- **Vectorized & Parallel Algorithms**: Numba-accelerated implementation for fast polyhedral gravity calculations on complex 3D geometries.
- **Exact Analytical Solutions**: Computation of gravitational potential, acceleration, and gradient tensor for arbitrary polyhedra.
- **Reproducible Results**: Complete scripts to reproduce all figures and tables from the manuscript.

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/LeyuanWu/VecWerSch.git
cd VecWerSch
conda install numpy numba matplotlib pyvista pygmt pyshtools
```

## 📂 Repository Structure

## 🔬 Reproducibility Guide

Complete specifications for each numerical example. Click to expand details.

<details>
<summary><strong>ex_Benchmark_WerSch_v1v2.py</strong> – WerSch Numba v1 vs v2 performance & accuracy benchmark</summary>

- **📊 Produces:** **Figure 1** in manuscript
- **Generated Outputs:** `Benchmark_WerSch_v1v2_nsub{nsub_max}.png`
- **Required Inputs:** None (meshes and observation points are generated programmatically via `pv.Icosphere` and Fibonacci sphere sampling)
- **Job Log:** [`joblog_Benchmark_v1v2_nsub8.out`](JobLogs/joblog_Benchmark_v1v2_nsub8.out)
- **Est. Runtime:**
  - `nsub_max=6`: ~25 min
  - `nsub_max=7`: ~1 h 40 min
  - `nsub_max=8`: ~7 h
</details>

<details>
<summary><strong>ex_Sphere_Polyhedron.py</strong> – Sphere approximation: icosahedron subdivision vs. geographic grid</summary>

- **📊 Produces:** **Figure 2** and **Table 1** in manuscript
- **Generated Outputs:** `Sphere_Polyhedron.png`
- **Required Inputs:** None (meshes generated programmatically via `pv.Icosphere` and `pv.Sphere`)
- **Job Log:** [`joblog_Sphere_Polyhedron_nsub12.out`](JobLogs/joblog_Sphere_Polyhedron_nsub12.out)
- **Key Parameters:** `nsub_max=12`, two meshing methods (recursive icosahedron subdivision, triangulated regular geographic grid)
- **Est. Runtime (`nsub_max=12`):** ~5 min
</details>



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Leyuan Wu** - [GitHub](https://github.com/LeyuanWu)

---

*If you find this repository useful, please consider starring it on GitHub!* ⭐


