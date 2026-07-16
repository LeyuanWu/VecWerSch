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

<!-- ## 📂 Repository Structure -->

## 🔬 Reproducibility Guide

Complete specifications for each numerical example. Click to expand details.

<details>
<summary><strong>ex_Benchmark_WerSch_v1v2.py</strong> – WerSch_numba v1 vs v2 performance & accuracy benchmark</summary>

- 📊 Produces: **Figure 1** in manuscript
- 📦 Required Inputs: No external files needed; meshes generated programmatically via `pv.Icosphere`; observation points generated via `fibonacci_sphere_points`
- 🖼️ Generated Outputs: `Benchmark_WerSch_v1v2_nsub{nsub_max}.png`
- 📋 Job Log: [`joblog_Benchmark_v1v2_nsub8.out`](JobLogs/joblog_Benchmark_v1v2_nsub8.out)
- ⏱️ Est. Runtime:
  - `nsub_max=6`: ~25 min
  - `nsub_max=7`: ~1 h 40 min
  - `nsub_max=8`: ~7 h
</details>

<details>
<summary><strong>ex_Sphere_Polyhedron.py</strong> – Sphere approximation: icosahedron subdivision vs. geographic grid</summary>

- 📊 Produces: **Figure 2** and **Table 1** in manuscript
- 📦 Required Inputs: No external files needed; meshes generated programmatically via `pv.Icosphere` and `pv.Sphere`
- 🖼️ Generated Outputs: `Sphere_Polyhedron.png`
- 📋 Job Log: [`joblog_Sphere_Polyhedron_nsub12.out`](JobLogs/joblog_Sphere_Polyhedron_nsub12.out)
- ⏱️ Est. Runtime:
  - `nsub_max=12`: ~5 min
</details>

<details>
<summary><strong>ex_IcoGeoSphere_Global.py</strong> – Global gravity forward modelling: icosahedron vs. geographic grid convergence</summary>

- 📊 Produces: **Figure 3** in manuscript
- 📦 Required Inputs: No external files needed; homogeneous Earth sphere model defined analytically; meshes generated via `pv.Icosphere` and `pv.Sphere`; observation points generated via `fibonacci_sphere_points` (5000 pts × 4 altitudes)
- 🖼️ Generated Outputs:
  - `IcoGeoSphere_Global_nsub12_nobs5000.png`
  - `output/IcoGeoSphere_Global_nsub12_nobs5000.npz` *(NPZ — cached results for replotting without recomputation)*
- 📋 Job Log: [`joblog_IcoGeoSphere_Global_nsub12_nobs5000.out`](JobLogs/joblog_IcoGeoSphere_Global_nsub12_nobs5000.out)
- 💾 Memory Requirement ($N_T = 20 \times 4^{nsub\_max}$):
  - `nsub_max=11` ($N_T \approx 8.4 \times 10^7$): v1 ~12.8 GB · v2 ~1.9 GB
  - `nsub_max=12` ($N_T \approx 3.4 \times 10^8$): ⚠️ v1 ~**51.2 GB** · v2 ~7.5 GB
- ⏱️ Est. Runtime:
  - `nsub_max=10`: ~35 min
  - `nsub_max=11`: ~2 h 20 min
  - `nsub_max=12`: ~19.5 h
</details>

<details>
<summary><strong>ex_EROS_GreenThirdID_1.py</strong> – Green’s 3rd Identity validation: EROS shape model preparation</summary>

- 📊 Produces: **Figure 4** in manuscript
- 📦 Required Inputs: `input/EROS.mat` *(MAT — EROS asteroid polyhedral shape models)*
- 🖼️ Generated Outputs: `EROS_Geometry.png`
- ⏱️ Est. Runtime: ~1–2 min
</details>

<details>
<summary><strong>ex_EROS_GreenThirdID_2.py</strong> – Green’s 3rd Identity validation: convergence & error analysis</summary>

- 📊 Produces: **Figure 5** and **Figure 6** in manuscript
- 📦 Required Inputs: `input/EROS.mat` *(MAT — EROS asteroid polyhedral shape model)*
- 🖼️ Generated Outputs:
  - `EROS_Convergence.png`
  - `EROS_ErrorMaps.png`
- 📋 Job Log: [`joblog_EROS_GreenThirdID_nsub6_nobs10242.out`](JobLogs/joblog_EROS_GreenThirdID_nsub6_nobs10242.out)
- ⏱️ Est. Runtime: ~4–5 min
</details>

<details>
<summary><strong>ex_Moon_TopoGrav_Comp_SH.py</strong> – Lunar topographic gravity: spherical harmonics spectral-domain computation</summary>

- 📊 Produces: No direct manuscript figure/table; generates NetCDF input for a downstream manuscript figure
- 📦 Required Inputs: `input/Moon_shape_719.sh` *(SHTOOLS binary — LOLA lunar shape model)*
- 🖼️ Generated Outputs:
  - `Moon_Topo_GMT.png` *(topography map; validation only)*
  - `Moon_TopoGz_GMT.png` *(vertical gravity component map; validation only)*
  - `output/moon_topo_gravity_Lshp359_nmax7.nc` *(NetCDF — gx/gy/gz grids at r=1748 km; **used as input for downstream SH vs PH comparison**)*
- 📋 Job Log: [`joblog_Moon_TopoGrav_Comp_SH_Lshp359_nmax7.out`](JobLogs/joblog_Moon_TopoGrav_Comp_SH_Lshp359_nmax7.out)
- ⏱️ Est. Runtime: ~31 min
</details>


<details>
<summary><strong>ex_Moon_TopoGrav_Comp_PH.py</strong> – Lunar topographic gravity: polyhedron spatial-domain computation</summary>

- 📊 Produces: No direct manuscript figure or table; generates NetCDF inputs for downstream scripts 
- 📦 Required Inputs: `input/Moon_shape_719.sh` *(SHTOOLS binary — LOLA lunar shape model)*
- 🖼️ Generated Outputs:
  - `output/moon_topo_gravity_in{in_res}arcmin_out15arcmin.nc` *(NetCDF — full tensor gravity grids at r=1748 km; generated for in_res={6,15} arc-min; **used as input for downstream global SH vs PH comparison**)*
  - `output/moon_topo_Lshp359_{in_res}arcmin.nc` *(NetCDF — topography grids at 8 resolutions: 1, 2, 3, 4, 5, 6, 10, 15 arc-min; **used as input for downstream mesh generation**)*
    - ⚡ **Quick-generate note:** Produced by running only the *Shape of Moon* section (first code cell block); no polyhedral gravity computation required. Runtime < 1 min per resolution.
- 📋 Job Log:
  - [`joblog_Moon_TopoGrav_Comp_PH_in15arcmin_out15arcmin.out`](JobLogs/joblog_Moon_TopoGrav_Comp_PH_in15arcmin_out15arcmin.out)
  - [`joblog_Moon_TopoGrav_Comp_PH_in6arcmin_out15arcmin.out`](JobLogs/joblog_Moon_TopoGrav_Comp_PH_in6arcmin_out15arcmin.out)
- ⏱️ Est. Runtime (output grid fixed at 15 arc-min, 1,038,961 obs pts):
  - `in_res=15 arc-min` (~2M faces): ~1 h 58 min
  - `in_res=6 arc-min` (~13M faces): ~15 h 10 min
</details>


<details>
<summary><strong>ex_Moon_TopoGrav_PH_vs_SH_1.py</strong> – Lunar topographic gravity: PH vs SH comparison at 15 arc-min resolution</summary>

- 📊 Produces: No direct manuscript figure or table; generates validation plot and error-location CSV for downstream multi-resolution analysis
- 📦 Required Inputs:
  - `output/moon_topo_gravity_Lshp359_nmax7.nc` *(NetCDF — SH gravity components; generated by `ex_Moon_TopoGrav_Comp_SH.py`)*
  - `output/moon_topo_gravity_in15arcmin_out15arcmin.nc` *(NetCDF — PH gravity components; generated by `ex_Moon_TopoGrav_Comp_PH.py`)*
  - `output/moon_topo_Lshp359_15arcmin.nc` *(NetCDF — lunar topography grid; generated by `ex_Moon_TopoGrav_Comp_PH.py`, shape-only section)*
- 🖼️ Generated Outputs:
  - `Moon_TopoGz_PH_vs_SH_1.png` *(Validation only — gz_SH map with error peak markers + Δgz difference map; not included in manuscript)*
  - `Moon_gNED_errors.csv` *(Top-10 largest |Δgz| locations with >1000 km separation; **used as fixed evaluation points for downstream comparison across 8 topography resolutions using `moon_topo_Lshp359_{in_res}arcmin.nc`**)*
- ⏱️ Est. Runtime: ~1–2 min *(post-processing only; no forward modelling)*
</details>




## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Leyuan Wu** - [GitHub](https://github.com/LeyuanWu)

---

*If you find this repository useful, please consider starring it on GitHub!* ⭐


