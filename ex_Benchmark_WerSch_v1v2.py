##################################################################
# # ! Benchmarking WerSch_numba_v1 vs WerSch_numba_v2
##################################################################
# %%
# # ! Set up
import time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import pyvista as pv
from gravity_forward_numba import WerSch_numba_v1, WerSch_numba_v2
# %% 
# # ! Start time
print("="*80)
print(f"Start time: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
print("="*80)
# %% 
# # ! Functions
def fibonacci_sphere_points(N, radius=2.0):
    """Generate N approximately uniformly distributed points on a sphere."""
    i = np.arange(N, dtype=np.float64)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / N)
    theta = np.pi * (1.0 + 5.0**0.5) * (i + 0.5)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack((x, y, z))
# %%
# # ! Number of faces and observation points for each level
nsub_max = 7
NSUBs = np.arange(nsub_max + 1)
NFs = 20 * 4**NSUBs
NPs = 20 * 4**NSUBs
rho = 2670.0
r_obs = 1.1
# %%
# # ! Precompute meshes and observation points
meshes = []
Ps = []
for i, nsub in enumerate(NSUBs):
    mesh = pv.Icosphere(radius=1.0, nsub=nsub)
    Verts = mesh.points.astype(np.float64)
    Faces = mesh.regular_faces.astype(np.int32)
    meshes.append((Verts, Faces))
    Np = NPs[i]
    P = fibonacci_sphere_points(int(Np), radius=r_obs)
    Ps.append(P)
# %%
# # ! Benchmark
tc_v1 = np.empty((len(NSUBs), len(NSUBs)), dtype=np.float64)
tc_v2 = np.empty((len(NSUBs), len(NSUBs)), dtype=np.float64)
MAD_gz = np.empty((len(NSUBs), len(NSUBs)), dtype=np.float64)
MAD_Tzz = np.empty((len(NSUBs), len(NSUBs)), dtype=np.float64)

print("Benchmarking WerSch_numba_v1 vs WerSch_numba_v2")
for i in range(len(NSUBs)):
    Verts, Faces = meshes[i]
    Nf = NFs[i]
    for j in range(len(NSUBs)):
        P = Ps[j]
        Np = NPs[j]
        sum_ij = i + j
        if sum_ij <= 4:
            n_runs = 500
        elif sum_ij <= 8:
            n_runs = 50
        elif sum_ij <= 10:
            n_runs = 5
        else:
            n_runs = 1
        print(f"NF={Nf:8d}, NP={Np:8d}, runs={n_runs:3d} ... ")
        
        # Time v1 multiple runs
        times_v1 = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            V1, gx1, gy1, gz1, Txx1, Txy1, Txz1, Tyy1, Tyz1, Tzz1 \
                = WerSch_numba_v1(P, Verts, Faces, rho)
            times_v1.append(time.perf_counter() - t0)
        if n_runs > 1:
            tc_v1[i, j] = np.mean(times_v1[1:])  # exclude first run
        else:
            tc_v1[i, j] = times_v1[0]
        
        # Time v2 multiple runs
        times_v2 = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            V2, gx2, gy2, gz2, Txx2, Txy2, Txz2, Tyy2, Tyz2, Tzz2 \
                = WerSch_numba_v2(P, Verts, Faces, rho)
            times_v2.append(time.perf_counter() - t0)
        if n_runs > 1:
            tc_v2[i, j] = np.mean(times_v2[1:])  # exclude first run
        else:
            tc_v2[i, j] = times_v2[0]
        
        # Max abs diff gz (compute once)
        MAD_gz[i, j] = np.max(np.abs(gz2 - gz1))
        MAD_Tzz[i, j] = np.max(np.abs(Tzz2 - Tzz1))
# %%
# # ! Print results
def print_table(title, data, fmt='8.1e'):
    print(f"\n{title}:")
    print("   NF\\NP", end=" ")
    for j in NSUBs:
        print(f"{NPs[j]:8d}", end=" ")
    print()
    for i in NSUBs:
        print(f"{NFs[i]:8d}", end=" ")
        for j in NSUBs:
            print(f"{data[i,j]:{fmt}}", end=" ")
        print()

print_table("Time WerSch_v1 (s)", tc_v1)
print_table("Time WerSch_v2 (s)", tc_v2)
print_table("t2/t1 Ratio", tc_v2 / tc_v1, fmt='8.2f')
print_table(f"Max Abs Diff gz (mGal), with Ref. std {np.std(gz1):.2f} mGal", 
            MAD_gz, fmt='8.2e')
print_table(f"Max Abs Diff Tzz (Eotvos), with Ref. std {np.std(Tzz1):.2f} Eotvos", 
            MAD_Tzz, fmt='8.2e')
# %%
# # ! Plotting
fig, axes = plt.subplots(2, 2, figsize=(9, 8))

# Time v1
X, Y = np.meshgrid(NSUBs, NSUBs)
im1 = axes[0,0].pcolormesh(X, Y, tc_v1, 
                           shading='nearest',
                           cmap='turbo', 
                           norm=LogNorm())
axes[0,0].set_xlabel(r'Obs level $i$: $N_P = 20 \times 4^{i}$')
axes[0,0].set_ylabel(r'Mesh level $j$: $N_F = 20 \times 4^{j}$')
axes[0,0].set_xticks(NSUBs)
axes[0,0].set_yticks(NSUBs)
plt.colorbar(im1, ax=axes[0,0])

# Time v2
im2 = axes[0,1].pcolormesh(X, Y, tc_v2, 
                           shading='nearest',
                           cmap='turbo', 
                           norm=LogNorm())
axes[0,1].set_xlabel(r'Obs level $i$: $N_P = 20 \times 4^{i}$')
axes[0,1].set_ylabel(r'Mesh level $j$: $N_F = 20 \times 4^{j}$')
axes[0,1].set_xticks(NSUBs)
axes[0,1].set_yticks(NSUBs)
plt.colorbar(im2, ax=axes[0,1])

# Ratio t2/t1
ratio = tc_v2 / tc_v1
vmin, vmax = np.min(ratio), np.max(ratio)
norm_ratio = TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)
im3 = axes[1,0].pcolormesh(X, Y, ratio, 
                           shading='nearest',
                           cmap='RdBu_r', 
                           norm=norm_ratio)
axes[1,0].set_xlabel(r'Obs level $i$: $N_P = 20 \times 4^{i}$')
axes[1,0].set_ylabel(r'Mesh level $j$: $N_F = 20 \times 4^{j}$')
axes[1,0].set_xticks(NSUBs)
axes[1,0].set_yticks(NSUBs)
plt.colorbar(im3, ax=axes[1,0])

# Max abs diff gz
im4 = axes[1,1].pcolormesh(X, Y, MAD_gz, 
                           shading='nearest',
                           cmap='viridis', 
                           norm=LogNorm())
axes[1,1].set_xlabel(r'Obs level $i$: $N_P = 20 \times 4^{i}$')
axes[1,1].set_ylabel(r'Mesh level $j$: $N_F = 20 \times 4^{j}$')
axes[1,1].set_xticks(NSUBs)
axes[1,1].set_yticks(NSUBs)
plt.colorbar(im4, ax=axes[1,1])

[ax.text(-0.15, 1.05, label, transform=ax.transAxes, 
         fontsize=14, fontweight='bold', va='top') 
         for ax, label in zip(axes.flat, ['(a)', '(b)', '(c)', '(d)'])]

plt.tight_layout(pad=2.5)
plt.savefig(f"Benchmark_WerSch_v1v2_nsub{nsub_max}.png", dpi=300, bbox_inches='tight')
# plt.show()
# %% 
# # ! End time
print("="*80)
print(f"End time: [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
print("="*80)