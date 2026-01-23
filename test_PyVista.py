import numpy as np
import pyvista as pv

# --- 可视化：表面积分布 ---
Ns = np.array([2, 3, 4, 5])
pl = pv.Plotter(shape=(2, 2), window_size=[1200, 1000])

# 预计算所有网格以获取全局面积范围（用于统一 colorbar）
all_areas = []
meshes = []
for n in Ns:
    icosph = pv.Icosphere(nsub=n, radius=1.0)  # <-- 单位球
    icoArea = icosph.compute_cell_sizes()
    areas = icoArea['Area']
    all_areas.append(areas)
    meshes.append(icoArea)

global_min = min(a.min() for a in all_areas)
global_max = max(a.max() for a in all_areas)

for iN, (n, mesh) in enumerate(zip(Ns, meshes)):
    pl.subplot(iN // 2, iN % 2)
    sargs = dict(
        title=f'Cell Area (nsub={n})',
        title_font_size=12,
        label_font_size=10,
        n_labels=3,
        italic=True,
        fmt='%.6f',
        font_family='arial',
        width=0.8,
        position_x=0.1
    )
    pl.add_mesh(
        mesh,
        scalars='Area',
        scalar_bar_args=sargs,
        clim=[global_min, global_max],  # 统一色阶便于比较
        show_edges=True
    )
pl.show()

# --- 数值收敛：体积与表面积 ---
print(f"{'Unit sphere volume:':<24} {4*np.pi/3:12.6f}")
print(f"{'Unit sphere area:':<24} {4*np.pi:12.6f}")
print("-" * 50)

Ns_full = np.arange(2, 10)
volumes = []
areas = []

for n in Ns_full:
    icosph = pv.Icosphere(nsub=n, radius=1.0)  # 单位球
    volumes.append(icosph.volume)
    areas.append(icosph.area)

for n, vol, area in zip(Ns_full, volumes, areas):
    print(f"Icosphere (N={n:<2}) → Vol: {vol:12.6f}, Area: {area:12.6f}")