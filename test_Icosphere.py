##################################################################
# # ! 球体的多面体近似（正二十面体递归剖分法）
##################################################################
# %%
# # ! Setup
import pyvista as pv;
import numpy as np;
# %%
# # ! 正二十面体递归剖分法 --> 球
N  = 3; # 递归深度
icosahedron = pv.Icosahedron();
icosahedron.clear_data();  # remove extra scalars
icosahedron_sub = icosahedron.subdivide(nsub=N);
icosphere = pv.Icosphere(nsub=N);
pl = pv.Plotter(shape=(1, 3));
pl.subplot(0, 0);
_ = pl.add_mesh(icosahedron, show_edges=True);
pl.subplot(0, 1);
_ = pl.add_mesh(icosahedron_sub, show_edges=True);
pl.subplot(0, 2);
_ = pl.add_mesh(icosphere, show_edges=True);
pl.show();
# %%
# # ! 表面积变化
Ns = np.array([2,3,4,5]);
pl = pv.Plotter(shape=(2, 2));
for iN, n in enumerate(Ns):
    icosph = pv.Icosphere(nsub=n);
    icoArea = icosph.compute_cell_sizes();
    pl.subplot(iN//2, iN%2);
    sargs = dict(title=f'Area (nsub={n})', title_font_size=12,
                label_font_size=10, n_labels=3,
                italic=True, fmt='%.6f', font_family='arial',
                width=0.8, position_x=0.1);
    pl.add_mesh(icoArea, scalars='Area', scalar_bar_args=sargs);
pl.show();
# %%
# # ! 表面积与体积的递归逼近
Ns = np.arange(2,10,1);
print(f'{"Sphere volume:":<{24}} {4*np.pi/3:12.6f}', flush=True);
for iN, n in enumerate(Ns):
    icosph = pv.Icosphere(nsub=n);
    print(f'{"Icosphere volume:":<{20}} N={n} {icosph.volume:12.6f}', flush=True);
print(f'{"Sphere area:":<{24}} {4*np.pi:12.6f}', flush=True);
for iN, n in enumerate(Ns):
    icosph = pv.Icosphere(nsub=n);
    print(f'{"Icosphere area:":<{20}} N={n} {icosph.area:12.6f}', flush=True);