import numpy as np

import matplotlib.pyplot as plt
import tutorial
import cmasher as cmr

colors = cmr.take_cmap_colors('cmr.dusk', 10)

tag = ''

print()
print(tag)
print()

radii = np.arange(1, 5.0, 0.5)
radial_bin = 3
hydro_a = np.load(f'./data/radial_a{tag}.npy')[:,radial_bin]
hydro_b = np.load(f'./data/radial_b{tag}.npy')[:,radial_bin]
hydro_c = np.load(f'./data/radial_c{tag}.npy')[:,radial_bin]

# import sys; sys.exit()

hydro_q = hydro_b / hydro_a
hydro_s = hydro_c / hydro_a
hydro_T = (1 - hydro_q**2) / (1 - hydro_s**2)

hydro = np.load(f'max_density_distance{tag}.npy')

# fig,axs = plt.subplots(2,1,figsize=(5,8))
fig,axs = plt.subplots(1,2,figsize=(8,4.5))
axs = axs.flatten()

bins = np.arange(0,1.5,0.05)
axs[0].hist(hydro[(hydro < 1.5)], bins=bins, color=colors[3], alpha=1, rasterized=True)

print('< 0.441 kpc')
print(sum(hydro < 0.441))

print()
print('d rho->phi')
print(np.median(hydro[(hydro < 1.5)]), np.percentile(hydro[(hydro < 1.5)], 84), np.percentile(hydro[(hydro < 1.5)], 16) )

axs[0].set_xlabel(r'$d_{{\rho_{\rm max}}\to{\Phi_{\rm min}}}~[{\rm kpc}]$')
axs[0].set_ylabel(r'${\rm Number~of~Halos}$')

axs[0].axvline(0.441, color='k', ls='--')
axs[0].text(0.32,0.7,r'$\varepsilon_{\rm soft}$',transform=axs[0].transAxes, color='k', rotation=90, fontsize=15)

bins = np.arange(0.4,1.0+0.02,0.04)
if tag == '_Nbody':
    bins = np.arange(0.1,1.0+0.02,0.04)
axs[1].hist(hydro_q, bins=bins, rasterized=True, color=colors[5], alpha=0.7)

print()
print('q, s, T')
print(np.nanmedian(hydro_q), np.nanpercentile(hydro_q, 84), np.nanpercentile(hydro_q, 16) )
print(np.nanmedian(hydro_s), np.nanpercentile(hydro_s, 84), np.nanpercentile(hydro_s, 16) )
print(np.nanmedian(hydro_T), np.nanpercentile(hydro_T, 84), np.nanpercentile(hydro_T, 16) )

print(hydro_q[np.nanargmin(hydro_s)])

axs[1].set_ylabel(r'${\rm Number~of~Halos}$')
# axs[1].set_xlabel(r'$q_{\rm 3D}~(%0.1f~{\rm kpc})$' %radii[radial_bin])
axs[1].set_xlabel(r'${\rm Axis~Ratio}~(%0.1f~{\rm kpc})$' %radii[radial_bin])

axs[1].hist(hydro_s, bins=bins, rasterized=True, color=colors[7], alpha=0.7)

axs[1].text(0.05,0.85,r'$q_{\rm 3D}$',transform=axs[1].transAxes,va='top',color=colors[5])
axs[1].text(0.05,0.75,r'$s_{\rm 3D}$',transform=axs[1].transAxes,va='top',color=colors[7])

no_nans = ~np.isnan(hydro_T)
mask = (hydro < 1.5) & (no_nans)

axs[1].axvline(1.0, color='k', ls='--')

ymin, ymax = axs[1].get_ylim()
axs[1].set_ylim(ymin, ymax*1.17)

axs[1].annotate(
    "",  # Keep text empty unless you want a label
    xy=(0.7, 0.85),  # Arrow head tip (ends at x=0.6, y=0.9)
    xytext=(0.3, 0.85),  # Arrow tail start (starts at x=0.4, y=0.9)
    xycoords="axes fraction",  # Coordinates are relative to the axis (0 to 1)
    arrowprops=dict(
        arrowstyle="->",  # Arrowhead style ('->', '-|>', 'fancy', etc.)
        lw=1,  # Line width
        color="black",  # Line color
    ),
)
axs[1].text(0.5,0.9,r'${\rm More~Spherical}$',transform=axs[1].transAxes, ha='center', fontsize=14)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig(f'./figs/3d_shape_compare{tag}.pdf',bbox_inches='tight')
