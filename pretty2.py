import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import healpy as hp
import cmasher as cmr
from matplotlib.patches import Ellipse, ConnectionPatch

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = 'true'
plt.rcParams['ytick.minor.visible'] = 'true'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.minor.width'] = 0.75
plt.rcParams['ytick.minor.width'] = 0.75
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['xtick.minor.size'] = 4.
plt.rcParams['ytick.minor.size'] = 4.
plt.rcParams['xtick.top']   = True
plt.rcParams['ytick.right'] = True
mpl.rcParams['lines.dotted_pattern'] = 1., 2.5

# Use default Latex Fonts
# Fonts
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'CMU Serif'
mpl.rcParams['font.sans-serif'] = 'CMU Sans Serif, DejaVu Sans, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif'

fname = 'pretty.hdf5'

fig = plt.figure(figsize=(10,5))

boxes = [23, 256]

mins = []
maxs = []

cmap_name = 'cmr.dusk'
cmap = cmr.get_sub_cmap(cmap_name, 0.15, 1.0)
cmap_cb = cmr.get_sub_cmap(cmap_name, 0.15, 1.0)
cmap.set_under('white')
cmap.set_bad('white')

gs = GridSpec(1, 3, width_ratios=[1, 1, 0.03], wspace=0.05)

shift = 0.013
main_axes = []

with h5py.File(fname, 'r') as f:
    for index, box in enumerate(boxes):
        ## Load in data from boxes
        this_box = f[f'box_{box:04d}']
        js = np.log10(np.array(this_box['s_wave']))
        nside = this_box['nside'][...]

        if index == 0:
            mins.append(20.75)
            maxs.append(np.max(js))

        ## use healpy to get map (and inner region of map)
        plt.figure(fig.number)
        hp.mollview(js, cmap=cmap, title='',
                    fig=fig, sub=(1, 3, 1+index), min=mins[0], max=maxs[0], cbar=False,
                    bgcolor='white')
        hp.graticule(color='lightgray', alpha=0.5)

        ax_main = fig.axes[-1]
        main_axes.append(ax_main)

        d = 10
        lon0, lat0 = 0, 0

        lons = np.concatenate([
            np.linspace(lon0 - d, lon0 + d, 20),
            np.repeat(lon0 + d, 20),
            np.linspace(lon0 + d, lon0 - d, 20),
            np.repeat(lon0 - d, 20)
        ])
        lats = np.concatenate([
            np.repeat(lat0 - d, 20),
            np.linspace(lat0 - d, lat0 + d, 20),
            np.repeat(lat0 + d, 20),
            np.linspace(lat0 + d, lat0 - d, 20)
        ])

        plt.sca(ax_main)
        hp.projplot(lons, lats, lonlat=True, color='k', linewidth=1.2, linestyle='-')

        fig_tmp = plt.figure() ## make a dummy figure to trick healpy
        proj_map = hp.cartview(
            js,
            lonra=[lon0 - d, lon0 + d],
            latra=[lat0 - d, lat0 + d],
            xsize=400,
            return_projected_map=True,
            cmap='cmr.dusk',
            cbar=False,
            notext=True,
            fig=fig_tmp,
            title=''
        )
        plt.close(fig_tmp)
        plt.figure(fig.number) ## switch back to regular figure

        ## Get inner few degrees and brightest pixel
        radius_rad = np.radians(d * np.sqrt(2))
        center_theta = np.radians(90 - lat0)
        center_phi = np.radians(lon0)
        vec = hp.ang2vec(center_theta, center_phi)

        pix_in_region = hp.query_disc(nside, vec, radius_rad)

        thetas, phis = hp.pix2ang(nside, pix_in_region)
        lats = 90 - np.degrees(thetas)
        lons = np.degrees(phis)
        lons = (lons + 180) % 360 - 180

        mask = (lats >= lat0 - d) & (lats <= lat0 + d) & \
               (lons >= lon0 - d) & (lons <= lon0 + d)

        box_pix = pix_in_region[mask]

        values_in_box = js[box_pix]
        # Find the index of the max value relative to the subset
        local_max_idx = np.argmax(values_in_box)
        # Map back to the actual HEALPix pixel index
        pixmax = box_pix[local_max_idx]

        theta, phi = hp.pix2ang(nside, pixmax)
        lat_max = np.degrees(0.5 * np.pi - theta)
        lon_max = np.degrees(phi)

        ## Plot an ellipse around the figure
        ellipse = Ellipse((0.5, 0.5), width=1.0, height=1.0,
                          transform=ax_main.transAxes,
                          fill=False,
                          edgecolor='black',
                          linewidth=1,
                          zorder=10)
        ax_main.add_patch(ellipse)

        ## Add inset plot
        left_offset = index * 0.33      # 0.0 for first panel, 0.5 for second
        inset_x = -0.05 + left_offset + (shift if index == 1 else 0)    # left edge of inset in figure coords
        inset_y = 0.18                 # bottom edge
        inset_w = 0.275                # width
        inset_h = 0.275                # height

        ax_inset = fig.add_axes([inset_x, inset_y, inset_w, inset_h])
        vmaxs = [25.75,25.4]
        ax_inset.imshow(
            proj_map,
            origin='lower',
            cmap=cmap,
            # vmin=24.7-index*0.75,
            vmax=vmaxs[index],
            extent=[-d, d, -d, d]
        )
        ax_inset.axhline(0.0, color='white',alpha=0.5)
        ax_inset.axvline(0.0, color='white',alpha=0.5)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('k')
            spine.set_linewidth(1.2)

        ax_inset.scatter( lat_max, lon_max, marker='+', color='k', s=100 )

        ax_inset.text(-0.1,0.5,r'$20^\circ$',transform=ax_inset.transAxes, rotation=90, va='center', ha='center', color='k', fontsize=12)
        ax_inset.text(0.5,-0.1,r'$20^\circ$',transform=ax_inset.transAxes, va='center', ha='center', color='k', fontsize=12)

        proj = hp.projector.MollweideProj()

        x_tl, y_tl = proj.ang2xy(lon0 + d, lat0 + d, lonlat=True)
        x_br, y_br = proj.ang2xy(lon0 - d, lat0 - d, lonlat=True)

        con_tl = ConnectionPatch(
            xyA=(0, 1), coordsA=ax_inset.transAxes,          # inset top-left (axes coords)
            xyB=(x_tl, y_tl), coordsB=ax_main.transData,     # map top-left (data coords)
            color='k', linewidth=1.2, linestyle=':', zorder=5
        )
        fig.add_artist(con_tl)

        con_br = ConnectionPatch(
            xyA=(1, 0), coordsA=ax_inset.transAxes,          # inset bottom-right (axes coords)
            xyB=(x_br, y_br), coordsB=ax_main.transData,     # map bottom-right (data coords)
            color='k', linewidth=1.2, linestyle=':', zorder=5
        )
        fig.add_artist(con_br)

pos = main_axes[1].get_position()
main_axes[1].set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])

axs = np.array(fig.axes)
for ax in fig.axes: ## rasterize Figure
    for im in ax.get_images():
        im.set_rasterized(True)

## add colorar
cax_s = fig.add_axes([0.7, 0.15, 0.015, 0.50])

sm_s = mpl.cm.ScalarMappable(cmap=cmap_cb,
                             norm=mpl.colors.Normalize(vmin=mins[0], vmax=maxs[0]))
sm_s.set_array([])

cb_s = fig.colorbar(sm_s, cax=cax_s, extend='both', extendfrac=0.05)

cb_s.set_label(r'$\log({\rm d}{\rm J}/{\rm d}\Omega~{\rm [GeV^2\,cm^{-5}]})$', labelpad=20, rotation=270)

cb_s.ax.xaxis.set_label_position('top')

#plt.tight_layout()
plt.subplots_adjust(hspace=0.0,wspace=-0.2,right=0.5)
plt.savefig(f'./figs/pretty.pdf',bbox_inches='tight')