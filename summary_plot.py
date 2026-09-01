import numpy as np
import h5py
import healpy as hp
import tutorial

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

from skimage.measure import EllipseModel
from skimage._shared.utils import FailedEstimationAccessError
import cmasher as cmr
import dreams_python 

base_dir = '/scratch/aku7cf/'

factor = 's_wave'
maps = {
    's_wave': 'cmr.dusk',
    'p_wave': 'cmr.ember',
    'd_wave': 'cmr.freeze',
    'd_factor': 'cmr.sepia'
}

runs = np.arange(1024)
# fname = f'{base_dir}/CDM_hydro_unrotated.hdf5'
fname = f'{base_dir}/healpix_gemini_cubic_spline.hdf5'
# fname = f'{base_dir}/healpix_gemini_Nbody.hdf5'

offset_lat, offset_lon = [], []
centroid_lat, centroid_lon = [], []
all_maps = []

xs, ys = [], []
a1s, b1s = [], []
ell_thetas = []

for index, run in enumerate(runs):
    print(run)

    if ('Nbody' in fname) and (run == 796):
        print('\tskip')
        all_maps.append( np.ones((500,500)) * np.nan )
        offset_lat.append(np.nan)
        offset_lon.append(np.nan)
        centroid_lat.append(np.nan)
        centroid_lon.append(np.nan)

        (xc, yc) = np.nan, np.nan
        (a_len, b_len) = np.nan, np.nan
        theta = np.nan

        xs.append(xc); ys.append(yc)
        a1s.append(a_len); b1s.append(b_len)
        ell_thetas.append(theta)
        continue

    my_map = None
    with h5py.File(fname, 'r') as f:
        this_group = f[f'box_{run:04d}']

        my_map = np.array(this_group[factor])

        nside = np.array(this_group['nside'])
    lat0, lon0 = 0, 0
    d = 10

    radius_rad = np.radians(d * np.sqrt(2)) # Radius to cover the square
    center_theta = np.radians(90 - lat0)
    center_phi = np.radians(lon0)
    vec = hp.ang2vec(center_theta, center_phi)

    pix_in_region = hp.query_disc(nside, vec, radius_rad)

    thetas, phis = hp.pix2ang(nside, pix_in_region)
    lats = 90 - np.degrees(thetas)
    lons = np.degrees(phis)
    # Adjust lons for wrap-around if necessary
    lons = (lons + 180) % 360 - 180 
    
    mask = (lats >= lat0 - d) & (lats <= lat0 + d) & \
           (lons >= lon0 - d) & (lons <= lon0 + d)
    
    box_pix = pix_in_region[mask]

    values_in_box = my_map[box_pix]
    # Find the index of the max value relative to the subset
    local_max_idx = np.argmax(values_in_box)
    # Map back to the actual HEALPix pixel index
    pixmax = box_pix[local_max_idx]

    theta, phi = hp.pix2ang(nside, pixmax)
    lat_max = np.degrees(0.5 * np.pi - theta)
    lon_max = np.degrees(phi)
    lon_max = (lon_max + 180) % 360 - 180
    
    offset_lat.append(-lat_max)
    offset_lon.append(-lon_max)

    #### centroid ####

    box_lats = lats[mask]
    box_lons = lons[mask]
    
    top_pct = 1.0   # top pixels by value
    thresh_val = np.percentile(values_in_box, 100.0 - top_pct)
    top_mask = values_in_box >= thresh_val

    vals_top = values_in_box[top_mask]
    lats_top = box_lats[top_mask]
    lons_top = box_lons[top_mask]

    w = vals_top / vals_top.sum()
    lat_cent = float(np.sum(w * lats_top))
    lon_cent = float(np.sum(w * lons_top))
    # lat_cent = float(np.mean(box_lats[top_mask]))
    # lon_cent = float(np.mean(box_lons[top_mask]))

    #### done centroid ####
    
    centroid_lat.append(-lat_cent)
    centroid_lon.append(-lon_cent)

    projected_map = hp.cartview(
        my_map,
        coord=None,
        lonra=[lon0 - d, lon0 + d],
        latra=[lat0 - d, lat0 + d],
        xsize=500,
        title="",
        return_projected_map=True,
        notext=True,
        cbar=False,
    )

    # if index <= 16:
    all_maps.append(projected_map)

    peak = projected_map.max()

    level_70 = np.log10(0.7 * peak)

    contours = plt.contour(
        np.log10(projected_map),
        levels=[level_70],
        colors='white',
        linewidths=0.6,
        extent=(-d,d,-d,d),
        origin='upper'
    )
    
    for seg in contours.allsegs[0]:
        x = seg[:, 0]
        y = seg[:, 1]

        try:
            model = EllipseModel.from_estimate(seg) ## fit ellipse
            (xc, yc) = model.center
            (a_len, b_len) = model.axis_lengths
            theta = model.theta ## Radians
        except FailedEstimationAccessError:
            print('\t didnt converge')
            (xc, yc) = np.nan, np.nan
            (a_len, b_len) = np.nan, np.nan
            theta = np.nan

        xs.append(xc); ys.append(yc)
        a1s.append(a_len); b1s.append(b_len)
        ell_thetas.append(theta)
        break

    plt.close()

log_stellar_masses = np.load('stellar_masses.npy')[:len(runs)]
sm_min = 10.6
sm_max = 10.78

mask = (log_stellar_masses > sm_max) # & (log_stellar_masses < sm_max)

fig = plt.figure(figsize=(11, 5))
gs = gridspec.GridSpec(4, 10, height_ratios=[1.4, 1, 1, 2],
                       width_ratios=[1, 1, 0.4, 1, 1, 1, 1, 0.4, 1, 1])

colors = cmr.take_cmap_colors('cmr.dusk', 10)

center_subspec = gs[0:4, 3:7]
inner_gs = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=center_subspec,
                                            wspace=0.0, hspace=0.0)
axs_center = np.empty((4, 4), dtype=object)
for i in range(4):
    for j in range(4):
        axs_center[i, j] = fig.add_subplot(inner_gs[i, j])

ax_left = fig.add_subplot(gs[0:2, 0:2])
ax_right = fig.add_subplot(gs[0:2, 8:10])

ax_left_bot = fig.add_subplot(gs[3, 0:2])
ax_right_bot = fig.add_subplot(gs[3, 8:10])

for ax in axs_center.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

n_levels = 10
for i in range(16):
    ax = axs_center.flatten()[i]

    this_map = np.log10(all_maps[i])
    ax.imshow( this_map, cmap='cmr.dusk', rasterized=True, extent=(-d,d,-d,d), aspect='equal' )

    peak = 10**this_map.max()

    levels = np.log10([
        0.1 * peak,
        0.3 * peak,
        0.5 * peak,
        0.7 * peak,
        0.9 * peak,
    ])
    
    ax.contour(
        this_map,
        levels=levels,
        colors='white',
        linewidths=0.6,
        extent=(-d,d,-d,d),
        origin='upper'
    )

    # ax.scatter( 0, 0, color='dodgerblue', marker='+', alpha=1  )
    ax.axhline(0,color='white',ls='--',alpha=0.5,lw=0.5)
    ax.axvline(0,color='white',ls='--',alpha=0.5,lw=0.5)
    ax.scatter( offset_lon[i], offset_lat[i], color='k', marker='+'  )
    # ax.scatter( centroid_lon[i], centroid_lat[i], color='k', marker='+'  )

    # xc, yc = xs[i], ys[i]
    # a_len = a1s[i]
    # b_len = b1s[i]
    # theta = ell_thetas[i]
    
    # ell = Ellipse(
    #     xy=(xc, yc),
    #     width=2*a_len,
    #     height=2*b_len,
    #     angle=np.rad2deg(theta),
    #     edgecolor='red',
    #     facecolor='none',
    #     lw=1.5,
    # )
    
    # ax.add_patch(ell)
    

avg_map = np.log10(np.nanmean(all_maps, axis=0))
ax_left.imshow( avg_map, cmap='cmr.dusk', rasterized=True, extent=(-d,d,-d,d), aspect='equal', origin='lower', alpha=0.75, zorder=-1 )
ax_left.set_xlabel(r'${\rm Longitude}$', fontsize=16)
ax_left.set_ylabel(r'${\rm Latitude}$', fontsize=16)

# levels = np.linspace(avg_map.min(),avg_map.max(),n_levels)
peak = 10**avg_map.max()

levels = np.log10([
    0.1 * peak,
    0.3 * peak,
    0.5 * peak,
    0.7 * peak,
    0.9 * peak,
])

contours = ax_left.contour(
    avg_map,
    levels=levels,
    colors='k',
    linewidths=2.2,
    alpha=1,
    extent=(-d, d, -d, d),
    zorder=5
)

for index, level_seg in enumerate(contours.allsegs):
    if index == 3:
        for seg in level_seg:
            x = seg[:, 0]
            y = seg[:, 1]
        
            model = EllipseModel.from_estimate(seg) ## fit ellipse
        
            (xc, yc) = model.center
            (a_len, b_len) = model.axis_lengths
            theta = model.theta ## Radians

qs_all = np.array( b1s ) / np.array( a1s )

bins = np.arange(0.3,1.0,0.025)
ax_left_bot.hist(qs_all[mask], color=colors[3], bins=bins, rasterized=True)

# np.save('q2d.npy',qs_all)

print('-- 2d Shape --')
print(f'\tMedian : {np.nanmedian(qs_all):0.3f}')
print(f'\t16th   : {np.nanpercentile(qs_all,16):0.3f}')
print(f'\t84th   : {np.nanpercentile(qs_all,84):0.3f}')
print()
print(f'\tAvg Map: {b_len/a_len:0.3f}')
print()

ax_left_bot.axvline( b_len/a_len, color=colors[6], ls='--', lw=3 )
ax_left_bot.set_xlabel(r'$q_{\rm 2D}$', fontsize=12)

ax_right.imshow( avg_map, cmap=plt.cm.Greys, rasterized=True,
                 extent=(-d,d,-d,d), aspect='equal', vmin=np.max(avg_map), origin='lower' )
offset_lon, offset_lat = np.array(offset_lon), np.array(offset_lat)
centroid_lon, centroid_lat = np.array(centroid_lon), np.array(centroid_lat)

ax_right.scatter( offset_lon[mask], offset_lat[mask], color=colors[6], marker='+'  )

ax_right.yaxis.tick_right()
ax_right.yaxis.set_label_position("right")
ax_right.tick_params(
    axis="y", left=True, right=True, labelleft=False, labelright=True, which='both'
)

ax_right.set_xlabel(r'${\rm Longitude}$', fontsize=16)
ax_right.set_ylabel(r'${\rm Latitude}$', fontsize=16)

bins = np.linspace(-10,10,30)

print('-- Offsets --')
print(f'\tLongitude Median: {np.nanmedian(offset_lon):0.3f}')
print(f'\tLongitude 16th  : {np.nanpercentile(offset_lon,16):0.3f}')
print(f'\tLongitude 84th  : {np.nanpercentile(offset_lon,84):0.3f}')
print()
print(f'\tLatitude Median : {np.nanmedian(offset_lat):0.3f}')
print(f'\tLatitude 16th   : {np.nanpercentile(offset_lat,16):0.3f}')
print(f'\tLatitude 84th   : {np.nanpercentile(offset_lat,84):0.3f}')
print()
print('-- Centroid --')
print(f'\tLongitude Median: {np.nanmedian(centroid_lon):0.3f}')
print(f'\tLongitude 16th  : {np.nanpercentile(centroid_lon,16):0.3f}')
print(f'\tLongitude 84th  : {np.nanpercentile(centroid_lon,84):0.3f}')
print()
print(f'\tLatitude Median : {np.nanmedian(centroid_lat):0.3f}')
print(f'\tLatitude 16th   : {np.nanpercentile(centroid_lat,16):0.3f}')
print(f'\tLatitude 84th   : {np.nanpercentile(centroid_lat,84):0.3f}')
print()

ax_right_bot.hist( offset_lon, bins=bins, color=colors[3], alpha=0.8, rasterized=True )
ax_right_bot.hist( offset_lat, bins=bins, color=colors[6], alpha=0.8, rasterized=True )
ax_right_bot.set_yscale('log')
ax_right_bot.set_xlabel(r'${\rm Offset~[degrees]}$',fontsize=12)

ax_right_bot.text(0.05,0.85,r'${\rm Longitude}$',transform=ax_right_bot.transAxes,color=colors[3],fontsize=12)
ax_right_bot.text(0.05,0.75,r'${\rm Latitude}$',transform=ax_right_bot.transAxes,color=colors[6],fontsize=12)

# ax_right.text(0.5,1.05,r'${\rm Offsets}$', transform=ax_right.transAxes,ha='center')

ax_right.axhline(0, color='gray', ls=':', alpha=0.5)
ax_right.axvline(0, color='gray', ls=':', alpha=0.5)

for ax in [ax_left, ax_right]:
    ax.tick_params(labelsize=12)

    xticks = ax.get_xticklabels()
    yticks = ax.get_yticklabels()
    ax.set_xticklabels([r'$%s^\circ$' %i.get_text().replace('$','') for i in xticks])
    ax.set_yticklabels([r'$%s^\circ$' %i.get_text().replace('$','') for i in yticks])

for ax in [ax_left_bot, ax_right_bot]:
    ax.tick_params(labelsize=12)

    if ax == ax_right_bot:
        xticks = ax.get_xticklabels()
        ax.set_xticklabels([r'$%s^\circ$' %i.get_text().replace('$','') for i in xticks])

plt.subplots_adjust(wspace=0.0, hspace=0.0)

name = './figs/summary_plot.pdf'
if 'Nbody' in fname:
    name = './figs/summary_plot_Nbody.pdf'
plt.savefig(name,bbox_inches='tight')