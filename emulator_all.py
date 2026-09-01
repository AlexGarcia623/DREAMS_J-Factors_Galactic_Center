import optuna
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28
plt.rcParams['axes.linewidth'] = 2.25*1.25
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = 'true'
plt.rcParams['ytick.minor.visible'] = 'true'
plt.rcParams['xtick.major.width'] = 1.5*1.25
plt.rcParams['ytick.major.width'] = 1.5*1.25
plt.rcParams['xtick.minor.width'] = 1.0*1.25
plt.rcParams['ytick.minor.width'] = 1.0*1.25
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4.5
plt.rcParams['ytick.minor.size'] = 4.5
plt.rcParams['xtick.top']   = True
plt.rcParams['ytick.right'] = True

import dreams_python
from dreams_python import emulation

offsets = np.load('max_density_distance.npy')
stellar_masses = np.load('stellar_masses.npy')

mw_cdm = dreams_python.DREAMS('/standard/DREAMS',suite='MW_zooms',DM_type='CDM',sobol_number=5)
params, header = mw_cdm.read_param_file('CDM_TNG_MW_SB5.txt')

radii = np.arange(1, 5.0, 0.5)
radial_bin = 3
hydro_a = np.load(f'./data/radial_a.npy')[:,radial_bin]
hydro_b = np.load(f'./data/radial_b.npy')[:,radial_bin]
hydro_c = np.load(f'./data/radial_c.npy')[:,radial_bin]

q_3d = hydro_b / hydro_a
s_3d = hydro_c / hydro_a

q_2d = np.load('q2d.npy')

mask = (offsets < 2) & (stellar_masses > 5) & ~(np.isnan(q_3d)) & ~(np.isnan(s_3d)) & ~(np.isnan(q_2d))

stellar_masses = stellar_masses[ mask ]
offsets = offsets[ mask ]
q_3d = q_3d[ mask ]
s_3d = s_3d[ mask ]
q_2d = q_2d[ mask ]

features = np.array([
    params[mask, 0],
    params[mask, 1],
    np.log10(params[mask, 2]), ## sn1, sn2, and AGN are log spaced
    np.log10(params[mask, 3]),
    np.log10(params[mask, 4])
]).T
labels = np.column_stack([
    stellar_masses,
    offsets,
    q_3d,
    s_3d,
    q_2d
])

name = 'everything_emulator_1'

omega_m_min, omega_m_max     = params[:, 0].min(), params[:, 0].max()
sigma_8_min, sigma_8_max     = params[:, 1].min(), params[:, 1].max()
sn1_min,     sn1_max         = np.log10(params[:, 2].min()), np.log10(params[:, 2].max())
sn2_min,     sn2_max         = np.log10(params[:, 3].min()), np.log10(params[:, 3].max())
agn1_min,    agn1_max        = np.log10(params[:, 4].min()), np.log10(params[:, 4].max())

flow = emulation.emulator(features, labels, name)

fid_omega_m = 0.31
fid_sigma_8 = 0.8159

fig, all_axs = plt.subplots(5, 3, figsize=(12, 12.5),sharey='row',sharex='col')

def median(x, y, dx=0.05):
    filter_infs = np.isfinite(y)
    
    x = x[filter_infs]
    y = y[filter_infs]
    
    xs = np.arange(np.min(x), np.max(x), dx)
    ys      = np.zeros_like(xs)
    ys_lo   = np.zeros_like(xs)
    ys_hi   = np.zeros_like(xs)
    
    for index, xval in enumerate(xs):
        within_dx = (x > xval) & (x < xval + dx)
        if sum(within_dx) < 10:
            ys[index]    = np.nan
            ys_lo[index] = np.nan
            ys_hi[index] = np.nan
        else:
            ys[index]    = np.nanmedian(y[within_dx])
            ys_lo[index] = np.nanpercentile(y[within_dx], 16)
            ys_hi[index] = np.nanpercentile(y[within_dx], 84)
    
    return xs, ys, ys_lo, ys_hi

n_samples = 50_000
n_per_point = 100
colors = cmr.take_cmap_colors('cmr.dusk', 10)
colors = [colors[2], colors[4], colors[6]]
var_labels = [r'$\log(\bar{e}_w)$',r'$\log(\kappa_w)$',r'$\log(\epsilon_{f,\,{\rm high}})$']
mult = [3.6, 7.4, 0.1]
for jjj in range(3):
    
    print(f'Generating {n_samples*n_per_point:,} samples ({n_samples:,} total with {n_per_point:,} redundancy)')
    if jjj == 0:
        x = np.log10(np.logspace(sn1_min,sn1_max,n_samples))
        context = np.column_stack([
            np.full(n_samples, fid_omega_m),
            np.full(n_samples, fid_sigma_8),
            x,
            np.full(n_samples, 0),
            np.full(n_samples, 0),
        ])
    elif jjj == 1:
        x = (np.log10(np.logspace(sn2_min,sn2_max,n_samples)))
        context = np.column_stack([
            np.full(n_samples, fid_omega_m),
            np.full(n_samples, fid_sigma_8),
            np.full(n_samples, 0),
            x,
            np.full(n_samples, 0),
        ])
    else:
        x = (np.log10(np.logspace(agn1_min,agn1_max,n_samples)))
        context = np.column_stack([
            np.full(n_samples, fid_omega_m),
            np.full(n_samples, fid_sigma_8),
            np.full(n_samples, 0),
            np.full(n_samples, 0),
            x
        ])
        
    posterior_draws = flow.make_prediction(context, n_samples=n_per_point)
    
    n_points = posterior_draws.shape[1]
    chosen = np.random.randint(0, n_per_point, size=n_points)
    ensemble_mean = posterior_draws[chosen, np.arange(n_points)]


    for iii in range(5):
        _x_, _y_, ys_lo, ys_hi = median(x, ensemble_mean[:, iii])

        _x_ = np.log10( 10**_x_ * mult[jjj] )
        
        ax = all_axs[iii, jjj]
        ax.plot(_x_, _y_, color=colors[jjj], lw=3)
        ax.fill_between(_x_, ys_lo, ys_hi, color=colors[jjj], alpha=0.5)

        ax.plot(_x_, ys_lo, color=colors[jjj], lw=1)
        ax.plot(_x_, ys_hi, color=colors[jjj], lw=1)


all_axs[4,0].set_xlabel(var_labels[0])
all_axs[4,1].set_xlabel(var_labels[1])
all_axs[4,2].set_xlabel(var_labels[2])

all_axs[1,0].axhline(0.441,color='gray',ls=':',alpha=0.75)
all_axs[1,1].axhline(0.441,color='gray',ls=':',alpha=0.75)
all_axs[1,2].axhline(0.441,color='gray',ls=':',alpha=0.75)
    
all_axs[0,0].set_ylabel(r'$\log(M_\star~[{\rm M}_\odot])$')
all_axs[1,0].set_ylabel(r'$d_{\rho\to\phi}~[{\rm kpc}]$')
all_axs[2,0].set_ylabel(r'$q_{\rm 3D}$')
all_axs[3,0].set_ylabel(r'$s_{\rm 3D}$')
all_axs[4,0].set_ylabel(r'$q_{\rm 2D}$')

plt.tight_layout()
plt.subplots_adjust(wspace=0.0, hspace=0.22)
plt.savefig('./figs/parameter_variations.pdf',bbox_inches='tight')
