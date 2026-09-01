import optuna
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True 
plt.rcParams['font.size'] = 16
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

mask = (offsets < 2) & (stellar_masses > 5)

stellar_masses = stellar_masses[ mask ]
offsets = offsets[ mask ]

features = np.array([
    params[mask, 0],
    params[mask, 1],
    np.log10(params[mask, 2]), ## sn1, sn2, and AGN are log spaced
    np.log10(params[mask, 3]),
    np.log10(params[mask, 4])
]).T
labels = np.column_stack([
    stellar_masses,
    offsets
])
name = '3d_offset_0'

omega_m_min, omega_m_max     = params[:, 0].min(), params[:, 0].max()
sigma_8_min, sigma_8_max     = params[:, 1].min(), params[:, 1].max()
sn1_min,     sn1_max         = np.log10(params[:, 2].min()), np.log10(params[:, 2].max())
sn2_min,     sn2_max         = np.log10(params[:, 3].min()), np.log10(params[:, 3].max())
agn1_min,    agn1_max        = np.log10(params[:, 4].min()), np.log10(params[:, 4].max())

flow = emulation.emulator(features, labels, name)

fid_omega_m = 0.31
fid_sigma_8 = 0.8159

fig, all_axs = plt.subplots(2, 3, figsize=(12, 5),sharey='row')

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

n_samples = 10_000
n_per_point = 100
colors = ['r','k','b']
var_labels = [r'$\log(\bar{e}_w)$',r'$\log(\kappa_w)$',r'$\log(\epsilon_{f,\,{\rm high}})$']
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

    _x_, _y_, ys_lo, ys_hi = median(x, ensemble_mean[:, 0])
    
    ax = all_axs[0, jjj]
    ax.plot(_x_, _y_, color=colors[jjj])
    ax.fill_between(_x_, ys_lo, ys_hi, color=colors[jjj], alpha=0.2)

    _x_, _y_, ys_lo, ys_hi = median(x, ensemble_mean[:, 1])
    
    ax = all_axs[1, jjj]
    ax.plot(_x_, _y_, color=colors[jjj])
    ax.fill_between(_x_, ys_lo, ys_hi, color=colors[jjj], alpha=0.2)
    ax.set_xlabel(var_labels[jjj])
    ax.axhline(0.441,color='gray',ls=':',alpha=0.75)
    
all_axs[0,0].set_ylabel(r'$\log(M_\star~[{\rm M}_\odot])$')
all_axs[1,0].set_ylabel(r'$d_{\rho\to\phi}~[{\rm kpc}]$')

plt.tight_layout()
plt.subplots_adjust(wspace=0.0)
plt.savefig('./figs/parameter_variations.pdf',bbox_inches='tight')
