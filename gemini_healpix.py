#!/home/aku7cf/.conda/envs/py3/bin/python
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from tqdm import tqdm
import tutorial
from scipy.spatial import cKDTree
import galaxy_transform as gt
import cmasher as cmr

auxtag  = 'MW_zooms'
sim     = 'CDM'
sb      = 4 if sim == 'WDM' else 5
basedir = '/standard/DREAMS/'
snap_path  = basedir + f'Sims/{sim}/'+auxtag+f'/SB{sb}/'
group_path = basedir + f'FOF_Subfind/{sim}/'+auxtag+f'/SB{sb}/'

snapnr    = 90
h         = 0.6909
DesNgb    = 32
scf       = tutorial.get_scf(snap_path, snapnr, 0)

kpc2cm  = 3.086e+21 # cm
Msun2kg = 1.989e+30 # kg
kg2GeV  = 1.783e-27 # GeV

def cubic_spline_W_q(q):
    q = np.asarray(q)
    Wq = np.zeros_like(q)
    mask1 = (q >= 0) & (q <= 1)
    mask2 = (q > 1) & (q <= 2)
    Wq[mask1] = 1.0 - 1.5 * q[mask1]**2 + 0.75 * q[mask1]**3
    Wq[mask2] = 0.25 * (2.0 - q[mask2])**3
    return Wq

def top_hat_W_q(q):
    q = np.asarray(q)
    Wq = np.zeros_like(q)
    Wq[q <= 1] = 1.0
    return Wq


from scipy.spatial.distance import pdist
def get_3d_density_velocity(pos, vel, Nngb=32):
    N = pos.shape[0] 
    tree = cKDTree(pos)
    
    dists, idxs = tree.query(pos, k=Nngb)
    hsml = dists[:, -1].copy()
    
    N  = vel.shape[0]
    v2 = np.zeros(N)
    v4 = np.zeros(N)
    for i in range(N):
        neighbors = idxs[i]
        v = vel[neighbors]

        vals  = pdist(v)**2
        v2[i] = np.mean(vals)
        v4[i] = np.mean(vals**2)
        
    return v2, v4

def get_particle_rho_and_moments(pos, vel, mass, Nngb=32):
    """Calculates local density and velocity moments for each particle."""
    tree = cKDTree(pos)
    dists, idxs = tree.query(pos, k=Nngb)
    hsml = dists[:, -1] # Smoothing length
    
    # Calculate local density rho = sum(mass * kernel)
    # Using a simplified SPH density estimate: rho ~ M_total_ngb / Vol_ngb
    # For more precision, you could apply the cubic spline to the neighbors here.
    vol = (4.0/3.0) * np.pi * hsml**3
    rho = (mass[idxs].sum(axis=1)) / vol
    
    v2 = np.zeros(len(pos))
    v4 = np.zeros(len(pos))
    for i in tqdm(range(len(pos))):
        neighbors = idxs[i]
        dv = vel[neighbors] - vel[i] # Velocity dispersion relative to particle
        vals = np.sum(dv**2, axis=1)
        v2[i] = np.mean(vals)
        v4[i] = np.mean(vals**2)
        
    return rho, v2, v4, hsml

def project_particles_to_healpix(pos, mass, rho_p, v2, v4, hsml, nside):
    npix = hp.nside2npix(nside)
    # Initialize maps
    js, jp, jd, d_map = np.zeros(npix), np.zeros(npix), np.zeros(npix), np.zeros(npix)

    dist = np.linalg.norm(pos, axis=1)
    mask = dist > 0.01 
    pos, mass, rho_p, hsml, dist = pos[mask], mass[mask], rho_p[mask], hsml[mask], dist[mask]
    v2, v4 = v2[mask], v4[mask]

    # Calculate angular radius of each particle (in radians)
    ang_radii = hsml / dist
    
    # Calculate the approximate radius of a single HEALPix pixel
    pixel_radius = np.sqrt(hp.nside2pixarea(nside) / np.pi)

    # Pre-calculate weights to save time in the loop
    weight_js_all = (mass * rho_p) / (dist**2)
    weight_jp_all = (mass * rho_p * v2) / (dist**2)
    weight_jd_all = (mass * rho_p * v4) / (dist**2)
    weight_d_all  = mass / (dist**2)

    for i in tqdm(range(len(pos))):
        # 1. Check if the particle is smaller than a pixel
        if ang_radii[i] < pixel_radius:
            # Point-source approximation (Fast)
            pix = hp.vec2pix(nside, pos[i,0], pos[i,1], pos[i,2])
            js[pix] += weight_js_all[i]
            jp[pix] += weight_jp_all[i]
            jd[pix] += weight_jd_all[i]
            d_map[pix] += weight_d_all[i]
        else:
            # Kernel spreading (Smooths out dead pixels)
            # Find all pixels within 2*h (the SPH support radius)
            # hp.query_disc returns indices of pixels within a radius
            pix_indices = hp.query_disc(nside, pos[i], radius=2.0*ang_radii[i])
            
            if len(pix_indices) == 0:
                continue

            # Get vectors for these pixels to calculate exact angular distance
            pix_vecs = np.array(hp.pix2vec(nside, pix_indices)).T
            # Angular distance between particle center and pixel centers
            # Using dot product for speed: cos(theta) = A.B / (|A||B|)
            cos_theta = np.dot(pix_vecs, pos[i]) / dist[i]
            cos_theta = np.clip(cos_theta, -1, 1)
            ang_dists = np.arccos(cos_theta)

            # Apply Kernel: W(q) where q = angular_dist / angular_hsml
            q = ang_dists / ang_radii[i]
            weights = cubic_spline_W_q(q) # Or cubic_spline_W_q
            
            # Normalize kernel weights so we don't create/destroy mass
            if np.sum(weights) > 0:
                weights /= np.sum(weights)
                
                js[pix_indices]   += weight_js_all[i] * weights
                jp[pix_indices]   += weight_jp_all[i] * weights
                jd[pix_indices]   += weight_jd_all[i] * weights
                d_map[pix_indices]+= weight_d_all[i] * weights

    omega_pix = hp.nside2pixarea(nside)
    return js/omega_pix, jp/omega_pix, jd/omega_pix, d_map/omega_pix

def process_box(box, fname):
    print(f'Processing box {box}')
    path   = f'{snap_path}/box_{box}/snap_{snapnr:03}.hdf5'
    dm_cat = tutorial.load_particle_data(path, ['Masses', 'Coordinates', 'Velocities'], [1, 2])
    star_cat = tutorial.load_particle_data(path, ['Masses', 'Coordinates', 'Velocities', 'GFM_StellarFormationTime'], [4])
    
    path = f'{group_path}/box_{box}/fof_subhalo_tab_{snapnr:03}.hdf5'
    keys = ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 'GroupMassType',
            'GroupPos', 'SubhaloLenType', 'SubhaloGrNr', 'GroupVel', 'Group_R_Crit200']
    grp_cat = tutorial.load_group_data(path, keys)
    
    mw_idx = tutorial.get_MW_idx(grp_cat) 
    _, fof_cat = tutorial.get_galaxy_data(dm_cat, grp_cat, mw_idx)
    
    dm_mass      = dm_cat['PartType1/Masses'] * 1.00E+10 / h
    dm_coords    = dm_cat['PartType1/Coordinates'] * scf / h
    dm_vels      = dm_cat['PartType1/Velocities'] * np.sqrt(scf)
    dm_lr_mass   = dm_cat['PartType2/Masses'] * 1.00E+10 / h
    dm_lr_coords = dm_cat['PartType2/Coordinates'] * scf / h
    dm_lr_vels   = dm_cat['PartType2/Velocities'] * np.sqrt(scf)
    star_age     = star_cat['PartType4/GFM_StellarFormationTime']
    star_mass    = star_cat['PartType4/Masses'][star_age > 0] * 1.00E+10 / h
    star_coords  = star_cat['PartType4/Coordinates'][star_age > 0] * scf / h
    star_vel     = star_cat['PartType4/Velocities'][star_age > 0] * np.sqrt(scf)
    
    gal_pos       = fof_cat['GroupPos'] * scf / h
    gal_vel       = fof_cat['GroupVel'] * np.sqrt(scf)
    dm_coords    -= gal_pos
    dm_lr_coords -= gal_pos
    star_coords  -= gal_pos
    dm_vels      -= gal_vel
    dm_lr_vels   -= gal_vel
    star_vel     -= gal_vel
    
    ## combine prt type 1 and 2
    dm_mass   = np.concatenate([dm_mass, dm_lr_mass])
    dm_coords = np.concatenate([dm_coords, dm_lr_coords], axis=0)
    dm_vel    = np.concatenate([dm_vels, dm_lr_vels], axis=0)
    
    # star_rad = np.sqrt(
    #     star_coords[:,0]**2 + star_coords[:,1]**2 + star_coords[:,2]**2
    # )

    # ri   = 0
    # ro   = 5
    # incl = gt.calc_incl(star_coords, star_vel, star_mass, ri, ro)
    
    # dm_coords = gt.trans(dm_coords, incl) ## rotate galaxy to "face-on"
    # dm_vel    = gt.trans(dm_vel, incl)
    
    rvir = fof_cat['Group_R_Crit200'] * scf / h
    lim  = rvir

    pos_norm = np.linalg.norm(dm_coords, axis=1)
    sorted_indices = np.argsort(pos_norm)

    pos_sorted      = dm_coords[sorted_indices, :]
    pos_norm_sorted = pos_norm[sorted_indices]
    mass_sorted     = dm_mass[sorted_indices]
    vel_sorted      = dm_vel[sorted_indices, :]

    pos_sorted[:,0] += 8
    
    x = np.array(pos_sorted[:, 0])
    y = np.array(pos_sorted[:, 1])
    z = np.array(pos_sorted[:, 2])
    
    mask_lims = (
        (x < lim) & (x > -lim) &
        (y < lim) & (y > -lim) &
        (z < lim) & (z > -lim)
    )
    pos_filtered  = pos_sorted[mask_lims, :]
    mass_filtered = mass_sorted[mask_lims]
    vel_filtered  = vel_sorted[mask_lims, :]

    rho_p, v2, v4, hsml = get_particle_rho_and_moments(pos_filtered, vel_filtered, mass_filtered)

    nside = 1024
    js, jp, jd, d = project_particles_to_healpix(pos_filtered, mass_filtered, rho_p, v2, v4, hsml, nside=nside)
    
    factor_j = ((Msun2kg / kg2GeV)**2) / (kpc2cm**5)
    factor_d = (Msun2kg / kg2GeV) / (kpc2cm**2)
    c2 = (3e5)**2

    js *= factor_j
    jp *= (factor_j / c2)
    jd *= (factor_j / c2**2)
    d  *= factor_d


    with h5py.File(fname, 'r+') as f:
        this_group = f.require_group(f'box_{box:04d}')

        this_group.create_dataset('d_factor', data=d)
        this_group.create_dataset('s_wave'  , data=js)
        this_group.create_dataset('p_wave'  , data=jp)
        this_group.create_dataset('d_wave'  , data=jd)
        this_group.create_dataset('nside'   , data=nside)
        this_group.create_dataset('rvir'    , data=rvir)

    return

    lat0, lon0 = 0, 0
        
    # print(lat0, lon0)
    
    d = 4.0  # half–width in degrees

    # plt.sca(axs[box])
    
    hp.cartview(
        np.log10(jd),
        coord=None,            # or ('G','C'), ('C','G'), etc.
        lonra=[lon0 - d, lon0 + d],
        latra=[lat0 - d, lat0 + d],
        # xsize=600,             # controls resolution of output image
        title="",
        return_projected_map=False,
        notext=True,
        cmap='cmr.freeze',
    )

    plt.savefig(f'./figs/hp_vel/box_{box:04d}.pdf',bbox_inches='tight')
    plt.close()
    
    return
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    hp.mollview(np.log10(js), cmap='cmr.dusk', title='',
            fig=fig, sub=(2, 2, 1))

    hp.mollview(np.log10(jp), cmap='cmr.ember', title='',
                fig=fig, sub=(2, 2, 2))
    
    hp.mollview(np.log10(jd), cmap='cmr.freeze', title='',
                fig=fig, sub=(2, 2, 3))

    for ax in axs:
        ax.axis('off')

    plt.savefig(f'./figs/hp_vel/box_{box:04d}.pdf',bbox_inches='tight')
    plt.close()

if __name__ == '__main__': 
    box   = int(sys.argv[1])

    base_dir = '/scratch/aku7cf/healpix_maps'
    fname = f'{base_dir}/box_{box:04d}_unrotated.hdf5'
    with h5py.File(fname, 'w') as f:
        pass

    process_box(box, fname)

    print('SUCCESS')
