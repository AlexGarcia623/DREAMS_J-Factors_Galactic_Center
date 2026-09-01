import os, sys
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import h5py
from scipy.spatial import KDTree

mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.linewidth'] = 2.25*1.25
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5*1.25
mpl.rcParams['ytick.major.width'] = 1.5*1.25
mpl.rcParams['xtick.minor.width'] = 1.0*1.25
mpl.rcParams['ytick.minor.width'] = 1.0*1.25
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4.5
mpl.rcParams['ytick.minor.size'] = 4.5
mpl.rcParams['xtick.top']   = True
mpl.rcParams['ytick.right'] = True

def get_MW_idx(cat, h=0.6909):
    """
    Selects the corrent MW-mass galaxy from each simulation given the group catalog.
    This function only works for z~0
    It selects the least contaminated halo with a mass within current uncertainties of the MW's mass
    
    Inputs 
     - cat - a dictionary containing the 'GroupMassType' field from the FOF catalogs
     
    Returns
     - mw_idx - the index into the group catalog for the target MW-mass galaxy
    """
    masses = cat['GroupMassType'] * 1e10 / h
    
    tot_masses = np.sum(masses,axis=1)
    mcut = (tot_masses > 5e11) & (tot_masses < 2.5e12)
    contamination = masses[:,2] / tot_masses
    idx = np.argmin(contamination[mcut])
    
    mw_idx = np.arange(len(masses))[mcut][idx]
    return mw_idx

def load_particle_data(path, keys, part_types):
    """
    Read particle data from the DREAMS simulations
    
    Inputs
      path - the absolute or relative path to the hdf5 file you want to read from
      keys - the data that you want to read from the simulation 
             see https://www.tng-project.org/data/docs/specifications/ for a list of available data)
      part_types - which particle types to load.
                   0 - gas
                   1 - high res dark matter
                   2 - low res dark matter
                   3 - tracers (not used in DREAMS)
                   4 - stars
                   5 - black holes
      
    Returns
      cat - a dictionary that contains all of the particle information for the specified keys and particle types
    """
    cat = dict()
    with h5py.File(path) as ofile:
        
        if type(part_types) == type(0):
            part_types = [part_types]
        
        for pt in part_types:
            for key in keys:
                if pt == 1 and key == 'Masses':
                    cat[f'PartType{pt}/{key}'] = np.ones(ofile['PartType1/ParticleIDs'].shape)*ofile['Header'].attrs['MassTable'][1]
                else:
                    cat[f'PartType{pt}/{key}'] = np.array(ofile[f'PartType{pt}/{key}'])
    return cat

def load_group_data(path, keys):
    """
    Read Group Data from the DREAMS simulations
    
    Inputs
      path - the absolute or relative path to the hdf5 file you want to read from
      keys - the data that you want to read from the simulation 
             see https://www.tng-project.org/data/docs/specifications/ for a list of available data)
      
    Returns
      cat - a dictionary that contains all of the group and subhalo information for the specified keys
    """
    cat = dict()
    with h5py.File(path) as ofile:
        for key in keys:
            if 'Group' in key:
                cat[key] = np.array(ofile[f'Group/{key}'])
            if 'Subhalo' in key:
                cat[key] = np.array(ofile[f'Subhalo/{key}'])
    return cat

def get_galaxy_data(part_cat, group_cat, fof_idx=-1, sub_idx=-1):
    """
    Given particle and group catalogs, return a new catalog that only contains data for a specified galaxy.
    If fof_idx is given but sub_idx is not, data for the FOF group and all satellites are returned
    If fof_idx is given and sub_idx is given, data for just the specified subhalo of that group
      e.g. fof_idx=3 sub_idx=5 will provide data for the fifth subhalo of group three
    If only sub_idx is supplied, the data for that subfind galaxy is returned, can be in any FOF group
    
    Inputs
      part_cat  - a dictionary containing particle data make from load_particle_data
      group_cat - a dictionary containing group data make from load_particle_data
                  must contain these fields: GroupLenType, GroupFirstSub, GroupNsubs, SubhaloLenType, SubhaloGrNr
      fof_idx   - the FOF group that you want data for
      sub_idx   - the Subfind galaxy that you want data for
      
    Returns
      new_part_cat  - a new dictionary with keys from part_cat but data only for the specified galaxy
      new_group_cat - a new dictionary with keys from group_cat but data only for the specified galaxy
    """
    
    if fof_idx < 0 and sub_idx < 0:
        return part_cat, group_cat
    
    if fof_idx < 0 and sub_idx >= 0:
        fof_idx = group_cat['SubhaloGrNr'][sub_idx]
    
    offsets = np.sum(group_cat['GroupLenType'][:fof_idx],axis=0)
    
    if sub_idx >= 1:
        start_sub = group_cat['GroupFirstSub'][fof_idx]
        offsets += np.sum(group_cat['SubhaloLenType'][start_sub:start_sub+sub_idx], axis=0)
    
    if sub_idx < 0:
        num_parts = group_cat['GroupLenType'][fof_idx]
        nsubs = group_cat['GroupNsubs'][fof_idx]
        sub_start = group_cat['GroupFirstSub'][fof_idx]
    else:
        num_parts = group_cat['SubhaloLenType'][sub_idx]
        nsubs = 1
        sub_start = sub_idx
    
    
    new_part_cat = dict()
    for key in part_cat:
        pt = int(key.split("/")[0][-1])
        new_part_cat[key] = part_cat[key][offsets[pt]:offsets[pt]+num_parts[pt]]
    
    new_group_cat = dict()
    for key in group_cat:
        if 'Group' in key:
            new_group_cat[key] = group_cat[key][fof_idx]
        else:
            new_group_cat[key] = group_cat[key][sub_start:sub_start+nsubs]
    
    return new_part_cat, new_group_cat

def get_scf(datadir, snapnr, box_to_process):
    scf = None
    with h5py.File(datadir+f'box_{box_to_process}/'+'snap_'+str(snapnr).zfill(3)+'.hdf5', 'r') as f:
        scf=f['Header'].attrs['Time']
    return scf

def get_params(file):
    """
    This function reads in the simulation parameters for all DREAMS sims and converts them into physical units
    
    Inputs
     - file - absolute or relative path to the file containing all simulation parameters
     
    Returns
     - params - an Nx4 ndarray with the following parameters
                index 0 - Omega_m
                index 1 - Sigma_8
                index 2 - SN1 (ew) wind energy
                index 3 - SN2 (kw) wind velocity
                index 4 - AGN1 (BHFF) black hole feedback factor
    """
    sample = np.loadtxt(file)
        
    fiducial = np.array([1,1,3.6,7.4,.1]) #fiducial TNG values
    params = sample * fiducial
    
    return params

def fibonacci_sphere(samples, r):
    """
    This function returns a set of points equally spaced on the surface of a sphere.
    This function was adapted from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    
    Inputs
      samples - the number of points on the sphere
      r       - the radius of the sphere that is sampled
      
    Returns
      points  - the coordinates of the points of the sphere with shape (samples,3)
    """
    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / (samples - 1)) * 2  
        radius = np.sqrt(1 - y * y) 

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    points = np.array(points) * r
        
    return points

def calc_density(idx, matter_mass, hsml):
    mass_enclosed = np.sum(matter_mass[idx], axis=1)
    density = mass_enclosed / (4 / 3 * np.pi * np.power(hsml,3))
    return density ## used to be np.average

def calc_vel_moment(idx, matter_vel, matter_mass, hsml, mw=True):
    vmom2 = np.zeros(len(idx))
    vmom4 = np.zeros(len(idx))
    for j, neighbors in enumerate(idx): ## for each 32 neighbors
        v = matter_vel[neighbors]
        
        diffs = v[:,None,:] - v[None,:,:] ## pairwise velocity differences
        
        vrel  = np.sqrt(np.sum(diffs**2, axis=2))
        vrel2 = vrel**2

        iu = np.triu_indices_from(vrel, k=1)  # indices for upper triangle, i<j
        vrel2_unique = vrel2[iu]
        
        vmom2[j] = np.mean(vrel2_unique)
        vmom4[j] = np.mean(vrel2_unique**2)
        # vmom2[j] = np.average(vrel2)
        # vmom4[j] = np.average(vrel2**2)
        
    return vmom2, vmom4

def load_particle_single(path, keys, part_types, subset=None, float32=False):
    """
    Read particle data for specified particle types and keys.
    If subset is provided, only loads the data for the subhalo of interest.

    Inputs:
      path - path to snapshot file
      keys - list of fields to load
      part_types - list of particle types (0 = gas, 1 = DM, 4 = stars, etc.)
      subset - dictionary with 'offsetType', 'lenType', and 'snapOffsets' from subhalo catalog
      float32 - if True, casts float64 fields to float32 to save memory

    Returns:
      cat - dict with requested data, only for the specified subhalo (if subset is provided)
    """
    cat = dict()
    with h5py.File(path, 'r') as f:
        if isinstance(part_types, int):
            part_types = [part_types]

        for pt in part_types:
            gName = f'PartType{pt}'

            if gName not in f:
                continue

            # Determine read range
            if subset is not None:
                offset = subset['offsetType'][pt]
                length = subset['lenType'][pt]
                if length == 0:
                    continue
            else:
                offset = 0
                length = f[gName + '/ParticleIDs'].shape[0]

            for key in keys:
                field_name = f'{gName}/{key}'

                if field_name not in f:
                    if pt == 1 and key == 'Masses':
                        # Low-res DM: use MassTable
                        cat[field_name] = np.ones(length) * f['Header'].attrs['MassTable'][pt]
                        continue
                    else:
                        raise Exception(f'Field {key} not in {path}')
                        # continue  # Skip if field doesn't exist

                dtype = f[field_name].dtype
                if dtype == np.float64 and float32:
                    data = f[field_name][offset:offset+length].astype(np.float32)
                else:
                    data = f[field_name][offset:offset+length]
                
                cat[field_name] = data

    return cat

if __name__ == "__main__":
    print('Hello World')