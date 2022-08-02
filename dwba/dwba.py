# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 12:26:59 2022

@author: gavin
"""

"""
    This module provides a function that calculates the target strength of 
    
"""

import sys
import math
import numpy as np
import xarray as xr
from scipy import ndimage

def phase_tracking_dwba(volume, frequencies, voxel_size, angles, densities, sound_speeds):
    """
    
    volume: 3D numpy array of integer type that contains the object to be modelled.
            Integers should be 0 for surrounding water, then increasing integers
            for other material types (i.e., 1, 2, 3, etc).
            
    densities: should be an iterable variable with the same number of entries as
               unique categories in 'volume'  [kg/m^3]
    sound_speeds: same as for densities [m/s]
    frequencies: iterable variable of modelling frequencies [Hz]
    angles: 2D ndarray containing angles ([0,:] = pitch, [1,:] = roll) [degree]

    Returns
    -------
    Xarray that contains the modelled target strength [dB re 1 m^2]

    """  
    
    # Make sure things are numpy arrayes
    densities = np.array(densities)
    sound_speeds = np.array(sound_speeds)
    voxel_size = np.array(voxel_size)
    
    # volume of the voxels [m^3]
    dv = voxel_size.prod()

    # input parameter checks
    # - that volume is 3D
    # - that unique values in volume are consecutive integers starting at 0
    # - that length of densities and sound_speeds are >= the number of unique values in volume
    # - that frequencies are positive 
    # - that angles (pitch) are -180 to +180
    # - that angles (roll) are -180 to +180
    # - that voxel_size is of length 3
    # - that voxel_size values are positive (and identical for the moment)

    # an intermediate output data structure
    fbs_dwba = xr.DataArray(np.zeros((angles.shape[1], len(frequencies)), dtype=np.complex128), 
                            dims = ('angle','frequency'),
                            coords = {'pitch': ('angle', angles[0,:]),
                                      'roll': ('angle', angles[1,:]),
                                      'frequency': ('frequency', frequencies)})
    
    fbs_dwba.frequency.attrs['units'] = 'Hz'
    fbs_dwba.angle.attrs['units'] = 'degrees'
    fbs_dwba.angle.attrs['long_name'] = 'Incident direction'
    fbs_dwba.angle.attrs['description'] = 'Tuple giving direction: [0] is pitch and [1] is roll'

    g = densities[1:] / densities[0]
    h = sound_speeds[1:] / sound_speeds[0] 
    
    for angle_i in np.arange(angles.shape[1]):
        pitch = angles[0,angle_i]
        roll = angles[1,angle_i]
        
        print(f'Running at pitch of {pitch}° and roll of {roll}° for {frequencies.min()/1e3} to {frequencies.max()/1e3} kHz')
        
        v = volume_rotate(volume, pitch, roll)
        
        categories = np.unique(v) # or just take the max?

        for f_i, f in enumerate(frequencies):
            #print(f'Running at {f/1e3} kHz')
        
            # wavenumbers in the various media
            k = 2*math.pi * f / sound_speeds
            
            # DWBA coefficients
            # amplitudes in media 1,2,...,n
            Cb = 1.0/(g * h**2) + 1.0/g - 2.0 # gamma_kappa - gamma_rho
            Ca = k[0]**2 * Cb / (4.0*math.pi) # summation coefficient
            
            # Differential phase for each voxel
            dph = np.zeros(v.shape)
            masks = []
            for i, category in enumerate(categories):
                masks.append(np.isin(v, category))
                #print(masks[i].shape)
                #print(k[i])
                #print(voxel_size[0])
                dph[masks[i]] = k[i] * voxel_size[0]
            masks.pop(0) # don't need to keep the category[0] mask
                
            phase = dph.cumsum(axis=2) - dph/2.0 
            dA = np.zeros(phase.shape, dtype=np.complex128)
    
            # differential phases for each voxel
            for i, m in enumerate(masks):
                dA[m] = Ca[i] * np.exp(2.0*1j*phase[m]) * dv
            
            # volume summation
            fbs_dwba[angle_i, f_i] = dA.sum()
        
    # Get the final output data structure
    ts = 20.0 * np.log10(np.abs(fbs_dwba))
    ts.attrs['units'] = 'dB'
    ts.attrs['long_name'] = 'target strength (re 1 m^2)'
    
    return ts
    
def volume_rotate(v, theta, phi):
    """
    Rotates the 3D ndarray by theta (pitch) and phi (roll) degrees using
    nearest neighbour interpolation (to preserve the categorial nature of the 
    data).

    Positive theta pitches head up.
    Positive phi rolls to the left. 

    Returns
    -------
    Rotated 3D ndarray.

    """    

    v = ndimage.rotate(v, -phi, axes=(0,1), order=0)
    v = ndimage.rotate(v, -theta, axes=(0,2), order=0)

    return v


def main():
    
    frequencies = np.arange(1000,5001,500)
    angles = [(0.0, 25.0), (0.0, 45.0)] ######
    voxel_size = [5e-4, 5e-4, 5e-4]
    densities = [1024., 1025., 1026.]
    sound_speeds = [1480., 1490., 1500.]
    volume = np.array([[[0,0,0],[0,1,2],[0,0,0]], 
                       [[0,0,0],[0,1,2],[0,0,0]], 
                       [[0,0,0],[0,1,2],[0,0,0]]])
    
    ts = phase_tracking_dwba(volume, frequencies, voxel_size, angles, densities, sound_speeds)
    
    print(ts)
    
    
if __name__ == '__main__':
    sys.exit(main())
    
    