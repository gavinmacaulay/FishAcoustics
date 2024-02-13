# -*- coding: utf-8 -*-
"""
Creates a plot of recommended minimum sphere calibration range given a
transduer beamwidth and acoustic frequency.

@author: Gavin Macaulay
"""

import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLines # see https://github.com/cphyc/matplotlib-label-lines

c = 1470. # [m/s] speed of sound in water
freq = np.array([12, 18, 38, 70, 120, 200, 333]) * 1e3 # [Hz] acoustic frequency
bw = np.linspace(2,20,50) # [deg] 3 dB two-way beamwidth

yticksAt = [1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20]

textLine = (yticksAt[-1]-yticksAt[0])/(bw[-1]-bw[0]) * 10**((bw-bw[0])/10) + yticksAt[0]

#%%
fig, ax = plt.subplots(1)

for f in freq:
    lambda_ = c / f # [m]
    k = 2 * np.pi * f / c # [m-1]

    d_t = 3.2 / (k * np.sin(np.deg2rad(bw/2.0))) # [m]
    r_nf  = 3 * np.pi * d_t**2.0 / (4.0 * lambda_) # [m]

    line = ax.plot(bw, r_nf, color='black', linewidth=2, label=f'{f/1000:.0f} kHz')

labelLines(ax.get_lines(), fontsize=7, xvals=(16,2))

plt.minorticks_on()
ax.grid(axis='x', which='minor', linewidth=0.5, linestyle=':')
ax.grid(axis='x', which='major')
ax.grid(axis='y', which='major')
ax.set_yscale('log')
ax.set_yticks(yticksAt)
ax.set_yticklabels((str(i) for i in yticksAt))
ax.set_ylim(yticksAt[0], yticksAt[-1])
ax.set_xlim(bw[0], bw[-1])
ax.set_xlabel('Transducer 3 dB beamwidth [°]')
ax.set_ylabel('Minimum sphere range [m]')
