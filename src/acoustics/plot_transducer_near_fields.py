"""Creates a plot of recommended minimum sphere calibration ranges.

Minimum calibration range varies as a function of transducer beamwidth and acoustic frequency.

The plot implements equations 2.1b, 2.2, and the factor 3 multiplier mentioned in the caption of
Figure 2.5. of Demer et al. (2015). An earlier version of this code (written in Matlab) was used
to generate Figure 2.5 Demer et al.

Demer, D.A., Berger, L., Bernasconi, M., Boswell, K.M., Chu, D., Domokos, R., Dunford, A.J.,
   Fässler, S.M.M., Gauthier, S., Hufnagle, L.T., Jech, J.M., Bouffant, N., Lebourges-Dhaussy, A.,
   Lurton, X., Macaulay, G.J., Perrot, Y., Ryan, T.E., Parker-Stetter, S., Stienessen, S.,
   Weber, T.C., Williamson, N.J., 2015. Calibration of acoustic instruments (ICES Cooperative
   Research Report 326). https://doi.org/10.17895/ices.pub.5494
"""
# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "matplotlib-label-lines",
# ]
# ///
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLines  # see https://github.com/cphyc/matplotlib-label-lines

c = 1470.  # [m/s] speed of sound in water # pylint: disable=invalid-name

# Calculate the near-field range at these frequencies
freq = np.array([12, 18, 26, 38, 50, 70, 120, 200, 333]) * 1e3  # [Hz]

# The range of transducer 3 dB two-way beamwidths to use
bw = np.linspace(2, 20, 50)  # [deg]

# Force y tick locations as the automatic ones aren't ideal
yticksAt = [1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20]  # [m]

fig, ax = plt.subplots(1)

for f in freq:
    lambda_ = c / f  # [m]
    k = 2 * np.pi * f / c  # [m-1]

    d_t = 3.2 / (k * np.sin(np.deg2rad(bw/2.0)))  # [m]
    r_nf = 3 * np.pi * d_t**2.0 / (4.0 * lambda_)  # [m]

    line = ax.plot(bw, r_nf, color='black', linewidth=2, label=f'{f/1000:.0f} kHz')

labelLines(ax.get_lines(), fontsize=7, xvals=(16, 2))

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

plt.savefig('near_field_ranges.png', dpi=300)
