

import numpy as np
import numpy.linalg as LA
from scipy import interpolate

import sys

import matplotlib.pyplot as plt
from matplotlib import cm, colors

# -----------------------------------------------------------------------------

from hex_harmonics import calc_hex_surf_value
from index_analysis import get_Cart3_PlaneNormal

# -----------------------------------------------------------------------------

large = 24; med = 20; small = 16;
params = {
          'axes.linewidth': '1',
          'xtick.major.size': '6',
          'ytick.major.size': '6',
          'xtick.major.width': '1',
          'ytick.major.width': '1',
          'xtick.minor.size': '3',
          'ytick.minor.size': '3',
          'xtick.minor.width': '0.5',
          'ytick.minor.width': '0.5',

          'font.family': ['sans-serif'],
          'font.sans-serif': ['Arial',
                              'DejaVu Sans',
                              'Liberation Sans',
                              'Bitstream Vera Sans',
                              'sans-serif'],

          'axes.titlesize': large,
          'legend.fontsize': med,
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': small,
          'ytick.labelsize': small,
          'figure.titlesize': large}

plt.rcParams.update(params)

# -----------------------------------------------------------------------------

def interpolate_CSE_conc_nolog(conc_x, directions, cs):

    cse_conc = np.zeros([len(directions[:, 0])])
    for K in range(len(directions[:, 0])):
        fcs = interpolate.interp1d(cs[:, 0], cs[:, K + 1], kind = 'linear')
        cse_conc[K] = fcs(conc_x)
    # end of for-K

    return cse_conc

def calc_fitting_err(coeffs, directions, cse_conc, syminfo, lattvec):

    error_vec = np.zeros(len(cse_conc))
    for K in range(len(cse_conc)):
        if syminfo == 'hex':
            hex3 = np.zeros([3])
            hex3[0] = directions[K, 0]
            hex3[1] = directions[K, 1]
            hex3[2] = directions[K, 3]
            cart3 = get_Cart3_PlaneNormal(hex3, lattvec)
            error_vec[K] = calc_hex_surf_value(coeffs, cart3) - cse_conc[K]
        elif syminfo == 'cubic':
            error_vec[K] = calc_cubic_surf_value(coeffs, directions[K, :]) - cse_conc[K]
        else:
            raise ValueError('Cannot recognize syminfo: ', syminfo)
        # end of if
    # end of for-K

    return np.sqrt(np.sum(error_vec**2)/len(error_vec))

def get_kspace_ECI(cs_coeffs, kpts, conc_x):

    if LA.norm(kpts) > 1E-3:
        EE = calc_hex_surf_value(cs_coeffs, kpts)
        keci = EE/(4.0*(1.0 - conc_x)*conc_x)
        return keci
    else:
        return 0.0
    # end if

# -----------------------------------------------------------------------------

