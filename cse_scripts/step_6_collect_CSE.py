
# ---------- import the python modules ------------------------------------------

from __future__ import division

import numpy as np
import numpy.linalg as LA
from scipy import interpolate

import sys
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# sys.path.append("/home/kw7rr/bin/kwlib/atat")
# from atat_io import read_atat_latt

#################################################################################

def poly1d_vectorial_getval(xmesh, coeff, method):

    if method == "BM4":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, -2.0/3)
        ymesh = ymesh + coeff[2]*np.power(xmesh, -4.0/3)
        ymesh = ymesh + coeff[3]*np.power(xmesh, -6.0/3)
    elif method == "BM5":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, -2.0/3)
        ymesh = ymesh + coeff[2]*np.power(xmesh, -4.0/3)
        ymesh = ymesh + coeff[3]*np.power(xmesh, -6.0/3)
        ymesh = ymesh + coeff[4]*np.power(xmesh, -8.0/3)
    elif method == "BM6":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, -2.0/3)
        ymesh = ymesh + coeff[2]*np.power(xmesh, -4.0/3)
        ymesh = ymesh + coeff[3]*np.power(xmesh, -6.0/3)
        ymesh = ymesh + coeff[4]*np.power(xmesh, -8.0/3)
        ymesh = ymesh + coeff[5]*np.power(xmesh, -10.0/3)
    elif method == "NegEven8":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, -2.0)
        ymesh = ymesh + coeff[2]*np.power(xmesh, -4.0)
        ymesh = ymesh + coeff[3]*np.power(xmesh, -6.0)
        ymesh = ymesh + coeff[4]*np.power(xmesh, -8.0)
    elif method == "2nd":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, 1.0)
        ymesh = ymesh + coeff[2]*np.power(xmesh, 2.0)
    elif method == "3rd":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, 1.0)
        ymesh = ymesh + coeff[2]*np.power(xmesh, 2.0)
        ymesh = ymesh + coeff[3]*np.power(xmesh, 3.0)
    elif method == "4th":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, 1.0)
        ymesh = ymesh + coeff[2]*np.power(xmesh, 2.0)
        ymesh = ymesh + coeff[3]*np.power(xmesh, 3.0)
        ymesh = ymesh + coeff[4]*np.power(xmesh, 4.0)
    elif method == "5th":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, 1.0)
        ymesh = ymesh + coeff[2]*np.power(xmesh, 2.0)
        ymesh = ymesh + coeff[3]*np.power(xmesh, 3.0)
        ymesh = ymesh + coeff[4]*np.power(xmesh, 4.0)
        ymesh = ymesh + coeff[5]*np.power(xmesh, 5.0)
    elif method == "6th":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, 1.0)
        ymesh = ymesh + coeff[2]*np.power(xmesh, 2.0)
        ymesh = ymesh + coeff[3]*np.power(xmesh, 3.0)
        ymesh = ymesh + coeff[4]*np.power(xmesh, 4.0)
        ymesh = ymesh + coeff[5]*np.power(xmesh, 5.0)
        ymesh = ymesh + coeff[6]*np.power(xmesh, 6.0)
    else:
        raise ValueError("Method " + method + " is not implemented in polyfit_1d!")
    # end of if

    return ymesh

def poly1d_vectorial_pinvfit(xmesh, ymesh, method):

    if method == "BM4":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, -2.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -4.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -6.0/3)))
    elif method == "BM5":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, -2.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -4.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -6.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -8.0/3)))
    elif method == "BM6":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, -2.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -4.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -6.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -8.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -10.0/3)))
    elif method == "NegEven8":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, -2.0)))
        vv = np.vstack((vv, np.power(xmesh, -4.0)))
        vv = np.vstack((vv, np.power(xmesh, -6.0)))
        vv = np.vstack((vv, np.power(xmesh, -8.0)))
    elif method == "2nd":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, 1.0)))
        vv = np.vstack((vv, np.power(xmesh, 2.0)))
    elif method == "3rd":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, 1.0)))
        vv = np.vstack((vv, np.power(xmesh, 2.0)))
        vv = np.vstack((vv, np.power(xmesh, 3.0)))
    elif method == "4th":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, 1.0)))
        vv = np.vstack((vv, np.power(xmesh, 2.0)))
        vv = np.vstack((vv, np.power(xmesh, 3.0)))
        vv = np.vstack((vv, np.power(xmesh, 4.0)))
    elif method == "5th":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, 1.0)))
        vv = np.vstack((vv, np.power(xmesh, 2.0)))
        vv = np.vstack((vv, np.power(xmesh, 3.0)))
        vv = np.vstack((vv, np.power(xmesh, 4.0)))
        vv = np.vstack((vv, np.power(xmesh, 5.0)))
    elif method == "6th":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, 1.0)))
        vv = np.vstack((vv, np.power(xmesh, 2.0)))
        vv = np.vstack((vv, np.power(xmesh, 3.0)))
        vv = np.vstack((vv, np.power(xmesh, 4.0)))
        vv = np.vstack((vv, np.power(xmesh, 5.0)))
        vv = np.vstack((vv, np.power(xmesh, 6.0)))
    else:
        raise ValueError("Method " + method + " is not implemented in polyfit_1d!")
    # end of if

    coeff = np.dot(LA.pinv(vv).transpose(), ymesh) # (4x1)=(4xn)*(nx1); the (Moore-Penrose) pseudo-inverse
    y_fit = poly1d_vectorial_getval(xmesh, coeff, method)
    err_rmse = np.sqrt(np.sum((y_fit - ymesh)**2)/len(ymesh))
    # print("RMSE (eV): {:.4f}".format(err_rmse))

    return coeff, err_rmse

#################################################################################

if __name__ == '__main__':

    search_method = "use_min"
    conc = np.linspace(0, 1, 101)
    natom = 2.0

    max_paral_stretch = [0.5, 1.75]
    ngrids_perp_1 = 15
    ngrids_perp_2 = 17
    ngrids_paral = 19

    directions = np.loadtxt('./dir_HCP.in', float)
    directions = np.reshape(directions, (-1, 10))

    paral_stretch = np.linspace(max_paral_stretch[0], max_paral_stretch[1], ngrids_paral)
    E_epi = np.zeros([len(conc), len(directions[:, 0])])

    for idir in range(len(directions[:, 0])):
        G_epi_4ind = directions[idir, :4]
        dir_string = '_' + str(int(G_epi_4ind[0])) + '_' + str(int(G_epi_4ind[1])) + '_' \
                     + str(int(G_epi_4ind[2])) + '_' + str(int(G_epi_4ind[3]))
        print("Analyze {} ...".format(dir_string))

        E_surface_AA = np.loadtxt("./0/DIR_{}/energy_surface.out".format(dir_string), float)
        E_surface_BB = np.loadtxt("./1/DIR_{}/energy_surface.out".format(dir_string), float)
        print(np.shape(E_surface_AA), np.shape(E_surface_BB))

        perp_1_mesh = range(ngrids_perp_1)
        perp_2_mesh = range(ngrids_perp_2)
        mesh_x, mesh_y = np.meshgrid(perp_1_mesh, perp_2_mesh)

        perp_1_mesh_dense = np.linspace(0, ngrids_perp_1 - 1, 101)
        perp_2_mesh_dense = np.linspace(0, ngrids_perp_2 - 1, 103)
        mesh_x_dense, mesh_y_dense = np.meshgrid(perp_1_mesh_dense, perp_2_mesh_dense)

        func_AA = interpolate.interp2d(perp_1_mesh, perp_2_mesh, np.transpose(E_surface_AA), kind = 'linear')
        E_surface_AA_dense = func_AA(perp_1_mesh_dense, perp_2_mesh_dense)

        func_BB = interpolate.interp2d(perp_1_mesh, perp_2_mesh, np.transpose(E_surface_BB), kind = 'linear')
        E_surface_BB_dense = func_BB(perp_1_mesh_dense, perp_2_mesh_dense)

        E_surface_AA_dense = E_surface_AA_dense - np.amin(np.amin(E_surface_AA_dense))
        E_surface_BB_dense = E_surface_BB_dense - np.amin(np.amin(E_surface_BB_dense))

        for ic in range(len(conc)):
            E_surf_avg = E_surface_AA_dense*(1 - conc[ic]) + E_surface_BB_dense*conc[ic]
            min_indices = np.unravel_index(E_surf_avg.argmin(), E_surf_avg.shape)
            print("min_indices = {}".format(min_indices))
            print(mesh_x_dense[min_indices], mesh_y_dense[min_indices])
            E_epi[ic, idir] = E_surf_avg[min_indices]

            if conc[ic] < 0.01 or conc[ic] > 0.99:
                E_epi[ic, idir] = 0.0
            else:
                E_epi[ic, idir] = E_epi[ic, idir]/natom
            # end of if
        # end of ic

    # end for-idir

    figsize = np.array([9, 8])
    fig = plt.figure(figsize = figsize)

    for idir in range(len(directions[:, 0])):
        label = "[" + str(int(directions[idir, 0])) + str(int(directions[idir, 1])) + str(int(directions[idir, 2])) + str(int(directions[idir, 3])) + "]"
        plt.plot(conc, E_epi[:, idir], '-', lw = 2, fillstyle = 'none',
                 markeredgecolor = 'k', markeredgewidth = 1.5, markersize = 8, label = label)
    # end of for-idir

    conc_2d = conc.reshape((len(conc), 1))
    np.savetxt('cs.log', np.hstack((conc_2d, E_epi)), fmt = '%10.8f')

    # figure options
    plt.xlim([0, 1.0])
    plt.ylim([0, 0.08])

    plt.xlabel('Molar fraction', fontsize = 16)
    plt.ylabel('$\Delta E_{CS}$ ($eV/atom$)', fontsize = 16)

    plt.legend(loc = 2, frameon = False, ncol = 1, numpoints = 1, fontsize = 16)
    plt.grid()

    # plt.text(0.7, 0.055, 'HCP Mg-Zn', fontsize = 24, weight = 'bold')

    # adjust the margins
    margins = {  #     vvv margin in inches
               "left"   :     1.5 / figsize[0],
               "bottom" :     1.5 / figsize[1],
               "right"  : 1 - 0.5 / figsize[0],
               "top"    : 1 - 1   / figsize[1]
              }
    fig.subplots_adjust(**margins)

    plt.savefig("cse.pdf")

    plt.show()


#################################################################

