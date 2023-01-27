
# ---------- import the python modules ------------------------------------------

from __future__ import division

import numpy as np
import numpy.linalg as LA

import sys
import os

# sys.path.append("/home/kw7rr/bin/kwlib/atat")
# from atat_io import read_atat_latt

#################################################################################

def poly1d_vectorial_getval(xmesh, coeff, method = "BM4"):

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
    elif method == "BM7":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, -2.0/3)
        ymesh = ymesh + coeff[2]*np.power(xmesh, -4.0/3)
        ymesh = ymesh + coeff[3]*np.power(xmesh, -6.0/3)
        ymesh = ymesh + coeff[4]*np.power(xmesh, -8.0/3)
        ymesh = ymesh + coeff[5]*np.power(xmesh, -10.0/3)
        ymesh = ymesh + coeff[6]*np.power(xmesh, -12.0/3)
    elif method == "BM8":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, -2.0/3)
        ymesh = ymesh + coeff[2]*np.power(xmesh, -4.0/3)
        ymesh = ymesh + coeff[3]*np.power(xmesh, -6.0/3)
        ymesh = ymesh + coeff[4]*np.power(xmesh, -8.0/3)
        ymesh = ymesh + coeff[5]*np.power(xmesh, -10.0/3)
        ymesh = ymesh + coeff[6]*np.power(xmesh, -12.0/3)
        ymesh = ymesh + coeff[7]*np.power(xmesh, -14.0/3)
    elif method == "BM9":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, -2.0/3)
        ymesh = ymesh + coeff[2]*np.power(xmesh, -4.0/3)
        ymesh = ymesh + coeff[3]*np.power(xmesh, -6.0/3)
        ymesh = ymesh + coeff[4]*np.power(xmesh, -8.0/3)
        ymesh = ymesh + coeff[5]*np.power(xmesh, -10.0/3)
        ymesh = ymesh + coeff[6]*np.power(xmesh, -12.0/3)
        ymesh = ymesh + coeff[7]*np.power(xmesh, -14.0/3)
        ymesh = ymesh + coeff[8]*np.power(xmesh, -16.0/3)
    elif method == "BM10":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, -2.0/3)
        ymesh = ymesh + coeff[2]*np.power(xmesh, -4.0/3)
        ymesh = ymesh + coeff[3]*np.power(xmesh, -6.0/3)
        ymesh = ymesh + coeff[4]*np.power(xmesh, -8.0/3)
        ymesh = ymesh + coeff[5]*np.power(xmesh, -10.0/3)
        ymesh = ymesh + coeff[6]*np.power(xmesh, -12.0/3)
        ymesh = ymesh + coeff[7]*np.power(xmesh, -14.0/3)
        ymesh = ymesh + coeff[8]*np.power(xmesh, -16.0/3)
        ymesh = ymesh + coeff[9]*np.power(xmesh, -18.0/3)
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
    elif method == "7th":
        ymesh = coeff[0]*np.power(xmesh, 0.0)
        ymesh = ymesh + coeff[1]*np.power(xmesh, 1.0)
        ymesh = ymesh + coeff[2]*np.power(xmesh, 2.0)
        ymesh = ymesh + coeff[3]*np.power(xmesh, 3.0)
        ymesh = ymesh + coeff[4]*np.power(xmesh, 4.0)
        ymesh = ymesh + coeff[5]*np.power(xmesh, 5.0)
        ymesh = ymesh + coeff[6]*np.power(xmesh, 6.0)
        ymesh = ymesh + coeff[7]*np.power(xmesh, 7.0)
    else:
        raise ValueError("Method " + method + " is not implemented in polyfit_1d!")
    # end of if

    return ymesh

def poly1d_vectorial_pinvfit(xmesh, ymesh, method = "BM4"):

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
    elif method == "BM7":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, -2.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -4.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -6.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -8.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -10.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -12.0/3)))
    elif method == "BM8":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, -2.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -4.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -6.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -8.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -10.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -12.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -14.0/3)))
    elif method == "BM9":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, -2.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -4.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -6.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -8.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -10.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -12.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -14.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -16.0/3)))
    elif method == "BM10":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, -2.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -4.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -6.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -8.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -10.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -12.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -14.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -16.0/3)))
        vv = np.vstack((vv, np.power(xmesh, -18.0/3)))
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
    elif method == "7th":
        vv = np.power(xmesh, 0.0)
        vv = np.vstack((vv, np.power(xmesh, 1.0)))
        vv = np.vstack((vv, np.power(xmesh, 2.0)))
        vv = np.vstack((vv, np.power(xmesh, 3.0)))
        vv = np.vstack((vv, np.power(xmesh, 4.0)))
        vv = np.vstack((vv, np.power(xmesh, 5.0)))
        vv = np.vstack((vv, np.power(xmesh, 6.0)))
        vv = np.vstack((vv, np.power(xmesh, 7.0)))
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

    search_method = "poly1d"

    max_paral_stretch = [0.5, 1.75]
    ngrids_perp_1 = 15
    ngrids_perp_2 = 17
    ngrids_paral = 19

    directions = np.loadtxt('./dir_HCP.in', float)
    directions = np.reshape(directions, (-1, 10))

    paral_mesh = np.linspace(max_paral_stretch[0], max_paral_stretch[1], ngrids_paral)
    paral_mesh_dense = np.linspace(max_paral_stretch[0], max_paral_stretch[1], 1001)

    for element in ['0', '1']:
        os.chdir(element)

        for idir in range(len(directions[:, 0])):
            G_epi_4ind = directions[idir, :4]
            dir_string = '_' + str(int(G_epi_4ind[0])) + '_' + str(int(G_epi_4ind[1])) + '_' \
                         + str(int(G_epi_4ind[2])) + '_' + str(int(G_epi_4ind[3]))
            os.chdir("DIR_{}".format(dir_string))
            # print(os.getcwd())

            for I in range(ngrids_perp_1):
                for J in range(ngrids_perp_2):
                    stretch_name = "stretch_{}_{}".format(I, J)
                    os.chdir(stretch_name)

                    E_parallel = np.loadtxt("stretch_energy.out", float)
                    Index_min = np.argmin(E_parallel[:, 1])
                    if (Index_min < 2) or (Index_min > ngrids_paral - 3):
                        raise ValueError("  energy minimum at the boundary in {}.".format(stretch_name))
                    else:
                        miminum_file = open("stretch_minimum.out", "w+")
                        if search_method == "use_min":
                            miminum_file.write(" {:>16.9E} {:>16.9E}\n".format(E_parallel[Index_min, 0],
                                               E_parallel[Index_min, 1]))
                        elif search_method == "poly1d":

                            # ["BM4", "BM5", "BM6", "BM7", "BM8", "NegEven8", "2nd", "3rd", "4th", "5th", "6th", "7th"]:
                            method = "BM10"
                            coeff, rmse = poly1d_vectorial_pinvfit(paral_mesh, E_parallel[:, 1], method = method)
                            E_parallel_dense = poly1d_vectorial_getval(paral_mesh_dense, coeff, method = method)
                            print("{} {:>16.9f}".format(method, rmse))

                            Index_min = np.argmin(E_parallel_dense)
                            miminum_file.write(" {:>16.9E} {:>16.9E}\n".format(paral_mesh_dense[Index_min],
                                               E_parallel_dense[Index_min]))

                        # end if
                        miminum_file.close()
                        
                    # end if

                    os.chdir("..")
                # end for-J
            # end for-I

            os.chdir("..")
        # end for-idir

        os.chdir("..")
    # end for-element

#################################################################

