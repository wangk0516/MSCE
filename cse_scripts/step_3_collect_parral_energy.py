
# ---------- import the python modules ------------------------------------------

from __future__ import division

import numpy as np
import numpy.linalg as LA

import sys
import os

# sys.path.append("/home/kw7rr/bin/kwlib/atat")
# from atat_io import read_atat_latt

#################################################################################

if __name__ == '__main__':

    max_paral_stretch = [0.5, 1.75]
    ngrids_perp_1 = 15
    ngrids_perp_2 = 17
    ngrids_paral = 19

    directions = np.loadtxt('./dir_HCP.in', float)
    directions = np.reshape(directions, (-1, 10))

    paral_stretch = np.linspace(max_paral_stretch[0], max_paral_stretch[1], ngrids_paral)

    for element in ['0', '1']:
        os.chdir(element)

        for idir in range(len(directions[:, 0])):
            G_epi_4ind = directions[idir, :4]
            dir_string = '_' + str(int(G_epi_4ind[0])) + '_' + str(int(G_epi_4ind[1])) + '_' \
                         + str(int(G_epi_4ind[2])) + '_' + str(int(G_epi_4ind[3]))
            os.chdir("DIR_{}".format(dir_string))
            print(os.getcwd())

            for I in range(ngrids_perp_1):
                for J in range(ngrids_perp_2):
                    stretch_name = "stretch_{}_{}".format(I, J)
                    os.chdir(stretch_name)

                    stretch_energy_file = open("stretch_energy.out", "w+")
                    for K in range(ngrids_paral):
                        ss = paral_stretch[K]
                        energy_file = "energy.{}".format(K)
                        if os.path.exists(energy_file):
                            if os.stat(energy_file).st_size > 0:
                                ee = np.loadtxt(energy_file, float)
                                stretch_energy_file.write(" {:>+12.6E} \t {:>+12.6E} \t {}\n".format(ss, ee, K))
                            else:
                                print("  {} is empty in {}.".format(energy_file, stretch_name))
                                os.remove(energy_file)
                                stretch_energy_file.write(" {:>+12.6E} \t {} \t {}\n".format(ss, "FILE_EMPTY", K))
                            # end if
                        else:
                            print("  {} does not exist in {}.".format(energy_file, stretch_name))
                            stretch_energy_file.write(" {:>+12.6E} \t {} \t {}\n".format(ss, "NOT_EXISTS", K))
                        # end if
                    # end for-K
                    stretch_energy_file.close()

                    os.chdir("..")
                # end for-J
            # end for-I

            os.chdir("..")
        # end for-idir


        os.chdir("..")
    # end for-element

#################################################################

