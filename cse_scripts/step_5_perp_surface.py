
# ---------- import the python modules ------------------------------------------

from __future__ import division

import numpy as np
import numpy.linalg as LA

import sys
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# sys.path.append("/home/kw7rr/bin/kwlib/atat")
# from atat_io import read_atat_latt

#################################################################################

if __name__ == '__main__':

    search_method = "use_min"

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

            perp_1_mesh = range(ngrids_perp_1)
            perp_2_mesh = range(ngrids_perp_2)
            mesh_x, mesh_y = np.meshgrid(perp_1_mesh, perp_2_mesh)

            energy_surface = np.zeros([ngrids_perp_1, ngrids_perp_2])
            for I in range(ngrids_perp_1):
                for J in range(ngrids_perp_2):

                    stretch_name = "stretch_{}_{}".format(I, J)
                    energy_surface[I, J] = np.loadtxt("{}/stretch_minimum.out".format(stretch_name), float, usecols = (1,))

                # end for-J
            # end for-I
            np.savetxt("mesh_x.out", mesh_x, fmt = "%+16.9E")
            np.savetxt("mesh_y.out", mesh_y, fmt = "%+16.9E")
            np.savetxt("energy_surface.out", energy_surface, fmt = "%+16.9E")

            fig = plt.figure()
            ax = fig.gca(projection = '3d')
            ax.scatter(mesh_x.flatten(), mesh_y.flatten(), np.transpose(energy_surface).flatten(), c = 'r', s = 50)
            ax.plot_surface(mesh_x, mesh_y, np.transpose(energy_surface), \
                            cmap = cm.coolwarm, linewidth = 0, antialiased = False)
            ax.set_title(" {}: {}".format(element, dir_string))
            plt.savefig("fig_energy_surface_{}_{}.pdf".format(element, dir_string))
            # plt.show()

            os.chdir("..")
        # end for-idir

        os.chdir("..")
    # end for-element

#################################################################

