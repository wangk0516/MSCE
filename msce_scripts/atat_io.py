
# ---------- import the python modules ------------------------------------------

from __future__ import division

import numpy as np
import sys
import os

# ---------- user defined functions  --------------------------------------------

def abs_cap(x, max_abs_val = 1.0):

    x_abs = np.abs(x)

    if x_abs < max_abs_val:
        x_cap = x
    else:
        x_cap = np.sign(x)*max_abs_val
    # end if

    return x_cap

def axes_from_lengths_and_angles(a, b, c, alpha, beta, gamma):
    """
    https://pymatgen.org/_modules/pymatgen/core/lattice.html
    https://wiki.fysik.dtu.dk/ase/_modules/ase/geometry/cell.html#cellpar_to_cell
    """

    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
                -b * sin_alpha * np.cos(gamma_star),
                b * sin_alpha * np.sin(gamma_star),
                b * cos_alpha,
               ]
    vector_c = [0.0, 0.0, float(c)]

    axes = np.zeros([3, 3])
    axes[0, :] = vector_a[:]
    axes[1, :] = vector_b[:]
    axes[2, :] = vector_c[:]

    return axes

def read_atat_str(filename):

    # open the atat str file
    atatstrfile = open(filename, "r")
    atatstr = atatstrfile.readlines()
    atatstrfile.close
    nline = len(atatstr)

    lines_0 = atatstr[0].split()
    axes = np.zeros([3, 3])
    axes_lines = 0
    if np.size(lines_0) == 3:
        # print('axes given by 3x3 matrix;')
        for K0 in range(3):
            templine = atatstr[K0].split()
            for K1 in range(3): axes[K0, K1] = float(templine[K1])
        # end of for-K0
        axes_lines = 3
    elif np.size(lines_0) == 6:
        a = float(lines_0[0])
        b = float(lines_0[1])
        c = float(lines_0[2])
        alpha = float(lines_0[3])
        beta = float(lines_0[4])
        gamma = float(lines_0[5])
        axes = axes_from_lengths_and_angles(a, b, c, alpha, beta, gamma)
        axes_lines = 1
    else:
        print('cannot recognize the axes; exit!')
        sys.exit()
    # end of if

    lattvec = np.zeros([3, 3])
    for K2 in range(3):
        templine = atatstr[axes_lines + K2].split()
        for K3 in range(3): lattvec[K2, K3] = float(templine[K3])
    # end of for-K2

    atompos = np.zeros([nline - axes_lines - 3, 3])
    atomlist = []
    for K4 in range(nline - axes_lines - 3):
        templine = atatstr[axes_lines + 3 + K4].split()
        # print(np.size(templine))
        for K5 in range(3): atompos[K4, K5] = float(templine[K5])
        atomlist.append(templine[3])
    # end of for-K4

    atomlist_sorted = sorted(atomlist)
    atomlabel = []
    for atom in atomlist_sorted:
        if atom not in atomlabel: atomlabel.append(atom)
    # end of for-atom

    atomnum = np.zeros(len(atomlabel))
    for atom in atomlist:
        ind = atomlabel.index(atom)
        atomnum[ind] += 1
    # end of for-atom

    conc = atomnum/float(np.sum(atomnum))

    atat_structure = {'axes': axes, 'lattvec': lattvec, \
            'atompos': atompos, 'atomlist': atomlist, \
            'atomlabel': atomlabel, 'atomnum': atomnum, \
            'conc': conc}

    # return the results
    return atat_structure

def write_atat_str(StructInfo, filename = 'struct.out'):

    axes = StructInfo['axes']
    lattvec = StructInfo['lattvec']
    atompos = StructInfo['atompos']
    atomlist = StructInfo['atomlist']

    structre_file = open(filename, "w+")
    for I in range(3):
        structre_file.write(" {:>15.9f} {:>15.9f} {:>15.9f}\n".format(axes[I, 0], axes[I, 1], axes[I, 2]))
    # end for-I
    for J in range(3):
        structre_file.write(" {:>15.9f} {:>15.9f} {:>15.9f}\n".format(lattvec[J, 0], lattvec[J, 1], lattvec[J, 2]))
    # end for-J
    for K in range(len(atompos[:, 0])):
        structre_file.write(" {:>15.9f} {:>15.9f} {:>15.9f} {}\n".format(atompos[K, 0],
                            atompos[K, 1], atompos[K, 2], atomlist[K]))
    # end for-K
    structre_file.close()

    return 0

def read_atat_latt(filename):

    # open the atat lattice file
    atatlattfile = open(filename, "r")
    atatlatt = atatlattfile.readlines()
    atatlattfile.close
    nline = len(atatlatt)

    lines_0 = atatlatt[0].split()
    axes = np.zeros([3, 3])
    axes_lines = 0
    if np.size(lines_0) == 3:
        # print('axes given by 3x3 matrix;')
        for K0 in range(3):
            templine = atatlatt[K0].split()
            for K1 in range(3): axes[K0, K1] = float(templine[K1])
        # end of for-K0
        axes_lines = 3
    elif np.size(lines_0) == 6:
        a = float(lines_0[0])
        b = float(lines_0[1])
        c = float(lines_0[2])
        alpha = float(lines_0[3])
        beta = float(lines_0[4])
        gamma = float(lines_0[5])
        axes = axes_from_lengths_and_angles(a, b, c, alpha, beta, gamma)
        axes_lines = 1
    else:
        print('cannot recognize the axes; exit!')
        sys.exit(1)
    # end of if

    lattvec = np.zeros([3, 3])
    for K2 in range(3):
        templine = atatlatt[axes_lines + K2].split()
        for K3 in range(3): lattvec[K2, K3] = float(templine[K3])
    # end of for-K2

    atompos = np.zeros([nline - axes_lines - 3, 3])
    siteoccu = []
    for K4 in range(nline - axes_lines - 3):
        templine = atatlatt[axes_lines + 3 + K4].replace(',', ' ').split()
        # print(np.size(templine))
        for K5 in range(3): atompos[K4, K5] = float(templine[K5])
        siteoccu.append(templine[3:])
    # end of for-K4

    atat_lattice = {'axes': axes, 'lattvec': lattvec, \
                    'sitepos': atompos, 'sitenum': len(atompos[:, 0]), 'siteoccu': siteoccu}

    # return the results
    return atat_lattice


#################################################################################

"""
atat_structure = read_atatstr('str.out')
print(atat_structure['axes'])
print(atat_structure['lattvec'])
print(atat_structure['atompos'])
print(atat_structure['atomlist'])
print(atat_structure['atomlabel'])
print(atat_structure['atomnum'])
print(atat_structure['conc'])


x = np.linspace(-2, 2, 31)
for f in x:
    print(" {:+.9f} {:+.9f}".format(f, abs_cap(f)))


a = 3.193 
b = 3.193
c = 5.181

alpha = 90
beta = 90
gamma = 120

axes = axes_from_lengths_and_angles(a, b, c, alpha, beta, gamma)
print('axes = \n', axes)

StructInfo = read_atat_str('lat.in')
for item in StructInfo:
    print("{} = {}\n".format(item, StructInfo[item]))

"""


