
# ---------- import the python modules ------------------------------------------

from __future__ import division

import numpy as np
import numpy.linalg as LA

import sys
import os

sys.path.append("/home/kw7rr/msce_scripts/")
from atat_io import read_atat_latt
from folder_utils import get_folder_names
from bz_utils import get_KS_ECI

#################################################################################

ParentLattInfo = read_atat_latt('lat.in')
for item in ParentLattInfo: print("{}:\n{}\n".format(item, ParentLattInfo[item]))
ParentLattInfo['syminfo'] = 'hex'

directions = np.loadtxt('dir_HCP.in', float, usecols = (0, 1, 2, 3))
cse = np.loadtxt('cs.log', float)

folders = get_folder_names(incl = ['str.out'], excl = [])
get_KS_ECI(folders, cse, directions, ParentLattInfo)


#################################################################

