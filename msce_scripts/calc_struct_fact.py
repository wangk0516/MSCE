
# ---------- import the python modules ------------------------------------------

from __future__ import division

import numpy as np
import numpy.linalg as LA

import sys
import os

sys.path.append("/home/kw7rr/msce_scripts")
from atat_io import read_atat_latt
from folder_utils import get_folder_names
from bz_utils import calculate_structural_factors

#################################################################################

ParentLattInfo = read_atat_latt('lat.in')
for item in ParentLattInfo: print("{}:\n{}\n".format(item, ParentLattInfo[item]))

folders = get_folder_names(incl = ['str.out'], excl = [])
calculate_structural_factors(ParentLattInfo, folders, ['Mg', 'Zn'])

#################################################################

