
import numpy as np
import os

# -----------------------------------------------------------------------------

def get_folder_names(prefix = './', incl = ['str.out', 'energy'], excl = ['error']):

    foldernames = []
    for content in os.listdir(prefix):
        if os.path.isdir(content):

            flag_incl = np.zeros([len(incl)])
            for I in range(len(incl)):
                if os.path.exists(content + '/' + incl[I]): flag_incl[I] = 1
            # end of for-I

            flag_excl = np.ones([len(excl)])
            for J in range(len(excl)):
                if not os.path.exists(content + '/' + excl[J]): flag_excl[J] = 0
            # end of for-J

            if np.sum(flag_incl) == len(flag_incl) and np.sum(flag_excl) == 0:
                foldernames.append(content)

    # foldernames.sort(key = int)
    foldernames.sort()

    return foldernames

# -----------------------------------------------------------------------------

