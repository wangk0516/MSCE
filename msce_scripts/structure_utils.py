
import numpy as np

ZERO_TOLERANCE = 1E-3

# -----------------------------------------------------------------------------
"""
def which_atom(atom_pos, pos, inv_cell):

    if len(np.shape(atom_pos)) == 1:
        natom = 1
    else:
        natom = np.shape(atom_pos)[0]
    # end of if

    for I in range(natom):
        if natom == 1:
            if equivalent_mod_cell(atom_pos, pos, inv_cell):
                return I
            # end of if
        else:
            if equivalent_mod_cell(atom_pos[I, :], pos, inv_cell):
                return I
            # end of if
        # end of if
    # end of for-I

    print("Not equivalent")
    return -1
"""

def which_atom(atom_pos, pos, inv_cell):

    atom_pos = np.reshape(atom_pos, (-1, 3))
    pos = np.reshape(pos, (-1, 3))

    for I in range(len(atom_pos[:, 0])):
        if equivalent_mod_cell(atom_pos[I, :], pos, inv_cell):
            return I
        # end if
    # end for-I

    return -1

def equivalent_mod_cell(kvec_a, kvec_b, inv_cell):

    # print('shape of kvec_a, kvec_b, inv_cell: ', np.shape(kvec_a), np.shape(kvec_b), np.shape(inv_cell))

    kvec_a = np.reshape(kvec_a, (3, 1))
    kvec_b = np.reshape(kvec_b, (3, 1))

    frac_a = np.matmul(inv_cell, kvec_a)
    frac_b = np.matmul(inv_cell, kvec_b)
    delta = frac_a - frac_b

    # delta_2 = np.matmul(inv_cell, kvec_a - kvec_b)
    # if LA.norm(delta - delta_2) > 1E-10: print('LA.norm(delta - delta_2) = ', LA.norm(delta - delta_2))

    for I in range(3): delta[I] = cylinder_norm_scalar(delta[I])

    if np.sum(delta**2) < ZERO_TOLERANCE:
        return 1
    else:
        return 0

def cylinder_norm_scalar(x):
    """
    If x is an integer (positive or negative), return 0.
    If the returned value is close to zero, x is close to an integer.
    """
    return np.abs((np.abs(x) + 0.5) % 1.0 - 0.5)

def test_cylinder_norm_scalar():

    import matplotlib.pyplot as plt

    figsize = np.array([7,  7])
    fig, ax = plt.subplots(figsize = figsize)

    x = np.linspace(-2, 2, 1001)
    y = cylinder_norm_scalar(x)

    ax.plot(x, y, 'b-', lw = 2)
    ax.grid()

    plt.show()

# -----------------------------------------------------------------------------

# test_cylinder_norm_scalar()

