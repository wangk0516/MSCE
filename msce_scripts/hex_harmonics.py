
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

reference = \
"""
[1] van de Walle, Axel, Q. Hong, L. Miljacic, C. Balaji Gopal, S. Demers, G. Pomrehn, A. Kowalski, and Pratyush Tiwary.
    "Ab initio calculation of anisotropic interfacial excess free energies." Physical Review B 89, no. 18 (2014): 184101.
[2] van de Walle, Axel, C. Balaji Gopal, S. Demers, Q. Hong, A. Kowalski, L. Miljacic, G. Pomrehn, and P. Tiwary.
    "Symmetry-adapted bases for the parametrization of anisotropic properties." arXiv preprint arXiv:1301.0168 (2013).
"""
# print("reference: \n", reference)

###############################################################################

def independent_hex_harmonics(R, vec):
    
    if len(vec) == 3:
        r = np.sqrt(np.sum(vec**2))
        x, y, z = vec[:]
        # print("x, y, z, r = {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(x, y, z, r))
    else:
        raise ValueError("data type of vector is wrong: ", vec)
    # end of if    
    
    if R == 0:
        H = 1
    elif R == 1:
        H = + 0.408248*(x**2) + 0.408248*(y**2) + -0.816497*(z**2)
        H = H/r**2
    elif R == 2:
        H = + 0.179284*(x**4) + 0.358569*(x**2)*(y**2) + 0.179284*(y**4) + -1.43427*(x**2)*(z**2) \
            + -1.43427*(y**2)*(z**2) + 0.478091*(z**4)
        H = H/r**4
    elif R == 3:
        H = + 0.194972*x**6 + -2.30011*x**4*y**2 + 2.50827*x**2*y**4 + -0.125587*y**6 + -0.624465*x**4*z**2 \
            + -1.24893*x**2*y**2*z**2 + -0.624465*y**4*z**2 + 0.832621*x**2*z**4 + 0.832621*y**2*z**4 + -0.111016*z**6
        H = H/r**6
    elif R == 4:
        H = + 1.34224*x**4*y**2 + -0.894825*x**2*y**4 + 0.149137*y**6 + -1.34224*x**4*z**2 + -2.68447*x**2*y**2*z**2 \
            + -1.34224*y**4*z**2 + 1.78965*x**2*z**4 + 1.78965*y**2*z**4 + -0.23862*z**6
        H = H/r**6
    elif R == 5:
        H = + 0.0751924*x**8 + -0.696672*x**6*y**2 + 0.118674*x**4*y**4 + 0.854904*x**2*y**6 + -0.0356344*y**8 \
            + -1.40871*x**6*z**2 + 9.73804*x**4*y**2*z**2 + -13.5356*x**2*y**4*z**2 + 0.142861*y**6*z**2 + 1.89878*x**4*z**4 \
            + 3.79756*x**2*y**2*z**4 + 1.89878*y**4*z**4 + -1.01268*x**2*z**6 + -1.01268*y**2*z**6 + 0.0723345*z**8
        H = H/r**8
    elif R == 6:
        H = + 0.595912*x**6*y**2 + 0.198637*x**4*y**4 + -0.331062*x**2*y**6 + 0.0662124*y**8 + -0.595912*x**6*z**2 \
            + -10.1305*x**4*y**2*z**2 + 3.77411*x**2*y**4*z**2 + -1.52289*y**6*z**2 + 3.17819*x**4*z**4 \
            + 6.35639*x**2*y**2*z**4 + 3.17819*y**4*z**4 + -1.69504*x**2*z**6 + -1.69504*y**2*z**6 + 0.121074*z**8
        H = H/r**8
    else:
        raise ValueError("only the first 6 hex harmonics are implemented!")
    # end of if

    return H

def calc_hex_surf_value(coeffs, vec):

    ee = 0.0
    for K in range(len(coeffs)): ee = ee + coeffs[K]*independent_hex_harmonics(K, vec)
 
    return ee

def plot_hex_harmonics(rnk, ngrids_phi = 181, ngrids_theta = 91):

    """
    Spherical coordinates (r, theta, phi) as commonly used in physics (ISO 80000-2:2019 convention):
    ---- radial distance r (distance to origin), The symbol (rho) is often used instead of r;
    ---- polar angle (theta) (angle with respect to polar axis);
    ---- azimuthal angle (phi) (angle of rotation from the initial meridian plane);
    """

    phi = np.linspace(0, 2*np.pi, ngrids_phi)
    theta = np.linspace(0, np.pi, ngrids_theta)
    phi_mesh, theta_mesh = np.meshgrid(phi, theta)
    rho_mesh = np.zeros(np.shape(phi_mesh))
    print('shape of phi and theta: ', np.shape(phi), np.shape(theta))
    print('shape of phi_mesh, theta_mesh, and rho_mesh: ', np.shape(phi_mesh), np.shape(theta_mesh), np.shape(rho_mesh))

    for I in range(len(theta)):
        for J in range(len(phi)):
            vec = np.array([np.sin(theta_mesh[I, J])*np.cos(phi_mesh[I, J]), np.sin(theta_mesh[I, J])*np.sin(phi_mesh[I, J]), np.cos(theta_mesh[I, J])])
            rho_mesh[I, J] = independent_hex_harmonics(rnk, vec)
        # end of for-J
    # end of for-I
    figtitle = "Hexgonal Harmonics $H_" + str(rnk) + "$"
    
    x = rho_mesh*np.sin(theta_mesh)*np.cos(phi_mesh)
    y = rho_mesh*np.sin(theta_mesh)*np.sin(phi_mesh)
    z = rho_mesh*np.cos(theta_mesh)
    
    norm = colors.Normalize()
    
    figsize = plt.figaspect(1.)
    fig, ax = plt.subplots(subplot_kw = dict(projection = '3d'), figsize = plt.figaspect(1.)*2)
    
    cmap = cm.ScalarMappable(cmap = cm.jet)
    ax.plot_surface(x, y, z, rstride = 1, cstride = 1, shade = False, facecolors = cm.jet(norm(rho_mesh)), zorder = 1)
    
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    ax.axis('off')
    
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    
    ax.set_title(figtitle)
                   
    cmap.set_array(rho_mesh)
    fig.colorbar(cmap, shrink = 0.75);
    
    ###########################################################################
    # adjust the margins
    
    margins = {  #     vvv margin in inches
        "left"   :     1.0 / figsize[0],
        "bottom" :     1.5 / figsize[1],
        "right"  : 1 - 1.0 / figsize[0],
        "top"    : 1 - 1   / figsize[1]
    }
    fig.subplots_adjust(**margins)
    
    ###########################################################################
    
    plt.savefig('fig_hexharm_' + str(rnk) + '.pdf')
    # plt.show()

    return 0

###############################################################################

