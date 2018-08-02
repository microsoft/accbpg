# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import matplotlib.pyplot as plt
from .algorithms import *


def compare_gamma(f, h, L, x0, gamma=[1, 1.5, 2], gammaDA=2, theta_eq=True, 
               restart=False, maxitrs=1000, dispskip=100, xlim=[], ylim=[]):
    """
    Compare BPG and ABPG with different TSE gamma, as well as ABDA.
    """
    
    F = []
    G = []
    labels = []
    
    (x_BPG, F_BPG) = BPG(f, h, L, x0, maxitrs, verbskip=dispskip)
    F.append(F_BPG)
    G.append(np.ones(len(F_BPG)))
    labels.append(r'BPG')
    
    for i in range(len(gamma)):    
        (x_ABPG, F_ABPG, G_ABPG) = ABPG(f, h, L, gamma[i], x0, maxitrs, 
                theta_eq=theta_eq, restart=restart, verbskip=dispskip)
        F.append(F_ABPG)
        G.append(G_ABPG)
        labels.append(r'ABPG $\gamma$={0:3.1f}'.format(gamma[i]))
        
    (x_ABDA, F_ABDA, G_ABDA) = ABDA(f, h, L, gammaDA, x0, maxitrs, 
                                    theta_eq=theta_eq, verbskip=dispskip)
    F.append(F_ABDA)
    G.append(G_ABDA)
    labels.append(r'ABDA $\gamma$={0:3.1f}'.format(gammaDA))

    plot_figures(F, G, labels, xlim=xlim, ylim=ylim,
                 linefmts=['k:', 'g-', 'b-.', 'k-', 'r--', 'k-', 'm-'],
                 linedash=[[1,2], [], [4,2,1,2], [], [4,2], [], []])

    
def compare_adapt(f, h, L, x0, gamma=2, theta_eq=True, restart=False, rho=1.2, 
                 checkdiv=False, maxitrs=1000, dispskip=100, xlim=[], ylim=[]):
    """
    Compare BPG and ABPG wither their adaptive version with line search.
    """
    
    F = []
    G = []
    labels = []
    
    (x_BPG, F_BPG) = BPG(f, h, L, x0, maxitrs, verbskip=dispskip)
    F.append(F_BPG)
    G.append(np.ones(len(F_BPG)))
    labels.append(r'BPG')
    
    (x_BPG, F_BPG, L_BPG) = BPG_LS(f, h, L, x0, maxitrs, ls_ratio=rho, 
                            ls_adapt=True, verbskip=dispskip, stop_eps=1e-24)
    F.append(F_BPG)
    G.append(L_BPG)
    labels.append(r'BPG-LS')
    
    (x_ABPG, F_ABPG, G_ABPG) = ABPG(f, h, L, gamma, x0, maxitrs, 
                theta_eq=theta_eq, restart=restart, verbskip=dispskip)
    F.append(F_ABPG)
    G.append(G_ABPG)
    labels.append(r'ABPG')

    (x_ABPG, F_ABPG, Gamma2, G_ABPG) = ABPG_expo(f, h, L, gamma+1, x0, maxitrs, 
                theta_eq=theta_eq, checkdiv=checkdiv, gainmargin=3, 
                restart=restart, verbskip=dispskip)
    F.append(F_ABPG)
    G.append(G_ABPG)
    labels.append(r'ABPG-e')
    
    (x_ABPG, F_ABPG, G_ABPG, Gdiv2) = ABPG_gain(f, h, L, gamma, x0, maxitrs, 
                                    G0=0.1, ls_adapt=True,
                                    ls_increment=rho, ls_decrement=rho,
                                    theta_eq=theta_eq, checkdiv=checkdiv, 
                                    restart=restart, verbskip=dispskip)
    F.append(F_ABPG)
    G.append(G_ABPG)
    labels.append(r'ABPG-g')

    plot_figures(F, G, labels, xlim=xlim, ylim=ylim,
                 linefmts=['k:', 'g-', 'b-.', 'k-', 'r--', 'k-', 'm-'],
                 linedash=[[1,2], [], [4,2,1,2], [], [4,2], [], []])
    
    
def compare_restart(f, h, L, x0, gamma=2, theta_eq=True, rho=1.2, 
                checkdiv=False, maxitrs=1000, dispskip=100, xlim=[], ylim=[]):
    """
    Compare BPG and ABPG with restart on problems with relative strong convex.
    """
    
    F = []
    G = []
    labels = []
    
    (x_BPG, F_BPG) = BPG(f, h, L, x0, maxitrs, verbskip=dispskip)
    F.append(F_BPG)
    G.append(np.ones(len(F_BPG)))
    labels.append(r'BPG')
    
    (x_BPG, F_BPG, L_BPG) = BPG_LS(f, h, L, x0, maxitrs, ls_ratio=rho, 
                            ls_adapt=True, verbskip=dispskip, stop_eps=1e-24)
    F.append(F_BPG)
    G.append(L_BPG)
    labels.append(r'BPG-LS')
    
    (x_ABPG, F_ABPG, G_ABPG) = ABPG(f, h, L, gamma, x0, maxitrs, 
                theta_eq=theta_eq, restart=False, verbskip=dispskip)
    F.append(F_ABPG)
    G.append(G_ABPG)
    labels.append(r'ABPG')

    (x_ABPG, F_ABPG, G_ABPG) = ABPG(f, h, L, gamma, x0, maxitrs, 
                theta_eq=theta_eq, restart=True, verbskip=dispskip)
    F.append(F_ABPG)
    G.append(G_ABPG)
    labels.append(r'ABPG RS')

    (x_ABPG, F_ABPG, G_ABPG, Gdiv2) = ABPG_gain(f, h, L, gamma, x0, maxitrs, 
                                    G0=0.1, ls_adapt=True,
                                    ls_increment=rho, ls_decrement=rho,
                                    theta_eq=theta_eq, checkdiv=checkdiv, 
                                    restart=False, verbskip=dispskip)
    F.append(F_ABPG)
    G.append(G_ABPG)
    labels.append(r'ABPG-g')
    
    (x_ABPG, F_ABPG, G_ABPG, Gdiv2) = ABPG_gain(f, h, L, gamma, x0, maxitrs, 
                                    G0=0.1, ls_adapt=True,
                                    ls_increment=rho, ls_decrement=rho,
                                    theta_eq=theta_eq, checkdiv=checkdiv, 
                                    restart=True, verbskip=dispskip)
    F.append(F_ABPG)
    G.append(G_ABPG)
    labels.append(r'ABPG-g RS')
    
    plot_figures(F, G, labels, xlim=xlim, ylim=ylim,
                 linefmts = ['k:', 'g-', 'b-', 'm-.', 'k-', 'r--'],
                 linedash = [[1,2], [], [4,2,1,2], [4,2,1,2,1,2], [], [4,2]])


def plot_figures(F, G, labels, fontsize=20, usetex=False, xlim=[], ylim=[],
                 linefmts=['k:', 'g-', 'b-.', 'k-', 'r--', 'k-', 'm-'],
                 linedash=[[1,2], [], [4,2,1,2], [], [4,2], [], []]):
    """
    Plot comparison figures using matplotlib.pyplot.
    """

    # use computer modern fonts for figures in paper
    if usetex:
        plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
        plt.rc('text', usetex=True)
    else:
        plt.rc('font', **{'family':'serif'})

    # set different fontsizes
    plt.rc('xtick', labelsize=fontsize-2)
    plt.rc('ytick', labelsize=fontsize-2)
    plt.rc('lines', linewidth=2)
    plt.rc('font', size=fontsize)
    plt.rc('legend',**{'fontsize':fontsize-4})

    # find the minimum value of all    
    Fmin = F[0].min()
    for i in range(len(F)):
        #Fmin = min(Fmin, F[i].min())
        Fmin = min(Fmin, F[i][-1])

    # semilogy plot of objective gap
    plt.figure()
    for i in range(len(F)):
        plt.semilogy(np.arange(len(F[i]))+1, F[i]-Fmin, linefmts[i], 
                     dashes=linedash[i], label=labels[i])
    plt.legend()
    plt.xlabel(r'iteration number $k$')
    plt.ylabel(r'$\phi(x_k)-\phi_\star$')
    if len(xlim) > 0:
        plt.xlim(xlim)
    if len(ylim) > 0:
        plt.ylim(ylim)
    plt.tight_layout()
    
    # loglog plot of objective gap
    plt.figure()
    for i in range(len(F)):
        plt.loglog(np.arange(len(F[i]))+1, F[i]-Fmin, linefmts[i], 
                   dashes=linedash[i], label=labels[i])
    plt.legend()
    plt.xlabel(r'iteration number $k$')
    plt.ylabel(r'$\phi(x_k)-\phi_\star$')
    if len(xlim) > 0:
        plt.xlim(xlim)
    if len(ylim) > 0:
        plt.ylim(ylim)
    plt.tight_layout()

     # plot of gains determining the step sizes
    plt.figure()
    for i in range(1,len(G)):
        plt.semilogy(np.arange(len(G[i])), G[i], linefmts[i], 
                     dashes=linedash[i], label=labels[i])
    plt.legend()
    #plt.legend(loc='center right')
    plt.xlabel(r'iteration number $k$')
    plt.ylabel(r'$G_k$')
    if len(xlim) > 0:
        plt.xlim(xlim)
    plt.tight_layout()
