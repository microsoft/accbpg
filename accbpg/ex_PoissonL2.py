# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


"""
Numerical experiments on L2-regularized Poisson regression problem:
    minimize_{x >= 0}  D_KL(b, Ax) + (lamda/2) * ||x||_2^2
where 
    A:  m by n nonnegative matrix
    b:  nonnegative vector of length m
    noise:  noise level to generate b = A * x + noise
    lambda: L2 regularization weight
    normalizeA: wether or not to normalize columns of A
    
The objective function D_KL(b,A*x) is ||b||_1-smooth relative to Burg entropy.

"""

from .applications import * 
from .comparisons import *

import matplotlib.pyplot as plt
plt.close('all')

randseed = 1        # random seed used to generate figures in the paper
#randseed = -1      # use different seed for each run

# the hard case, sublinear convergence
m = 200
n = 100
lamda = 0.0001      # lamda !=0 even for overdeterminted case
gamma = 2

f, h, L, x0 = Poisson_regrL2(m, n, noise=0.001, lamda=lamda, randseed=randseed)

compare_gamma(f, h, L, x0, gamma=[1, 1.5, 2], gammaDA=gamma, theta_eq=True, 
           restart=False, maxitrs=5000, xlim=[], ylim=[])

compare_adapt(f, h, L, x0, gamma=gamma, theta_eq=False, restart=False, 
              rho=1.5, checkdiv=False, maxitrs=5000, xlim=[], ylim=[])

compare_restart(f, h, L, x0, gamma=gamma, theta_eq=False, rho=1.5, 
                checkdiv=False, maxitrs=5000, xlim=[], ylim=[])

# the hard case, sublinear convergence
m = 100
n = 1000
lamda = 0.001
gamma = 2

f, h, L, x0 = Poisson_regrL2(m, n, noise=0.001, lamda=lamda, randseed=randseed)

compare_gamma(f, h, L, x0, gamma=[1, 1.5, 2], gammaDA=gamma, theta_eq=True, 
              restart=False, maxitrs=10000, xlim=[], ylim=[])

compare_adapt(f, h, L, x0, gamma=gamma, theta_eq=False, restart=False, rho=1.5, 
              checkdiv=False, maxitrs=10000, xlim=[-20,2000], ylim=[1e-9,1e-1])

compare_restart(f, h, L, x0, gamma=gamma, theta_eq=False, rho=1.5, 
                checkdiv=False, maxitrs=5000, xlim=[], ylim=[])

