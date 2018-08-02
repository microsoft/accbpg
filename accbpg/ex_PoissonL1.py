# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


"""
Numerical experiments on L1-regularized Poisson regression problem:
    minimize_{x >= 0}  D_KL(b, Ax) + lamda * ||x||_1
where 
    A:  m by n nonnegative matrix
    b:  nonnegative vector of length m
    noise:  noise level to generate b = A * x + noise
    lambda: L2 regularization weight
    normalizeA: wether or not to normalize columns of A
    
The objective function D_KL(b,A*x) is ||b||_1-smooth relative to Burg entropy.

!!! NOT well-formulated if m < n because the proximal mapping may be infinite.

"""

from .applications import * 
from .comparisons import *

import matplotlib.pyplot as plt
plt.close('all')

randseed = 1       # random seed used to generate figures in the paper
#randseed = -1      # use different seed for each run

# the hard case, sublinear convergence
m = 200
n = 100
lamda = 0

f, h, L, x0 = Poisson_regrL1(m, n, noise=0.0001, lamda=lamda, randseed=randseed)

compare_gamma(f, h, L, x0, gamma=[1, 1.5, 2], gammaDA=2, theta_eq=True,
              restart=False, maxitrs=10000, dispskip=1000, 
              xlim=[-50,5000], ylim=[1e-6,200])

compare_adapt(f, h, L, x0, gamma=2, theta_eq=True, restart=False, rho=1.5,
              maxitrs=10000, dispskip=1000, xlim=[-50,5000], ylim=[1e-6,10])

compare_restart(f, h, L, x0, gamma=2, theta_eq=True, rho=1.5, maxitrs=10000, 
                dispskip=1000, xlim=[-50,5000], ylim=[3e-7,10])

