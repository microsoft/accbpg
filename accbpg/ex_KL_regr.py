# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


"""
Numerical experiments on nonnegative regression with KL-divergence.
    minimize_x D_KL(A*x, b)
            minimize_{x >= 0}  D_KL(Ax, b) + lamda * ||x||_1
    where 
        A:  m by n nonnegative matrix
        b:  nonnegative vector of length m
        noise:  noise level to generate b = A * x + noise
        lambda: L2 regularization weight
        normalizeA: wether or not to normalize columns of A

The objective D_KL(A*x,b) is L-relative smooth relative to Shannon entropy,
wher L = max(sum(A, axis=0)), which is the maximum column sum of matrix A.

"""

from .applications import * 
from .comparisons import *

import matplotlib.pyplot as plt
plt.close('all')

randseed = 1        # random seed used to generate figures in the paper
#randseed = -1      # use different seed for each run

# the easy case? sublinear convergence
m = 1000
n = 100
lamda = 0.001

f, h, L, x0 = KL_nonneg_regr(m, n, noise=0.01, lamdaL1=lamda, 
                             randseed=randseed, normalizeA=False)

compare_gamma(f, h, L, x0, gamma=[1, 1.5, 2], gammaDA=2.0, theta_eq=False, 
              restart=False, maxitrs=5000, xlim=[-20,3000], ylim=[])

compare_adapt(f, h, L, x0, gamma=2, theta_eq=True, restart=False, rho=1.2,
              maxitrs=5000, xlim=[-20,3000], ylim=[])

compare_restart(f, h, L, x0, gamma=2, theta_eq=True, rho=1.2, maxitrs=5000, 
                xlim=[-20,2000], ylim=[1e-10,1e2])

# the hard case, sublinear convergence
m = 100
n = 1000
lamda = 0.001

f, h, L, x0 = KL_nonneg_regr(m, n, noise=0.01, lamdaL1=lamda, 
                             randseed=randseed, normalizeA=True)

compare_gamma(f, h, L, x0, gamma=[1, 1.5, 2], gammaDA=2.0, theta_eq=False, 
              restart=False, maxitrs=5000, xlim=[-30, 3000], ylim=[])

compare_adapt(f, h, L, x0, gamma=2, theta_eq=False, restart=False, rho=1.2,
              maxitrs=5000, xlim=[-30,3000], ylim=[])

compare_restart(f, h, L, x0, gamma=2, theta_eq=False, rho=1.2, maxitrs=5000, 
                xlim=[-20,2000], ylim=[1e-10,1e-1])
