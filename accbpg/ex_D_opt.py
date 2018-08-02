# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


"""
Numerical experiments on the D-Optimal design problem.
    minimize_x f(x) = - log(det(H*diag(x)*H'))
    subject to: x in unit simplex in R^n
The objective f is 1-relative smooth relative to Burg entropy.

"""

from .applications import * 
from .comparisons import *

import matplotlib.pyplot as plt
plt.close('all')

randseed = 10       # random seed used to generate figures in the paper
#randseed = -1      # use different seed to explore random experiments

# the hard case, sublinear convergence
m = 80
n = 200

f, h, L, x0 = D_opt_design(m, n, randseed)

compare_gamma(f, h, L, x0, gamma=[1, 1.5, 2.0], gammaDA=2.2, theta_eq=True, 
              restart=False, maxitrs=3000, xlim=[0,1000], ylim=[1e-5,2])

compare_adapt(f, h, L, x0, gamma=2, theta_eq=True, restart=False, 
              maxitrs=3000, xlim=[0,1000], ylim=[1e-5,2])

# the easy case, linear convergence with restart
m = 80
n = 140

f, h, L, x0 = D_opt_design(m, n, randseed)

compare_restart(f, h, L, x0, gamma=2, theta_eq=True, maxitrs=200, 
                xlim=[0,100], ylim=[1e-10,1])

# even easier, and faster linear convergence
m = 80
n = 120

f, h, L, x0 = D_opt_design(m, n, randseed)

compare_restart(f, h, L, x0, gamma=2, theta_eq=True, maxitrs=200, 
                xlim=[0,50], ylim=[1e-11,1])
