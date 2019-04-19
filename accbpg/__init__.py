# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from .functions import *
from .algorithms import BPG, ABPG, ABPG_expo, ABPG_gain, ABDA
from .applications import D_opt_design, D_opt_KYinit, Poisson_regrL1, Poisson_regrL2, KL_nonneg_regr
from .comparisons import compare_gamma, compare_adapt, compare_restart
from .D_opt_alg import D_opt_FW, D_opt_WATY
from .trianglescaling import plotTSE
from .plotfigs import plot_comparisons