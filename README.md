# Accelerated Bregman Proximal Gradient Methods

Accelerated first-order algorithms for solving relatively-smooth convex optimization problems of the form

    minimize { f(x) + P(x) | x in C }

with a reference function h(x), where

* h(x) is convex and essentially smooth on C
* f(x) is convex and differentiable, and L-smooth relative to h(x), that is, f(x)-L*h(x) is convex
* P(x) is convex and closed (lower semi-continuous)
* C is a closed convex set

### Implemented algorithms in [HRX2018](https://arxiv.org/abs/1808.03045)

* BPG_LS (Bregman proximal gradient) method with line search
* ABPG (Accelerated BPG) method
* ABPG-expo (ABPG with exponent adaption)
* ABPG-gain (ABPG with gain adaption)
* ABDA (Accelerated Bregman dual averaging) method

## Installation

Clone or fork from GitHub. Or install from PyPI:

    pip install accbpg

## Usage

    import accbpg

    # generate a random instance of D-optimal design problem
    f, h, L, x0 = accbpg.D_opt_design(80, 200)

    # solve the problem instance using BPG with line search
    x1, F1, G1 = accbpg.BPG_LS(f, h, L, x0, maxitrs=1000, verskip=100)

    # solve it again using ABPG_gain with gamma=2
    x2, F2, G2, D2 = accbpg.ABPG_gain(f, h, L, 2, x0, maxitrs=1000, verbskip=100)

compare the two methods by visualization

    import matplotlib.pyplot as plt
    Fmin = min(F1.min(), F2.min())
    plt.semilogy(range(len(F1)), F1-Fmin, range(len(F2)), F2-Fmin)

## Examples in [HRX2018](https://arxiv.org/abs/1808.03045)

**D-optimal experiment design**
    
    import accbpg.ex_D_opt

**Nonnegative regression with KL-divergence**

    import accbpg.ex_KL_regr

**Poisson linear inverse problems**

    import accbpg.ex_PoissonL1
    import accbpg.ex_PoissonL2

