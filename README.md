# Accelerated Bregman Proximal Gradient Methods

A Python package of accelerated first-order algorithms for solving relatively-smooth convex optimization problems

    minimize { f(x) + P(x) | x in C }

with a reference function h(x), where C is a closed convex set and

* h(x) is convex and essentially smooth on C
* f(x) is convex and differentiable, and L-smooth relative to h(x), that is, f(x)-L*h(x) is convex
* P(x) is convex and closed (lower semi-continuous)

### Implemented algorithms in [HRX2018](https://arxiv.org/abs/1808.03045)

* BPG(Bregman proximal gradient) method with line search option
* ABPG (Accelerated BPG) method
* ABPG-expo (ABPG with exponent adaption)
* ABPG-gain (ABPG with gain adaption)
* ABDA (Accelerated Bregman dual averaging) method

Additional algorithms for solving D-Optimal Experiment Design problems:

* D_opt_FW (basic Frank-Wolfe method)
* D_opt_FW_away (Frank-Wolfe method with away steps)

## Install

Clone or fork from GitHub. Or install from PyPI:

    pip install accbpg

## Usage

Example: generate a random instance of D-optimal design problem and solve it using two different methods

    import accbpg

    # generate a random instance of D-optimal design problem of size 80 by 200
    f, h, L, x0 = accbpg.D_opt_design(80, 200)

    # solve the problem instance using BPG with line search
    x1, F1, G1, T1 = accbpg.BPG(f, h, L, x0, maxitrs=1000, verbskip=100)

    # solve it again using ABPG with gamma=2
    x2, F2, G2, T2 = accbpg.ABPG(f, h, L, x0, gamma=2, maxitrs=1000, verbskip=100)

    # solve it again using adaptive variant of ABPG with gamma=2
    x3, F3, G3, _, _, T3 = accbpg.ABPG_gain(f, h, L, x0, gamma=2, maxitrs=1000, verbskip=100)

A complete example is given in this [Jupyter Notebook](ipynb/ex_Dopt_random.ipynb)

### Additional examples

* All examples in [HRX2018](https://arxiv.org/abs/1808.03045) can be found in the [ipynb](ipynb/) directory.

* Comparisons with the Frank-Wolfe method can be found in [ipynb/ABPGvsFW](ipynb/ABPGvsFW/)
