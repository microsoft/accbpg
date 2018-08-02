# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
from .functions import *


def D_opt_design(m, n, randseed=-1):
    """
    Generate a random instance of the D-Optimal Design problem
        m, n: size of design matrix H is m by n wiht m < n
    Return f, h, L, x0:
        f:  f(x) = - log(det(H*diag(x)*H'))
        h:  Burg Entrop with Simplex constraint
        L:  L = 1
        x0: initial point is center of simplex
    """

    if randseed > 0:
        np.random.seed(randseed)
    H = np.random.randn(m,n)

    f = DOptimalObj(H)
    h = BurgEntropySimplex()
    L = 1.0
    x0 = (1.0/n)*np.ones(n)
    
    return f, h, L, x0


def Poisson_regrL1(m, n, noise=0.01, lamda=0, randseed=-1, normalizeA=True):
    """
    Generate a random instance of L1-regularized Poisson regression problem
            minimize_{x >= 0}  D_KL(b, Ax) + lamda * ||x||_1
    where 
        A:  m by n nonnegative matrix
        b:  nonnegative vector of length m
        noise:  noise level to generate b = A * x + noise
        lambda: L1 regularization weight
        normalizeA: wether or not to normalize columns of A
    
    Return f, h, L, x0: 
        f: f(x) = D_KL(b, Ax)
        h: Burg entropy with L1 regularization 
        L: L = ||b||_1
        x0: initial point, scaled version of all-one vector
    """
    
    if randseed > 0:
        np.random.seed(randseed)
    A = np.random.rand(m,n)
    if normalizeA:
        A = A / A.sum(axis=0)   # scaling to make column sums equal to 1
    x = np.random.rand(n) / n
    xavg = x.sum() / x.size
    x = np.maximum(x - xavg, 0) * 10
    b = np.dot(A, x) + noise * (np.random.rand(m) - 0.5)
    assert b.min() > 0, "need b > 0 for nonnegative regression."

    f = PoissonRegression(A, b)
    # L1 regularization often not enough for convergence!
    h = BurgEntropyL1(lamda)
    L = b.sum()
    # Initial point should be far from 0 in order for ARDA to work well!
    x0 = (1.0/n)*np.ones(n) * 10

    return f, h, L, x0


def Poisson_regrL2(m, n, noise=0.01, lamda=0, randseed=-1, normalizeA=True):
    """
    Generate a random instance of L2-regularized Poisson regression problem
            minimize_{x >= 0}  D_KL(b, Ax) + (lamda/2) * ||x||_2^2
    where 
        A:  m by n nonnegative matrix
        b:  nonnegative vector of length m
        noise:  noise level to generate b = A * x + noise
        lambda: L2 regularization weight
        normalizeA: wether or not to normalize columns of A
    
    Return f, h, L, x0: 
        f: f(x) = D_KL(b, Ax)
        h: Burg entropy with L1 regularization 
        L: L = ||b||_1
        x0: initial point is center of simplex
    """

    if randseed > 0:
        np.random.seed(randseed)
    A = np.random.rand(m,n)
    if normalizeA:
        A = A / A.sum(axis=0)   # scaling to make column sums equal to 1
    x = np.random.rand(n) / n
    xavg = x.sum() / x.size
    x = np.maximum(x - xavg, 0) * 10
    b = np.dot(A, x) + noise * (np.random.rand(m) - 0.5)
    assert b.min() > 0, "need b > 0 for nonnegative regression."

    f = PoissonRegression(A, b)
    h = BurgEntropyL2(lamda)
    L = b.sum()
    # Initial point should be far from 0 in order for ARDA to work well!
    x0 = (1.0/n)*np.ones(n)

    return f, h, L, x0


def KL_nonneg_regr(m, n, noise=0.01, lamdaL1=0, randseed=-1, normalizeA=True):
    """
    Generate a random instance of L1-regularized KL regression problem
            minimize_{x >= 0}  D_KL(Ax, b) + lamda * ||x||_1
    where 
        A:  m by n nonnegative matrix
        b:  nonnegative vector of length m
        noise:  noise level to generate b = A * x + noise
        lambda: L2 regularization weight
        normalizeA: wether or not to normalize columns of A
    
    Return f, h, L, x0: 
        f: f(x) = D_KL(Ax, b)
        h: h(x) = Shannon entropy (with L1 regularization as Psi)
        L: L = max(sum(A, axis=0)), maximum column sum
        x0: initial point, scaled version of all-one vector
    """
    if randseed > 0:
        np.random.seed(randseed)
    A = np.random.rand(m,n)
    if normalizeA:
        A = A / A.sum(axis=0)   # scaling to make column sums equal to 1
    x = np.random.rand(n)
    b = np.dot(A, x) + noise * (np.random.rand(m) - 0.5)
    assert b.min() > 0, "need b > 0 for nonnegative regression."

    f = KLdivRegression(A, b)
    h = ShannonEntropyL1(lamdaL1)
    L = max( A.sum(axis=0) )    #L = 1.0 if columns of A are normalized
    x0 = 0.5*np.ones(n)
    #x0 = (1.0/n)*np.ones(n)

    return f, h, L, x0
