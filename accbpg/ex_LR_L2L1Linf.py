# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


"""
Example of logistic regression with L1 regularization and Linf bounds

    minimize_x  f(x) = (1/m) * sum_{i=1}^m log(1 + exp(-b_i*(ai'*x)))
    subject to  x in R^n, and ||x||_inf <= B  

The objective f is 1-relative smooth relative to (1/2)||x||_2^2.

"""

import numpy as np
from functions import RSmoothFunction, L2L1Linf, SquaredL2Norm
from algorithms import BPG, ABPG_gain

class LogisticRegression(RSmoothFunction):
    """
    f(x) = (1/m)*sum_{i=1}^m log(1 + exp(-b_i*(ai'*x))) with ai in R^n, bi in R
    """
    def __init__(self, A, b):
        assert len(b) == A.shape[0], "Logistic Regression: len(b) != m"
        self.bA = np.reshape(b, [len(b),1]) * A
        self.m = A.shape[0]
        self.n = A.shape[1]
        
    def __call__(self, x):
        return self.func_grad(x, flag=0)
        
    def gradient(self, x):
        return self.func_grad(x, flag=1)
        
    def func_grad(self, x, flag=2):
        assert x.size == self.n, "Logistic Regression: x.size not equal to n"

        bAx = np.dot(self.bA, x)
        
        loss = - bAx
        mask = bAx > -50
        loss[mask] = np.log(1 + np.exp(-bAx[mask]))
        f = np.sum(loss) / self.m

        if flag == 0:
            return f
        
        p = -1/(1+np.exp(bAx)) 
        g = np.dot(p, self.bA) / self.m
        
        if flag == 1:
            return g

        return f, g        


def test_L2L1Linf():

    m = 100
    n = 200
    A = np.random.randn(m, n)
    #b = np.sign(A[:, 0])
    b = np.sign(np.random.rand(m,1))

    f = LogisticRegression(A, b)
    #h = SquaredL2Norm()
    h = L2L1Linf(lamda=1.0/m, B=1)

    L = 0.25
    x0 = np.zeros(n)
    maxitrs = 100

    x1, F1, G1, _ = BPG(f, h, L, x0, maxitrs, verbskip=10)
    
    x2, F2, G2, _, _, _ = ABPG_gain(f, h, L, x0, gamma=2, maxitrs=maxitrs,
                                 restart=False, verbskip=10)
    
if __name__ == "__main__":
    test_L2L1Linf()    