# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import matplotlib.pyplot as plt
from .functions import *


def plotTSE(h, dim=10, nTriples=10, nThetas=100, R=1, onSimplex=True, 
            randseed=-1):
    """
    Plot estimated triangle scaling exponents of Bregman distance.
    """
    
    if randseed >= 0:
        np.random.seed(randseed)
    
    plt.figure()

    for k in range(nTriples):
        x = R * np.random.rand(dim)
        y = R * np.random.rand(dim)
        z = R * np.random.rand(dim)
        if onSimplex:
            x = x / x.sum()
            y = y / y.sum()
            z = z / z.sum()
        
        theta = np.arange(1.0/nThetas, 1, 1.0/nThetas)
        expnt = np.zeros(theta.shape)
        dyz = h.divergence(y, z)

        for i in range(theta.size):
            c = theta[i]
            dtheta = h.divergence((1-c)*x+c*y, (1-c)*x+c*z)
            expnt[i] = np.log(dtheta / dyz) / np.log(c)
            #expnt[i] = (np.log(dtheta) - np.log(dyz)) / np.log(c)
        plt.plot(theta, expnt)

    plt.xlim([0,1])
    #plt.ylim([0,5])
    #plt.xlabel(r'$\theta$')
    #plt.ylabel(r'$\hat{\gamma}(\theta)$')
    plt.tight_layout()
    

if __name__ == "__main__":

    h = ShannonEntropy()
    #h = BurgEntropy()
    #h = SquaredL2Norm()
    #h = SumOf2nd4thPowers(1)

    plotTSE(h)
