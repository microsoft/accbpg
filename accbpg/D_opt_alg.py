# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np


def D_opt_FW(V, x0, eps, maxitrs, verbose=True, verbskip=1):
    """
    Solve the D-optimal design problem by the Frank-Wolfe algorithm
        minimize     - log(det(V*diag(x)*V'))
        subject to   x >= 0  and sum_i x_i=1
    where V is m by n matrix and x belongs to n-dimensional simplex

    """
    m, n = V.shape
    F = np.zeros(maxitrs)

    x = np.copy(x0)
    VXVT = np.dot(V*x, V.T)
    detVXVT = np.linalg.det(VXVT)
    H = np.linalg.inv(VXVT)

    if verbose:
        print("\nFrank-Wolfe method for D-optimal design")
        print("     k      F(x)     w_max/m-1   1-w_min/m")
        
    for k in range(maxitrs):
        F[k] = - np.log(detVXVT)

        # compute w = - gradient # This step cost m^2*n
        w = np.sum(V * np.dot(H, V), axis=0)
        # check approximate optimality conditions        
        i = np.argmax(w)
        w_xpos = w[x>0]
        j = np.argmin(w_xpos)

        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}".format(
                    k, F[k], w[i]/m-1, 1-w_xpos[j]/m))

        if w[i] <= (1 + eps) * m and w_xpos[j] >= (1 - eps) * m:
            break
        
        t = (w[i] / m - 1) / (w[i] - 1)
        x *= (1 - t)
        x[i] += t
        HVi = np.dot(H, V[:,i])
        H = (H - (t / (1 + t * (w[i] - 1))) * np.outer(HVi, HVi)) / (1 - t)
        detVXVT *= np.power(1 - t, m - 1) * (1 + t * (w[i] - 1)) 
   
    F = F[0:k+1]
    return x, F

def D_opt_WATY(V, x0, eps, maxitrs, verbose=True, verbskip=1):
    """
    Solve the D-optimal design problem by the Frank-Wolfe algorithm
        minimize     - log(det(V*diag(x)*V'))
        subject to   x >= 0  and sum_i x_i=1
    where V is m by n matrix and x belongs to n-dimensional simplex

    """
    m, n = V.shape
    F = np.zeros(maxitrs)

    x = np.copy(x0)
    VXVT = np.dot(V*x, V.T)
    detVXVT = np.linalg.det(VXVT)
    H = np.linalg.inv(VXVT)

    if verbose:
        print("\nFrank-Wolfe method for D-optimal design")
        print("     k      F(x)     w_max/m-1   1-w_min/m")
        
    for k in range(maxitrs):
        #F[k] = - np.log(detVXVT)
        F[k] = np.log(np.linalg.det(H))

        # compute w = - gradient # This step cost m^2*n
        w = np.sum(V * np.dot(H, V), axis=0)
        # check approximate optimality conditions        
        i = np.argmax(w)
        ww = w - w[i]   # shift the array so that ww.max() = 0
        j = np.argmin(ww * [x > 1.0e-8])

        eps_pos = w[i] / m - 1
        eps_neg = 1 - w[j] / m
    
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}".format(
                    k, F[k], eps_pos, eps_neg))

        if eps_pos <= eps and eps_neg <= eps:
            break

        if eps_pos >= eps_neg:
            t = (w[i] / m - 1) / (w[i] - 1)
            x *= (1 - t)
            x[i] += t
            HVi = np.dot(H, V[:,i])
            H = (H - (t / (1 - t + t * w[i])) * np.outer(HVi, HVi)) / (1 - t)
            detVXVT *= np.power(1 - t, m - 1) * (1 + t * (w[i] - 1)) 
        else: # Wolfe's awaystep
            t = min((1 - w[j] / m) / (w[j] - 1), x[j] / (1 - x[j]))
            x *= (1 + t)
            x[j] -= t
            HVj = np.dot(H, V[:,j])
            H = (H + (t / (1 + t - t * w[j])) * np.outer(HVj, HVj)) / (1 + t)
            detVXVT *= np.power(1 + t, m - 1) * (1 + t - t * w[i]) 

    F = F[0:k+1]
    return x, F

