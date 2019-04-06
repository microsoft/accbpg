# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import time


def D_opt_FW(V, x0, eps, maxitrs, verbose=True, verbskip=1):
    """
    Solve the D-optimal design problem by the Frank-Wolfe algorithm
        minimize     - log(det(V*diag(x)*V'))
        subject to   x >= 0  and sum_i x_i=1
    where V is m by n matrix and x belongs to n-dimensional simplex

    """
    start_time = time.time()
    
    m, n = V.shape
    F = np.zeros(maxitrs)
    SP = np.zeros(maxitrs)
    SN = np.zeros(maxitrs)
    T = np.zeros(maxitrs)

    x = np.copy(x0)
    VXVT = np.dot(V*x, V.T)
    detVXVT = np.linalg.det(VXVT)
    H = np.linalg.inv(VXVT)

    # compute w = - gradient # This step cost m^2*n
    w = np.sum(V * np.dot(H, V), axis=0)

    if verbose:
        print("\nSolving D-opt design problem using Frank-Wolfe method")
        print("     k      F(x)     pos_slack   neg_slack    time")
        
    for k in range(maxitrs):
        F[k] = - np.log(detVXVT)
        T[k] = time.time() - start_time

        # compute w = - gradient # This step cost m^2*n
        #w = np.sum(V * np.dot(H, V), axis=0)
        
        # check approximate optimality conditions        
        i = np.argmax(w)
        w_xpos = w[x>0]
        j = np.argmin(w_xpos)

        eps_pos = w[i] / m - 1
        eps_neg = 1 - w_xpos[j] / m
        SP[k] = eps_pos
        SN[k] = eps_neg

        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:6.1f}".format(
                    k, F[k], eps_pos, eps_neg, T[k]))

        if eps_pos <= eps and eps_neg <= eps:
            break
        
        t = (w[i] / m - 1) / (w[i] - 1)
        x *= (1 - t)
        x[i] += t
        HVi = np.dot(H, V[:,i])
        H = (H - (t / (1 + t * (w[i] - 1))) * np.outer(HVi, HVi)) / (1 - t)
        detVXVT *= np.power(1 - t, m - 1) * (1 + t * (w[i] - 1)) 
        # compute w more efficiently # This step cost m*n
        w = (w - (t / (1 + t * (w[i] - 1))) * np.dot(HVi, V)**2 ) / (1 - t)
   
    F = F[0:k+1]
    SP = SP[0:k+1]
    SN = SN[0:k+1]
    T = T[0:k+1]
    return x, F, SP, SN, T


def D_opt_WATY(V, x0, eps, maxitrs, verbose=True, verbskip=1):
    """
    Solve the D-optimal design problem by Frank-Wolfe (Wolfe-Atwood) algorithm
        minimize     - log(det(V*diag(x)*V'))
        subject to   x >= 0  and sum_i x_i=1
    where V is m by n matrix and x belongs to n-dimensional simplex

    """
    start_time = time.time()

    m, n = V.shape
    F = np.zeros(maxitrs)
    SP = np.zeros(maxitrs)
    SN = np.zeros(maxitrs)
    T = np.zeros(maxitrs)

    x = np.copy(x0)
    VXVT = np.dot(V*x, V.T)
    detVXVT = np.linalg.det(VXVT)
    H = np.linalg.inv(VXVT)

    # compute w = - gradient # This step cost m^2*n
    w = np.sum(V * np.dot(H, V), axis=0)

    if verbose:
        print("\nSolving D-opt design problem using Frank-Wolfe method with away steps")
        print("     k      F(x)     pos_slack   neg_slack    time")
        
    for k in range(maxitrs):
        F[k] = np.log(np.linalg.det(H))
        # the following can be much faster but often inaccurate!
        #F[k] = - np.log(detVXVT)
        T[k] = time.time() - start_time

        # compute w = - gradient # This step cost m^2*n
        #w = np.sum(V * np.dot(H, V), axis=0)

        # check approximate optimality conditions        
        i = np.argmax(w)
        ww = w - w[i]   # shift the array so that ww.max() = 0
        j = np.argmin(ww * [x > 1.0e-8])

        eps_pos = w[i] / m - 1
        eps_neg = 1 - w[j] / m
        SP[k] = eps_pos
        SN[k] = eps_neg
   
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:6.1f}".format(
                    k, F[k], eps_pos, eps_neg, T[k]))

        if eps_pos <= eps and eps_neg <= eps:
            break

        if eps_pos >= eps_neg:
            t = (w[i] / m - 1) / (w[i] - 1)
            x *= (1 - t)
            x[i] += t
            HVi = np.dot(H, V[:,i])
            H = (H - (t / (1 - t + t * w[i])) * np.outer(HVi, HVi)) / (1 - t)
            detVXVT *= np.power(1 - t, m - 1) * (1 + t * (w[i] - 1)) 
            # compute w more efficiently # This step cost m*n
            w = (w - (t / (1 - t + t * w[i])) * np.dot(HVi, V)**2 ) / (1 - t)
        else: # Wolfe's awaystep
            t = min((1 - w[j] / m) / (w[j] - 1), x[j] / (1 - x[j]))
            x *= (1 + t)
            x[j] -= t
            HVj = np.dot(H, V[:,j])
            H = (H + (t / (1 + t - t * w[j])) * np.outer(HVj, HVj)) / (1 + t)
            detVXVT *= np.power(1 + t, m - 1) * (1 + t - t * w[i]) 
            # compute w more efficiently # This step cost m*n
            w = (w + (t / (1 + t - t * w[j])) * np.dot(HVj, V)**2 ) / (1 + t)

    F = F[0:k+1]
    SP = SP[0:k+1]
    SN = SN[0:k+1]
    T = T[0:k+1]
    return x, F, SP, SN, T

