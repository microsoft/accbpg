# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np


def BPG(f, h, L, x0, maxitrs, verbose=True, verbskip=1, stop_eps=1e-16):
    """   
    Bregman Proximal Gradient (BPG) method for min_{x in C} f(x) + Psi(x): 
    x(k+1) = argmin_{x in C} { Psi(x) + < f'(x(k)), x > + L*D_h(x,x(k)) }
    
    Inputs: 
        f, h, L:  f is L-smooth relative to h, and Psi is defined with h
        x0:       initial point to start algorithm
        maxitrs:  maximum number of iterations
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays
        stop_eps: stop if F(x[k])-F(x[k-1]) < stop_eps, where F(x)=f(x)+Psi(x)
    
    Returns (x, Fx):
        x:  the last iterate of BPG
        Fx: array storing F(x[k]) for all k
    """

    if verbose:
        print("\nBPG method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)")
    
    Fx = np.zeros(maxitrs)
    
    x = x0
    for k in range(maxitrs):
        fx, g = f.func_grad(x)
        Fx[k] = fx + h.extra_Psi(x)
        
        x = h.div_prox_map(x, g, L)
        
        # store and display computational progress
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}".format(k, Fx[k]))

        # stopping criteria
        if k > 0 and abs(Fx[k]-Fx[k-1]) < stop_eps:
            break;

    Fx = Fx[0:k+1]
    return x, Fx        


def BPG_LS(f, h, L, x0, maxitrs, linesearch=True, ls_ratio=2, ls_adapt=True,
           verbose=True, verbskip=1, stop_eps=1e-16):
    """
    Bregman Proximal Gradient (BGP) method for min_{x in C} f(x) + Psi(x): 
    x(k+1) = argmin_{x in C} { Psi(x) + <f'(x(k)), x> + L*D_h(x,x(k))}
 
    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined with h
        x0:       initial point to start algorithm
        maxitrs:  maximum number of iterations
        linesearch:  whether or not perform line search (True or False)
        ls_ratio: backtracking line search parameter >= 1
        ls_adapt: whether or not use adaptive line search (True or False)
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays
        stop_eps: stop if F(x[k])-F(x[k-1]) < stop_eps, where F(x)=f(x)+Psi(x)

    Returns (x, Fx, Ls):
        x:  the last iterate of BPG
        Fx: array storing F(x[k]) for all k
        Ls: array storing local Lipschitz constants obtained by line search
    """

    if verbose:
        print("\nBPG_LS method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)          Lk")
    
    Fx = np.zeros(maxitrs)
    Ls = np.ones(maxitrs) * L
    
    x = x0
    for k in range(maxitrs):
        fx, g = f.func_grad(x)
        Fx[k] = fx + h.extra_Psi(x)
        
        if linesearch:
            if ls_adapt:
                L = L / ls_ratio
            x1 = h.div_prox_map(x, g, L)
            while f(x1) > fx + np.dot(g, x1-x) + L*h.divergence(x1, x):
                L = L * ls_ratio
                x1 = h.div_prox_map(x, g, L)
            x = x1
        else:
            x = h.div_prox_map(x, g, L)

        # store and display computational progress
        Ls[k] = L
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3}".format(k, Fx[k], L))
            
        # stopping criteria
        if k > 0 and abs(Fx[k]-Fx[k-1]) < stop_eps:
            break;

    Fx = Fx[0:k+1]
    Ls = Ls[0:k+1]
    return x, Fx, Ls


def solve_theta(theta, gamma, gainratio=1):
    """
    solve theta_k1 from the equation
    (1-theta_k1)/theta_k1^gamma = gainratio * 1/theta_k^gamma
    using Newton's method, starting from theta
    
    """
    ckg = theta**gamma / gainratio
    cta = theta
    eps = 1e-6 * theta
    phi = cta**gamma - ckg*(1-cta)
    while abs(phi) > eps:
        drv = gamma * cta**(gamma-1) + ckg
        cta = cta - phi / drv
        phi = cta**gamma - ckg*(1-cta)
        
    return cta
      

def ABPG(f, h, L, gamma, x0, maxitrs, theta_eq=False, restart=False, 
         verbose=True, verbskip=1, stop_eps=1e-14):
    """
    Accelerated Bregman Proximal Gradient (ABPG) method for solving 
            minimize_{x in C} f(x) + Psi(x): 

    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined with h
        gamma:    triangle scaling exponent (TSE) for Bregman div D_h(x,y)
        x0:       initial point to start algorithm
        maxitrs:  maximum number of iterations
        theta_eq: calculate theta_k by solving equality using Newton's method
        restart:  restart the algorithm when overshooting (True or False)
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays
        stop_eps: stop if D_h(z[k],z[k-1]) < stop_eps

    Returns (x, Fx, Ls):
        x:  the last iterate of BPG
        Fx: array storing F(x[k]) for all k
        Gdiv: triangle scaling gains D(xk,yk)/D(zk,zk_1)/theta_k^gamma
    """

    if verbose:
        print("\nABPG method for minimize_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta" + 
              "        TSG       D(x+,y)     D(z+,z)")
    
    Fx = np.zeros(maxitrs)
    Gdiv = np.zeros(maxitrs)
    
    x = x0
    z = x0
    theta = 1.0     # initialize theta = 1 for updating with equality 
    kk = 0          # separate counter for theta_k, easy for restart
    for k in range(maxitrs):
        # function value at previous iteration
        fx = f(x)   
        Fx[k] = fx + h.extra_Psi(x)
        
        # Update three iterates x, y and z
        z_1 = z
        x_1 = x     # only required for restart mode
        if theta_eq and kk > 0:
            theta = solve_theta(theta, gamma)
        else:
            theta = gamma / (kk + gamma)

        y = (1-theta)*x + theta*z_1
        g = f.gradient(y)
        z = h.div_prox_map(z_1, g, theta**(gamma-1) * L)
        x = (1-theta)*x + theta*z

        # compute triangle scaling quantities
        dxy = h.divergence(x, y)
        dzz = h.divergence(z, z_1)
        Gdr = dxy / dzz / theta**gamma

        # store and display computational progress
        Gdiv[k] = Gdr
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}".format(
                    k, Fx[k], theta, Gdr, dxy, dzz))

        # restart if gradient predicts objective increase
        kk += 1
        if restart:
            #if k > 0 and Fx[k] > Fx[k-1]:
            if np.dot(g, x-x_1) > 0:
                theta = 1.0     # reset theta = 1 for updating with equality
                kk = 0          # reset kk = 0 for theta = gamma/(kk+gamma)
                z = x           # in either case, reset z = x and also y

        # stopping criteria
        if dzz < stop_eps:
            break;

    Fx = Fx[0:k+1]
    Gdiv = Gdiv[0:k+1]
    return x, Fx, Gdiv


def ABPG_expo(f, h, L, gamma0, x0, maxitrs, delta=0.2, theta_eq=True, 
              checkdiv=False, gainmargin=2, restart=False, 
              verbose=True, verbskip=1, stop_eps=1e-14):
    """
    Accelerated Bregman Proximal Gradient method with exponent adaption for
            minimize_{x in C} f(x) + Psi(x) 
 
    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined with h
        gamma0:   initial triangle scaling exponent(TSE) for D_h(x,y) (>2)
        x0:       initial point to start algorithm
        maxitrs:  maximum number of iterations
        delta:    amount to decrease TSE for exponent adaption
        theta_eq: calculate theta_k by solving equality using Newton's method
        checkdiv: check triangle scaling inequality for adaption (True/False)
        gainmargin: extra gain margin allowed for checking TSI
        restart:  restart the algorithm when overshooting (True or False)
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays
        stop_eps: stop if D_h(z[k],z[k-1]) < stop_eps

    Returns (x, Fx, Ls):
        x:  the last iterate of BPG
        Fx: array storing F(x[k]) for all k
        Gamma: gamma_k obtained at each iteration
        Gdiv:  triangle scaling gains D(xk,yk)/D(zk,zk_1)/theta_k^gamma_k
    """
    
    if verbose:
        print("\nABPG_expo method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta       gamma" +
              "        TSG       D(x+,y)     D(z+,z)")
    
    Fx = np.zeros(maxitrs)
    Gdiv = np.zeros(maxitrs)
    Gamma = np.ones(maxitrs) * gamma0
    
    gamma = gamma0
    x = x0
    z = x0
    theta = 1.0     # initialize theta = 1 for updating with equality 
    kk = 0          # separate counter for theta_k, easy for restart
    for k in range(maxitrs):
        # function value at previous iteration
        fx = f(x)   
        Fx[k] = fx + h.extra_Psi(x)
        
        # Update three iterates x, y and z
        z_1 = z
        x_1 = x
        if theta_eq and kk > 0:
            theta = solve_theta(theta, gamma)
        else:
            theta = gamma / (kk + gamma)

        y = (1-theta)*x_1 + theta*z_1
        #g = f.gradient(y)
        fy, g = f.func_grad(y)
        
        condition = True
        while condition:    # always execute at least once per iteration 
            z = h.div_prox_map(z_1, g, theta**(gamma-1) * L)
            x = (1-theta)*x_1 + theta*z

            # compute triangle scaling quantities
            dxy = h.divergence(x, y)
            dzz = h.divergence(z, z_1)
            Gdr = dxy / dzz / theta**gamma

            if checkdiv:
                condition = (dxy > gainmargin * (theta**gamma) * dzz )
            else:
                condition = (f(x) > fy + np.dot(g, x-y) + theta**gamma*L*dzz)
                
            if condition and gamma > 1:
                gamma = max(gamma - delta, 1)
            else: 
                condition = False
               
        # store and display computational progress
        Gdiv[k] = Gdr
        Gamma[k] = gamma
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:10.3e}".format(
                    k, Fx[k], theta, gamma, Gdr, dxy, dzz))

        # restart if gradient predicts objective increase
        kk += 1
        if restart:
            #if k > 0 and Fx[k] > Fx[k-1]:
            if np.dot(g, x-x_1) > 0:
                theta = 1.0     # reset theta = 1 for updating with equality
                kk = 0          # reset kk = 0 for theta = gamma/(kk+gamma)
                z = x           # in either case, reset z = x and also y

        # stopping criteria
        if dzz < stop_eps:
            break;

    Fx = Fx[0:k+1]
    Gamma = Gamma[0:k+1]
    Gdiv = Gdiv[0:k+1]
    return x, Fx, Gamma, Gdiv


def ABPG_gain(f, h, L, gamma, x0, maxitrs, G0=1, 
              ls_increment=1.2, ls_decrement=1.2, ls_adapt=True, 
              theta_eq=True, checkdiv=False, restart=False, 
              verbose=True, verbskip=1, stop_eps=1e-14):
    """
    Accelerated Bregman Proximal Gradient (ABPG) method with gain adaption for 
            minimize_{x in C} f(x) + Psi(x): 
    
    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined with h
        gamma:    triangle scaling exponent(TSE) for Bregman distance D_h(x,y)
        x0:       initial point to start algorithm
        G0:       initial value for triangle scaling gain
        maxitrs:  maximum number of iterations
        ls_increment: factor of increasing gain (>=1)
        ls_decrement: factor of decreasing gain (>=1)
        ls_adapt: whether or not automatically decreasing gain (True or False)
        theta_eq: calculate theta_k by solving equality using Newton's method
        checkdiv: check triangle scaling inequality for adaption (True/False)
        restart:  restart the algorithm when overshooting (True/False)
        verbose:  display computational progress (True/False)
        verbskip: number of iterations to skip between displays
        stop_eps: stop if D_h(z[k],z[k-1]) < stop_eps

    Returns (x, Fx, Ls):
        x:  the last iterate of BPG
        Fx: array storing F(x[k]) for all k
        Gain: triangle scaling gains G_k obtained by LS at each iteration
        Gdiv: triangle scaling gains D(xk,yk)/D(zk,zk_1)/theta_k^gamma_k
    """
    if verbose:
        print("\nABPG_gain method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta         Gk" + 
              "         TSG       D(x+,y)     D(z+,z)")
    
    Fx = np.zeros(maxitrs)
    Gain = np.ones(maxitrs) * G0
    Gdiv = np.zeros(maxitrs)
    
    x = x0
    z = x0
    G = G0
    theta = 1.0     # initialize theta = 1 for updating with equality 
    kk = 0          # separate counter for theta_k, easy for restart
    for k in range(maxitrs):
        # function value at previous iteration
        fx = f(x)   
        Fx[k] = fx + h.extra_Psi(x)
        
        # Update three iterates x, y and z
        z_1 = z
        x_1 = x
        # adaptive option: always try a smaller Gain first before line search
        G_1 = G
        theta_1 = theta
        
        if ls_adapt:
            G = G / ls_decrement
        
        condition = True
        while condition:
            if kk > 0:
                if theta_eq:
                    theta = solve_theta(theta_1, gamma, G / G_1)
                else:
                    alpha = G / G_1
                    theta = theta_1*((1+alpha*(gamma-1))/(gamma*alpha+theta_1))

            y = (1-theta)*x_1 + theta*z_1
            #g = f.gradient(y)
            fy, g = f.func_grad(y)
        
            z = h.div_prox_map(z_1, g, theta**(gamma-1) * G * L)
            x = (1-theta)*x_1 + theta*z

            # compute triangle scaling quantities
            dxy = h.divergence(x, y)
            dzz = h.divergence(z, z_1)
            if dzz < stop_eps:
                break
            
            Gdr = dxy / dzz / theta**gamma

            if checkdiv:
                condition = (Gdr > G )
            else:
                condition = (f(x) > fy + np.dot(g,x-y) + theta**gamma*G*L*dzz)
                
            if condition:
                G = G * ls_increment
               
        # store and display computational progress
        Gain[k] = G
        Gdiv[k] = Gdr
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}  {6:10.3e}".format(
                    k, Fx[k], theta, G, Gdr, dxy, dzz))

        # restart if gradient predicts objective increase
        kk += 1
        if restart:
            if k > 0 and Fx[k] > Fx[k-1]:
            #if np.dot(g, x-x_1) > 0:
                theta = 1.0     # reset theta = 1 for updating with equality
                kk = 0          # reset kk = 0 for theta = gamma/(kk+gamma)
                z = x           # in either case, reset z = x and also y

        # stopping criteria
        if dzz < stop_eps:
            break;

    Fx = Fx[0:k+1]
    Gain = Gain[0:k+1]
    Gdiv = Gdiv[0:k+1]
    return x, Fx, Gain, Gdiv


def ABDA(f, h, L, gamma, x0, maxitrs, theta_eq=True,
           verbose=True, verbskip=1, stop_eps=1e-14):
    """
    Accelerated Bregman Dual Averaging (ABDA) method for solving
            minimize_{x in C} f(x) + Psi(x) 
    
    Inputs:
        f, h, L:  f is L-smooth relative to h, and Psi is defined with h
        gamma:    triangle scaling exponent (TSE) for Bregman distance D_h(x,y)
        x0:       initial point to start algorithm
        maxitrs:  maximum number of iterations
        theta_eq: calculate theta_k by solving equality using Newton's method
        verbose:  display computational progress (True or False)
        verbskip: number of iterations to skip between displays
        stop_eps: stop if D_h(z[k],z[k-1]) < stop_eps

    Returns (x, Fx, Ls):
        x:  the last iterate of BPG
        Fx: array storing F(x[k]) for all k
        Gdiv: triangle scaling gains D(xk,yk)/D(zk,zk_1)/theta_k^gamma
    """
    # Simple restart schemes for dual averaging method do not work!
    restart = False
    
    if verbose:
        print("\nABDA method for min_{x in C} F(x) = f(x) + Psi(x)")
        print("     k      F(x)       theta" + 
              "        TSG       D(x+,y)     D(z+,z)")
    
    Fx = np.zeros(maxitrs)
    Gdiv = np.zeros(maxitrs)
    
    x = x0
    z = x0
    theta = 1.0     # initialize theta = 1 for updating with equality 
    kk = 0          # separate counter for theta_k, easy for restart
    gavg = np.zeros(x.size)
    csum = 0
    for k in range(maxitrs):
        # function value at previous iteration
        fx = f(x)   
        Fx[k] = fx + h.extra_Psi(x)
        
        # Update three iterates x, y and z
        z_1 = z
        x_1 = x
        if theta_eq and kk > 0:
            theta = solve_theta(theta, gamma)
        else:
            theta = gamma / (kk + gamma)

        y = (1-theta)*x_1 + theta*z_1
        g = f.gradient(y)
        gavg = gavg + theta**(1-gamma) * g
        csum = csum + theta**(1-gamma)
        z = h.prox_map(gavg/csum, L/csum)
        x = (1-theta)*x_1 + theta*z

        # compute triangle scaling quantities
        dxy = h.divergence(x, y)
        dzz = h.divergence(z, z_1)
        Gdr = dxy / dzz / theta**gamma

        # store and display computational progress
        Gdiv[k] = Gdr
        if verbose and k % verbskip == 0:
            print("{0:6d}  {1:10.3e}  {2:10.3e}  {3:10.3e}  {4:10.3e}  {5:10.3e}".format(
                    k, Fx[k], theta, Gdr, dxy, dzz))

        kk += 1
        # restart does not work for ABDA (restart = False)
        if restart:
            if k > 0 and Fx[k] > Fx[k-1]:
            #if np.dot(g, x-x_1) > 0:   # this does not work for dual averaging
                theta = 1.0     # reset theta = 1 for updating with equality
                kk = 0          # reset kk = 0 for theta = gamma/(kk+gamma)
                z = x           # in either case, reset z = x and also y
                gavg = np.zeros(x.size) # this is why restart does not work
                csum = 0

        # stopping criteria
        if dzz < stop_eps:
            break;

    Fx = Fx[0:k+1]
    Gdiv = Gdiv[0:k+1]
    return x, Fx, Gdiv
