# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np


class RSmoothFunction:
    """
    Relatively-Smooth Function, can query f(x) and gradient
    """
    def __call__(self, x):
        assert 0, "RSmoothFunction: __call__(x) is not defined"
        
    def gradient(self, x):
        assert 0, "RSmoothFunction: gradient(x) is not defined"
 
    def func_grad(self, x, flag):
        """
        flag=0: function, flag=1: gradient, flag=2: function & gradient 
        """
        assert 0, "RSmoothFunction: func_grad(x, flag) is not defined"


class DOptimalObj(RSmoothFunction):
    """
    f(x) = - log(det(H*diag(x)*H')) where H is an m by n matrix, n > m
    """
    def __init__(self, H):
        self.H = H
        self.m = H.shape[0]
        self.n = H.shape[1]
        assert self.m < self.n, "DOptimalObj: need m < n"
        
    def __call__(self, x):
        return self.func_grad(x, flag=0)
        
    def gradient(self, x):
        return self.func_grad(x, flag=1)
        
    def func_grad(self, x, flag=2):
        assert x.size == self.n, "DOptimalObj: x.size not equal to n"
        assert x.min() >= 0,     "DOptimalObj: x needs to be nonnegative"
        sx = np.sqrt(x)
        Hsx = self.H*sx;    # using numpy array broadcast
        HXHT = np.dot(Hsx,Hsx.T)
        
        if flag == 0:       # only return function value
            f = -np.log(np.linalg.det(HXHT))
            return f
        
        Hsx = np.linalg.solve(HXHT, self.H)
        g = np.empty(self.n)
        for i in range(self.n):
            g[i] = - np.dot(self.H[:,i], Hsx[:,i])
            
        if flag == 1:       # only return gradient
            return g
        
        # return both function value and gradient
        f = -np.log(np.linalg.det(HXHT))
        return f, g


class PoissonRegression(RSmoothFunction):
    """
    f(x) = D_KL(b, Ax) for linear inverse problem A * x = b
    """
    def __init__(self, A, b):
        assert A.shape[0] == b.shape[0], "A and b sizes not matching"
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.n = A.shape[1]
        
    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)
    
    def func_grad(self, x, flag=2):
        assert x.size == self.n, "PoissonRegression: x.size not equal to n."
        Ax = np.dot(self.A, x)
        if flag == 0:
            fx = sum( self.b * np.log(self.b / Ax) + Ax - self.b )
            return fx

        # use array broadcasting
        g = ((1-self.b/Ax).reshape(self.m, 1) * self.A).sum(axis=0)
        # same as the following code
        #g = np.zeros(x.shape)
        #for i in range(self.m):
        #    g += (1 - self.b[i]/np.dot(self.A[i,:], x)) * self.A[i,:]
        if flag == 1:
            return g
        
        # return both function value and gradient
        fx = sum( self.b * np.log(self.b / Ax) + Ax - self.b )
        return fx, g


class KLdivRegression(RSmoothFunction):
    """
    f(x) = D_KL(Ax, b) for linear inverse problem A * x = b
    """
    def __init__(self, A, b):
        assert A.shape[0] == b.shape[0], "A and b size not matching"
        self.A = A
        self.b = b
        self.m = A.shape[0]
        self.n = A.shape[1]
        
    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)
    
    def func_grad(self, x, flag=2):
        assert x.size == self.n, "NonnegRegression: x.size not equal to n."
        Ax = np.dot(self.A, x)
        if flag == 0:
            fx = sum( Ax * np.log(Ax / self.b) - Ax + self.b )
            return fx

        # use array broadcasting
        g = (np.log(Ax/self.b).reshape(self.m, 1) * self.A).sum(axis=0)
        # same as the following code
        #g = np.zeros(x.shape)
        #for i in range(self.m):
        #    g += np.log(Ax[i]/self.b[i]) * self.A[i,:]
        if flag == 1:
            return g
        
        # return both function value and gradient
        fx = sum( Ax * np.log(Ax / self.b) - Ax + self.b )
        return fx, g
           
           
#######################################################################


class LegendreFunction:
    """
    Function of Legendre type, used as the kernel of Bregman divergence.
    Include an extra Psi(x) for convenience of composite optimization.
    """
    def __call__(self, x):
        assert 0, "LegendreFunction: __call__(x) is not defined."
        
    def extra_Psi(self, x):
        return 0
        
    def gradient(self, x):
        assert 0, "LegendreFunction: gradient(x) is not defined."

    def divergence(self, x, y):
        """
        Return D(x,y) = h(x) - h(y) - <h'(y), x-y>
        """
        assert 0, "LegendreFunction: divergence(x,y) is not defined."

    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * h(x) }
        """
        assert 0, "LegendreFunction: prox_map(x, L) is not defined."

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { Psi(x) + <g, x> + L * D(x,y)  } 
        """
        assert 0, "LegendreFunction: div_prox_map(y, g, L) is not defined."


class BurgEntropy(LegendreFunction):
    """
    h(x) = - sum_{i=1}^n log(x[i]) for x > 0
    """
    def __call__(self, x):
        assert x.min()>0, "BurgEntropy only takes positive arguments."
        return -sum(np.log(x))
    
    def gradient(self, x):
        assert x.min()>0, "BurgEntropy only takes positive arguments."
        return -1/x
    
    def divergence(self, x, y):
        assert x.shape == y.shape, "Vectors x and y are of different sizes."
        assert x.min() > 0 and y.min() > 0, "Entries of x or y not positive."
        return sum(x/y - np.log(x/y) - 1)        

    def prox_map(self, g, L):
        """
        Return argmin_{x > 0} { <g, x> + L * h(x) } 
        This function needs to be replaced with inheritance
        """
        assert L > 0, "BurgEntropy prox_map only takes positive L value."
        assert g.min() > 0, "BurgEntropy prox_map only takes positive value."
        return L / g
           
    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x > C} { <g, x> + L * D(x,y) }
        This is a general function that works for all derived classes
        """
        assert y.shape == g.shape, "Vectors y and g are of different sizes." 
        assert y.min() > 0 and L > 0, "Either y or L is not positive."
        gg = g/L - self.gradient(y)
        return self.prox_map(gg, 1)


class BurgEntropyL1(BurgEntropy):
    """
    h(x) = - sum_{i=1}^n log(x[i]) used in context of solving the problem 
            min_{x > 0} f(x) + lamda * ||x||_1 
    """
    def __init__(self, lamda=0, x_max=1e4):
        assert lamda >= 0, "BurgEntropyL1: lambda should be nonnegative."
        self.lamda = lamda
        self.x_max = x_max

    def extra_Psi(self, x):
        """
        return lamda * ||x||_1
        """
        return self.lamda * x.sum()

    def prox_map(self, g, L):
        """
        Return argmin_{x > 0} { lambda * ||x||_1 + <g, x> + L h(x) }
        !!! This proximal mapping may have unbounded solution x->infty
        """
        assert L > 0, "BurgEntropyL1: prox_map only takes positive L."
        assert g.min() > -self.lamda, "Not getting positive solution."
        #g = np.maximum(g, -self.lamda + 1.0 / self.x_max)
        return L / (self.lamda + g)

       
class BurgEntropyL2(BurgEntropy):
    """
    h(x) = - sum_{i=1}^n log(x[i]) used in context of solving the problem 
            min_{x > 0} f(x) + (lambda/2) ||x||_2^2 
    """
    def __init__(self, lamda=0):
        assert lamda >= 0, "BurgEntropyL2: lamda should be nonnegative."
        self.lamda = lamda

    def extra_Psi(self, x):
        """
        return (lamda/2) * ||x||_2^2
        """
        return (self.lamda / 2) * np.dot(x, x)

    def prox_map(self, g, L):
        """
        Return argmin_{x > 0} { (lamda/2) * ||x||_2^2 + <g, x> + L h(x) }
        """
        assert L > 0, "BurgEntropyL2: prox_map only takes positive L value."
        gg = g / L
        lamda_L = self.lamda / L
        return (np.sqrt(gg*gg + 4*lamda_L) - gg) / (2 * lamda_L)

       
class BurgEntropySimplex(BurgEntropy):
    """
    h(x) = - sum_{i=1}^n log(x[i])  used in the context of solving 
    min_{x \in C} f(x)  where C is the standard simplex, with  Psi(x) = 0
    """
    def __init__(self, eps=1e-8):
        # eps is precision for solving prox_map using Newton's method
        assert eps > 0, "BurgEntropySimplex: eps should be positive."
        self.eps = eps
     
    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { <g, x> + L h(x) } where C is unit simplex
        """
        assert L > 0, "BergEntropySimplex prox_map only takes positive L."
        gg = g / L
        cmin = -gg.min()    # choose cmin to ensure min(gg+c) >= 0
        # first use bisection to find c such that sum(1/(gg+c)) > 0
        c = cmin + 1        
        while sum(1/(gg+c))-1 < 0:
            c = (cmin + c) / 2.0
        # then use Newton's method to find optimal c
        fc = sum(1/(gg+c))-1
        while abs(fc) > self.eps:
            fpc = sum(-1.0/(gg+c)**2)
            c = c - fc / fpc
            fc = sum(1/(gg+c))-1
        x = 1.0/(gg+c)
        return x
       

class ShannonEntropy(LegendreFunction):
    """
    h(x) = sum_{i=1}^n x[i]*log(x[i]) for x >= 0, note h(0) = 0
    """
    def __init__(self, delta=1e-20):
        self.delta = delta
        
    def __call__(self, x):
        assert x.min() >= 0, "ShannonEntropy takes nonnegative arguments."
        xx = np.maximum(x, self.delta)
        return sum( xx * np.log(xx) )

    def gradient(self, x):         
        assert x.min() >= 0, "ShannonEntropy takes nonnegative arguments."
        xx = np.maximum(x, self.delta)
        return 1.0 + np.log(xx)

    def divergence(self, x, y):
        assert x.shape == y.shape, "Vectors x and y are of different shapes."
        assert x.min() >= 0 and y.min() >= 0, "Some entries are negative."
        #for i in range(x.size):
        #    if x[i] > 0 and y[i] == 0:
        #        return np.inf 
        return sum(x*np.log((x+self.delta)/(y+self.delta))) + (sum(y)-sum(x))        
        
    def prox_map(self, g, L):
        """
        Return argmin_{x >= 0} { <g, x> + L * h(x) }
        """
        assert L > 0, "ShannonEntropy prox_map require L > 0."
        return np.exp(-g/L - 1)

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x >= 0} { <g, x> + L * D(x,y) }
        """
        assert y.shape == g.shape, "Vectors y and g are of different sizes." 
        assert y.min() >= 0 and L > 0, "Some entries of y are negavie."
        #gg = g/L - self.gradient(y)
        #return self.prox_map(gg, 1)
        return y * np.exp(-g/L)
   

class ShannonEntropyL1(ShannonEntropy):
    """
    h(x) = sum_{i=1}^n x[i]*log(x[i]) for x >= 0, note h(0) = 0
    used in the context of  min_{x >=0 } f(x) + lamda * ||x||_1
    """
    def __init__(self, lamda=0, delta=1e-20): 
        ShannonEntropy.__init__(self, delta)
        self.lamda = lamda
        
    def extra_Psi(self, x):
        """
        return lamda * ||x||_1
        """
        return self.lamda * x.sum()
       
    def prox_map(self, g, L):
        """
        Return argmin_{x >= 0} { lamda * ||x||_1 + <g, x> + L * h(x) }
        """
        return ShannonEntropy.prox_map(self, self.lamda + g, L)

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x >= 0} { lamda * ||x||_1 + <g, x> + L * D(x,y) }
        """
        return ShannonEntropy.div_prox_map(self, y, self.lamda + g, L)
   
       
class ShannonEntropySimplex(ShannonEntropy):
    """
    h(x) = sum_{i=1}^n x[i]*log(x[i]) for x >= 0, note h(0) = 0
    used in the context of  min_{x in C } f(x) where C is standard simplex 
    """
    
    def prox_map(self, g, L):
        """
        Return argmin_{x in C} { <g, x> + L * h(x) } where C is unit simplex
        """
        assert L > 0, "ShannonEntropy prox_map require L > 0."
        x = np.exp(-g/L - 1)
        return x / sum(x)

    def div_prox_map(self, y, g, L):
        """
        Return argmin_{x in C} { <g, x> + L*d(x,y) } where C is unit simplex
        """
        assert y.shape == g.shape, "Vectors y and g are of different shapes."
        assert y.min() > 0 and L > 0, "prox_map needs positive arguments."
        x = y * np.exp(-g/L)
        return x / sum(x)
   

class SquaredL2Norm(LegendreFunction):
    """
    h(x) = (1/2)||x||_2^2
    """       
    def __call__(self, x):
        return 0.5*np.dot(x, x)

    def gradient(self, x):         
        return x

    def divergence(self, x, y):
        assert x.shape == y.shape, "SquaredL2Norm: x and y not same shape."
        xy = x - y
        return 0.5*np.dot(xy, xy)

    def prox_map(self, g, L):
        assert L > 0, "SquaredL2Norm: L should be positive."
        return -(1/L)*g
        
    def div_prox_map(self, y, g, L):
        assert y.shape == g.shape and L > 0, "Vectors y and g not same shape."
        return y - (1/L)*g


class SumOf2nd4thPowers(LegendreFunction):
    """
    h(x) = (1/2)||x||_2^2 + (M/4)||x||_2^4
    """       
    def __init__(self, M):
        self.M = M
    
    def __call__(self, x):
        normsq = np.dot(x, x)
        return 0.5 * normsq + (self.M / 4) * normsq**2

    def gradient(self, x):
        normsq = np.dot(x, x)         
        return (1 + self.M * normsq) * x

    def divergence(self, x, y):
        assert x.shape == y.shape, "Bregman div: x and y not same shape."
        return self.__call__(x) - (self.__call__(y) 
                                   + np.dot(self.gradient(y), x-y))
        