import numpy as np
import torch 
import math

''' GP Kernels '''
class SqExp_kernel: 
    def __init__(self,lenscale, sigma):
        self.lenscale = lenscale
        self.sigma = sigma
    def c(self,x1,x2):
        diff = (x1.unsqueeze(-1)-x2.unsqueeze(-2)).pow(2)
        res = diff.mul(-1/(2*self.lenscale**2)).exp_().mul(self.sigma**2)
        return res
    
class Periodic_kernel: 
    def __init__(self,lenscale, sigma, period):
        self.lenscale = lenscale
        self.period = period
        self.sigma = sigma
    def c(self,x1,x2):
        diff = (x1.unsqueeze(-1)-x2.unsqueeze(-2)).abs()*math.pi/self.period
        res = diff.sin().pow(2).mul(-2/self.lenscale**2).exp_().mul(self.sigma**2)
        return res
    
class additive_composite_kernel:
    def __init__(self,kernels):
            self.kernels = kernels
    def c(self,x1,x2):
        res = torch.zeros(len(x1),len(x2)).type(torch.double)
        for k in self.kernels:
            res += k.c(x1,x2)
        return res
            
''' GP structure '''
class GP:
    
    def __init__(self,mean,kernel,noise=0):
        # We'll set the entire process in terms of a FIXED mean, 
        # and a kernel class with a correlation function c
        # also specify a tiny jitter term to make sure we can perform matrix inversions properly
        self.mean = mean
        self.kernel = kernel
        self.jitter = torch.tensor(1e-5).type(torch.double)
        self.noise = noise
        
    def prior(self,X): 
        # the prior is very simple. We only need to calculate c(X,X)
        K11 = self.kernel.c(X, X) + self.jitter*torch.eye(X.shape[0])
        m = self.mean.expand([X.shape[0]]).type(torch.double)
        return m, K11 
    
    def posterior_conj(self,X_1,X_2,Y_1): 
        # conjugate posterior predictive with y~N(f,e), f~MVN
        K11 = self.kernel.c(X_1,X_1) + self.jitter*torch.eye(X_1.shape[0])
        K11_= torch.inverse(K11+self.noise*torch.eye(X_1.shape[0]))
        K21 = self.kernel.c(X_2,X_1)
        K22 = self.kernel.c(X_2,X_2)
        posterior_mean = self.mean + K21.matmul(K11_).matmul(Y_1-self.mean)
        posterior_cov  = K22       - K21.matmul(K11_).matmul(K21.T) 
        return posterior_mean, posterior_cov 
    
    def posterior_cond(self,X_1,X_2,f): 
        #generalized posterior with q(f)~MVN (gives actual confidence bounds)
        K11 = self.kernel.c(X_1,X_1)+ self.jitter*torch.eye(X_1.shape[0])
        K11_= torch.inverse(K11+self.noise*torch.eye(X_1.shape[0]))
        K21 = self.kernel.c(X_2,X_1)
        K22 = self.kernel.c(X_2,X_2)
        posterior_mean = self.mean + K21.matmul(K11_).matmul(f-self.mean)
        posterior_cov  = K22       - K21.matmul(K11_).matmul(K21.T) 
        return posterior_mean, posterior_cov
    
    def posterior(self,X_1,X_2,m,S): 
        #generalized posterior with q(f)~MVN (gives actual confidence bounds)
        K11 = self.kernel.c(X_1,X_1)+ self.jitter*torch.eye(X_1.shape[0])
        K11_= torch.inverse(K11+self.noise*torch.eye(X_1.shape[0]))
        K21 = self.kernel.c(X_2,X_1)
        K22 = self.kernel.c(X_2,X_2)
        A = K21.matmul(K11_)
        B = K22 - K21.matmul(K11_).matmul(K21.T)
        posterior_mean = self.mean + A.matmul(m-self.mean)
        posterior_cov  = B + A.matmul(S).matmul(A.T)  
        return posterior_mean, posterior_cov + self.jitter*torch.eye(X_2.shape[0])