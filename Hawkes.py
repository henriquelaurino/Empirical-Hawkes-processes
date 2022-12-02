import numpy as np
import torch 
import math
from torch.distributions import Uniform
torch.set_default_dtype(torch.double)

''' Kernels '''

class Blank_Kernel:
    def __init__(self):
        return
    def bound(self,t,T,H):
        return torch.zeros(1)
    def density(self,t,history):
        return torch.zeros(len(t))
    def cumulative(self,T,history):
        return torch.zeros(1)

class Exp_Kernel:
    def __init__(self,params):
        self.α = params[0]
        self.β = params[1]
    def bound(self,t,T,H):
        grid = torch.linspace(t[0],T,int(T-t[0]+1)).reshape(-1,1) 
        deltas = grid - H.reshape(1,-1).repeat(grid.shape[0],1)
        raw = self.α*torch.exp(-self.β*deltas)
        aggregate = torch.sum((deltas>0)*raw,1)
        return max(aggregate)
    def density(self,t,history):
        history_deltas = t.reshape(-1,1) - history
        timing_mask = history_deltas>0
        raw = self.α*torch.exp(-self.β*(history_deltas*timing_mask))
        return torch.sum(raw*timing_mask,1)
    def cumulative(self,T,history):
        timing_mask = T-history>0
        return torch.sum(timing_mask*(self.α/self.β)*(1-torch.exp(-self.β*(T-history))))

class Delay_Exp_Kernel:
    def __init__(self,params):
        self.α = params[0]
        self.β = params[1]
        self.γ = params[2]
    def bound(self,t,T,H):
        grid = torch.linspace(t[0],T,int(T-t[0]+1)).reshape(-1,1) 
        deltas = grid - H.reshape(1,-1).repeat(grid.shape[0],1)
        raw = self.α*(1+self.γ*deltas)*torch.exp(-self.β*deltas)
        aggregate = torch.sum((deltas>0)*raw,1)
        return max(aggregate)
    def density(self,t,history):
        history_deltas = t.reshape(-1,1) - history
        timing_mask = history_deltas>0
        raw = self.α*(1+self.γ*history_deltas)*torch.exp(-self.β*history_deltas*timing_mask)
        return torch.sum(raw*timing_mask,1)
    def cumulative(self,T,history):
        deltas = T-history
        timing_mask = deltas>0
        C = (self.α/(self.β**2))*(
            (self.β+self.γ)*(1-torch.exp(-deltas*self.β*timing_mask))
            -deltas*self.β*self.γ*torch.exp(-deltas*self.β*timing_mask)
        )
        return torch.sum(timing_mask*C)

    
''' Base Intensity'''

class GP_intensity:
    def __init__(self,gp,Z,f_z):
        self.gp = gp
        self.Z  = Z
        self.f_z= f_z
    def density(self,t):
        return (self.gp.posterior_cond(self.Z,t,self.f_z)[0]).pow(2)
    def integral(self,t0,t1):
        p1 = (t1-t0)/2
        p2 = (t1+t0)/2
        [x,w] = np.polynomial.legendre.leggauss(20)
        x = torch.tensor(x)
        w = torch.tensor(w)
        integral = p1*torch.sum(w*(self.gp.posterior_cond(self.Z,p1*x + p2,self.f_z)[0].pow(2)))
        return integral

class Constant_intensity:
    def __init__(self,mu):
        self.mu = mu.pow(2)
    def density(self,t):
        return self.mu.expand(len(t))
    def integral(self,t0,t1):
        return self.mu*(t1-t0)
    
class Ad_intensity:
    def __init__(self,params,spacings,history):
        self.ϕ=params[0]
        self.ν=params[1]
        self.δ=params[2]
        self.η=params[3]
        self.spacings = spacings
        self.history = history
    def density(self,t):
        deltas = t.reshape(-1,1) - self.history
        timing_mask = deltas>0
        full_scale = self.ϕ
        rescaling = torch.exp(timing_mask*(-self.η/self.spacings))        
        decay = self.ν*(timing_mask*deltas/self.δ).pow((self.ν-1)*timing_mask)*(1+(timing_mask*deltas/self.δ).pow(timing_mask*self.ν)).pow(-2)
        ad_response = full_scale*rescaling*decay
        return torch.sum(timing_mask*ad_response,1)
    def integral(self,t0,t1):
        deltas = t1 - self.history
        timing_mask = deltas>0
        full_scale = self.ϕ
        rescaling = torch.exp(timing_mask*(-self.η/self.spacings))
        decay = self.δ*(1-(1+(timing_mask*deltas/self.δ).pow(self.ν*timing_mask)).pow(-1))
        ad_response = full_scale*rescaling*decay
        return torch.sum(timing_mask*ad_response)
    
class Composite_intensity:
    def __init__(self,components):
        self.components = components
    def density(self,t):
        density = self.components[0].density(t)
        for component in self.components[1:]:
            density += component.density(t)
        return density
    def integral(self,t0,t1):
        integral = self.components[0].integral(t0,t1)
        for component in self.components[1:]:
            integral += component.integral(t0,t1)
        return integral
    

''' Simulation Algorithms '''

def Thinning_Sim(T, Base_Rate,Kernel=Blank_Kernel()):
    sample = np.array([])
    t = np.zeros(1) 
    while t<T:
        h0 = max(Base_Rate(torch.linspace(0,T,5*(T+1)))) + Kernel.bound(torch.linspace(0,T,5*(T+1)),T,torch.tensor(sample))
        u  = Uniform(0,1).sample()
        t1  = -torch.log(u)/h0
        t = min(T,t+np.array(t1))
        h1 = Base_Rate(torch.tensor(t)) + Kernel.density(torch.tensor(t),torch.tensor(sample)) 
        s = Uniform(0,1).sample()
        if s<=h1/h0:
            sample = np.append(sample, t)
    return sample
    
''' Likelihoods '''

class General_Point_Process:
    def __init__(self,intensity,T):
        self.intensity = intensity
        self.T = T
    def log_prob(self, timeseries):
        safety = torch.tensor(1e-9)
        h = self.intensity.density(timeseries)
        H = self.intensity.integral(0,self.T)
        ll = torch.sum(torch.log(safety + h)) - H
        return ll

class Hawkes_Process:
    def __init__(self,intensity,feedback,T):
        self.intensity = intensity
        self.feedback = feedback
        self.T = T
    def log_prob(self, timeseries):
        safety = torch.tensor(1e-15)
        h = self.intensity.density(timeseries)
        #g = torch.nan_to_num(self.feedback.density(timeseries,timeseries))
        H = self.intensity.integral(0.,self.T)
        #G = torch.nan_to_num(self.feedback.cumulative(self.T,timeseries))
        #ll = torch.sum((h>0)*torch.log(safety + h + g)) - H - G
        ll = torch.sum(torch.log(safety + h)) - H
        return ll