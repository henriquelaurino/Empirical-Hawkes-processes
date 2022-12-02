from Auxiliary import Hawkes, GP
import numpy as np
import torch
import pyro
from pyro.infer import SVI,Trace_ELBO,TraceMeanField_ELBO
from torch.distributions.transforms import SoftplusTransform
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta
torch.set_default_dtype(torch.double)
        
        
class GP_point_process_final: 
    def __init__(self,data,ads,spacings,starting_vals,minibatch,u):
        self.data       = data
        self.ads        = ads
        self.spacings   = spacings
        self.minibatch  = minibatch
        self.u          = u
        self.t0         = 0
        self.T          = 78
        counts = []
        for i in range(len(data)):
            counts.append(len(data[i]))
        self.mu = np.sqrt(np.array(counts).mean()/self.T)
        self.starting   = starting_vals
        self.starting_Z = dist.Uniform(self.t0,self.T).sample([self.u])
        
    def model(self,subsample):
        # set prior
        Hyper_Mean   = pyro.sample("Mean"        ,dist.Normal(self.mu,.01)).type(torch.double)
        Kernel_hyp   = pyro.sample("Kernel"      ,dist.Normal(self.starting[ 0:6 ],torch.tensor([.1]*6)).to_event(1)).type(torch.double)
        Feedback_hyp = pyro.sample("Feedback_hyp",dist.Normal(self.starting[ 6:12],torch.tensor([.1]*6)).to_event(1)).type(torch.double)
        Response_hyp = pyro.sample("Response_hyp",dist.Normal(self.starting[12:20],torch.tensor([.1]*8)).to_event(1)).type(torch.double)
        Z_loc        = pyro.sample("Z_loc"       ,dist.Normal(self.starting_Z  ,torch.tensor(1.)*self.u).to_event(1)).type(torch.double)
        # set GP
        kernel_SE = GP.SqExp_kernel(Kernel_hyp[0].exp(),Kernel_hyp[1].exp())
        #kernel_P1 = GP.Periodic_kernel(torch.exp(Kernel_hyp[2]),torch.exp(Kernel_hyp[3]),torch.tensor([1]).type(torch.double))
        #kernel_P7 = GP.Periodic_kernel(torch.exp(Kernel_hyp[4]),torch.exp(Kernel_hyp[5]),torch.tensor([7]).type(torch.double))
        kernel    = GP.additive_composite_kernel([kernel_SE#,kernel_P1,kernel_P7
                                                 ])
        gp        = GP.GP(Hyper_Mean,kernel)
        Mean_z,Covariance_z = gp.prior(Z_loc)
        # only subsample a few users
        for i in pyro.plate("Obs",len(self.data),subsample = subsample):
            # sample local
            feedback_params  = pyro.sample("Feedback_{}".format(i), 
                                           dist.Normal(Feedback_hyp[0:3],torch.exp(Feedback_hyp[3:6])).to_event()).type(torch.double)
            f_z              = pyro.sample("f_z_{}".format(i), dist.MultivariateNormal(Mean_z,Covariance_z)).type(torch.double)
            response_params  = pyro.sample("Response_{}".format(i), 
                                           dist.Normal(Response_hyp[0:4],torch.exp(Response_hyp[4:8])).to_event()).type(torch.double)
            # compute ll
            endog_intensity= Hawkes.Delay_Exp_Kernel(torch.exp(feedback_params))
            intensity_0    = Hawkes.GP_intensity(gp,Z_loc,f_z)
            intensity_1    = Hawkes.Ad_intensity(torch.exp(response_params),self.spacings[i],self.ads[i])
            exog_intensity = intensity_0  #Hawkes.Composite_intensity([intensity_0,intensity_1])
            process        = Hawkes.Hawkes_Process(exog_intensity,endog_intensity,self.T)
            pyro.sample('Points_{}'.format(i), process,obs=self.data[i])
        
    def guide(self,subsample):
        hyperparams = pyro.param("Hyper",self.starting).type(torch.double)
        meanparam   = pyro.param("Hyper_Mean",torch.tensor(self.mu)).type(torch.double)
        errors      = pyro.param("Errors",torch.exp(torch.tensor(-3)).expand(22)).type(torch.double).pow(2)
        Hyper_Mean    = pyro.sample("Mean"        ,dist.Normal(meanparam,errors[0])).type(torch.double)
        Kernel_hyp    = pyro.sample("Kernel"      ,dist.Normal(hyperparams[ 0:6 ],errors[1:7]).to_event(1)).type(torch.double)
        Feedback_hyp  = pyro.sample("Feedback_hyp",dist.Normal(hyperparams[ 6:12],errors[7:13]).to_event(1)).type(torch.double)
        Response_hyp  = pyro.sample("Response_hyp",dist.Normal(hyperparams[12:20],errors[13:21]).to_event(1)).type(torch.double)
        Z             = pyro.sample("Z_loc"       ,dist.Normal(pyro.param("Z",self.starting_Z).type(torch.double),errors[21].expand([self.u])).to_event(1))
        # set GP
        kernel_SE = GP.SqExp_kernel(torch.exp(Kernel_hyp[0]),torch.ones(1))
        #kernel_P1 = GP.Periodic_kernel(torch.exp(Kernel_hyp[2]),torch.exp(Kernel_hyp[3]),torch.tensor([1]).type(torch.double))
        #kernel_P7 = GP.Periodic_kernel(torch.exp(Kernel_hyp[4]),torch.exp(Kernel_hyp[5]),torch.tensor([7]).type(torch.double))
        kernel    = GP.additive_composite_kernel([kernel_SE
                                                  #,kernel_P1,kernel_P7
                                                 ])
        # find u at inducing points
        m = pyro.param("m",dist.Normal(Hyper_Mean,.001).sample([self.u])).type(torch.double)
        S = pyro.param("S",torch.linalg.cholesky(kernel.c(Z,Z)+1e-4*torch.eye(self.u)),constraint=dist.constraints.lower_cholesky).type(torch.double)
        for i in pyro.plate("Obs",len(self.data),subsample = subsample):
            pyro.sample("Feedback_{}".format(i),dist.Normal(Feedback_hyp[0:3],torch.exp(Feedback_hyp[3:6])).to_event()).type(torch.double)
            pyro.sample("f_z_{}".format(i)     ,dist.MultivariateNormal(m,scale_tril=S)).type(torch.double)
            pyro.sample("Response_{}".format(i),dist.Normal(Response_hyp[0:4],torch.exp(Response_hyp[4:8])).to_event()).type(torch.double)


''' '''''' '''''' '''''' '''
''' Prototype tests '''
''' '''''' '''''' '''''' '''
class Poisson_test_bench: 
    def __init__(self,data,minibatch):
        self.data       = data
        self.minibatch  = minibatch
        self.T          = 78
        self.mu = torch.sqrt(self.data.type(torch.double).mean()/self.T)*2
        self.sig= torch.sqrt(self.data.type(torch.double).var()/self.T)*2
    def model(self,subsample):
        for i in pyro.plate("Obs",len(self.data),subsample = subsample):
            Hyper_Mean   = pyro.sample("Mean_{}".format(i),dist.Normal(self.mu,self.sig.pow(2))).type(torch.double)
            pyro.sample('Points_{}'.format(i), dist.Poisson(Hyper_Mean.pow(2)*self.T),obs=self.data[i])
    def guide(self,subsample):
        meanparam   = pyro.param("Hyper_Mean",self.mu.clone()).type(torch.double)
        errors      = pyro.param("Errors"    ,self.sig.clone()).type(torch.double).pow(2)
        for i in pyro.plate("Obs",len(self.data),subsample = subsample):
            Hyper_Mean    = pyro.sample("Mean_{}".format(i),dist.Normal(meanparam,errors)).type(torch.double)
            
            
        
class GP_point_process: 
    def __init__(self,data,ads,spacings,starting_vals,minibatch,u):
        self.data       = data
        self.ads        = ads
        self.spacings   = spacings
        self.minibatch  = minibatch
        self.u          = u
        self.t0         = 0
        self.T          = 78
        counts = []
        for i in range(len(data)):
            counts.append(len(data[i]))
        self.mu = np.sqrt(np.array(counts).mean()/self.T)*2
        self.sig= np.sqrt(np.array(counts).var() /self.T)*2
        self.starting   = starting_vals
        self.starting_Z = dist.Uniform(self.t0,self.T).sample([self.u])
        
    def model(self,subsample):
        # set prior
        Hyper_Mean   = pyro.sample("Mean"        ,dist.Normal(self.mu,.01)).type(torch.double)
        Kernel_hyp   = pyro.sample("Kernel"      ,dist.Normal(self.starting[ 0:6 ],torch.tensor([1.]*6)).to_event(1)).type(torch.double)
        Feedback_hyp = pyro.sample("Feedback_hyp",dist.Normal(self.starting[ 6:12],torch.tensor([1.]*6)).to_event(1)).type(torch.double)
        Response_hyp = pyro.sample("Response_hyp",dist.Normal(self.starting[12:20],torch.tensor([1.]*8)).to_event(1)).type(torch.double)
        Z_loc        = pyro.sample("Z_loc"       ,dist.Normal(self.starting_Z  ,torch.tensor(1.)*self.u).to_event(1)).type(torch.double)
        noise        = pyro.sample("Noise"       ,dist.Normal(torch.ones(1)*-2,torch.ones(1)*1e-5)).type(torch.double)
        # set GP
        kernel_SE = GP.SqExp_kernel(Kernel_hyp[0].exp(),Kernel_hyp[1].exp())
        #kernel_P1 = GP.Periodic_kernel(torch.exp(Kernel_hyp[2]),torch.exp(Kernel_hyp[3]),torch.tensor([1]).type(torch.double))
        #kernel_P7 = GP.Periodic_kernel(torch.exp(Kernel_hyp[4]),torch.exp(Kernel_hyp[5]),torch.tensor([7]).type(torch.double))
        kernel    = GP.additive_composite_kernel([kernel_SE#,kernel_P1,kernel_P7
                                                 ])
        gp        = GP.GP(Hyper_Mean,kernel,noise)
        Mean_z,Covariance_z = gp.prior(Z_loc)
        # subsample
        for i in pyro.plate("Obs",len(self.data),subsample = subsample):
            # sample local
            feedback_params  = pyro.sample("Feedback_{}".format(i), dist.Normal(Feedback_hyp[0:3],torch.exp(Feedback_hyp[3:6])).to_event()).type(torch.double)
            f_z              = pyro.sample("f_z_{}".format(i), dist.MultivariateNormal(Mean_z,Covariance_z)).type(torch.double)
            response_params  = pyro.sample("Response_{}".format(i), dist.Normal(Response_hyp[0:4],torch.exp(Response_hyp[4:8])).to_event()).type(torch.double)
            # compute ll
            endog_intensity= Hawkes.Delay_Exp_Kernel(torch.exp(feedback_params))
            intensity_0    = Hawkes.GP_intensity(gp,Z_loc,f_z)
            intensity_1    = Hawkes.Ad_intensity(torch.exp(response_params),self.spacings[i],self.ads[i])
            exog_intensity = intensity_0  #Hawkes.Composite_intensity([intensity_0,intensity_1])
            process        = Hawkes.Hawkes_Process(exog_intensity,endog_intensity,self.T)
            pyro.sample('Points_{}'.format(i), process,obs=self.data[i])
        
    def guide(self,subsample):
        # set params
        hyperparams = pyro.param("Hyper"     ,self.starting).type(torch.double)
        meanparam   = pyro.param("Hyper_Mean",torch.tensor(self.mu)).type(torch.double)
        errors      = pyro.param("Errors"    ,torch.tensor(self.sig).expand(22)).type(torch.double).pow(2)
        Z_param     = pyro.param("Z",self.starting_Z).type(torch.double)
        noise_param = pyro.param("noise_param",torch.ones(1)*-2).type(torch.double)
        # draw global
        Hyper_Mean    = pyro.sample("Mean"        ,dist.Normal(meanparam,errors[0])).type(torch.double)
        Kernel_hyp    = pyro.sample("Kernel"      ,dist.Normal(hyperparams[ 0:6 ],errors[ 1: 7]).to_event(1)).type(torch.double)
        Feedback_hyp  = pyro.sample("Feedback_hyp",dist.Normal(hyperparams[ 6:12],errors[ 7:13]).to_event(1)).type(torch.double)
        Response_hyp  = pyro.sample("Response_hyp",dist.Normal(hyperparams[12:20],errors[13:21]).to_event(1)).type(torch.double)
        Z             = pyro.sample("Z_loc"       ,dist.Normal(Z_param,errors[21].expand([self.u])).to_event(1))
        noise         = pyro.sample("Noise"       ,dist.Normal(noise_param, torch.ones(1)*1e-5)).type(torch.double)
        # set GP
        kernel_SE = GP.SqExp_kernel(Kernel_hyp[0].exp(),Kernel_hyp[1].exp())
        #kernel_P1 = GP.Periodic_kernel(torch.exp(Kernel_hyp[2]),torch.exp(Kernel_hyp[3]),torch.tensor([1]).type(torch.double))
        #kernel_P7 = GP.Periodic_kernel(torch.exp(Kernel_hyp[4]),torch.exp(Kernel_hyp[5]),torch.tensor([7]).type(torch.double))
        kernel    = GP.additive_composite_kernel([kernel_SE
                                                  #,kernel_P1,kernel_P7
                                                 ])
        m = pyro.param("m",dist.Normal(meanparam,.1).sample([self.u])).type(torch.double)
        S = pyro.param("S",torch.linalg.cholesky(kernel.c(Z,Z)+1e-4*torch.eye(self.u)),constraint=dist.constraints.lower_cholesky).type(torch.double)
        # subsample
        for i in pyro.plate("Obs",len(self.data),subsample = subsample):
            pyro.sample("Feedback_{}".format(i),dist.Normal(Feedback_hyp[0:3],torch.exp(Feedback_hyp[3:6])).to_event())
            pyro.sample("f_z_{}".format(i)     ,dist.MultivariateNormal(m,scale_tril=S))
            pyro.sample("Response_{}".format(i),dist.Normal(Response_hyp[0:4],torch.exp(Response_hyp[4:8])).to_event())


        