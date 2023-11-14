
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gammainc,gamma
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(seed = 12009837)
"""
Created on Tue Apr 26 16:21:38 2022

@author: s1984454
"""
class LoKi_samp:
    
    def __init__(self,model,**kwargs ):
        
        self._set_kwargs(model,**kwargs)
        
        self.sample_positions()
        self.sample_velocities()
        self.set_masses()
        
        if(self.scale_nbody):
            self.scale_output()
        
        if (self.print):
            self.print_results()
        
        
    def _set_kwargs(self, model, **kwargs):
        
        self.model = model
        self.N = 1000
        self.scale_nbody = True
        self.print = False
        self.fname = 'LoKi_samples.csv'
        self.plot = False
    
        if kwargs is not None:
            for key,value in kwargs.items():
                setattr(self,key,value)
    
    def sample_positions(self):
    
        rand1 = np.random.rand(self.N)

        self.r = np.interp(rand1, self.model.M_r/self.model.M_hat, self.model.rhat)

        rand2 = np.random.rand(self.N)
        rand3 = np.random.rand(self.N)

        r2 = self.r ** 2
        self.z = (1-2*rand2) * self.r
        self.x = np.sqrt(r2 - self.z**2)*np.cos(2*np.pi*rand3)
        self.y = np.sqrt(r2 - self.z**2)*np.sin(2*np.pi*rand3)

        self.psi_samples = np.interp(self.r, self.model.rhat, self.model.psi)
        
    def sample_velocities(self):
        
        def cdf_v(v, max_v, psi):
            
            numerator =   np.sqrt(2)*np.exp(psi)*gamma(3/2)*gammainc(3/2,v**2/2) - v**3/3
            denominator = np.sqrt(2)*np.exp(psi)*gamma(3/2)*gammainc(3/2,max_v**2/2) - max_v**3/3
            
            return numerator/denominator
            
        self.v = np.zeros(self.N)

        for j in range(self.N):

            psi_j = self.psi_samples[j]
            
            v_max = np.sqrt(2*psi_j)
            vs = np.linspace(0,v_max,10000)
            cdf = cdf_v(vs, v_max, psi_j)

            rand4 = np.random.rand()
            self.v[j] = np.interp(rand4, cdf, vs)

        rand6 = np.random.rand(self.N)
        rand7 = np.random.rand(self.N)

        v2 = self.v ** 2
        self.vz = (1-2*rand6) * self.v
        self.vx = np.sqrt(v2 - self.vz**2)*np.cos(2*np.pi*rand7)
        self.vy = np.sqrt(v2 - self.vz**2)*np.sin(2*np.pi*rand7) 
            
        self.E_hat_samp = 0.5*self.v**2 - self.psi_samples
            
    def set_masses(self):
        
        model = self.model
        
        if(self.scale_nbody):
            
            self.m = np.ones(self.N)/self.N
            
        else:
            
            self.m = np.ones(self.N)*model.M_hat/self.N
                
    def scale_nody(self):
        
        model = self.model
        
        self.r_k = -2*model.U_hat/np.power(model.M_hat, 2)
        self.a = -9*model.U_hat/(2*np.pi*model.M_hat)
        sqrt_a = np.sqrt(self.a)
        self.E_0 = -1/(self.a*self.r_k*model.rt)
        self.Ae = -3*np.power(self.a,3/2)*np.power(model.M_hat,5)/(64*np.sqrt(2)*np.pi*np.power(model.U_hat,3)*model.rho_hat[0])
        
        
        self.x,self.y,self.z = self.x*self.r_k, self.y*self.r_k, self.z*self.r_k
        self.vz,self.vx,self.vy = self.vz/sqrt_a, self.vx/sqrt_a, self.vy/sqrt_a
    
    def print_results(self):
    
        results = pd.DataFrame(data  = {'x':self.x, 'y':self.y, 'z':self.z, 'vx':self.vx, 'vy':self.vy, 'vz':self.vz,'m':self.m})
        results.to_csv(self.fname,index = False,columns = ['x','y','z','vx','vy','vz','m'])

    def print_validation_figs(self):
        
        model = self.model
        
        if(self.plot and self.scale_nbody):
            
            def density_normalisation(r,rho):
                integrand = r**2*rho
                return np.trapz(y = integrand, x = r)
            
            fig1,ax1 = plt.subplots(1,1)
            ax1.plot(model.rhat,model.rhat**2*model.rho_hat/density_normalisation(model.rhat,model.rho_hat))
            ax1.hist(self.r,density = True,bins = 20)
            ax1.set_xlabel('$\hat{r}$')
            ax1.set_ylabel('Radius probability')
            
            def dimensionless_dM_dE(E_hats,psi,r,npoints):
                r_psi = interp1d(x = psi,y = r, bounds_error = False, fill_value = (r[-1],r[0]))
                psi_r = interp1d(x = r, y = psi,bounds_error = False, fill_value = (psi[0],psi[-1]))
                
                integrand = np.zeros((len(E_hats),npoints))
                r_grid = np.zeros((len(E_hats),npoints))
                
                for i in range(len(E_hats)):
                    r_max = r_psi(-E_hats[i])
                    
                    r_grid[i,:] = np.linspace(r[0],r_max, npoints)
                    integrand[i,:] = (np.exp(-E_hats[i])-1) * np.power(r_grid[i,:],2)*np.nan_to_num(np.sqrt(2*(E_hats[i] + psi_r(r_grid[i,:]))))
                
                integral = np.trapz(y = integrand, x = r_grid, axis = 1)
            
                return 16*np.pi**2 * integral
            
            E_hat = np.linspace(-model.Psi,0,100)
            theoretical_dimensionless_dM_dE = dimensionless_dM_dE(E_hat,model.psi,model.rhat,1000)

            E = E_hat/self.a + self.E_0
            theoretical_dM_dE = self.Ae * self.r_k**3 * np.power(self.a,-0.5) * theoretical_dimensionless_dM_dE
            
            fig2,ax2 = plt.subplots(1,1)
            ax2.hist(self.E_hat_samp/self.a + self.E_0,density = True,bins = 20)
            ax2.plot(E,theoretical_dM_dE)
            ax2.set_xlabel('E')
            ax2.set_ylabel('dM/dE')
            
            return fig1,fig2
        
        else:
            raise Exception("scale_nbody and plot must both be true in order to print validation figures currently")
        
        
        
 
    
    
        
        
        