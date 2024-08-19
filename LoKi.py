#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp,simps
from scipy.special import gammainc,gamma
from scipy.interpolate import interp1d

"""
Created on Thu Apr 21 15:21:33 2022

@author: Samuel Bonsor
"""

class LoKi:
    
    def __init__(self, mu, epsilon, Psi, **kwargs ):
        
        self._set_kwargs(mu, epsilon, Psi, **kwargs)
        
        self.solve_poisson()
        
        if (self.scale):
            self.scale()
            
        if (self.project):
            
            self.d = self.rhat
            self.proj_dens = self.project_quantity(self.rho_hat)
            self.proj_press = self.project_quantity(self.P_hat)
            self.proj_vel_disp = self.proj_press/self.proj_dens
            
    
    def _set_kwargs(self, mu, epsilon, Psi, **kwargs):
        
        if (Psi - 9*mu/(4*np.pi*epsilon) < 0):
            raise ValueError("a_0 must not be negative")
            
        self.Psi = Psi
        self.epsilon = epsilon
        self.mu = mu
        self.scale = False
        self.project = False
        self.max_r = 1e9
        self.ode_atol = 1e-30
        self.ode_rtol = 1e-8
        self.pot_only = False
        self.asymptotics = False
        self.model = 'K'
        
        if kwargs is not None:
            for key,value in kwargs.items():
                setattr(self,key,value)
        
    def solve_poisson(self):
        
        def odes(r,y):
            RHS = np.zeros(2,)
    
            RHS[0] = y[1]
            RHS[1] = -(2/r)*y[1]-9*(self.density(y[0])/self.density(self.Psi))
            
            if(self.pot_only == False):
                RHS = np.append(RHS,4*np.pi*r*(self.density(y[0])/self.density(self.Psi))*(self.mu + (4*np.pi/9)*r**2*y[1]))
                RHS = np.append(RHS,2*np.pi*r**2*self.pressure(y[0]))
                
            return RHS
        
        def initial_conditions():
            IC = np.zeros(2,)
            
            IC[0] = self.Psi
            IC[1] = -(9*self.mu)/(4*np.pi*np.power(self.epsilon,2))
            
            if(self.pot_only == False):
                IC = np.append(IC,np.zeros(2,))
                
            return IC
        
        def zero_cross(r, y): return y[0]
        zero_cross.terminal = True
        zero_cross.direction = -1
        
        def solve_odes():
            
               poisson_solution = solve_ivp(fun = odes, t_span = (self.epsilon,self.max_r), y0 = initial_conditions(), method = 'RK45',
                                dense_output = True,rtol = self.ode_rtol, atol=self.ode_atol, events = zero_cross)
               return poisson_solution
        
        poisson_solution = solve_odes()
        
        self.psi = poisson_solution.y[0,:]
        self.dpsi_dr = poisson_solution.y[1,:]
        self.rhat = poisson_solution.t
        self.rt = self.rhat[-1]
        
        if(self.pot_only == False):
            
            self.U_r = poisson_solution.y[2,:]
            self.K_r = poisson_solution.y[3,:]
            self.M_r = -self.mu-self.dpsi_dr*np.power(self.rhat,2)*(4*np.pi/9)
            self.M_hat = self.M_r[-1]
            self.U_hat = self.U_r[-1]
            self.rho_hat = self.density(self.psi)
            self.P_hat = self.pressure(self.psi)
        
    def scale(self):
        
         self.r_k = -2*self.U_hat/np.power(self.M_hat, 2)
         self.a = -9*self.U_hat/(2*np.pi*self.M_hat)
         self.E_0 = -1/(self.a*self.r_k*self.rt)
         self.Ae = -3*np.power(self.a,3/2)*np.power(self.M_hat,5)/(64*np.sqrt(2)*np.pi*np.power(self.U_hat,3)*self.density(self.Psi))
         self.A_hat = 8*np.sqrt(2)*np.pi*self.Ae/(3*np.power(self.a,3/2))
        
    def density(self,psi):
        if self.model == 'K':
            density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
            density = np.nan_to_num(density,copy = False)
            
        if self.model == 'PT':
            density = (3/2)*np.exp(psi)*gamma(3/2)*gammainc(3/2,psi)
            density = np.nan_to_num(density,copy = False)
            
        if self.model == 'W':
            density = (2/5)*np.exp(psi)*gamma(7/2)*gammainc(7/2,psi)
            density = np.nan_to_num(density,copy = False) 
            
        return density
    
    def pressure(self,psi):
        
        if self.model == 'K':
            pressure = (2/5)*np.exp(psi)*gamma(7/2)*gammainc(7/2,psi)
            pressure = np.nan_to_num(pressure,copy = False)
            
        if self.model == 'PT':
            pressure = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
            pressure = np.nan_to_num(pressure,copy = False)
            
        if self.model == 'W':
            pressure = (4/35)*np.exp(psi)*gamma(9/2)*gammainc(9/2,psi)
            pressure = np.nan_to_num(pressure,copy = False)
            
        return pressure
    
    def project_quantity(model,f):
        d = model.rhat
        proj_f = np.zeros(len(d))
        
        for i in range(len(d)):
            idx = (model.rhat >= d[i])
            r = model.rhat[idx]
            z = np.sqrt(abs(r**2 - d[i]**2))
            
            proj_f[i] = 2.0*abs(simps(f[idx], x=z))
            
        return proj_f

                    
    