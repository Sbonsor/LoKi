
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gammainc,gamma
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import root_scalar
"""
Created on Tue Apr 26 16:21:38 2022

@author: s1984454
"""
class LoKi_samp:
    
    def __init__(self, model, **kwargs ):
        
        self._set_kwargs(model,**kwargs)
        
        self.generate_dimensionless_samples()

        self.set_masses()
        
        if(self.scale_nbody_flag):
            self.scale_nbody()
        
        if(self.scale_LoKi_flag):
            self.scale_LoKi()
        
        if(self.save):
            self.print_results()
            
        if(self.validate):
            self.print_validation_figures()
        
        
    def _set_kwargs(self, model, **kwargs):
        
        self.model = model
        self.N = 1000
        self.scale_nbody_flag = False
        self.scale_LoKi_flag = False
        self.save = False
        self.fname = 'LoKi_samples.csv'
        self.plot = False
        self.M = None
        self.rK = None
        self.G = 4.3009e-3
        self.validate = False
    
        if kwargs is not None:
            for key,value in kwargs.items():
                setattr(self,key,value)
        
        if(self.scale_nbody_flag and self.scale_LoKi_flag):
            raise ValueError('Both scale flags can not be True.')
        
        if(self.scale_LoKi_flag and (self.M == None or self.rK == None)):
            raise ValueError('Please enter the two dimensional scales M, and rK required for the selected dimensions.')
            
    def set_masses(self):
        
        model = self.model
        
        if(self.scale_nbody_flag):
            
            self.m = np.ones(self.N)/self.N
            
        elif(self.scale_LoKi_flag):
            
            self.m = np.ones(self.N)*self.M/self.N
            
        else:
            
            self.m = np.ones(self.N)*model.M_hat/self.N
            
    def sample_positions(self, N):
    
        rand1 = np.random.rand(N)

        r = np.interp(rand1, self.model.M_r/self.model.M_hat, self.model.rhat)

        rand2 = np.random.rand(N)
        rand3 = np.random.rand(N)

        r2 = r ** 2
        z = (1-2*rand2) * r
        x = np.sqrt(r2 - z**2)*np.cos(2*np.pi*rand3)
        y = np.sqrt(r2 - z**2)*np.sin(2*np.pi*rand3)

        psi_samples = np.interp(r, self.model.rhat, self.model.psi)
        
        return x, y, z, psi_samples
        
    def sample_velocities(self, N, psi_samples):
        
        def cdf_v(v, max_v, psi):
            
            numerator =   np.sqrt(2)*np.exp(psi)*gamma(3/2)*gammainc(3/2,v**2/2) - v**3/3
            denominator = np.sqrt(2)*np.exp(psi)*gamma(3/2)*gammainc(3/2,max_v**2/2) - max_v**3/3
            
            return numerator/denominator
            
        v = np.zeros(N)

        for j in range(N):

            psi_j = psi_samples[j]
            
            v_max = np.sqrt(2*psi_j)
            vs = np.linspace(0,v_max,10000)
            cdf = cdf_v(vs, v_max, psi_j)

            rand4 = np.random.rand()
            v[j] = np.interp(rand4, cdf, vs)

        rand6 = np.random.rand(N)
        rand7 = np.random.rand(N)

        v2 = v ** 2
        vz = (1-2*rand6) * v
        vx = np.sqrt(v2 - vz**2)*np.cos(2*np.pi*rand7)
        vy = np.sqrt(v2 - vz**2)*np.sin(2*np.pi*rand7)
        
        return vx, vy, vz
    
    def generate_dimensionless_samples(self):
        
        self.dimensionless_samples = []
    
        while (len(self.dimensionless_samples) < self.N):
            
            N = self.N - len(self.dimensionless_samples)
            
            xhats, yhats, zhats, psi_samples = self.sample_positions(N)
            vxhats, vyhats, vzhats = self.sample_velocities(N, psi_samples)
            
            for i in range(N):
    
                xhat = xhats[i]
                yhat = yhats[i]
                zhat = zhats[i]
                vxhat = vxhats[i]
                vyhat = vyhats[i]
                vzhat = vzhats[i]
    
                v_t = np.sqrt(vxhat**2 + vyhat**2)
                vhat = np.sqrt(vxhat**2 + vyhat**2 + vzhat**2)
                rhat = np.sqrt(xhat**2 + yhat**2 + zhat**2)
                psi_interp = interp1d(self.model.rhat, self.model.psi, fill_value = (self.model.Psi, 0), bounds_error = False)
    
                Jhat = v_t * rhat
                Ehat = np.clip(0.5* vhat**2 - psi_interp(rhat), a_max = 0, a_min = None)
    
                def f(r, Ehat, Jhat, psi_interp):
                    
                    function = 2*(Ehat + psi_interp(r)) - (Jhat/r)**2
                    
                    return function
            
                try:
                    r_grid = np.linspace(self.model.rhat[0], self.model.rhat[-1], 10000)
                    function = 2*(Ehat + psi_interp(r_grid)) - (Jhat/r_grid)**2
                    upper_bracket = r_grid[np.where(function == max(function))[0][0]]
                    r_p = root_scalar(f = f, args = (Ehat, Jhat, psi_interp), bracket = [self.model.rhat[0], upper_bracket], method = 'bisect').root
                    
                except:
                    print('Tried to sample a point without a resolved r_p, retrying calculation with finer radial grid.')
                    r_grid = np.linspace(self.model.rhat[0], self.model.rhat[-1], 1000000)
                    function = 2*(Ehat + psi_interp(r_grid)) - (Jhat/r_grid)**2
                    
                    if function[0]<=0:
                        upper_bracket = r_grid[np.where(function == max(function))[0][0]]
                        r_p = root_scalar(f = f, args = (Ehat, Jhat, psi_interp), bracket = [self.model.rhat[0], upper_bracket], method = 'bisect').root
                    else:
                        r_p = 0
                        
                if (r_p >= self.model.epsilon):
                    self.dimensionless_samples.append([xhat, yhat, zhat, vxhat, vyhat, vzhat])
                else:
                    print(f'Sample at r_p={r_p} rejected.')
                print(len(self.dimensionless_samples))
                
        self.dimensionless_samples = np.array(self.dimensionless_samples)        
        self.x_hat = self.dimensionless_samples[:,0]
        self.y_hat = self.dimensionless_samples[:,1]
        self.z_hat = self.dimensionless_samples[:,2]
        self.vx_hat = self.dimensionless_samples[:,3]
        self.vy_hat = self.dimensionless_samples[:,4]
        self.vz_hat = self.dimensionless_samples[:,5]
            
            
    def scale_nbody(self):
        
        model = self.model
        
        self.rK = -2*model.U_hat/np.power(model.M_hat, 2)
        self.a = -9*model.U_hat/(2*np.pi*model.M_hat)
        self.sqrt_a = np.sqrt(self.a)
        self.E_0 = -1/(self.a*self.rK*model.rt)
        self.Ae = -3*np.power(self.a,3/2)*np.power(model.M_hat,5)/(64*np.sqrt(2)*np.pi*np.power(model.U_hat,3)*model.rho_hat[0])
                
        self.dimensional_samples = np.zeros((self.N, 6))

        self.dimensional_samples[:,0] = self.x_hat * self.rK
        self.dimensional_samples[:,1] = self.y_hat * self.rK
        self.dimensional_samples[:,2] = self.z_hat * self.rK
        self.dimensional_samples[:,3] = self.vx_hat / self.sqrt_a
        self.dimensional_samples[:,4] = self.vy_hat / self.sqrt_a
        self.dimensional_samples[:,5] = self.vz_hat / self.sqrt_a
        
    def scale_LoKi(self):
        
        self.A_hat = self.M/(self.model.M_hat * self.model.density(self.model.Psi) * self.rK**3)
        self.sqrt_a = np.sqrt( 9/(4 * np.pi * self.G * self.A_hat * self.model.density(self.model.Psi) * self.rK**2) )
        
        self.dimensional_samples = np.zeros((self.N, 6))
        
        self.x  = self.dimensional_samples[:,0] = self.x_hat * self.rK
        self.y  = self.dimensional_samples[:,1] = self.y_hat * self.rK
        self.z  = self.dimensional_samples[:,2] = self.z_hat * self.rK
        self.vx = self.dimensional_samples[:,3] = self.vx_hat / self.sqrt_a
        self.vy = self.dimensional_samples[:,4] = self.vy_hat / self.sqrt_a
        self.vz = self.dimensional_samples[:,5] = self.vz_hat / self.sqrt_a
                    
        return 1
    
    def print_results(self):
        
        dimensionless_results = pd.DataFrame(self.dimensionless_samples, columns = ['x_hat','y_hat','z_hat','vx_hat','vy_hat','vz_hat'])
        dimensionless_results.to_csv(f'dimensionless_{self.fname}', index = False, columns = ['x_hat','y_hat','z_hat','vx_hat','vy_hat','vz_hat'])
        
        if(self.scale_nbody_flag or self.scale_LoKi_flag):
            dimensional_results = pd.DataFrame(self.dimensional_samples, columns = ['x','y','z','vx','vy','vz'])
            dimensional_results.to_csv(f'dimensional_{self.fname}', index = False, columns = ['x','y','z','vx','vy','vz'])
            
    def print_validation_figures(self):
                
        def rad_prob_density_profile(rhats, vhats):
            
            def density_normalisation(r,rho):
                integrand = r**2*rho
                return np.trapz(y = integrand, x = r)

            fig1,ax1 = plt.subplots(1,1)
            ax1.plot(model.rhat,model.rhat**2*model.rho_hat/density_normalisation(model.rhat,model.rho_hat))
            ax1.hist(rhats,density = True,bins = 50)
            ax1.set_xlabel('$\hat{r}$')
            ax1.set_ylabel('Radial probability density')
            ax1.set_title(f'N = {len(rhats)}')
            
            nks, bin_edges = np.histogram(rhats, density=False, bins = 50)        
            rks = np.zeros(len(nks))
            for i in range(len(nks)):
                rks[i] = (bin_edges[i+1]+bin_edges[i])/2
                
            drs = [bin_edges[i+1]-bin_edges[i] for i in range(len(nks))]
            rho_ks = nks/(4*np.pi*rks**2*drs)

            dr0 = drs[0]
            n0 = np.sum(rhats<dr0)
            rho_0 = n0/((4/3) * np.pi * dr0**3)

            fig2, ax2 = plt.subplots(1,1)
            ax2.scatter(x = rks, y = rho_ks/rho_0, marker = 'x')
            ax2.plot(model.rhat, model.density(model.psi)/model.density(model.Psi), color = 'r')
            ax2.set_yscale('log')
            ax2.set_xlabel('\\hat{r}')
            ax2.set_ylabel('$log\\hat\\rho/\\rho_0$')
            ax2.set_title(f'N = {len(rhats)}')
            
            return rks, rho_ks/rho_0, rho_ks, rho_0
        
        def pressure_profile(self, rhats, vhats):
            
            rs = rhats
            vs = vhats

            xs = self.x_hat
            ys = self.y_hat
            zs = self.z_hat
            vxs = self.vx_hat
            vys = self.vy_hat
            vzs = self.vz_hat
            
            nks, bin_edges = np.histogram(rhats, density=False, bins = 50)        
            rks = np.zeros(len(nks))
            rks = np.zeros(len(nks))
            for i in range(len(nks)):
                rks[i] = (bin_edges[i+1]+bin_edges[i])/2
            
            drs = [bin_edges[i+1]-bin_edges[i] for i in range(len(nks))]
            rho_ks = nks/(4*np.pi*rks**2*drs)
            
            Pks = np.zeros(len(nks))
            for i in range(len(nks)):
                in_shell = (rs < bin_edges[i+1])&(rs>bin_edges[i])
                
                v_rsi = (xs[in_shell] * vxs[in_shell] + ys[in_shell] * vys[in_shell] + zs[in_shell] * vzs[in_shell])/rs[in_shell]
                Pks[i] = (rho_ks[i]/nks[i]) * np.sum(v_rsi**2)
            
            dr0 = drs[0]
            n0 = np.sum(rhats<dr0)
            rho_0 = n0/((4/3) * np.pi * dr0**3)
            
            in_sphere = (rs < dr0)
            v_rsi = (xs[in_sphere] * vxs[in_sphere] + ys[in_sphere] * vys[in_sphere] + zs[in_sphere] * vzs[in_sphere])/rs[in_sphere]
            P0 = (rho_0/n0)*np.sum(v_rsi**2)
            
            return rho_ks, rho_0, Pks, P0 
        
        model = self.model
        
        rhats = np.sqrt(self.x_hat**2 + self.y_hat**2 + self.z_hat**2)
        vhats = np.sqrt(self.vx_hat**2 + self.vy_hat**2 + self.vz_hat**2)
        
        rks, normalised_density, rho_ks, rho_0 = rad_prob_density_profile(rhats, vhats)
        
        rho_ks, rho_0, Pks, P0 = pressure_profile(self, rhats, vhats)
        
        fig3, ax3 = plt.subplots(1,1)
        ax3.scatter(x = rks, y =Pks/P0, marker = 'x')
        ax3.plot(model.rhat, model.pressure(model.psi)/model.pressure(model.Psi), color = 'r')
        ax3.set_yscale('log')
        ax3.set_xlabel('\\hat{r}')
        ax3.set_ylabel('$logP/P_0$')
        ax3.set_title(f'N = {len(rhats)}')
        
        

        
        
        
 
    
    
        
        
        
