#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:11:48 2023

@author: s1984454
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import gammainc
from scipy.special import gamma
from scipy.integrate import solve_ivp
from LoKi import LoKi

class LoKi_asymp:
    
    def __init__(self, LoKi_model, **kwargs ):
        
        self._set_kwargs(LoKi_model, **kwargs)
                
        if self.regime_I_flag:
            self.solve_regime_I()
            
        if self.regime_II_flag:
            self.solve_regime_II()
            
        if self.regime_III_flag:
            self.solve_regime_III()
    
    def _set_kwargs(self, LoKi_model, **kwargs):
            
        self.regime_I_flag = True
        self.regime_II_flag = True
        self.regime_III_flag = True
        
        self.Psi = LoKi_model.Psi
        self.epsilon = LoKi_model.epsilon
        self.mu = LoKi_model.mu
        
        self.a_0 = self.Psi - 9*self.mu/(4*np.pi*self.epsilon)
        self.kappa = 18/(5*self.rho_hat(self.Psi))
        
        if kwargs is not None:
            for key,value in kwargs.items():
                setattr(self,key,value)
    
    def rho_hat(self, psi):
        
       if(psi > 0):
           
           density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
           
       else:
           
           density = 0
 
       return density
    
    def solve_regime_I(self):
        
        def region_1_ODEs(r,y):
            
            RHS= np.zeros(4,)
            RHS[0] = y[1]
            RHS[1] = -(2/r)*y[1]
            RHS[2] = y[3]
            RHS[3] = -(2/r)*y[3] - 9*(self.rho_hat(y[0])/self.rho_hat(self.Psi))
            
            return RHS
        
        def zero_cross_region_1(r,y):
            return y[0] + self.epsilon**2 * y[2]
        zero_cross_region_1.terminal = False
        
        def solve_region_1(self, final_radius = 1e9 ):
            
            solution = solve_ivp(fun = region_1_ODEs, t_span = (1,final_radius/self.epsilon),
                                 y0 = (self.Psi,-9*self.mu/(4*np.pi*self.epsilon),0,0),method = 'RK45',
                                    dense_output = True,rtol = 1e-8, atol = 1e-30,events = zero_cross_region_1)
            
            self.psi_0_regime_I_region_1 = solution.y[0,:]
            self.psi_0_grad_regime_I_region_1 = solution.y[1,:]
            self.psi_2_regime_I_region_1 = solution.y[2,:]
            self.psi_2_grad_regime_I_region_1 = solution.y[3,:]
            self.r_1_regime_I = solution.t
            
            return 1
        
        def region_2_ODEs(r,y):
            
            RHS = np.zeros(2,)
            RHS[0] = y[1]
            RHS[1] = -(2/r)*y[1]-9*self.rho_hat(y[0])/self.rho_hat(self.Psi)
                
            return RHS
        def zero_cross_region_2(r,y):
            return y[0]
        zero_cross_region_2.terminal = True
        
        def solve_region_2(self, final_radius = 1e9):
            solution = solve_ivp(fun = region_2_ODEs, t_span=(np.spacing(1),final_radius), y0=(self.a_0,0),method = 'RK45',
                                         dense_output = True,rtol=1e-8,atol=1e-30,events = zero_cross_region_2)
            self.psi_0_regime_I_region_2 = solution.y[0,:]
            self.psi_0_grad_regime_I_region_2 = solution.y[1,:]
            self.r_2_regime_I = solution.t
            
            self.r_2t = solution.t_events[0][0]
            
            return 1
        

        solve_region_1(self)
        solve_region_2(self)
        
        return 1

    def solve_regime_II(self):
        
        def region_1_ODEs(r,y):
            
            RHS= np.zeros(6,)
            
            RHS[0] = y[1]
            RHS[1] = -(2/r)*y[1]
            RHS[2] = y[3]
            RHS[3] = -(2/r)*y[3] - (9/self.rho_hat(self.Psi)) * self.rho_hat(y[0])
            RHS[4] = y[5]
            RHS[5] = -(2/r)*y[5] - (9/self.rho_hat(self.Psi)) * (self.rho_hat(y[0]) + np.power(y[0],3/2)) * y[2]
            
            return RHS
        
        def zero_cross_region_1(r,y):
            return y[0] + self.epsilon**2 * y[2]
        zero_cross_region_1.terminal = True
        
        def solve_region_1(self, final_radius = 1e9):
            poisson_solution = solve_ivp(fun = region_1_ODEs, t_span =(1,final_radius/self.epsilon), y0 = (self.Psi,-self.Psi,0,self.A_0,0,0),
                                         method = 'RK45', dense_output = True, rtol = 1e-8, atol = 1e-30, events = zero_cross_region_1)
            self.psi_0_regime_II_region_1 = poisson_solution.y[0,:]
            self.psi_0_grad_regime_II_region_1 = poisson_solution.y[1,:]
            self.psi_2_regime_II_region_1 = poisson_solution.y[2,:]
            self.psi_2_grad_regime_II_region_1 = poisson_solution.y[3,:]
            self.psi_4_regime_II_region_1 = poisson_solution.y[4,:]
            self.psi_4_grad_regime_II_region_1 = poisson_solution.y[5,:]
            self.r_1_regime_II = poisson_solution.t
            
            self.psi_regime_II_region_1 = self.psi_0_regime_II_region_1 + self.epsilon**2 * self.psi_2_regime_II_region_1
            self.region_1_events = poisson_solution.t_events
            return 1
        
        def region_2_ODEs(r,y):
            
            RHS=np.zeros(2,)
            
            RHS[0] = y[1]
            RHS[1] = -(2/r)*y[1] - self.kappa*np.power(self.a_2+(self.Psi/r),5/2)
            
            return RHS
            
            return 1
        
        def zero_cross_region_2(r,y):
            return y[0]
        zero_cross_region_2.terminal = True
        
        def solve_region_2(self, final_radius = 1e9):
            inner_limit = np.power(self.epsilon,2)
            
            poisson_solution = solve_ivp(fun = region_2_ODEs, t_span = (inner_limit,final_radius*self.epsilon), 
                                         y0 =(4*self.kappa*np.power(self.Psi,5/2)*np.power(inner_limit,-1/2),-2*self.kappa*np.power(self.Psi,5/2)*np.power(inner_limit,-3/2)),
                                         method = 'RK45', dense_output = True ,rtol = 1e-8, atol = 1e-30)
            
            self.r_2_regime_II = poisson_solution.t
            self.phi_0_regime_II_region_2 = self.Psi/self.r_2_regime_II +self.a_2
            self.phi_0_grad_regime_II_region_2 = -self.Psi/self.r_2_regime_II**2
            self.phi_1_regime_II_region_2 = poisson_solution.y[0,:]
            self.phi_1_grad_regime_II_region_2 = poisson_solution.y[1,:]
            
            
            self.psi_regime_II_region_2 = (self.phi_0_regime_II_region_2 + self.epsilon*self.phi_1_regime_II_region_2)*np.power(self.epsilon,2)
            
            return 1
        
        def region_3_ODEs(r,y):
            
            RHS=np.zeros(2,)
            
            RHS[0] = y[1]
            RHS[1] = -(2/r)*y[1]-9*(2/5)*np.nan_to_num(np.power(y[0],5/2),copy=False)/self.rho_hat(self.Psi)
            
            return RHS
        
        def zero_cross_region_3(r,y):
            return y[0]
        zero_cross_region_3.terminal = True
        
        def solve_region_3(self,final_radius = 1e9):
            
            inner_boundary = self.epsilon*np.power(self.epsilon,3/2)
            solution = solve_ivp(fun = region_3_ODEs, t_span = (inner_boundary , np.power(self.epsilon,3/2)*final_radius), 
                                 y0 = (self.a_2-(self.kappa/6)*np.power(self.a_2,5/2)*np.power(inner_boundary, 2), -(self.kappa/3)*np.power(self.a_2,5/2)* inner_boundary ), 
                                 method = 'RK45', dense_output = True, rtol=1e-8, atol=1e-30, events = zero_cross_region_3)
            self.phi_0_regime_II_region_3 = solution.y[0,:]
            self.phi_0_grad_regime_II_region_3 = solution.y[1,:]
            self.r_3_regime_II = solution.t
            
            self.psi_regime_II_region_3 = np.power(self.epsilon,2) * self.phi_0_regime_II_region_3
            self.r_t_regime_II = solution.t_events[0][0]* self.epsilon**(-3/2)
            return 1
        
        self.A_0 = self.a_0 * self.epsilon**-2
        
        solve_region_1(self)
        
        b_2_func= -2*self.kappa*np.power(self.Psi,5/2)*np.power(self.r_1_regime_II,1/2) - np.power(self.r_1_regime_II,2) * self.psi_2_grad_regime_II_region_1
        self.b_2 = b_2_func[-1]
        
        a_2_func = self.psi_2_regime_II_region_1 - 4*self.kappa*np.power(self.Psi,5/2)*np.power(self.r_1_regime_II,-1/2) - self.b_2/self.r_1_regime_II
        self.a_2 = a_2_func[-1]
        
        if(self.a_2 <= 0):
            self.r_t_regime_II = self.region_1_events[0][0]*self.epsilon
        else:
            solve_region_2(self)
            solve_region_3(self)
        
        return 1
    
    def solve_regime_III(self):
        
        self.A_2 = self.epsilon**-2 * self.a_2
        self.C_Psi = self.a_2 - self.A_0
        
        def region_1_ODEs(r,y):
            
            RHS= np.zeros(6,)
            
            RHS[0] = y[1]
            RHS[1] = -(2/r)*y[1]
            RHS[2] = y[3]
            RHS[3] = -(2/r)*y[3]-(9/self.rho_hat(self.Psi))*self.rho_hat(y[0])
            RHS[4] = y[5]
            RHS[5] = -(2/r)*y[5]-(9/self.rho_hat(self.Psi))*(self.rho_hat(y[0]) + np.power(y[0],3/2))*y[2]
            
            return RHS
        
        def solve_region_1(self, final_radius = 1e9):
            
            solution = solve_ivp(fun = region_1_ODEs, t_span =(1,final_radius/self.epsilon),y0 = (self.Psi,-self.Psi,0,-self.C_Psi,0,self.A_2),
                                         method = 'RK45', dense_output = True,rtol = 1e-8, atol = 1e-30)
            
            self.psi_0_regime_III_region_1  = solution.y[0,:]
            self.psi_0_grad_regime_III_region_1  = solution.y[1,:]
            self.psi_2_regime_III_region_1  = solution.y[2,:]
            self.psi_2_grad_regime_III_region_1  = solution.y[3,:]
            self.psi_4_regime_III_region_1  = solution.y[4,:]
            self.psi_4_grad_regime_III_region_1  = solution.y[5,:]
            self.r_1_regime_III = solution.t
            
            self.psi_regime_III_region_1 = self.psi_0_regime_III_region_1 + self.epsilon**2 * self.psi_2_regime_III_region_1 + self.epsilon**4 * self.psi_4_regime_III_region_1
            
            return 1
        
        solve_region_1(self)
        
        psi_4_large_radius = 4*np.power(self.kappa,2)*np.power(self.Psi,5)/self.r_1_regime_III + 4*np.power(self.kappa,2)*np.power(self.Psi,5)*np.log(self.r_1_regime_III)/self.r_1_regime_III - 10*np.power(self.kappa,2)*np.power(self.Psi,4)*np.log(self.r_1_regime_III) - (4/3)*self.kappa*self.b_2*np.power(self.Psi,5/2)*np.power(self.r_1_regime_III,-3/2) + 10*self.kappa*self.b_2*np.power(self.Psi,3/2)*np.power(self.r_1_regime_III,-1/2)
        
        a_4_func = self.psi_4_regime_III_region_1 - psi_4_large_radius
        self.a_4 = a_4_func[-1]
        self.D_Psi = self.a_4 - self.A_2
        
        def region_2_ODEs(r,y):
            
            RHS = np.zeros(2,)
            
            RHS[0] = y[1]
            RHS[1] = -(2/r)*y[1]-self.kappa*np.power(y[0],5/2) 
            
            return RHS
        
        def zero_cross_region_2(r,y):
            return y[0]
        zero_cross_region_2.terminal = True
        
        def solve_region_2(self, final_radius = 1e9, BC_loc = 0.00000001):
            
            BC1  = self.a_4 + 40 * self.kappa**2 * self.Psi**4 *np.log(self.epsilon) + self.Psi/BC_loc + 4*self.kappa*np.power(self.Psi,5/2)*np.power(BC_loc,-1/2) - 10*np.power(self.kappa,2)*np.power(self.Psi,4)*np.log(BC_loc)
            
            BC2 = -self.Psi*np.power(BC_loc,-2) - 2*self.kappa*np.power(self.Psi,5/2)*np.power(BC_loc,-3/2)-10*np.power(self.kappa,2)*np.power(self.Psi,4)/BC_loc
            
            solution = solve_ivp(fun = region_2_ODEs , t_span = (BC_loc,final_radius), y0 = (BC1,BC2), method = 'RK45',
                                        dense_output = True,rtol = 1e-10,atol=1e-30)
            self.r_2_regime_III = solution.t
            self.phi_0_regime_III_region_2 = solution.y[0,:]
            self.phi_0_grad_regime_III_region_2 = solution.y[1,:]
            
            self.psi_regime_III_region_2 = self.epsilon**4 * self.phi_0_regime_III_region_2
            
            return 1
        
        solve_region_2(self)
        
        return 1
    
def determine_critical_alpha(Psi, alpha_min = -40 , alpha_max = 100, plot = False):
    
    LoKi_model = LoKi(0, 0.1, Psi)
    LoKi_asymptotics =  LoKi_asymp(LoKi_model)
    
    alpha_range = np.linspace(alpha_min, alpha_max,500)
    kappa = 18/(5*LoKi_asymptotics.rho_hat(Psi))
    
    def regime_III_region_2_ODEs(r,y):
        
        RHS = np.zeros(2,)
        
        RHS[0] = y[1]
        RHS[1] = -(2/r)*y[1] - kappa*np.power(y[0],5/2) 
        
        return RHS
    
    # def zero_cross_region_2(r,y):
    #     return y[0]
    # zero_cross_region_2.terminal = True
    
    def solve_region_2(alpha,final_radius = 1e13, BC_loc = 0.00000001):
        
        BC1  = alpha + Psi/BC_loc + 4*kappa*np.power(Psi,5/2)*np.power(BC_loc,-1/2) - 10*np.power(kappa,2)*np.power(Psi,4)*np.log(BC_loc)
        
        BC2 = -Psi*np.power(BC_loc,-2) - 2*kappa*np.power(Psi,5/2)*np.power(BC_loc,-3/2)-10*np.power(kappa,2)*np.power(Psi,4)/BC_loc
        
        solution = solve_ivp(fun = regime_III_region_2_ODEs , t_span = (BC_loc,final_radius), y0 = (BC1,BC2), method = 'RK45',
                                    dense_output = True,rtol = 1e-10,atol=1e-30)
        r_2_regime_III = solution.t
        phi_0_regime_III_region_2 = solution.y[0,:]
        phi_0_grad_regime_III_region_2 = solution.y[1,:]
        
        
        return r_2_regime_III, phi_0_regime_III_region_2, phi_0_grad_regime_III_region_2
    
    r_2_ts = []
    
    for alpha in alpha_range:
        
        r_2_regime_III, phi_0_regime_III_region_2, phi_0_grad_regime_III_region_2 = solve_region_2(alpha,final_radius = 1e13, BC_loc = 0.00000001)
        r_2_ts.append(r_2_regime_III[-1])
    
    alpha_c = alpha_range[np.where(r_2_ts==max(r_2_ts))[0][0]]
            
    C_Psi = LoKi_asymptotics.C_Psi
    D_Psi = LoKi_asymptotics.D_Psi
    
    ###### Test with full numerical solutions
    
    if(plot == True):
        fig,ax = plt.subplots(1,1)
        ax.plot(alpha_range,r_2_ts,color = 'k')
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$r_{t}$')
        ax.set_title('$\\Psi = $ '+ str(Psi)) 
        ax.axvline(x = alpha_c, color = 'k', linestyle = '--')
        
        epsilons = [0.5,0.1,0.01]
        
        for epsilon in epsilons:
            
            epsilon = 0.01
            
            A_2s = alpha_range-D_Psi- 40*np.power(kappa,2)*np.power(Psi,4)*np.log(epsilon)
            a_0s = (A_2s*np.power(epsilon,2)-C_Psi)*np.power(epsilon,2)
            mus = (Psi-a_0s)*(4*np.pi*epsilon)/9
            
            rts = []
            i=0
            for mu in mus:
                model = LoKi(mu,epsilon,Psi)
                rts.append(model.rt)
                i+=1
                print(i)
            
            scaled_rts = np.array(rts)*np.power(epsilon,3)
            
            ax.scatter(alpha_range, scaled_rts, marker = 'x')
        
    return alpha_c, C_Psi, D_Psi, kappa
    
    
        
            
