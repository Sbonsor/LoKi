#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:11:02 2024

@author: s1984454
"""

from LoKi import LoKi
import numpy as np
import matplotlib.pyplot as plt
from LoKi_asymptotics import LoKi_asymp, determine_critical_alpha
import pickle
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.integrate import quadrature, simps

plt.rc('text', usetex=True)
plt.rcParams['font.size'] = 16

##### Fig. 2.1 Top

# Psi = 5
# epsilon = 0.1
# mu_max = 4*np.pi*epsilon*Psi/9
# mu_fracs = np.linspace(0,1,5)

# fig, ax = plt.subplots(1,1)
# ax.set_title(f'$(\\Psi,\\epsilon)=$({Psi},{epsilon})')
# ax.set_xlabel('$r$')
# ax.set_ylabel('$\\psi$')

# rt = 0
# for mu_frac in mu_fracs:
    
#     mu = mu_frac*mu_max
#     model = LoKi(mu, epsilon, Psi)
#     ax.plot(model.rhat, model.psi, label = '$\\mu/\\mu_{max}' + f'=${mu_frac}')
#     rt = max(rt, model.rt)

# ax.legend(loc = 'upper right')
# ax.set_xlim(epsilon, rt)
# ax.set_ylim(0,Psi)
# ax.set_xscale('log')

# ##### Fig. 2.1 Bottom
# a_0 = 3
# epsilon = 0.1
# Psis = np.linspace(3,8,5)

# fig, ax = plt.subplots(1,1)
# ax.set_title(f'$(a_0,\\epsilon)=$({Psi},{epsilon})')
# ax.set_xlabel('$r$')
# ax.set_ylabel('$\\psi$')

# rt = 0
# for Psi in Psis:
    
#     mu = (Psi - a_0) *(4*np.pi*epsilon)/9
#     model = LoKi(mu, epsilon, Psi)
#     ax.plot(model.rhat, model.psi, label = f'$\\Psi=${Psi}')
#     rt = max(rt, model.rt)

# ax.legend(loc = 'upper right')
# ax.set_xlim(epsilon, rt)
# ax.set_ylim(0,max(Psis))
# ax.set_xscale('log')

##### Fig. 2.2

# Psi = 5
# epsilon = 0.1
# mu_fracs = np.linspace(0,1,500)

# mu_max = 4*np.pi*epsilon*Psi/9

# rts = []
# M_Cs = []

# for mu_frac in mu_fracs:
    
#     mu = mu_frac*mu_max
#     model = LoKi(mu, epsilon, Psi)
    
#     rts.append(model.rt)
#     M_Cs.append(model.M_hat)
    
#     print(mu_frac)
    
# fig, ax = plt.subplots(1,1)
# ax.set_yscale('log')
# ax.set_title(f'$(\\Psi, \\epsilon)=$({Psi},{epsilon})')
# ax.set_xlabel('$\\mu/\\mu_{max}$')
# ax.set_ylabel('$r_t$')
# lines1 = ax.plot(mu_fracs, rts, label = '$r_t$', linewidth = 1, color = 'blue')

# ax2 = ax.twinx()
# ax2.set_ylabel('$M_C$')
# ax2.set_yscale('log')
# lines2 = ax2.plot(mu_fracs, M_Cs, label = '$M_C$', linewidth = 1, color = 'orange')

# lines = lines1 +lines2
# labs = [l.get_label() for l in lines]

# ax.legend(lines, labs, loc = 'center left')

##### Fig. 2.3

# Psi = 5
# epsilon = 0.1

# alpha_c, C_Psi, D_Psi, kappa = determine_critical_alpha(Psi, alpha_min = -40 , alpha_max = 100, plot = False)
# A2 = alpha_c - D_Psi - 40*kappa**2*Psi**4*np.log(epsilon)
# A0 = epsilon**2 * A2 - C_Psi
# mu_c = (4*np.pi*epsilon)/9 * (Psi - A0*epsilon**2)

# displacements = np.linspace(-0.007, 0.007, 500)

# rts = []
# M_Cs = []

# for displacement in displacements:
    
#     mu = mu_c + displacement
#     model = LoKi(mu, epsilon, Psi)
    
#     rts.append(model.rt)
#     M_Cs.append(model.M_hat)
    
#     print(displacement)
    
# fig, ax = plt.subplots(1,1)
# ax.set_yscale('log')
# ax.set_title(f'$(\\Psi, \\epsilon)=$({Psi},{epsilon})')
# ax.set_xlabel('$\\mu-\\mu_c$')
# ax.set_ylabel('$r_t$')
# ax.plot(displacements, rts, label = '$r_t$', linewidth = 1, color = 'blue')

# ax2 = ax.twinx()
# ax2.set_ylabel('$M_C$')
# ax2.set_yscale('log')
# ax2.plot(displacements, M_Cs, label = '$M_C$', linewidth = 1, color = 'orange')

# ax.axvspan(-0.001, 0.001, alpha=0.3, color='red', label = '$a_2=O(\\epsilon^2)$')
# ax.axvspan(0.001, 0.007, alpha=0.3, color='blue', label = '$a_0 = O(\\epsilon^2)$, $a_2 = O(1)$ ')
# ax.axvspan(-0.006, -0.001, alpha=0.3, color='blue')
# ax.axvspan(-0.007, -0.006, alpha=0.3, color='gray', label = '$a_0=O(1)$')
# ax.set_xlim(-0.007, 0.007)

# h1, l1 = ax.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()

# ax.legend(h1+h2, l1+l2, loc = 'center left')

##### Fig. 2.4

# Psi = 5
# epsilon = 0.1
# mu_frac = 0.8

# mu_max = 4*np.pi*epsilon*Psi/9

# mu = mu_frac * mu_max

# model = LoKi(mu, epsilon, Psi)
# asymp_model = LoKi_asymp(model)

# fig, ax = plt.subplots(1,1)
# ax.set_title('$(\\Psi, \\mu/\\mu_{max},\\epsilon)=$' + f'({Psi},{mu_frac},{epsilon})')
# ax.set_xlabel('$r$')
# ax.set_ylabel('$\\psi$')
# ax.set_xscale('log')
# ax.set_xlim(epsilon, model.rt)
# ax.set_ylim(0,Psi)

# ax.plot(model.rhat, model.psi, label = 'Numerical solution')
# ax.plot(asymp_model.r_1_regime_I * epsilon, asymp_model.psi_0_regime_I_region_1, label = 'Region 1 solution', linestyle = 'dashed')
# ax.plot(asymp_model.r_2_regime_I, asymp_model.psi_0_regime_I_region_2, label = 'Region 2 solution', linestyle = 'dotted')
# ax.axvline(asymp_model.r_2t, color = 'k')
# ax.axvline(model.rt)
# ax.axvline(asymp_model.r_2_regime_I[-1], color = 'r')

##### Fig. 2.5

# Psi = 5
# epsilon = 0.1
# mu_fracs = np.linspace(0,0.999,100)

# mu_max = 4*np.pi*epsilon*Psi/9

# rts = []
# asymp_rts = []

# for mu_frac in mu_fracs:
    
#     mu = mu_frac*mu_max
#     model = LoKi(mu, epsilon, Psi)
#     asymp_model = LoKi_asymp(model, regime_I_flag = True, regime_II_flag = False, regime_III_flag = False)
    
#     rts.append(model.rt)
#     asymp_rts.append(asymp_model.r_2t)
    
#     print(mu_frac)
    
# fig, ax = plt.subplots(1,1)
# ax.set_yscale('log')
# ax.set_title(f'$(\\Psi, \\epsilon)=$({Psi},{epsilon})')
# ax.set_xlabel('$\\mu/\\mu_{max}$')
# ax.set_ylabel('$r_t$')
# lines1 = ax.plot(mu_fracs, rts, label = 'Numerical solution', linewidth = 1, color = 'blue')
# lines2 = ax.plot(mu_fracs, asymp_rts, label = 'Regime I solution', linewidth = 1, color = 'orange')

# lines = lines1 +lines2
# labs = [l.get_label() for l in lines]

# ax.legend(lines, labs, loc = 'center left')

##### Fig. 2.6

# Psi = 5
# epsilon = 0.1
# mu_frac = 0.9

# mu_max = 4*np.pi*epsilon*Psi/9

# mu = mu_frac * mu_max

# model = LoKi(mu, epsilon, Psi)
# asymp_model = LoKi_asymp(model)

# fig, ax = plt.subplots(1,1)
# ax.set_title('$(\\Psi, \\mu/\\mu_{max},\\epsilon)=$' + f'({Psi},{mu_frac},{epsilon})')
# ax.set_xlabel('$r$')
# ax.set_ylabel('$\\psi$')
# ax.set_xscale('log')
# ax.set_xlim(epsilon, model.rt)
# ax.set_ylim(1e-4,Psi)
# ax.set_yscale('log')

# ax.plot(model.rhat, model.psi, label = 'Numerical solution')
# ax.plot(asymp_model.r_1_regime_II * epsilon, asymp_model.psi_regime_II_region_1, label = 'Region 1 solution', linestyle = 'dashed')
# ax.plot(asymp_model.r_2_regime_II * epsilon**(-1), asymp_model.psi_regime_II_region_2, label = 'Region 2 solution', linestyle = 'dotted')
# ax.plot(asymp_model.r_3_regime_II * epsilon**(-3/2), asymp_model.psi_regime_II_region_3, label = 'Region 2 solution', linestyle = 'dashdot')

# ax.legend(loc = 'lower left')

##### Fig. 2.7

# Psi = 5
# epsilon = 0.1
# mu_fracs = np.linspace(0,0.999,100)

# mu_max = 4*np.pi*epsilon*Psi/9

# rts = []
# asymp_rts = []

# for mu_frac in mu_fracs:
    
#     mu = mu_frac*mu_max
#     model = LoKi(mu, epsilon, Psi)
#     asymp_model = LoKi_asymp(model, regime_I_flag = False, regime_II_flag = True, regime_III_flag = False)
    
#     rts.append(model.rt)
#     asymp_rts.append(asymp_model.r_t_regime_II)
    
#     print(mu_frac)
    
# fig, ax = plt.subplots(1,1)
# ax.set_yscale('log')
# ax.set_title(f'$(\\Psi, \\epsilon)=$({Psi},{epsilon})')
# ax.set_xlabel('$\\mu/\\mu_{max}$')
# ax.set_ylabel('$r_t$')
# lines1 = ax.plot(mu_fracs, rts, label = 'Numerical solution', linewidth = 1, color = 'blue')
# lines2 = ax.plot(mu_fracs, asymp_rts, label = 'Regime II solution', linewidth = 1, color = 'orange')

# lines = lines1 +lines2
# labs = [l.get_label() for l in lines]

# ax.legend(lines, labs, loc = 'center left')

##### Fig. 2.8

# def calculate_rt(Psi,n_mu,n_eps):
    
#     epsilons = np.linspace(0.01, 0.5, n_eps)
    
#     mu_matrix = np.zeros((n_mu,n_eps))
#     rts = np.zeros((n_mu,n_eps))

#     j = 0
    
#     for epsilon in epsilons:

#         mus = np.linspace(0,4*np.pi*epsilon*Psi/9-1e-6,n_mu)
#         mu_matrix[:,j] = mus
        
#         i = 0    
#         for mu in mus:
            
#             print(f'{Psi}_{epsilon}_{mu}')
            
#             model = LoKi(mu, epsilon, Psi, pot_only = True)
#             rts[i,j] = model.rt

#             i = i+1            
#         j = j+1
        
#     results_dict = {'Psi':Psi, 'epsilons':epsilons,'mu_matrix':mu_matrix, 'rts':rts }
    
#     fname = f'rt_cmap_Psi_{Psi}_nmus_{n_mu}_neps_{n_eps}'
    
#     with open('Data/'+ fname + '.pkl', 'wb') as handle:
#         pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#     return 1

# def generate_cplots_parameter_exploration(Psis, nmus, neps):
    
#     fig, ax = plt.subplots(2,2)

#     for i,Psi in enumerate(Psis):
        
#         with open(f'Data/rt_cmap_Psi_{Psi}_nmus_{nmus}_neps_{neps}.pkl', 'rb') as f:
#             data = pickle.load(f)
        
#         rts = data['rts']
    
#         mu_matrix = data['mu_matrix']
    
#         epsilons = np.reshape(data['epsilons'], (1,neps))
#         epsilon_matrix = np.repeat(epsilons, nmus, axis = 0)
    
#         axis = ax.flatten()[i]
#         contour1 = axis.contourf(epsilon_matrix, mu_matrix, np.log10(rts), levels = 100)
#         axis.set_xlabel('$\\epsilon$')
#         axis.set_ylabel('$\\mu$')
#         cbar1 = fig.colorbar(contour1, ax=axis)
#         cbar1.set_label('$\\log_{10}(r_t)$')
#         axis.set_title(f'$\\Psi=$ {Psi}')
        
#         alpha_c, C_Psi, D_Psi, kappa = determine_critical_alpha(Psi, alpha_min = -40 , alpha_max = 100, plot = False)
        
#         epsilons = np.linspace(0.01, 0.5, n_eps)
#         a_2_line = 4*np.pi*epsilons/9 * (Psi + epsilons**2 * C_Psi)
#         axis.plot(epsilons, a_2_line, color = 'r', linewidth = 0.5)
    
#     return 0

# Psis = [3,5,7,9]
# n_mu = 500
# n_eps = 500

# ### Only run this loop if the relevant output files are not in the Data directory. This loop can take a long time to run
# # for Psi in Psis:
    
# #     print(Psi)
# #     calculate_rt(Psi,n_mu,n_eps)

# generate_cplots_parameter_exploration(Psis, n_mu, n_eps)

##### Fig. 2.9

# Psi = 5
# alpha_c, C_Psi, D_Psi, kappa = determine_critical_alpha(Psi, alpha_min = -40 , alpha_max = 100, plot = True)

##### Fig. 2.10

# Psi = 5
# epsilon = 0.1
# mu_fracs = np.linspace(0,1,5)

# mu_max = 4*np.pi*epsilon*Psi/9

# fig, ax = plt.subplots(2,1)

# ax[0].set_xlabel('$r$')
# ax[0].set_ylabel('$\\rho/\\rho_0$')
# ax[0].set_title('$(\\Psi,\\epsilon)=$' + f'({Psi},{epsilon})')
# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
# ax[0].set_ylim(1e-4,1)

# for mu_frac in mu_fracs:
#     mu = mu_frac * mu_max
    
#     model = model = LoKi(mu, epsilon, Psi)
    
#     ax[0].plot(model.rhat, model.density(model.psi)/model.density(Psi), label = '$\\mu/\\mu_{max}=$' + f'{mu_frac}')
    
# ax[0].legend(loc = 'lower left')

# a0 = 3
# epsilon = 0.1
# Psis = np.linspace(3,8,5)

# ax[1].set_xlabel('$r$')
# ax[1].set_ylabel('$\\rho/\\rho(a_0)$')
# ax[1].set_title('$(\\Psi,\\epsilon)=$' + f'({Psi},{epsilon})')
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
# ax[1].set_ylim(1e-4,150)

# for Psi in Psis:

#     mu = 4*np.pi*epsilon/9 * (Psi - a0)
    
#     model = model = LoKi(mu, epsilon, Psi)
    
#     ax[1].plot(model.rhat, model.density(model.psi)/model.density(a0), label = '$\\Psi=$' + f'{Psi}')
    
# ax[1].legend(loc = 'lower left')

##### Fig. 2.11

# Psi = 5
# epsilon = 0.1
# mu_fracs = np.linspace(0,1,5)

# mu_max = 4*np.pi*epsilon*Psi/9

# fig, ax = plt.subplots(2,1)

# ax[0].set_xlabel('$r$')
# ax[0].set_ylabel('$\\sigma^2/\\sigma^2_0$')
# ax[0].set_title('$(\\Psi,\\epsilon)=$' + f'({Psi},{epsilon})')
# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
# ax[0].set_ylim(1e-2,1.6)

# for mu_frac in mu_fracs:
#     mu = mu_frac * mu_max
    
#     model = model = LoKi(mu, epsilon, Psi)
    
#     normalised_vel_disp = model.pressure(model.psi)*model.density(Psi) / (model.density(model.psi) * model.pressure(Psi))
    
#     ax[0].plot(model.rhat, normalised_vel_disp, label = '$\\mu/\\mu_{max}=$' + f'{mu_frac}')
    
# ax[0].legend(loc = 'lower left')

# a0 = 3
# epsilon = 0.1
# Psis = np.linspace(3,8,5)

# ax[1].set_xlabel('$r$')
# ax[1].set_ylabel('$\\sigma^2/\\sigma(a_0)^2$')
# ax[1].set_title('$(\\Psi,\\epsilon)=$' + f'({Psi},{epsilon})')
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
# ax[1].set_ylim(1e-2,1.6)

# for Psi in Psis:

#     mu = 4*np.pi*epsilon/9 * (Psi - a0)
    
#     model = model = LoKi(mu, epsilon, Psi)
    
#     normalised_vel_disp = model.pressure(model.psi)*model.density(a0) / (model.density(model.psi) * model.pressure(a0))
    
#     ax[1].plot(model.rhat, normalised_vel_disp, label = '$\\Psi=$' + f'{Psi}')
    
# ax[1].legend(loc = 'lower left')

##### Fig. 2.12

# Psi = 5
# epsilon = 0.1
# mu_fracs = np.linspace(0,1,5)

# mu_max = 4*np.pi*epsilon*Psi/9

# fig, ax = plt.subplots(2,1)

# ax[0].set_xlabel('$r$')
# ax[0].set_ylabel('$\\Sigma/\\Sigma_0$')
# ax[0].set_title('$(\\Psi,\\epsilon)=$' + f'({Psi},{epsilon})')
# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
# ax[0].set_ylim(1e-4,1)

# for mu_frac in mu_fracs:
#     mu = mu_frac * mu_max
    
#     model = model = LoKi(mu, epsilon, Psi, project = True)
    
#     ax[0].plot(model.rhat, model.proj_dens/model.proj_dens[0], label = '$\\mu/\\mu_{max}=$' + f'{mu_frac}')
    
# ax[0].legend(loc = 'lower left')

# a0 = 3
# epsilon = 0.1
# Psis = np.linspace(3,8,5)

# ax[1].set_xlabel('$r$')
# ax[1].set_ylabel('$\\Sigma/\\Sigma_0$')
# ax[1].set_title('$(\\Psi,\\epsilon)=$' + f'({Psi},{epsilon})')
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
# ax[1].set_ylim(1e-4,1)

# for Psi in Psis:

#     mu = 4*np.pi*epsilon/9 * (Psi - a0)
    
#     model = model = LoKi(mu, epsilon, Psi, project = True)
    
#     ax[1].plot(model.rhat, model.proj_dens/model.proj_dens[0], label = '$\\Psi=$' + f'{Psi}')
    
# ax[1].legend(loc = 'lower left')

##### Fig. 2.13

# Psi = 5
# epsilon = 0.1
# mu_fracs = np.linspace(0,1,5)

# mu_max = 4*np.pi*epsilon*Psi/9

# fig, ax = plt.subplots(2,1)

# ax[0].set_xlabel('$r$')
# ax[0].set_ylabel('$\\sigma^2_d/\\sigma^2_d0$')
# ax[0].set_title('$(\\Psi,\\epsilon)=$' + f'({Psi},{epsilon})')
# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
# ax[0].set_ylim(1e-2,1.1)

# for mu_frac in mu_fracs:
#     mu = mu_frac * mu_max
    
#     model = model = LoKi(mu, epsilon, Psi, project = True)
    
#     ax[0].plot(model.rhat, model.proj_vel_disp/model.proj_vel_disp[0], label = '$\\mu/\\mu_{max}=$' + f'{mu_frac}')
    
# ax[0].legend(loc = 'lower left')

# a0 = 3
# epsilon = 0.1
# Psis = np.linspace(3,8,5)

# ax[1].set_xlabel('$r$')
# ax[1].set_ylabel('$\\sigma^2_d/\\sigma^2_d0$')
# ax[1].set_title('$(\\Psi,\\epsilon)=$' + f'({Psi},{epsilon})')
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
# ax[1].set_ylim(1e-2,1.1)

# for Psi in Psis:

#     mu = 4*np.pi*epsilon/9 * (Psi - a0)
    
#     model = model = LoKi(mu, epsilon, Psi, project = True)
    
#     ax[1].plot(model.rhat, model.proj_vel_disp/model.proj_vel_disp[0], label = '$\\Psi=$' + f'{Psi}')
    
# ax[1].legend(loc = 'lower left')
 
##### Fig. 2.14

# Psi = 5
# epsilon = 0.1
# mu_max = 4*np.pi*epsilon*Psi/9
# mu = 0.5*mu_max
# npoints = 100

# def energy_function(r,E,J,psi): 
#       function = 2*(E+psi(r))/np.power(J,2) - np.power(r,-2)
#       return function
 
# def Period_integrand(r,E,J,psi):
#     v_squared = abs(2*(E+psi(r))-np.power(J/r,2))
#     return np.power(v_squared,-1/2)

# def DF(E):
#     if E<0:
#         f = np.exp(-E)-1
#     else:
#         f = 0
#     return f

# model = LoKi(mu,epsilon, Psi)
# psi = interp1d(x=model.rhat,y=model.psi)

# E_grid = np.linspace(-Psi,0,npoints)
# E_J_f = []
# f = np.zeros((npoints,npoints))
# J_centrifugal_boundary = []
# J_inner_boundary = []

# for E in E_grid:
    
#     J_circ = model.rhat*np.sqrt(2*(E+model.psi))
#     J_min = epsilon*np.sqrt(2*(E+Psi))
    
#     J_centrifugal_boundary.append(max(J_circ))
#     J_inner_boundary.append(J_min)

# J_grid =np.linspace(0,max(J_centrifugal_boundary),npoints)
# J_inner_boundary = np.array(J_inner_boundary)
# J_centrifugal_boundary = np.array(J_centrifugal_boundary)

# for i in range(npoints):
#     E = E_grid[i]
#     J_max = J_centrifugal_boundary[i]
#     J_min = J_inner_boundary[i]

#     for j in range(npoints):
#         J = J_grid[j]
#         if(J<J_min or J>J_max):
#             f[i][j] = 0
#         else:             
#             if J == 0:
#                 f[i][j] = 0
                
#             else:
#                 r = np.logspace(np.log10(epsilon),np.log10(model.rt),100000)
#                 r[0] = epsilon
#                 r[-1] = model.rt
                
#                 v_r_function = 2*(E+psi(r))/np.power(J,2) - np.power(r,-2)
#                 v_r_function[np.where(abs(v_r_function)<1e-10)] = 0
                
#                 max_idx = np.where(v_r_function == max(v_r_function))[0][0]
                
#                 ## Checks if this is a circular orbit at r=epsilon, or if r_p<epsilon which is not an allowed orbit. Setting the r_a=r_p in this case ensures that the density is zero in this case.
#                 if (r[max_idx]==model.epsilon):
#                     r_p=r_a=epsilon
                    
#                 ## Checks if r_p = epsilon in which case manually set r_p as the r_p routine won't work in this case.    
#                 elif(v_r_function[0]==0):
#                     r_p = epsilon
#                     r_a = brentq(energy_function, a = r[max_idx], b = model.rt, args = (E,J,psi))
#                 else:
#                     r_p = brentq(energy_function, a = model.epsilon, b = r[max_idx], args = (E,J,psi))
#                     r_a = brentq(energy_function, a = r[max_idx], b = model.rt, args = (E,J,psi))
                
#                 ## Checks for a circular or disallowed orbit and sets the density there to zero.
#                 if(r_a == r_p):
#                     density = 0
#                 else:             
#                     T_r,T_r_error = quadrature(Period_integrand,r_p,r_a, args = (E,J,psi), maxiter = 500,tol = 1e-5,rtol=1e-8)
#                     density = 16*np.pi**2*J*DF(E)*T_r
                    
#                 f[i][j] = density

# # ##### Obtain a,A,E_0
# M_hat = model.M_hat
# U_hat = model.U_hat

# r_k = -2*U_hat/np.power(M_hat,2)
# rho_0 = -np.power(M_hat,5)/(8*np.power(U_hat,3))
# a = -9*U_hat/(2*np.pi*M_hat)
# Ae = -3*np.power(a,3/2)*np.power(M_hat,5)/(64*np.sqrt(2)*np.pi*np.power(U_hat,3)*model.density(Psi))
# E_0 = -1/(a*r_k*model.rt)

# x = J_grid*(r_k/np.sqrt(a))
# y = E_grid/a +E_0
# z = f*np.power(r_k,2)*Ae
# integral_of_f = simps(simps(z, y), x)

# for i in range(len(z)):
#     for j in range(len(z)):
#         if(z[i][j] == 0):
#             z[i][j] = np.nan

# z_min = np.min(z[np.nonzero(z)])
# z_max = np.max(z)
# fig2,ax2 = plt.subplots(1,1)
# ax2.contour(x,y,z,levels=10,linewidths=0.5,colors='k')
# cmap = plt.cm.jet
# cmap.set_under('w')
# CP2 = ax2.contourf(x,y,z,levels = 10,cmap=cmap)
# ax2.plot(J_inner_boundary*(r_k/np.sqrt(a)),E_grid/a +E_0,linestyle = '--',color = 'red')
# ax2.plot(J_centrifugal_boundary*(r_k/np.sqrt(a)),E_grid/a + E_0,linestyle = '--',color = 'red')
# ax2.set_xlabel('J')
# ax2.set_ylabel('E')
# cbar2 = fig2.colorbar(CP2)
# cbar2.ax.set_ylabel('f(E,J)')
# ax2.set_title('$(\\Psi,\\mu/\\mu_{max},\\epsilon) = $'+'('+str(Psi)+','+str(0.5)+','+str(epsilon)+')')

##### Fig. 2.16

# def df(v,psi_r):
#     if (0.5*v**2-psi_r < 0):
#         df=np.exp(psi_r-0.5*v**2)-1
#     else:
#         df=0
#     return df

# def v_integrand(v,psi_r):
#     DF = df(v,psi_r)
#     if(DF == 0):
#         return 0
#     else:
#         return v**2 * (DF*np.log(DF))
    
# def entropy_n_body(model,npoints):
#     psi = interp1d(x=model.rhat,y=model.psi)
    
#     r_grid = np.linspace(model.epsilon, model.rt, npoints)
#     r_integrand = np.zeros(len(r_grid))
    
#     for i in range(len(r_grid)):
#         r = r_grid[i]
#         psi_r = psi(r)
#         if psi_r<0:
#             r_integrand[i] = 0 
#         else:
            
#             upper_vlim = np.sqrt(2*psi_r)
#             r_integrand[i] =-16*np.pi**2 *  r**2 * quadrature(v_integrand, 0, upper_vlim, args=(psi_r), tol=1.49e-08, rtol=1.49e-08, maxiter=50,vec_func = False)[0]
        
        
#     S_hat = np.trapz(x=r_grid,y=r_integrand)
    
#     M_hat = model.M_hat
#     U_hat = model.U_hat
    
#     r_k = -2*U_hat/np.power(M_hat,2)
#     a = -9*U_hat/(2*np.pi*M_hat)
#     Ae = -3*np.power(a,3/2)*np.power(M_hat,5)/(64*np.sqrt(2)*np.pi*np.power(U_hat,3)*model.density(model.Psi))
    
#     prefactor = np.power(r_k,3)*Ae/np.power(a,3/2)
#     shift = -8*np.sqrt(2)*np.pi*model.density(model.Psi)*M_hat*np.log(Ae)/3
    
#     S = prefactor*(S_hat + shift)
#     return S

# ### Top
# a0s = np.linspace(1,20,50)
# epsilon = np.spacing(1)

# mus = [0, 0.05]
# npoints = 1000
# entropies = []

# fig,ax = plt.subplots(1,1)
# ax.set_xlabel('$\\Psi$')
# ax.set_ylabel('$S$')

# for mu in mus:
    
#     Psis = a0s + (9*mu)/(4*np.pi*epsilon)
    
#     for Psi in Psis:
#         model = LoKi(mu, epsilon, Psi)
#         S = entropy_n_body(model, npoints)
#         entropies.append(S)
#         print(Psi)
        
#     ax.plot(Psis,entropies, label = '$\\mu=$' + f'{mu}')

#### Bottom

# Psi = 5
# epsilon = 0.1
# Nmus = 100
# mus = np.linspace(0,4*np.pi*epsilon*Psi/9,Nmus)
# mus_zoom = np.linspace(0.65,4*np.pi*epsilon*Psi/9,Nmus)
# npoints = 1000
# entropies = []
# entropies_zoom = []

# for i in range(Nmus):
#     mu = mus[i]
#     mu_zoom = mus_zoom[i]
    
#     model = LoKi(mu, epsilon, Psi)
#     model_zoom = LoKi(mu_zoom, epsilon, Psi)
    
#     S = entropy_n_body(model,npoints = 1000)
#     S_zoom = entropy_n_body(model_zoom,npoints = 1000)
    
#     entropies.append(S)
#     entropies_zoom.append(S_zoom)
    
#     print(mu)

# ### Determining critical mu
# alpha_c, C_Psi, D_Psi, kappa = determine_critical_alpha(Psi, alpha_min = -40 , alpha_max = 100, plot = True)
# mu_crit = 4*np.pi*epsilon*(Psi+epsilon**2*C_Psi)/9

# ##plotting figure
# fig1,ax1 = plt.subplots(1,1)
# ax1.scatter(mus,entropies,marker ='x')
# ax1.set_title('$(\\Psi,\\epsilon) = ($'+str(Psi)+','+str(epsilon)+')')
# ax1.set_xlabel('$\\mu$')
# ax1.set_ylabel('$S$')
# ax1.axvline(x=mu_crit,linestyle='--',color='k')

# ax2 = fig1.add_axes([0.2,0.4, 0.4, 0.4])
# ax2.axvline(x=mu_crit,linestyle='--',color='k')
# ax2.scatter(mus_zoom,entropies_zoom,marker='x')

##### Fig. 2.17

# def beta_energy(mu,epsilon, Psi):
    
#     non_loaded_model = LoKi(mu, epsilon, Psi)
#     beta = ((81 * 3**(2/3))/(16*np.pi**2) ) * (non_loaded_model.M_hat**(4/3)) / non_loaded_model.density(Psi)**(2/3)
#     energy = -2*np.pi*non_loaded_model.density(Psi)**(2/3)*non_loaded_model.U_hat / (3**(8/3) * non_loaded_model.M_hat**(7/3))
    
#     return beta, energy

# E_scale = 8/3
# beta_scale = 4/3

# fig, [ax1,ax2] = plt.subplots(2,1)
# axins1 = fig.add_axes([0.65, 0.7, 0.2, 0.15])

# #### First add the classical King behaviour

# epsilon = 1e-6
# mu = 0
# a0s = np.concatenate((np.linspace(epsilon**2, 1, 50), np.linspace(1, 25, 200)))

# Es = np.zeros(len(a0s))
# betas = np.zeros(len(a0s))

# for i, a0 in enumerate(a0s):
    
#     Psi = a0 + (9*mu)/(4*np.pi*epsilon)
#     betas[i], Es[i] = beta_energy(mu,epsilon, Psi)
#     print(i)

# ax1.plot(Es, betas, label = '$\\epsilon$ = 0', color = 'k')#, marker = 'x')

# ### Add polytropic behaviour
# ax1.plot(Es, 5.77*Es**0.5, color = 'k', linestyle = '--')

# ### Add loaded lines

# epsilons = [0.1, 0.05, 0.01]

# Psi = 5
# alpha_c, C_Psi, D_Psi, kappa = determine_critical_alpha(Psi, alpha_min = -40 , alpha_max = 100, plot = False)
# epsilon = 0.1
# mu_c = (4*np.pi*epsilon/9) * (Psi + epsilon**2 * C_Psi + epsilon**4*np.log(epsilon) * 40 * Psi**4 * kappa**2 - epsilon**4 * (alpha_c - D_Psi))

# mu_epsilon = mu_c/0.1

# for i, epsilon in enumerate(epsilons):
    
#     if (i ==2):
#         a0s = np.concatenate((np.logspace(np.log10(0.000503637), np.log10(0.000505208), 300),np.linspace(0.00049688, 0.12*epsilon , 300) ,np.linspace(0.12*epsilon, 25, 300)))
        
#     if(i == 1):
#         a0s = np.concatenate((np.logspace(np.log10(4.9*epsilon**2), np.log10(0.255*epsilon), 300), np.linspace(0.255*epsilon, 25, 300)))

#     if(i == 0):
#         a0s = np.concatenate((np.logspace(np.log10(4.6*epsilon**2), np.log10(0.52*epsilon), 300), np.linspace(0.52*epsilon, 25, 300)))

#     mu = mu_epsilon * epsilon
#     Es = np.zeros(len(a0s))
#     betas = np.zeros(len(a0s))
    
#     scaled_Es = np.zeros(len(a0s))
#     scaled_betas = np.zeros(len(a0s))

#     for j, a0 in enumerate(a0s):
        
#         Psi = a0 + (9*mu)/(4*np.pi*epsilon)
#         betas[j], Es[j] = beta_energy(mu,epsilon, Psi)
        
#         scaled_betas[j], scaled_Es[j] = epsilon **(-beta_scale) * betas[j], epsilon ** (-E_scale) * Es[j]
#         print(f'({i},{j})')
#     ax1.plot(Es, betas, label = f'$\\epsilon$ = {epsilon}')#, marker = 'x')
#     axins1.plot(scaled_Es, scaled_betas)
    
# ax1.set_xlim(0,2)
# ax1.set_ylim(0,1.7)
# ax1.set_xlabel('$\\mathcal{E}$')
# ax1.set_ylabel('$\\beta$')
# ax1.set_title('')
# ax1.legend(loc='lower right')

# axins1.set_ylabel('$\\beta \\epsilon^{-4/3}$') 
# axins1.set_xlabel('$\\mathcal{E}\\epsilon^{-8/3}$')       
# axins1.set_xlim(0,1000)
# axins1.set_ylim(0,0.5)

# ###### Second panel
# Psi = 5
# alpha_c, C_Psi, D_Psi, kappa = determine_critical_alpha(Psi, alpha_min = -40 , alpha_max = 100, plot = True)

# epsilons = [0.1, 0.05, 0.01]
# for i, epsilon in enumerate(epsilons):
    
#     mu_max = 4*np.pi*epsilon*Psi/9
#     mu_c = (4*np.pi*epsilon/9) * (Psi + epsilon**2 * C_Psi + epsilon**4*np.log(epsilon) * 40 * Psi**4 * kappa**2 - epsilon**4 * (alpha_c - D_Psi))
    
#     critical_model_beta, critical_model_E = beta_energy(mu_c,epsilon, Psi)
    
#     width = mu_max-mu_c
    
#     mus = np.concatenate((np.linspace(0,mu_c-width/2,500) ,np.linspace(mu_c - width/2, mu_c+width/2, 500),np.linspace(mu_c+width/2, mu_c+width, 100)))
#     Es = np.zeros(len(mus))
#     betas = np.zeros(len(mus))
    
#     scaled_Es = np.zeros(len(mus))
#     scaled_betas = np.zeros(len(mus))
    
#     for j, mu in enumerate(mus):
        
#         betas[j], Es[j] = beta_energy(mu,epsilon, Psi)
#         scaled_betas[j], scaled_Es[j] = epsilon **(-4/3) * betas[j], epsilon ** (-8/3) * Es[j]
        
#     ax2.plot(Es, betas, label = f'$\\epsilon$ = {epsilon}')
#     ax2.scatter(critical_model_E, critical_model_beta, marker = 'x', color = 'r')
    
# ax2.set_xlabel('$\\mathcal{E}$')
# ax2.set_ylabel('$\\beta$')
# ax2.set_title(f"$\\Psi =$ {Psi}")
# ax2.set_xlim(0,1.5)
# ax2.set_ylim(0,2)
# classical_model_beta, classical_model_E = beta_energy(0,1e-6, Psi)
# ax2.scatter(classical_model_E, classical_model_beta, marker = 'x', color = 'k', label = 'Classical model')
# ax2.legend(loc = 'lower right')
# plt.savefig('fig7_remade')

##### Fig. 2.18

# Psi = 5
# epsilon = 0.1
# mu = 0.4

# models = ['K', 'PT', 'W']

# fig, ax = plt.subplots(1,1)
# ax.set_xlabel('$r$')
# ax.set_ylabel('$\\psi$')
# ax.set_xscale('log')
# ax.set_title('$(\\Psi, \\mu,\\epsilon)=$'+f'({Psi},{mu},{epsilon})')

# for model in models:
    
#     LoKi_model = LoKi(mu, epsilon, Psi, model = model)
    
#     ax.plot(LoKi_model.rhat, LoKi_model.psi, label = model)

# zero_order = (Psi - (9*mu)/(4*np.pi*epsilon)) + (9*mu)/(4*np.pi*LoKi_model.rhat)
# ax.plot(LoKi_model.rhat, zero_order, linestyle = '--', color = 'r', label = '$\\psi_0^{(I)}$') 

# ax.legend(loc = 'upper right')   
