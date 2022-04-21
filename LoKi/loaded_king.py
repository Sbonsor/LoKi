import numpy as np
import sys
import matplotlib.pyplot as plt
import utils
import ODEs
from scipy.interpolate import interp1d
from scipy.special import gammainc
from scipy.special import gamma
plt.rc('text', usetex=True)

class LoadedKing:
    
    def __init__(self,mu,x_0,central_concentration,model,final_radius,asymptotics):
        
        self.mu, self.x_0, self.central_concentration = mu, x_0, central_concentration
        
        self.epsilon = central_concentration - (9*mu/(4*np.pi*x_0))
        
#        if(self.epsilon < 0): 
#            print("Error: $\\epsilon$ must be greater than or equal to 0. $\\epsilon = $"+str(self.epsilon))
#            print(str(mu) + ' ' + str(x_0) + ' ' +str(central_concentration))
#            sys.exit(0)
            
        self.alpha = self.epsilon*np.power(x_0,-2)
        
        self.psi,self.rhat,self.psi_grad,self.rt,self.U_hat,self.K_hat = ODEs.solve_full_poisson(mu,x_0,central_concentration,model,final_radius)
        
        self.dimensionless_density = ODEs.dimensionless_density(self.psi,model)
        self.dimensionless_pressure = ODEs.dimensionless_pressure(self.psi,model)
        self.dimensionless_velocity_dispersion = ODEs.dimensionless_velocity_dispersion(self.psi,model)
        
        self.mass_profile = -mu-self.psi_grad*np.power(self.rhat,2)*(4*np.pi/9)
        
        self.total_mass = self.mass_profile[-1];
        self.total_potential_energy = self.U_hat[-1]
        self.total_kinetic_energy =  self.K_hat[-1]
        
        
#        self.density_slope = self.psi_grad*(ODEs.dimensionless_density(self.psi,model) + np.power(self.psi,3/2))
        
        self.probability_density = self.dimensionless_density/self.total_mass        
        
#        
#        self.projected_radius,self.projected_density = utils.project_density(self.rhat,self.dimensionless_density)
#        self.half_mass_radius = utils.half_mass(self.rhat,self.mass_profile)
            
        #######################################  Asymptotics 
        if(asymptotics):
            
            self.psi_inner,self.r,psi_inner_grad, self.psi_inner_2,self.psi_inner_2_grad, self.simple_asymp_rt = ODEs.solve_inner(mu,x_0,central_concentration,model,final_radius)
            
            if(self.epsilon < 0):
                print('epsilon<0')
                self.psi_outer = self.R = psi_outer_grad = None
            else:  
                self.psi_outer,self.R,self.psi_outer_grad,self.simple_asymp_rt = ODEs.solve_outer(mu,x_0,central_concentration,model,final_radius)
            
            self.kappa= kappa = -9*(2/5)/ODEs.dimensionless_density(central_concentration,model)
            
            self.psi_0_reg_1,self.psi_0_reg_1_grad,self.psi_2_reg_1,self.psi_2_reg_1_grad,self.psi_4_reg_1,self.psi_4_reg_1_grad,self.r_1,self.asymp_rt = ODEs.solve_region1(mu,x_0,central_concentration,self.alpha,model,final_radius)
            
            self.psi_reg_1 = self.psi_0_reg_1 + np.power(x_0,2)*self.psi_2_reg_1
            self.psi_reg_1_grad = (self.psi_0_reg_1_grad + np.power(x_0,2)*self.psi_2_reg_1_grad)*np.power(x_0,-1) 
            
            self.next_order = self.psi_0_reg_1 + np.power(x_0,2)*self.psi_2_reg_1 + np.power(x_0,4)*self.psi_4_reg_1
            self.next_order_grad = (self.psi_0_reg_1_grad + np.power(x_0,2)*self.psi_2_reg_1_grad + np.power(x_0,4)*self.psi_4_reg_1_grad)*np.power(x_0,-1)
            
            self.B_func= 2*kappa*np.power(central_concentration,5/2)*np.power(self.r_1,1/2) - np.power(self.r_1,2)*self.psi_2_reg_1_grad
            self.B = self.B_func[-1]
            
            self.A_func = self.psi_2_reg_1 +4*kappa*np.power(central_concentration,5/2)*np.power(self.r_1,-1/2) - self.B/self.r_1
            self.A = self.A_func[-1]
            
            if(self.A<0):
                print("A<0")
                self.psi_reg_2 = self.r_2 = self.psi_reg_3 = self.psi_0_reg_3 = self.psi_1_reg_3 = self.r_3 = None
            else:
                phi_1,phi_1_grad,self.r_2 = ODEs.solve_region2(mu,x_0,central_concentration,self.A,kappa,final_radius,None)
        
                self.psi_reg_2 = (central_concentration/self.r_2 +self.A + x_0*phi_1)*np.power(x_0,2)
                
                phi_0_reg_3, self.phi_0_reg_3_grad, phi_1_reg_3, phi_1_reg_grad, self.r_3, self.asymp_rt = ODEs.solve_region3(x_0,central_concentration,self.A,kappa,model,final_radius)
        
                self.psi_reg_3 = (phi_0_reg_3 + np.power(x_0,1/2)*phi_1_reg_3)*np.power(x_0,2)
                self.psi_0_reg_3 = phi_0_reg_3*np.power(x_0,2)
                
                
        #########################################  


    def plot_all(self,vlines = False):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('font', weight='bold')
        plt.rc('font', size=14)
        
        fig1,ax1 = plt.subplots(1,1)

        ax1.axhline( y = 0 )
        ax1.set_ylim(0,self.central_concentration)
        ax1.set_xlim(self.x_0,max(self.rhat))
        ax1.plot(self.rhat,self.psi, label = 'Full solution')
        ax1.plot(self.r*self.x_0,self.psi_inner, label = 'Simple inner')
        ax1.plot(self.r_1*self.x_0, self.psi_reg_1, label = 'Region I')
        
        if(self.epsilon >0):
            ax1.plot(self.R,self.psi_outer,label = 'Simple outer')
        
        if(self.A>0):
            #ax1.plot(self.r_2*np.power(self.x_0,-1),self.psi_reg_2, label = 'Region II')
            ax1.plot(self.r_3*np.power(self.x_0,-3/2),self.psi_0_reg_3, label = 'Region III')
            #ax1.plot(self.r_3*np.power(self.x_0,-3/2),self.psi_reg_3, label = 'Region III to first order')
            
        #ax1.set_title('$(\\alpha,a_0,\\mu,\\beta)$=('+utils.truncate(self.alpha,4)+','+utils.truncate(self.A,4)+','+utils.truncate(self.mu,4)+','+utils.truncate(self.epsilon,4)+')')
        ax1.set_title('$(\\Psi,\\mu,\\epsilon)=$'+'('+str(self.central_concentration)+','+str(self.mu)+','+str(self.x_0)+')' )
        ax1.legend()
        ax1.set_xlabel('$\hat{r}$')
        ax1.set_ylabel('$\psi$')
        ax1.set_xlim(0,self.rt)
                
        if(vlines):
            ax1.axvline(x=self.rt, color = 'b')
            ax1.axvline(x=self.simple_asymp_rt, color = 'r')
            ax1.axvline(x=self.asymp_rt, color = 'k')
            
        plt.show()
        

        
    

