import numpy as np
from scipy.special import gammainc
from scipy.special import gamma
from scipy.integrate import solve_ivp

def dimensionless_density(psi,model):
    
    if model == 'K':
        density = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
        density = np.nan_to_num(density,copy = False)
        
    if model == 'PT':
        density = (3/2)*np.exp(psi)*gamma(3/2)*gammainc(3/2,psi)
        density = np.nan_to_num(density,copy = False)
        
    if model == 'W':
        density = (2/5)*np.exp(psi)*gamma(7/2)*gammainc(7/2,psi)
        density = np.nan_to_num(density,copy = False) 
        
    return density

def dimensionless_pressure(psi,model):
    
    if model == 'K':
        pressure = (2/5)*np.exp(psi)*gamma(7/2)*gammainc(7/2,psi)
        pressure = np.nan_to_num(pressure,copy = False)
        
    if model == 'PT':
        pressure = np.exp(psi)*gamma(5/2)*gammainc(5/2,psi)
        pressure = np.nan_to_num(pressure,copy = False)
        
    if model == 'W':
        pressure = (4/35)*np.exp(psi)*gamma(9/2)*gammainc(9/2,psi)
        pressure = np.nan_to_num(pressure,copy = False)
        
    return pressure

def dimensionless_velocity_dispersion(psi,model):
    
    if model == 'K':
        dispersion = (2/5)*gamma(7/2)*gammainc(7/2,psi)/(gamma(5/2)*gammainc(5/2,psi))
        dispersion = np.nan_to_num(dispersion,copy = False)
        
    if model == 'PT':
        dispersion = (2/3)*gamma(5/2)*gammainc(5/2,psi)/(gamma(3/2)*gammainc(3/2,psi))
        dispersion = np.nan_to_num(dispersion,copy = False)
        
    if model == 'W':
        dispersion = (2/7)*gamma(9/2)*gammainc(9/2,psi)/(gamma(7/2)*gammainc(7/2,psi))
        dispersion = np.nan_to_num(dispersion,copy = False)
    return dispersion


def zero_cross(r, y,central_concentration,model,mu): return y[0]
zero_cross.terminal = True
zero_cross.direction = -1

def zero_cross_no_mu(r, y,central_concentration,model): return y[0]
zero_cross.terminal = True
zero_cross.direction = -1

### Full Equation
def full_poisson(r,y,central_concentration,model,mu):
    RHS= np.zeros(4,)
    RHS[0] = y[1]
    RHS[1] = -(2/r)*y[1]-9*(dimensionless_density(y[0],model)/dimensionless_density(central_concentration,model))
    RHS[2] = 4*np.pi*r*(dimensionless_density(y[0],model)/dimensionless_density(central_concentration,model))*(mu + (4*np.pi/9)*r**2*y[1])   
    RHS[3] = 2*np.pi*r**2*dimensionless_pressure(y[0], model) 
    return RHS

def solve_full_poisson(mu,x_0,central_concentration,model,final_radius):
    
    poisson_solution = solve_ivp(fun = full_poisson, t_span = (x_0,final_radius), y0 = (central_concentration,-(9*mu)/(4*np.pi*np.power(x_0,2)),0,0), method = 'RK45',
                                dense_output = True, args = [central_concentration,model,mu],rtol = 1e-8,atol=1e-30, events = zero_cross)
    psi = poisson_solution.y[0,:]
    psi_grad = poisson_solution.y[1,:]
    rhat = poisson_solution.t
    U_hat = poisson_solution.y[2,:]
    K_hat = poisson_solution.y[3,:]
    
    if len(poisson_solution.t_events[0] == 1):    
        rt = poisson_solution.t_events[0][0]
    else:
        rt = final_radius
    return psi,rhat,psi_grad,rt,U_hat,K_hat

### Simple asymptotics
def inner(r,y,central_concentration,model):
    RHS= np.zeros(4,)
    RHS[0] = y[1]
    RHS[1] = -(2/r)*y[1]
    RHS[2] = y[3]
    RHS[3] = -(2/r)*y[3] -9*(dimensionless_density(y[0],model)/dimensionless_density(central_concentration,model))
    return RHS

def solve_inner(mu,x_0,central_concentration,model,final_radius):
    poisson_solution = solve_ivp(fun = inner, t_span = (1,final_radius/x_0),y0 = (central_concentration,-9*mu/(4*np.pi*x_0),0,0),method = 'RK45',
                                 dense_output = True,args = [central_concentration,model],rtol = 1e-8, atol = 1e-30,events = zero_cross_no_mu)
    psi_0 = poisson_solution.y[0,:]
    psi_0_grad = poisson_solution.y[1,:]
    psi_2 = poisson_solution.y[2,:]
    psi_2_grad = poisson_solution.y[3,:]
    r = poisson_solution.t
    
    if len(poisson_solution.t_events[0] == 1):    
        rt = x_0 * poisson_solution.t_events[0][0]
    else:
        rt = final_radius
    
    return psi_0,r,psi_0_grad,psi_2,psi_2_grad,rt

def outer(r,y,central_concentration,model):
    RHS=np.zeros(2,)
    RHS[0] = y[1]
    RHS[1] = -(2/r)*y[1]-9*dimensionless_density(y[0],model)/dimensionless_density(central_concentration,model)
        
    return RHS

def solve_outer(mu,x_0,central_concentration,model,final_radius):
    poisson_solution = solve_ivp(fun = outer, t_span=(np.spacing(1),final_radius), y0=(central_concentration-(9*mu/(4*np.pi*x_0)),0),method = 'RK45',
                                 dense_output = True, args = [central_concentration,model],rtol=1e-8,atol=1e-30,events = zero_cross_no_mu)
    psi = poisson_solution.y[0,:]
    psi_grad = poisson_solution.y[1,:]
    R = poisson_solution.t
    
    if len(poisson_solution.t_events[0] == 1):    
        rt = poisson_solution.t_events[0][0]
    else:
        rt = final_radius
    
    return psi,R,psi_grad,rt
    

### Region 1
def zero_cross2(r,y,central_concentration,x_0,model): ## Detects zero crossing for the region I solution
    return y[0] + np.power(x_0,2)*y[2]

def region1(r,y,central_concentration,x_0,model):
    RHS= np.zeros(6,)
    
    RHS[0] = y[1]
    RHS[1] = -(2/r)*y[1]
    RHS[2] = y[3]
    RHS[3] = -(2/r)*y[3]-(9/dimensionless_density(central_concentration,model))*dimensionless_density(y[0],model)
    RHS[4] = y[5]
    RHS[5] = -(2/r)*y[5]-(9/dimensionless_density(central_concentration,model))*(dimensionless_density(y[0],model)+np.power(y[0],3/2))*y[2]
    
    return RHS

def solve_region1(mu,x_0,central_concentration,alpha,model,final_radius):
    poisson_solution = solve_ivp(fun = region1, t_span =(1,final_radius/x_0),y0 = (central_concentration,-central_concentration,0,alpha,0,0),
                                 method = 'RK45', dense_output = True, args = [central_concentration,x_0,model],rtol = 1e-8, atol = 1e-30, events = zero_cross2)
    psi_0_inner = poisson_solution.y[0,:]
    psi_0_inner_grad = poisson_solution.y[1,:]
    psi_2_inner = poisson_solution.y[2,:]
    psi_2_inner_grad = poisson_solution.y[3,:]
    psi_4_inner = poisson_solution.y[4,:]
    psi_4_inner_grad = poisson_solution.y[5,:]
    r = poisson_solution.t
    
    if len(poisson_solution.t_events[0] == 1):    
        rt = x_0*poisson_solution.t_events[0][0]
    else:
        rt = final_radius
    
    return psi_0_inner,psi_0_inner_grad,psi_2_inner,psi_2_inner_grad,psi_4_inner,psi_4_inner_grad,r,rt

### Region 2
def region2(R,y,central_concentration,A,kappa):
    RHS=np.zeros(2,)
    
    RHS[0] = y[1]
    RHS[1] = -(2/R)*y[1] + kappa*np.power(A+(central_concentration/R),5/2)
    
    return RHS

def solve_region2(mu,x_0,central_concentration,A,kappa,final_radius,eval_r):
    inner_limit = np.power(x_0,2)
    poisson_solution = solve_ivp(fun = region2, t_span = (inner_limit,final_radius*x_0), 
                                 y0 =(-4*kappa*np.power(central_concentration,5/2)*np.power(inner_limit,-1/2),2*kappa*np.power(central_concentration,5/2)*np.power(inner_limit,-3/2)),
                                 method = 'RK45', dense_output = True, args = [central_concentration,A,kappa],rtol = 1e-8, atol = 1e-30, t_eval = eval_r)
    phi_1 =poisson_solution.y[0,:]
    phi_1_grad = poisson_solution.y[1,:]
    R = poisson_solution.t
    return phi_1,phi_1_grad,R


def region3(r,y,central_concentration,model):
    RHS=np.zeros(4,)
    RHS[0] = y[1]
    RHS[1] = -(2/r)*y[1]-9*(2/5)*np.nan_to_num(np.power(y[0],5/2),copy=False)/dimensionless_density(central_concentration,model)
    RHS[3] = -(2/r)*y[3]-(9*(np.nan_to_num(np.power(y[0],3/2),copy=False)*y[2])/dimensionless_density(central_concentration,model))
    RHS[2] = y[3]
    
    return RHS


def solve_region3(x_0,central_concentration,A,kappa,model,final_radius):
    inner = x_0*np.power(x_0,3/2) # inner boundary of integration in terms of rhat (corresponding to the first value in unscaled coordinates).

    poisson_solution = solve_ivp(fun = region3, t_span = (inner,np.power(x_0,3/2)*final_radius),y0 = (A+(kappa/6)*np.power(A,5/2)*np.power(inner,2),
                                                          (kappa/3)*np.power(A,5/2)*inner,central_concentration*np.power(inner,-1),-central_concentration*np.power(inner,-2)),
                                                          method = 'RK45',dense_output = True,args = [central_concentration,model],rtol=1e-8,atol=1e-30,events=zero_cross_no_mu)
    phi_region_3_0 = poisson_solution.y[0,:]
    phi_region_3_0_grad = poisson_solution.y[1,:]
    phi_region_3_1 = poisson_solution.y[2,:]
    phi_region_3_1_grad = poisson_solution.y[3,:]
    Rhat = poisson_solution.t
    
    if len(poisson_solution.t_events[0] == 1):    
        rt = np.power(x_0,-3/2)*poisson_solution.t_events[0][0]
    else:
        rt = final_radius
        
    return phi_region_3_0,phi_region_3_0_grad,phi_region_3_1,phi_region_3_1_grad,Rhat,rt