from scipy.interpolate import interp1d
from scipy import integrate
import numpy as np
import ODEs

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

#def zero_crossing(x,y,final_radius):
#    
#    if(y[-1]<=0):
#        
#        f = interp1d(y,x)
#        zero_cross = f(0)
#        
#        return zero_cross
#    
#    if(y[-1]>0):
#        
#        print('No axis crossing, setting the final radius')
#        
#        return final_radius
    
def half_mass(r,mass_profile):
    
    f = interp1d(mass_profile/mass_profile[-1],r)
    
    return f(0.5)
    

def A_double_integral(central_concentration,r,model):
    integrand = 9*np.power(r,2)*ODEs.dimensionless_density(central_concentration/r,model)/ODEs.dimensionless_density(central_concentration,model)

    single_integral = integrate.cumtrapz(integrand,r,initial = 0)*np.power(r,-2)
    
    double_integral = np.trapz(y=single_integral,x = r)
    
    return double_integral

def project_quantity(r,f,rt,x_0):
    npoints_in_projection = 10000
    npoints_integral_approximation = 10000
    
    f_interp = interp1d(r,f,fill_value = 0,bounds_error = False)
    
    d = np.linspace(x_0,rt,npoints_in_projection)
    projected_f = np.zeros(len(d))
    
    for i in range(len(projected_f)):
        
        x = np.linspace(0,np.sqrt(np.square(d[i])+np.square(rt)),npoints_integral_approximation)
        
        r_x = np.sqrt(np.square(d[i])+np.square(x))
        
        projected_f[i] = 2*np.trapz(y = f_interp(r_x), x = x)

    return d,projected_f

def projected_density_gradient(r,psi,psi_grad,rho,rt,x_0):

    npoints_in_projection = 10000
    npoints_integral_approximation = 10000
    
    rho_grad = np.nan_to_num(psi_grad * (rho + np.power(psi,3/2)))    
    density_gradient = interp1d(r,rho_grad,fill_value = 0,bounds_error = False)

    d = np.linspace(x_0,rt,npoints_in_projection)
    projected_density_gradient = np.zeros(len(d))
    
    for i in range(len(projected_density_gradient)):
        
        x = np.linspace(0,np.sqrt(np.square(d[i])+np.square(rt)),npoints_integral_approximation)
        
        r_x = np.sqrt(np.square(d[i])+np.square(x))
        
        integrand = density_gradient(r_x)/r_x
        
        projected_density_gradient[i] = 2*d[i]*np.trapz(y = integrand, x = x)

    return projected_density_gradient

def projected_density_gradient2(r,psi,psi_grad,rho,rt,x_0,model):

    npoints_in_projection = 10000
    npoints_integral_approximation = 10000
    
    rho_grad = np.nan_to_num(psi_grad * (rho + np.power(psi,3/2)))
    psi_grad2 = (-(2/r)*psi_grad-(9/ODEs.dimensionless_density(psi[0],model))*rho)
    rho_grad2 = np.nan_to_num(psi_grad2*(rho + np.power(psi,3/2)) + np.power(psi_grad,2)*(rho + np.power(psi,3/2) + (3/2)*np.power(psi,1/2)))
    
    density_gradient = interp1d(r,rho_grad,fill_value = 0,bounds_error = False)
    density_gradient2 = interp1d(r,rho_grad2,fill_value = 0,bounds_error = False)

    d = np.linspace(x_0,rt,npoints_in_projection)
    projected_density_gradient2 = np.zeros(len(d))
    
    for i in range(len(projected_density_gradient2)):
        
        x = np.linspace(0,np.sqrt(np.square(d[i])+np.square(rt)),npoints_integral_approximation)
        
        r_x = np.sqrt(np.square(d[i])+np.square(x))
        
        integrand = density_gradient(r_x)/r_x - (d[i]**2) * density_gradient(r_x)/np.power(r_x,3) + (d[i]**2) * density_gradient2(r_x)/np.power(r_x,2)
        
        projected_density_gradient2[i] = 2*np.trapz(y = integrand, x = x)

    return projected_density_gradient2