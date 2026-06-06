import numpy as np

# Scaled constants for numerical stability
M_TOT = 1.0 
A = 1.0
G = 1.0

# Plummer mass formula
def analytical_mass(r, M_tot=M_TOT, a=A):
    return M_tot * (r**3) / (r**2 + a**2)**(1.5)

# Plummer density formula
def density(r, M_tot=M_TOT, a=A):
    return (3 * M_tot / (4 * np.pi * a**3)) * (1 + r**2 / a**2)**(-2.5)

def integrand(r, M_tot=M_TOT, a=A):
    return density(r, M_tot, a) * 4 * np.pi * r**2

# Checks if AI-predicted gravity matches actual mass density
def poisson_residual(r, phi, dphi_dr, d2phi_dr2):
    eps = 1e-4 
    r_safe = r + eps
    
    rho = density(r)    
    laplacian = d2phi_dr2 + (2.0 / r_safe) * dphi_dr
    
    return laplacian - 4 * np.pi * G * rho