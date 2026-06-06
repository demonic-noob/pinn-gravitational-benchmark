import numpy as np
from scipy.integrate import quad

def analytical_mass(r, M_tot, a):
    return M_tot * (r**3) / (r**2 + a**2)**(1.5)

def density(r, M_tot, a):
    return (3 * M_tot / (4 * np.pi * a**3)) * (1 + r**2 / a**2)**(-2.5)

def integrand(r, M_tot, a):
    return density(r, M_tot, a) * 4 * np.pi * r**2

M_tot, a = 1.0, 1.0
r_test = 2.0

analytical = analytical_mass(r_test, M_tot, a)
numerical, error = quad(integrand, 0, r_test, args=(M_tot, a))

print(f"Analytical Result: {analytical}")
print(f"Numerical Result:  {numerical}")

