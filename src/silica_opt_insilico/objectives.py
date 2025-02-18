"""
Objective functions to replace the physical synthesis process(eg for particle diameter, PDI)
"""
import numpy as np



def particle_diameter(x_1, x_2, x_3):
    const = 57.36
    a = const
    b = 6*const
    c = 6*const

    diameter = a*np.log(x_1) + b*np.exp(x_2)+c*x_3

    
    return diameter


def particle_pdi_gaussian(x_1, x_2, x_3, mu_x=0.007, mu_y=0.025, mu_z=0.03, sigma_x1=0.08, sigma_x2=0.1, sigma_x3=5, amplitude=1, offset=1.1):
    """
    Computes the value of a 3D Gaussian function at (x, y, z).
    
    Parameters:
    x, y, z : float or np.array
        Input coordinates where the function is evaluated.
    mu_x, mu_y, mu_z : float
        Center (mean) of the Gaussian in each dimension.
    sigma_x, sigma_y, sigma_z : float
        Standard deviations in x, y, and z directions.
    amplitude : float
        Peak value of the Gaussian.
    offset : float
        Baseline offset added to the function.

    Returns:
    float or np.array
        The computed Gaussian function value.
    """
    exponent = -(((x_1 - mu_x)**2 / (2 * sigma_x1**2)) + 
                 ((x_2 - mu_y)**2 / (2 * sigma_x2**2)) + 
                 ((x_3 - mu_z)**2 / (2 * sigma_x3**2)))
    return -amplitude * np.exp(exponent) + offset