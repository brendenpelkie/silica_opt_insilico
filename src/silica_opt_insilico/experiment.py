"""
Simulated experiment execution
- noisy synthesis
- scattering "measurement"
"""
import numpy as np
from silica_opt_insilico import objectives
from saxs_data_processing import io, manipulate, target_comparison, subtract, sasview_fitting

sld_silica = 8.575
sld_etoh = 9.611

def syringe_precision(vol_frac, noise_frac):
    """
    Mimics noise of our digital syringe tools. This is hard coded in here

    vol_frac - volume fraction to be noised
    noise-frac - what fraction of full syringe uncertainty to use (0 is noise free, 1 is full)
    """
    if vol_frac <= 0.029:
        precision = noise_frac* (-3551*vol_frac + 110)/100
    else:
        precision = noise_frac*0.0073

    return precision

def noisy_dispense(vol_frac, noise_frac):
    """
    calculate the 'noisy' amount to 'dispense' when calculating diameter, pdi

    vol frac - volume fraction 'called for' by planner
    noise_frac - noise parameter, [0,1], 1 is syringe noise, 0 is noise free

    will not return less than 0.003 to prevent negative diameters
    """

    vol_std = syringe_precision(vol_frac, noise_frac)*vol_frac
    sampled_vol_frac = np.random.normal(loc = vol_frac, scale=vol_std)
    if sampled_vol_frac < 0.003:
        sampled_vol_frac = 0.003
    return sampled_vol_frac


# something provides volume fractions to use
def run_experiment(sample_point, noise_frac, q_grid_nonlog, sld_silica, sld_etoh):
    """
    Takes an ideal sample point, returns scattering and real point sampled 

    Params:
    - sample point (teos, ammonia, water)
    - noise frac 0 is perfect, 1 is syringe precision
    - q_grid_nonlog q grid to scatter on, not log scale
    - sld silic
    - sld etoh

    Returns:
    - scattering vector
    - Actual point sampled at 
    - diameter
    - pdi
    """
    noisy_sample_point = [noisy_dispense(sample_point[0], noise_frac), noisy_dispense(sample_point[1], noise_frac), noisy_dispense(sample_point[2], noise_frac)]
    
    diameter = objectives.particle_diameter(noisy_sample_point[0], noisy_sample_point[1], noisy_sample_point[2])
    pdi = objectives.particle_pdi_gaussian(noisy_sample_point[0], noisy_sample_point[1], noisy_sample_point[2])
    
    target_r_angs = diameter*10/2
    
    scattering = target_comparison.target_intensities(q_grid_nonlog, target_r_angs, pdi, sld_silica, sld_etoh)

    return scattering, noisy_sample_point, diameter, pdi
