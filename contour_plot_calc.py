import sys
sys.path.append('src/silica_opt_insilico/')
from silica_opt_insilico import experiment, execute, data_processing
from saxs_data_processing import target_comparison
    
import numpy as np
import matplotlib.pyplot as plt

import pickle
import time

# physical experiment bounds

teos_min_vf = 0.005
ammonia_min_vf = 0.005
water_min_vf = 0.005

teos_max_vf = 0.1
ammonia_max_vf = 0.1
water_max_vf = 0.15

noise_frac = 0.5 # what fraction of experimental noise to use

target_r_nm = 40 # particle size target

# q range to consider up to and including spline fit step
q_min_subtract = 0.002
q_max_subtract = 0.035

# q range to interpolate spline fit on and perform distance metric calculation with
q_min_spl = 0.003
q_max_spl = 0.03
n_interpolate_gridpts = 1001 # number of grid points to interpolate q on.


target_r_angs = target_r_nm*10
target_pdi = 0.1
sld_silica = 8.575
sld_etoh = 9.611

q_grid = np.linspace(np.log10(q_min_spl), np.log10(q_max_spl), n_interpolate_gridpts)

q_grid_nonlog = 10**q_grid
target_I = target_comparison.target_intensities(q_grid_nonlog, target_r_angs, target_pdi, sld_silica, sld_etoh)
target_I = np.log10(target_I)


def contour_eval(teos, ammonia, water, return_scatter = False):

    #teos = sample[0]
    #ammonia = sample[1]
    #water = sample[2]

    noise_level = 0
    amplitude_weight = 0.1
    sample_point = (teos, ammonia, water)
    #print(sample_point)
    scattering, real_sample_point, diameter, pdi = experiment.run_experiment(sample_point, noise_level, q_grid_nonlog, experiment.sld_silica, experiment.sld_etoh)
    #print(diameter)
    # Process measurement
    ap_dist, ap_dist_report, I_scaled = data_processing.process_measurement(scattering, target_I, q_grid, amplitude_weight)

    print('ap dist ', ap_dist)
    if return_scatter:
        return ap_dist, scattering
    else:
        return ap_dist



## Build contour plot
n_grid = 50
# Define grid
teos = np.linspace(teos_min_vf, teos_max_vf, n_grid)
ammonia = np.linspace(ammonia_min_vf, ammonia_max_vf, n_grid)

water = 4.000e-02 # set 3rd fixed value to optima
Teos, Ammonia = np.meshgrid(teos, ammonia)
t1 = time.time()
# Compute contour values
Z = np.vectorize(contour_eval)(Teos, Ammonia, water)


with open('TEOS_ammonia_waterOptima_gridvals_80nm_ogfuncs_50.npy', 'wb') as f:
    np.save(f, Z)
print(f'finished grid in {time.time() - t1} s')


## Build contour plot

# Define grid
teos = np.linspace(teos_min_vf, teos_max_vf, n_grid)
water = np.linspace(water_min_vf, water_max_vf, n_grid)

ammonia = 2.021e-02
Teos, Water = np.meshgrid(teos, water)

# Compute contour values
Z = np.vectorize(contour_eval)(Teos, ammonia, Water)


with open('TEOS_water_ammoniaOptima_gridvals_80nm_ogfuncs_50.npy', 'wb') as f:
    np.save(f, Z)



