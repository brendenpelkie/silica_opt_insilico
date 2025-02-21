"""
Helper funcs to run simulated campaigns
"""
import experiment
import data_processing
import init_sampling
import bayesian_optimization as bo

import numpy as np

from joblib import Parallel, delayed

import torch
import pickle

def process_sample(uuid_val, sample, target_I, q_grid, amplitude_weight, noise_level, characterization = 'SAXS', distance_metric = 'apdist', pdi_weight = 0.5, target_d = None, target_pdi = None):
    """ Runs the experiment for a single sample and returns the results. """
    sample_point = [sample['teos_vol_frac'], sample['ammonia_vol_frac'], sample['water_vol_frac']]
    q_grid_nonlog = 10**q_grid
    scattering, real_sample_point, diameter, pdi = experiment.run_experiment(sample_point, noise_level, q_grid_nonlog, experiment.sld_silica, experiment.sld_etoh )
    
    #print('process sample distance metric: ', distance_metric)
    # Process measurement
    dist, ap_dist_report, I_scaled = data_processing.process_measurement(scattering, target_I, q_grid, amplitude_weight, distance_metric = distance_metric)

    if characterization == 'DLS':
        dist = data_processing.process_measurement_DLS(diameter, pdi, pdi_weight, target_d, target_pdi) 
    return uuid_val, {
        'scattering_I': scattering,
        'real_sampled_point': real_sample_point,
        'diameter': diameter,
        'pdi': pdi,
        'distance': dist,
        'ap_distance_reporting': ap_dist_report,
        'I_scaled': I_scaled
    }

def batch_experiment(batch, target_I, q_grid, amplitude_weight, noise_level, characterization, pdi_weight, distance_metric, target_d, target_pdi, n_jobs=-1):
    """ Runs experiments in parallel using joblib. """
    print('batch distance metric: ', distance_metric)
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_sample)(uuid_val, sample, target_I, q_grid, amplitude_weight, noise_level, characterization, distance_metric, pdi_weight, target_d, target_pdi) for uuid_val, sample in batch.items()
    )

    # Update batch with results
    for uuid_val, result in results:
        batch[uuid_val].update(result)

def run_grouped_trials(target_I, q_grid, batch_size, amplitude_weight, m_samples, lower_bounds, upper_bounds, trial_name, noise_level, budget, target_d, target_pdi characterization = 'SAXS', distance_metric = 'apdist', pdi_weight = 0.5, n_replicates = 3, sobol_seed = 42, NUM_RESTARTS = 50, RAW_SAMPLES = 512, nu = 5/2, ard_num_dims = 3):
    """Run a batch of replicates of a trial"""
    
    if batch_size == 0:
        n_batches = 0
    else:
        n_batches = int(np.ceil(budget/batch_size))
    
    bounds_torch_norm = torch.tensor([(lower_bounds[0], upper_bounds[0]), (lower_bounds[1], upper_bounds[1]), (lower_bounds[2], upper_bounds[2])]).transpose(-1, -2)
    bounds_torch_opt = torch.tensor([[0, 0, 0], [1.0, 1.0, 1.0]], dtype = torch.float32)

    print(f'Running optimization for trial {trial_name}')
    for rep_num in range(n_replicates):
        print(f'### Replicate {rep_num} ###')

        print('starting initial samples')
        initial_samples = init_sampling.sobol_sample(m_samples, sobol_seed, lower_bounds, upper_bounds)
        
        # 2. 'measure' sobol samples
        
        batch_experiment(initial_samples, target_I, q_grid, amplitude_weight, noise_level, characterization, pdi_weight, distance_metric, target_d, target_pdi)
        
        # 3. start experiment loop:
        data = initial_samples
        for i in range(n_batches):
            print(f'starting batch {i+1}')
            # 3a. Prepare and run BO
            x_train, y_train = bo.bo_preprocess(data, bounds_torch_norm)
            candidates = bo.bayesian_optimize(x_train, y_train, batch_size, NUM_RESTARTS, RAW_SAMPLES, nu, ard_num_dims, bounds_torch_opt, bounds_torch_norm)
            candidates = bo.bo_postprocess(candidates)
        
            # run experiment
            batch_experiment(candidates, target_I, q_grid, amplitude_weight, noise_level, characterization, pdi_weight, distance_metric, target_d, target_pdi)
        
            # update running data tally
            for uuid_val, sample in candidates.items():
                data[uuid_val] = sample

        
        with open(f'{trial_name}_replicate_{rep_num}.pkl', 'wb') as f:
            pickle.dump(data, f)
