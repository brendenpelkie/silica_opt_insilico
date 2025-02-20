import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import experiment
import data_processing

import pd_utils

def composition_distance(measured_point, true_optima):
    return np.sqrt(np.sum([(measured_point[i] - true_optima[i])**2 for i in range(len(measured_point))]))

def process_distances(trial_name, params, true_min_composition, budget = 100, convergence_threshold = 0.05, fp = './', n_replicates = 3):
    """
    First pass through data
    """

    batch_size = params['batch_size']
    amplitude_weight = params['amplitude_weight']
    m_samples = params['m_samples']
    lower_bounds = params['lower_bounds']
    upper_bounds = params['upper_bounds']
    noise_level = params['noise_level']

        
    if batch_size == 0:
        n_batches = 0
    else:
        n_batches = int(np.ceil(budget/batch_size))

    data_complete = {}
    data_campaigns = []
    best_distances_list = []
    best_uuids_list = []
    converge_iterations = []
    best_composition_dist = []
    for i in range(n_replicates):
        with open(f'{fp}{trial_name}_replicate_{i}.pkl', 'rb') as f:
            data = pickle.load(f)
        
        name_bounds = {}
        name_bounds['random'] = (0, 2**m_samples)
        for i in range(n_batches):
            name_bounds[f'Batch {i+1}'] = ((i)*batch_size + 2**m_samples, (i+1)*batch_size + 2**m_samples)

        best_distances_ap = []
        best_uuid = []
        converge_its = None
        
        for i, (uuid_val, sample) in enumerate(data.items()):
            data_complete[uuid_val] = sample
            dist = sample['ap_distance_reporting']
            if len(best_distances_ap) == 0:
                best_distances_ap.append(dist)
                best_distances_ap.append(dist)
                best_uuid.append(uuid_val)
            else:
                best_distances_ap.append(min(best_distances_ap[-1], dist))
        
            
            if dist < best_distances_ap[-2]:
                #print('new min found')
                best_uuid.append(uuid_val)

            if dist < convergence_threshold and converge_its is None:
                converge_its = i
                converge_iterations.append(i)
        if converge_its is None:
            converge_iterations.append('Not converged')
            
        best_distances_list.append(best_distances_ap)
        best_uuids_list.append(best_uuid)
        data_campaigns.append(data)

        # get best composition distance
        best_sample = data[best_uuid[-1]]
        best_sample_comp = [best_sample['teos_vol_frac'], best_sample['ammonia_vol_frac'], best_sample['water_vol_frac']]
        comp_dist = composition_distance(best_sample_comp, true_min_composition)
        best_composition_dist.append(comp_dist)

    return data_complete, data_campaigns, best_distances_list, best_uuids_list, converge_iterations, best_composition_dist, name_bounds
    

def convergence_plot(data_complete, best_distances_list, best_uuids_list, name_bounds, trial_name):
                         
    fig, ax = plt.subplots()

    for name, bounds in name_bounds.items():
        ax.fill_between(bounds, 0, max([max(dists) for dists in best_distances_list]), alpha = 0.2)
    #    ax.text(np.mean(bounds), 0.1, name, rotation = 'vertical')

    for best_distances_ap in best_distances_list:
        ax.plot(best_distances_ap)
    
    
    ax.set_xlabel('Sample number')
    ax.set_ylabel('amplitude-phase distance')
    ax.set_title(trial_name)
    
    plt.tight_layout()
    #plt.savefig('Campaign_convergence_plot.png', dpi = 300)

    return fig


def best_scatterer_plots(data_complete, best_uuids_list, q_grid_nonlog, target_I, trial_name):

    fig, ax = plt.subplots(1,len(best_uuids_list), figsize = (12,12/len(best_uuids_list)))

    # handle 1 replicate case
    if not isinstance(ax, np.ndarray):
        ax = np.array(ax).reshape(-1)

    for i, best_uuid in enumerate(best_uuids_list):

        I = data_complete[best_uuid[-1]]['I_scaled']
    
        ax[i].loglog(q_grid_nonlog, I)
        ax[i].loglog(q_grid_nonlog, target_I, ls ='--')

    fig.suptitle(trial_name)


def phase_diagram(data, lower_bounds, upper_bounds, q_grid):
    """
    Phase plot, for 1 campaign
    """

    x_name = 'TEOS'
    y_name = 'water'
    y_2_name = 'ammonia'

    x_min = lower_bounds[0]
    x_max = upper_bounds[0]
    y_min = lower_bounds[2]
    y_max = upper_bounds[2]

    y_2_min = lower_bounds[1]
    y_2_max = upper_bounds[1]

    bounds = np.array([[x_min, y_min], [x_max, y_max]])
    bounds_2 = np.array([[x_min, y_2_min], [x_max, y_2_max]])

    x_key = 'teos_vol_frac'
    y_key = 'water_vol_frac'
    y_2_key = 'ammonia_vol_frac'


    fig, ax = plt.subplots(1,2, figsize = (12, 6))

    c_1 = []
    s = []
    c_2 = []

    colors_1 = []
    colors_2 = []
    for uuid_val, sample in data.items():

        x_val = sample[x_key]
        y_val = sample[y_key]
        y_2_val = sample[y_2_key]

        
        I = np.log10(sample['scattering_I'])

        c_1.append([x_val, y_val])
        s.append(I)
        c_2.append([x_val, y_2_val])
        colors_1.append(get_colormap_color(y_2_val, y_2_min, y_2_max))
        colors_2.append(get_colormap_color(y_val, y_min, y_max))

            

    s = np.array(s)

    pd_utils.plot_phasemap(bounds, ax[0], c_1, s, colors = colors_1)
    pd_utils.plot_phasemap(bounds_2, ax[1], c_2, s, colors = colors_2)


    ax[0].set_xlabel('TEOS volume fraction')
    ax[0].set_ylabel('Water volume fraction')
    ax[1].set_xlabel('TEOS volume fraction')
    ax[1].set_ylabel('Ammonia volume fraction')

    cmap_name = 'viridis'
    norm = mcolors.Normalize(vmin=y_2_min, vmax=y_2_max)
    norm2 = mcolors.Normalize(vmin=y_min, vmax = y_max)
    cmap = cm.get_cmap(cmap_name)
    cmap2 = cm.get_cmap(cmap_name)

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[0])
    cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm2, cmap = cmap2), ax = ax[1])

    cbar.set_label(y_2_key)
    cbar2.set_label(y_key)


    fig.suptitle('Round 2 Optimization Batch 3 - APdist')
    #plt.savefig('Phaseplot_apdist_batch3_80nm.png', dpi = 300)
    return fig

def get_colormap_color(value, vmin, vmax, cmap_name='viridis'):
    """
    Maps a scalar value to an RGB color using a specified Matplotlib colormap.
    
    Parameters:
        value (float): The scalar value to be mapped to a color.
        vmin (float): The minimum bound of the scalar range.
        vmax (float): The maximum bound of the scalar range.
        cmap_name (str): The name of the Matplotlib colormap to use (default: 'viridis').
    
    Returns:
        tuple: An (R, G, B) color tuple, where values are in the range [0, 1].
    """
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    return cmap(norm(value))[:3] 

    


    

