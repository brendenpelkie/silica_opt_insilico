from saxs_data_processing import io, manipulate, target_comparison, subtract, sasview_fitting
import numpy as np


def process_measurement(scattering, target_I, q_grid, amplitude_weight, distance_metric = 'apdist', n_avg = 100, optim = "DP", grid_dim = 10):
    """
    Do scaling and amplitdue_phase calculation on scattering

    Params:
    - scattering - 'measured' scattering
    - target_I - target scattering
    - q_grid - q on log scale
    - amplitude weight - how much to weight amplitude in ap sum
    - n_avg - scaling average
    - optim, grid_dim - ap params

    Returns:
    - ap_sum
    - ap_sum_report - ap sum with 0.1 weight
    - I_scaled - scaled I
    """
    # processing works in log(I) space 

    
    scattering = np.log10(scattering)
    I_scaled = manipulate.scale_intensity_highqavg(scattering, target_I, n_avg = n_avg)
    amplitude, phase = target_comparison.ap_distance(q_grid, I_scaled, target_I, optim = optim, grid_dim = grid_dim)

    if distance_metric == 'rmse':
        distance = target_comparison.rmse_distance(I_scaled, target_I, log = False) # alsready in log10 space
    elif distance_metric == 'apdist':
        distance =  amplitude_weight*amplitude + (1-amplitude_weight)*phase
    else:
        raise AssertionError(f'invalid distance name {distance_metric}')
    
    ap_sum_report = 0.1 * amplitude + 0.9*phase

    return distance, ap_sum_report, I_scaled

def process_measurement_DLS(diameter, PDI, pdi_weight, target_d, target_pdi):
    """
    """
    diam_d = np.abs(target_d - diameter)/target_d
    pdi_d = np.abs(target_pdi - PDI)/target_pdi



    distance = pdi_d*pdi_weight + diam_d*(1-pdi_weight)
    return distance
