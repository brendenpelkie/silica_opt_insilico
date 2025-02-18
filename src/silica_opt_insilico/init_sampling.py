from scipy.stats.qmc import Sobol
from scipy.stats import qmc
import uuid

def sobol_sample(m_samples, seed, lower_bounds, upper_bounds):
    """
    Generate sobol sample and set up data structure
    """
    sampler = Sobol(d=3, seed = seed)
    sampled_points = sampler.random_base2(m_samples)

    sampled_volume_fractions = qmc.scale(sampled_points, lower_bounds, upper_bounds)


    sobol_samples = {}
    for i in range(len(sampled_volume_fractions)):
        uuid_val = str(uuid.uuid4())
        sample = {}
        sample['teos_vol_frac'] = sampled_volume_fractions[i,0]
        sample['ammonia_vol_frac'] = sampled_volume_fractions[i,1]
        sample['water_vol_frac'] = sampled_volume_fractions[i,2]

        sobol_samples[uuid_val] = sample

    return sobol_samples
        