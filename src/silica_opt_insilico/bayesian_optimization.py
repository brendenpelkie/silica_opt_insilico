import torch
import numpy as np
import uuid

from typing import Optional

from botorch.models.transforms.input import Normalize
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from sklearn.manifold import TSNE

from gpytorch.kernels import MaternKernel
from gpytorch.priors import GammaPrior

import time
import warnings

from botorch import fit_gpytorch_mll
from botorch.acquisition import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qUpperConfidenceBound
)
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler

import os

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
dtype = torch.double
SMOKE_TEST = os.environ.get("SMOKE_TEST")


def initialize_model(train_x, train_y, state_dict=None, nu = 5/2, ard_num_dims = None):
    # define models for objective and constraint
    kernel = MaternKernel(nu = nu, ard_num_dims = ard_num_dims)
    model_obj = SingleTaskGP(
        train_x,
        train_y,
        #train_Yvar=assumed_noise*torch.ones_like(train_y),
        input_transform=Normalize(d=train_x.shape[-1]),
        covar_module=kernel
    ).to(train_x)

    # combine into a multi-output GP model
    mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
    # load state dict if it is passed
    if state_dict is not None:
        model_obj.load_state_dict(state_dict)
    return mll, model_obj

def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]

def bayesian_optimize(x_train, y_train, batch_size, num_restarts, raw_samples, nu, ard_num_dims, bounds_torch_opt, bounds_torch_norm, acqf = 'qLogNEI', return_model = False):
    ## init model
    mll_nei, model_nei = initialize_model(x_train, y_train, ard_num_dims = ard_num_dims)    
    fit_mll = fit_gpytorch_mll(mll_nei)

    ## run acq opt
    # define the qEI and qNEI acquisition modules using a QMC sampler
    t_acqf = time.time()
    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([raw_samples]))

    objective = GenericMCObjective(objective=obj_callable)
    
    # for best_f, we use the best observed noisy values as an approximation
    if acqf == 'qLogNEI':
        acqfunc = qLogNoisyExpectedImprovement(
            model=model_nei,
            X_baseline=x_train,
            sampler=qmc_sampler,
            objective=objective,
            prune_baseline=True,
        )

    elif acqf == 'qLogEI':
        acqfunc = qLogExpectedImprovement(
            model = model_nei,
            best_f = y_train.max()[0],
            X_baseline = x_train,
            sampler = qmc_sampler,
            objective = objective,
            prune_baseline = True
        )
    
    # optimize for new candidates
    candidates, _ = optimize_acqf(
        acq_function=acqfunc,
        bounds=bounds_torch_opt,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # used for intialization heuristic
        #options={"batch_limit": 5, "maxiter": 200},
    )

    print(f'Optimized acqf in {time.time() - t_acqf} s')
    x_fractions = unnormalize(candidates, bounds_torch_norm)

    if return_model:
        return x_fractions, model_nei
    else:
        return x_fractions

def bo_preprocess(current_dataset, bounds_torch_norm):
    """
    Get current data into good form for BO
    """
    compositions = []
    apdist_vals = []
    for uiud_val, sample in current_dataset.items():
        comp = [sample['teos_vol_frac'], sample['ammonia_vol_frac'], sample['water_vol_frac']]
        compositions.append(comp)
        apdist_vals.append(sample['distance'])

    y_data = - torch.tensor(np.array(apdist_vals)).reshape(-1,1)
    x_data_torch = torch.tensor(np.array(compositions))


    x_data_norm = normalize(x_data_torch, bounds_torch_norm)
    y_data_norm = normalize(y_data, (y_data.min(), y_data.max())).reshape(-1,1)

    return x_data_norm, y_data_norm

def bo_postprocess(candidates):
    batch = {}
    for i in range(len(candidates)):
        sample = {}
        uuid_val = str(uuid.uuid4())

        sample['teos_vol_frac'] = candidates[i,0]
        sample['ammonia_vol_frac'] = candidates[i,1]
        sample['water_vol_frac'] = candidates[i,2]

        batch[uuid_val] = sample

    return batch
