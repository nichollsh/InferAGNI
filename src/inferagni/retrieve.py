from __future__ import annotations

import numpy as np
import emcee
from multiprocessing import Pool, cpu_count

from .grid import Grid

def log_prior(theta, gr:Grid):
    """Uniform prior: checks if parameters are within grid boundaries."""
    if np.all((theta >= gr._bounds[:, 0]) & (theta <= gr._bounds[:, 1])):
        return 0.0  # Log(1)
    return -np.inf  # Log(0) outside the grid

def log_likelihood(theta:list, obs_data:dict, obs_err:list, gr:Grid):
    """
    theta: Current N-dimensional parameter set proposed by MCMC.
    obs_data: The measured values of the observables.
    obs_err: The uncertainties on those measurements.

    For now, assuming that only one observable is possible: Radius
    """

    # Get the interpolated model values for these N parameters
    model_vals = np.array(gr.interp_eval(theta))

    # Standard Gaussian Log-Likelihood
    # ln L = -0.5 * sum((data - model)^2 / error^2)
    diff = obs_data - model_vals
    chi_sq = np.sum((diff / obs_err)**2)

    return -0.5 * chi_sq

def log_probability(theta, obs_data, obs_err, gr:Grid):
    """Combined Posterior: Prior + Likelihood"""
    lp = log_prior(theta, gr)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, obs_data, obs_err, gr)

def run(obs_data, obs_err, gr:Grid, n_walkers:int=32, n_steps:int=4000, n_procs:int=5):
    """Executes the MCMC retrieval"""

    # Check number of CPUs
    # n_cpus = cpu_count()
    # n_procs = max(1,n_procs)
    # if n_procs >= n_cpus:
    #     print("Warning: decreased n_procs from {n_procs} to {n_cpus}")
    #     n_procs = n_cpus


    # Number of dimensions
    n_dim = len(gr.input_keys)

    # Need at least 2*ND walkers to make system well-conditioned
    if n_walkers < n_dim*2:
        raise ValueError(f"Need at ≥{n_dim*2} walkers for {n_dim} system; got {n_walkers}.")

    # Convert type
    obs_data = np.array(obs_data, copy=True, dtype=gr._interp_dtype)
    obs_err  = np.array(obs_err,  copy=True, dtype=gr._interp_dtype)

    # Randomly sample initial positions from the uniform prior (within grid bounds)
    pos = np.random.uniform(
        low=gr._bounds[:, 0],
        high=gr._bounds[:, 1],
        size=(n_walkers, n_dim)
    )
    print(f'Initial guess:')
    for i,k in enumerate(gr.input_keys):
        print(f'  [{i+1}]  {k}: {pos[:,i]}')

    # Run the sampler
    # with Pool(n_procs) as pool:
    sampler = emcee.EnsembleSampler(n_walkers, n_dim,
                                        log_probability,
                                        args=(obs_data, obs_err, gr) )#, pool=pool)
    sampler.run_mcmc(pos, n_steps, progress=True)


    print(f"Running {n_steps} steps...")
    sampler.run_mcmc(pos, n_steps, progress=True)

    return sampler
