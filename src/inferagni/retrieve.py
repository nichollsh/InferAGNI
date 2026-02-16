from __future__ import annotations

import emcee
import numpy as np
import multiprocessing as mp
from copy import deepcopy

from .grid import Grid

global gr_glo
gr_glo:Grid = None

def log_prior(theta:list):
    """Uniform prior: checks if parameters are within grid boundaries."""

    global gr_glo

    if np.all((theta >= gr_glo._bounds[:, 0]) & (theta <= gr_glo._bounds[:, 1])):
        return 0.0  # Log(1)
    return -np.inf  # Log(0) outside the grid


def log_likelihood(theta: list, obs:dict):
    """
    theta: Current N-dimensional parameter set proposed by MCMC.
    obs: The measured values and errors of the observables.

    For now, assuming that only one observable is possible: Radius
    """

    global gr_glo

    chi_sq = 0.0

    for k in obs.keys():
        # Organise the observation values
        obs_val = obs[k][0]
        obs_err = obs[k][1]

        # Evaluate the model
        model_val = gr_glo.interp_eval(theta, k)

        # Standard Gaussian Log-Likelihood
        # ln L = -0.5 * sum((data - model)^2 / error^2)
        diff = obs_val - model_val
        chi_sq += (diff/obs_err)**2

    return -0.5 * chi_sq


def log_probability(theta:list, obs:dict):
    """Combined Posterior: Prior + Likelihood"""

    global gr_glo

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, obs)


def run(gr:Grid, obs:dict, n_walkers: int = 32, n_steps: int = 4000, n_procs: int = 1) -> emcee.EnsembleSampler:
    """Executes the MCMC retrieval"""

    # Copy grid into global scope. Required for multiprocessing to work.
    global gr_glo
    gr_glo = deepcopy(gr)

    # Check number of CPUs
    n_cpus = mp.cpu_count()
    n_procs = max(1,n_procs)
    if n_procs >= n_cpus:
        print("Warning: decreased n_procs from {n_procs} to {n_cpus}")
        n_procs = n_cpus

    # Number of dimensions
    n_dim = len(gr_glo.input_keys)

    # Need at least 2*ndim walkers for system to be well-conditioned
    if n_walkers < n_dim * 2:
        raise ValueError(f"Need at ≥{n_dim * 2} walkers for {n_dim} system; got {n_walkers}")
    print(f"Using {n_walkers} walkers and {n_procs} CPUs")

    # Check observables
    print("Observables:")
    for i,o in enumerate(obs.keys()):
        print(f"    {o}")
        # if o in gr_glo.input_keys:
        #     raise ValueError(f"Observable {o} is also a parameter!")

    # Randomly sample initial positions from the uniform prior (within grid bounds)
    pos = np.random.uniform(
        low=gr_glo._bounds[:, 0], high=gr_glo._bounds[:, 1], size=(n_walkers, n_dim)
    )
    print("Initial guess:")
    for i, k in enumerate(gr_glo.input_keys):
        print(f"    {k:18s}: range [{np.amin(pos[:, i]):10g}, {np.amax(pos[:, i]):10g}] across {n_walkers} walkers")

    # Run the sampler
    print(f"Running {n_steps} steps...")
    with mp.Pool(processes=n_procs) as pool:
        sampler = emcee.EnsembleSampler(
                        n_walkers, n_dim,
                        log_probability, args=(obs,) , pool=pool)
        sampler.run_mcmc(pos, n_steps, progress=True)

    print("Done")

    return sampler
