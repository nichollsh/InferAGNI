from __future__ import annotations

import emcee
import numpy as np
import multiprocessing as mp

from .grid import Grid


def log_prior(theta, gr: Grid):
    """Uniform prior: checks if parameters are within grid boundaries."""
    if np.all((theta >= gr._bounds[:, 0]) & (theta <= gr._bounds[:, 1])):
        return 0.0  # Log(1)
    return -np.inf  # Log(0) outside the grid


def log_likelihood(theta: list, obs:dict, gr: Grid):
    """
    theta: Current N-dimensional parameter set proposed by MCMC.
    obs: The measured values and errors of the observables.

    For now, assuming that only one observable is possible: Radius
    """

    chi_sq = 0.0

    for k in obs.keys():
        # Organise the observation values
        obs_val = obs[k][0]
        obs_err = obs[k][1]

        # Evaluate the model
        model_val = gr.interp_eval(theta, k)

        # Standard Gaussian Log-Likelihood
        # ln L = -0.5 * sum((data - model)^2 / error^2)
        diff = obs_val - model_val
        chi_sq += (diff/obs_err)**2

    return -0.5 * chi_sq


def log_probability(theta, obs:dict, gr: Grid):
    """Combined Posterior: Prior + Likelihood"""
    lp = log_prior(theta, gr)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, obs, gr)


def run(obs:dict, gr: Grid, n_walkers: int = 32, n_steps: int = 4000, n_procs: int = 1):
    """Executes the MCMC retrieval"""

    # Check number of CPUs
    n_cpus = mp.cpu_count()
    n_procs = max(1,n_procs)
    if n_procs >= n_cpus:
        print("Warning: decreased n_procs from {n_procs} to {n_cpus}")
        n_procs = n_cpus

    # Number of dimensions
    n_dim = len(gr.input_keys)

    # Need at least 2*ndim walkers for system to be well-conditioned
    if n_walkers < n_dim * 2:
        raise ValueError(f"Need at ≥{n_dim * 2} walkers for {n_dim} system; got {n_walkers}")
    print(f"Using {n_walkers} walkers and {n_procs} CPUs")

    # Check observables
    print("Observables:")
    for i,o in enumerate(obs.keys()):
        print(f"    {o}")
        # if o in gr.input_keys:
        #     raise ValueError(f"Observable {o} is also a parameter!")

    # Randomly sample initial positions from the uniform prior (within grid bounds)
    pos = np.random.uniform(
        low=gr._bounds[:, 0], high=gr._bounds[:, 1], size=(n_walkers, n_dim)
    )
    print("Initial guess:")
    for i, k in enumerate(gr.input_keys):
        print(f"    {k:18s}: range [{np.amin(pos[:, i]):10g}, {np.amax(pos[:, i]):10g}] across {n_walkers} walkers")

    # Run the sampler
    print(f"Running {n_steps} steps...")
    # with mp.get_context("spawn").Pool() as pool:
    sampler = emcee.EnsembleSampler(
                    n_walkers, n_dim,
                    log_probability, args=(obs, gr) ) #, pool=pool)
    sampler.run_mcmc(pos, n_steps, progress=True)

    print("Done")

    return sampler
