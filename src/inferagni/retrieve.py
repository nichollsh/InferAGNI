from __future__ import annotations

import multiprocessing as mp
from copy import deepcopy

import emcee
import numpy as np

from .grid import Grid
from .util import undimen, varprops

global gr_glo
gr_glo: Grid = None


def log_prior(theta: list):
    """Uniform prior: checks if parameters are within grid boundaries."""

    global gr_glo

    if np.all((theta >= gr_glo.bounds[:, 0]) & (theta <= gr_glo.bounds[:, 1])):
        return 0.0  # Log(1)
    return -np.inf  # Log(0) outside the grid


def log_likelihood(theta: list, obs: dict) -> float:
    """Logarithm of the likelihood function

    Parameters
    ------------
    - theta : list, Current N-dimensional parameter set proposed by MCMC.
    - obs : dict, The measured values and errors of the observables.
    """

    global gr_glo

    chi_sq = 0.0

    # Transform coordinates from log-scaled to physical
    theta_eval = deepcopy(theta)
    for i, k in enumerate(gr_glo.input_keys):
        if varprops[k].log:
            theta_eval[i] = 10 ** (theta_eval[i])

    for k in obs.keys():
        # Organise the observation values
        obs_val = obs[k][0]
        obs_err = obs[k][1]

        # Evaluate the model (returns scaled value)
        model_val = gr_glo.interp_eval(theta_eval, k)
        if varprops[k].log:
            model_val = np.log10(model_val)

        # Standard Gaussian Log-Likelihood
        # ln L = -0.5 * sum((data - model)^2 / error^2)
        diff = obs_val - model_val
        chi_sq += (diff / obs_err) ** 2

    return -0.5 * chi_sq


def log_probability(theta: list, obs: dict):
    """Logarithm of the combined posterior: Prior + Likelihood

    Parameters
    ------------
    - theta : list, Current N-dimensional parameter set proposed by MCMC.
    - obs : dict, The measured values and errors of the observables.
    """

    global gr_glo

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, obs)


def run(
    gr: Grid,
    obs: dict,
    n_steps: int = 4000,
    n_walkers: int | None = None,
    n_procs: int | None = None,
) -> emcee.EnsembleSampler:
    """Executes the MCMC retrieval"""

    # Copy grid into global scope. Required for multiprocessing to work.
    global gr_glo, obs_glo
    gr_glo = deepcopy(gr)
    obs_cpy = deepcopy(obs)

    # Check number of CPUs
    n_cpus = mp.cpu_count()
    if not n_procs:
        n_procs = n_cpus - 1
    else:
        n_procs = max(1, int(n_procs))
    if n_procs >= n_cpus:
        print("Warning: decreased n_procs from {n_procs} to {n_cpus}")
        n_procs = n_cpus

    # Initialising interpolators on original grid object
    print("Prepare interpolators")
    for k in obs_cpy.keys():
        gr.interp_init(vkey=k, reinit=False)
    print(" ")

    # Need at least 2*ndim walkers for system to be well-conditioned
    n_dim = len(gr_glo.input_keys)
    if not n_walkers:
        n_walkers = int(n_dim * 3)
    if n_walkers < n_dim * 2:
        raise ValueError(f"Need at ≥{n_dim * 2} walkers for {n_dim} system; got {n_walkers}")
    print(f"Using {n_walkers} walkers and {n_procs} CPUs")

    # Check observables (values,errors)
    print("Observables:")
    for i, k in enumerate(obs_cpy.keys()):
        print(f"    {k:18s}: {obs_cpy[k][0]:10g} ± {obs_cpy[k][1]:10g}")
        if varprops[k].log:
            obs_cpy[k] = np.log10(obs_cpy[k])

    # Convert bounds to logarithmic values where appropriate
    for i, k in enumerate(gr_glo.input_keys):
        if varprops[k].log:
            gr_glo.bounds[i] = np.log10(gr_glo.bounds[i])

    # Randomly sample initial positions from the uniform prior (within grid bounds)
    pos = np.random.uniform(
        low=gr_glo.bounds[:, 0], high=gr_glo.bounds[:, 1], size=(n_walkers, n_dim)
    )
    print("Initial guess:")
    for i, k in enumerate(gr_glo.input_keys):
        pos[:, i] = undimen(pos[:, i], k)
        print(
            f"    {k:18s}: range [{np.amin(pos[:, i]):10g}, {np.amax(pos[:, i]):10g}] w/ {n_walkers} walkers (log10={varprops[k].log})"
        )

    # Run the sampler
    print(f"Running {n_steps} steps...")
    with mp.Pool(processes=n_procs) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_probability, args=(obs_cpy,), pool=pool
        )
        sampler.run_mcmc(pos, n_steps, progress=True)

    print("Done")

    return sampler
