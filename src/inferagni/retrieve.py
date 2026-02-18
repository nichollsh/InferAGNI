from __future__ import annotations

import warnings  # noqa

warnings.simplefilter(action="ignore", category=FutureWarning)  # noqa

import contextlib
import multiprocessing as mp
from copy import deepcopy

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

from .grid import Grid
from .util import varprops

global gr_glo, obs_glo
gr_glo: Grid = None
obs_glo: dict = None
name_glo: str = "Unnamed planet"


def log_prior(theta: list):
    """Uniform prior: checks if parameters are within grid boundaries."""

    global gr_glo

    if np.all((theta >= gr_glo.bounds[:, 0]) & (theta <= gr_glo.bounds[:, 1])):
        return 0.0  # Log(1)
    return -np.inf  # Log(0) outside the grid


def log_likelihood(theta: list) -> float:
    """Logarithm of the likelihood function

    See paper: https://doi.org/10.1093/rasti/rzaf052

    Parameters
    ------------
    - theta : list, Current N-dimensional parameter set proposed by MCMC.
    """

    global gr_glo, obs_glo

    # Transform coordinates from log-scaled to physical
    theta_eval = deepcopy(theta)
    for i, k in enumerate(gr_glo.input_keys):
        if varprops[k].log:
            theta_eval[i] = 10 ** (theta_eval[i])

    ln_L = 0.0

    # Iterate through observables to handle potential asymmetry
    for k, (obs_val, obs_err) in obs_glo.items():
        # Evaluate the model (returns scaled value)
        model_val = gr_glo.interp_eval(theta_eval, k)
        if varprops[k].log:
            model_val = np.log10(model_val)

        # Obs_err can be a scalar, or a size-2 tuple
        if np.isscalar(obs_err) or len(obs_err) == 1:
            # Standard Symmetric Case
            sig = obs_err
            norm = -0.5 * np.log(2 * np.pi * sig**2)
            ln_L += norm - 0.5 * ((model_val - obs_val) / sig) ** 2
        else:
            # Asymmetric Case: err = (sigma_plus, sigma_minus)
            sig_hi, sig_lo = obs_err
            sig = sig_hi if model_val > obs_val else sig_lo

            # Normalization constant for Split-Normal
            norm = np.log(np.sqrt(2 / np.pi) / (sig_hi + sig_lo))
            ln_L += norm - 0.5 * ((model_val - obs_val) / sig) ** 2

    return ln_L


def log_probability(theta: list):
    """Logarithm of the combined posterior: Prior + Likelihood

    Parameters
    ------------
    - theta : list, Current N-dimensional parameter set proposed by MCMC.
    """

    global gr_glo, obs_glo

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def run(
    gr: Grid,
    obs: dict,
    n_steps: int | None = None,
    n_walkers: int | None = None,
    n_procs: int | None = None,
    n_burn: int | None = None,
    thin: int | None = None,
    extra_keys: list = [],
) -> tuple:
    """Executes the MCMC retrieval"""

    global obs_glo, gr_glo, name_glo

    # Extra keys should not include the parameters, but should include the observables
    try:
        name_glo = obs["_name"]
    except KeyError:
        name_glo = "Unnamed planet"
    obs_glo = {k: v for k, v in obs.items() if not k[0] == "_"}
    extra_keys = list((set(extra_keys) | set(obs_glo.keys())) - set(gr.input_keys))

    # Initialising interpolators on original grid object
    print("Prepare interpolators")
    for k in set(extra_keys) | set(gr.input_keys):
        gr.interp_init(vkey=k, reinit=False)
    print(" ")

    # Copy grid into module's global scope. Required for multiprocessing to work.
    print("Copy grid object into module global scope")
    gr_glo = deepcopy(gr)

    # Check number of CPUs
    n_cpus = mp.cpu_count()
    if not n_procs:
        n_procs = n_cpus - 1
    if n_procs >= n_cpus:
        print(f"Warning: decreased n_procs from {n_procs} to {n_cpus - 1}")
        n_procs = n_cpus - 1
    n_procs = max(1, int(n_procs))

    # Need at least 2*ndim walkers for system to be well-conditioned
    n_dim = len(gr_glo.input_keys)
    if not n_walkers:
        n_walkers = int(np.ceil(n_dim * 3))
    if n_walkers < n_dim * 2:
        raise ValueError(f"Need at ≥{n_dim * 2} walkers for {n_dim} system; got {n_walkers}")

    # Check observables (values,errors)
    print("Observables:")
    for k, (obs_val, obs_err) in obs_glo.items():
        if np.isscalar(obs_err):
            print(f"    {k:16s}: {obs_val:10g} ± {obs_err:<10g}")
        else:
            print(f"    {k:16s}: {obs_val:10g} (+ {obs_err[0]:<10g} - {obs_err[1]:<10g})")
        if varprops[k].log:
            obs_glo[k] = np.log10(obs_glo[k])
    print("")

    # Convert bounds to logarithmic values where appropriate
    for i, k in enumerate(gr_glo.input_keys):
        if varprops[k].log:
            gr_glo.bounds[i] = np.log10(gr_glo.bounds[i])

    # Randomly sample initial positions from the uniform prior (within grid bounds)
    pos = np.random.uniform(
        low=gr_glo.bounds[:, 0], high=gr_glo.bounds[:, 1], size=(n_walkers, n_dim)
    )
    print("Initial guesses for parameters:")
    for i, k in enumerate(gr_glo.input_keys):
        print(
            f"    {k:16s}: {' log10' if varprops[k].log else 'linear'} [{np.amin(pos[:, i]):10g}, {np.amax(pos[:, i]):10g}] w/ {n_walkers} walkers"
        )
    print("")

    # Default values
    if not thin:
        thin = 10
    thin = max(1, thin)
    if not n_burn:
        n_burn = 200
    n_burn = max(1, n_burn)
    if not n_steps:
        n_steps = 4000
    n_steps = max(1, n_steps)
    n_steps += n_burn

    # Run sampler
    print(f"Performing {n_steps} steps with {n_walkers} walkers using {n_procs} processes")
    print("Starting MCMC retrieval...")
    with mp.Pool(processes=n_procs) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
        sampler.run_mcmc(pos, n_steps, progress=True)

    # Estimate autocorrelation
    with contextlib.redirect_stderr(None):
        tau = sampler.get_autocorr_time(quiet=True)

    print("    done")
    print(" ")

    # 1. Extract the flattened samples, discarding the initial burn-in period
    samples = sampler.get_chain(discard=n_burn, flat=True, thin=thin)
    print(f"Discarded {n_burn} burn-in samples and thinned by {thin}")
    print(f"Samples: {samples.shape}, length {samples.size}")
    print("")

    # Convert results to physical values
    for i, k in enumerate(gr_glo.input_keys):
        if varprops[k].log:
            samples[:, i] = 10 ** (samples[:, i])

    print("    Quantity    :    Median         (Uncertainty)             Autocorrelation")
    for i, key in enumerate(gr_glo.input_keys):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(f"{key:16s}: {mcmc[1]:10g}  (+ {q[1]:<10g}  - {q[0]:<10g})    {tau[i]:.4f}")
    print("")

    print(f"Postprocessing grid with extra keys: {extra_keys}")
    output_samples = []
    for i, sam in enumerate(samples):
        output_samples.append([gr_glo.interp_eval(sam, vkey=k) for k in extra_keys])
    all_samples = np.hstack([samples, output_samples])
    all_keys = list(gr_glo.input_keys) + extra_keys

    print("    done")

    return all_keys, all_samples


def plot_chain(samples: np.ndarray, save: str = None, show: bool = False):

    global gr_glo, obs_glo, name_glo

    print(f"Plot retrieval chain with {samples.shape[0]} samples")

    # Diagnostic plot: Check if walkers converged or are still wandering.
    fig, axes = plt.subplots(len(gr_glo.input_keys), figsize=(10, 7), sharex=True)

    for i, k in enumerate(gr_glo.input_keys):
        ax = axes[i]
        ax.plot(samples[:, i], color="k", alpha=0.2, lw=0.5)
        ax.set_xlim(0, len(samples))
        ax.set_title(varprops[k].label, fontsize=9)

    axes[-1].set_xlabel("Step Number")

    fig.tight_layout()
    if save:
        print(f"    Saving plot to '{save}'")
        fig.savefig(save)
    if show:
        print("    Showing plot GUI")
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_corner(keys: list, samples: np.ndarray, save: str = None, show: bool = False):

    global gr_glo, obs_glo, name_glo

    print(f"Plot retrieval corner with {samples.shape[0]} samples")

    axes_truths = []
    axes_scale = []
    axes_labels = []
    for i, k in enumerate(keys):
        try:
            axes_truths.append(obs_glo[k][0])
        except KeyError:
            axes_truths.append(None)
        axes_scale.append("log" if varprops[k].log else "linear")
        axes_labels.append(varprops[k].label)

    # axes_range = gr_glo.bounds.tolist() + [0.99]*len(keys)
    # print(axes_range)

    # 2. Create the corner plot
    fig = plt.figure(figsize=(11, 9))
    fig = corner.corner(
        samples,
        fig=fig,
        labels=axes_labels,
        quantiles=[0.16, 0.5, 0.84],  # Shows 1-sigma boundaries
        titles=[l + "\n" for l in axes_labels],
        show_titles=True,
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9, "labelpad": 7.0},
        color="#1F2F3E",
        axes_scale=axes_scale,
        # range = axes_range,
        truths=axes_truths,
        truth_color="orangered",
    )

    # indicate which variables are observables
    ii = -1
    for i in range(len(keys)):
        for j in range(len(keys)):
            ii += 1
            if (keys[i] in obs_glo.keys()) or (keys[j] in obs_glo.keys()):
                fig.axes[ii].set_facecolor("beige")

    # tick fontsize
    for ax in fig.axes:
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.tick_params(axis="both", which="minor", labelsize=8)

    # annotate on figure
    text = "Retrieval corner plot\n"
    text += f"{name_glo}\n"
    text += f"{samples.shape[0]} samples\n"
    fig.text(
        0.8,
        0.8,
        text,
        fontsize=16,
        ha="center",
        va="top",
        transform=fig.transFigure,
    )

    # format figure and save
    fig.subplots_adjust(hspace=0.012, wspace=0.012)
    if save:
        print(f"    Saving plot to '{save}'")
        fig.savefig(save)
    if show:
        print("    Showing plot GUI")
        plt.show()
    else:
        plt.close(fig)

    return fig
