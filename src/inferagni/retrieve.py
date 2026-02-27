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
import pandas as pd

from .grid import Grid
from .util import print_sep_min, redimen, varprops
from .plot import DPI, truth_color, samples_color

global gr_glo, obs_glo
gr_glo: Grid = None
obs_glo: dict = None
name_glo: str = "Unnamed_planet"



def log_prior(theta: list):
    """Uniform prior: checks if parameters are within grid boundaries.


    Parameters
    ------------
    - theta : list, Current N-dimensional parameter set proposed by MCMC.

    Returns
    ------------
    - ln_prior : float, Logarithm of the prior probability of the model
    """

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

    Returns
    ------------
    - ln_L : float, Logarithm of the likelihood of the model given the data
    """

    global gr_glo, obs_glo

    # Transform coordinates from log-scaled to physical
    theta_eval = deepcopy(theta)
    for i, k in enumerate(gr_glo.input_keys):
        if varprops[k].log:
            theta_eval[i] = 10 ** (theta_eval[i])

    ln_L = 0.0

    # Check if this regime is physical in the grid, Log(0) for unphysical regimes
    if ('succ' in gr_glo._succ_mode) and (gr_glo.interp_eval(theta_eval, vkey="succ") < 0.5):
        return -np.inf
    if ('fmed' in gr_glo._succ_mode) and (gr_glo.interp_eval(theta_eval, vkey="flux_loss_med") > gr_glo._flux_loss_crit):
        return -np.inf

    # Iterate through observables to handle potential asymmetry
    for k, (obs_val, obs_err) in obs_glo.items():
        # Evaluate the model (returns scaled value)
        model_val = gr_glo.interp_eval(theta_eval, k)
        if varprops[k].log:
            model_val = np.log10(model_val)

        # Obs_err can be a scalar, or a size-2 tuple
        if obs_err == '<': # must be greater than obs_val
            if model_val < obs_val:
                return -np.inf  # Log(0)

        elif obs_err == '>': # must be less than obs_val
            if model_val > obs_val:
                return -np.inf  # Log(0)

        elif np.isscalar(obs_err):
            # Standard symmetric scalar case
            sig = obs_err
            norm = -0.5 * np.log(2 * np.pi * sig**2)
            ln_L += norm - 0.5 * ((model_val - obs_val) / sig) ** 2

        else:
            # Asymmetric scalar case
            # err = (sigma_plus, sigma_minus)
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

    Returns
    ------------
    - ln_post : float, Logarithm of the posterior probability of the model
    """

    global gr_glo, obs_glo

    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


def run_retrieval(
    gr: Grid,
    obs: dict,
    n_steps: int | None = None,
    n_walkers: int | None = None,
    n_procs: int | None = None,
    n_burn: int | None = None,
    thin: int | None = None,
    extra_keys: list = [],
) -> tuple:
    """Executes the MCMC retrieval

    Parameters
    ------------
    - gr: Grid, The grid object containing the model and interpolation methods.
    - obs: dict, The observed parameters of the planet, with uncertainties.
    - n_steps: int, Total number of MCMC steps to run (including burn-in).
    - n_walkers: int, Number of MCMC walkers to use.
    - n_procs: int, Number of CPU processes to use for parallelization.
    - n_burn: int, Number of initial steps to discard as burn-in.
    - thin: int, Thinning factor to reduce autocorrelation in samples.
    - extra_keys: list, Additional keys to evaluate from the grid for each sample.


    Returns
    ------------
    - all_keys: list, The list of parameter and observable keys corresponding to the samples.
    - all_samples: np.ndarray, The MCMC samples for the parameters and extra keys.
    """

    global obs_glo, gr_glo, name_glo

    # check grid is ok
    if gr.data is None:
        print("Cannot perform retrieval because grid data is not available")
        print("    please run 'inferagni update' command first.")
        return None, None

    # Extra keys should not include the parameters, but should include the observables
    try:
        name_glo = obs["_name"]
    except KeyError:
        name_glo = "Unnamed_planet"
    obs_glo = {k: deepcopy(v) for k, v in obs.items() if not k[0] == "_"}
    extra_keys = list((set(extra_keys) | set(obs_glo.keys())) - set(gr.input_keys))

    # Initialising interpolators on original grid object
    print("Prepare interpolators")
    #    required
    if 'succ' in gr._succ_mode:
        gr.interp_init(vkey="succ",method='nearest')
    if 'fmed' in gr._succ_mode:
        gr.interp_init(vkey="flux_loss_med",method='linear')
    #    optional
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
        obs_val = np.abs(obs_val)
        if obs_err == '<':
            print(f"    {k:16s}:    >{obs_val:g}")
        elif obs_err == '>':
            print(f"    {k:16s}:    <{obs_val:g}")
        else:
            obs_err = np.abs(obs_err)
            if np.isscalar(obs_err):
                print(f"    {k:16s}: {obs_val:10g} ± {obs_err:<10g}")
            else:
                print(f"    {k:16s}: {obs_val:10g} (+ {obs_err[0]:<10g} - {obs_err[1]:<10g})")
            if varprops[k].log:
                obs_glo[k][0] = np.log10(obs_glo[k][0])
                obs_glo[k][1] = np.log10(obs_glo[k][1])
    print("")

    # Convert bounds to logarithmic values where appropriate
    for i, k in enumerate(gr_glo.input_keys):
        if varprops[k].log:
            gr_glo.bounds[i] = np.log10(gr_glo.bounds[i])

    # Randomly sample initial positions from the uniform prior (within grid bounds)
    theta_ini = []
    print("Initial guesses for parameters:")
    for i, k in enumerate(gr_glo.input_keys):

        # centre guess around truth
        if k in obs_glo.keys() and (obs_glo[k][1] not in ['<', '>']):
            this_ini = np.random.normal(
                obs_glo[k][0], scale=np.abs(np.median(obs_glo[k][1]))/2, size=n_walkers
            )

        # otherwise use uniform guess
        else:
            this_ini = np.random.uniform(
                low=gr_glo.bounds[i, 0], high=gr_glo.bounds[i, 1], size=n_walkers
            )

        this_ini = np.clip(this_ini, gr_glo.bounds[i, 0], gr_glo.bounds[i, 1])

        print(
            f"    {k:16s}: {' log10' if varprops[k].log else 'linear'} [{np.amin(this_ini):10g}, {np.amax(this_ini):10g}] w/ {n_walkers} walkers"
        )

        theta_ini.append(this_ini)
    theta_ini = np.array(theta_ini, dtype=gr_glo._dtype).T
    print("")

    # Default values
    if not thin:
        thin = 10
    thin = max(1, thin)
    if not n_steps:
        n_steps = 4000
    n_steps = max(1, n_steps)
    if not n_burn:
        n_burn = max(200, int(n_steps * 0.25))
    n_burn = max(1, n_burn)
    n_steps += n_burn

    # Run sampler
    print(f"Will do {n_steps} steps with {n_walkers} walkers using {n_procs} processes")
    print("Running MCMC retrieval...")
    with mp.Pool(processes=n_procs) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, pool=pool)
        sampler.run_mcmc(theta_ini, n_steps, progress=True)

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

    print(f"Postprocessing grid with extra keys: {extra_keys}")
    output_samples = []
    for i, sam in enumerate(samples):
        output_samples.append([gr_glo.interp_eval(sam, vkey=k) for k in extra_keys])
    all_samples = np.hstack([samples, output_samples])
    all_keys = list(gr_glo.input_keys) + extra_keys
    print("    done")
    print("")

    # Filter to cases consistent with inequality constraints
    print("Filtering samples to satisfy inequality constraints")
    for k in obs_glo.keys():
        obs_val, obs_err = obs_glo[k]
        if obs_err == '<': # must be greater than obs_val
            print(f"    {k} > {obs_val:g}")
            mask = all_samples[:, all_keys.index(k)] >= obs_val
            all_samples = all_samples[mask]
        elif obs_err == '>': # must be less than obs_val
            print(f"    {k} < {obs_val:g}")
            mask = all_samples[:, all_keys.index(k)] <= obs_val
            all_samples = all_samples[mask]
    print("New sample size after filtering: "+ str(all_samples.shape[0]))
    print("")

    print("    Quantity    :    Median         (Uncertainty)             Autocorrelation")
    for i, key in enumerate(all_keys):
        mcmc = np.percentile(all_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        info_row = f"{key:16s}: {mcmc[1]:10g}  (+ {q[1]:<10g}  - {q[0]:<10g})"
        if i < len(tau):
            info_row += f"    {tau[i]:.4f}"
        print(info_row)
    print(print_sep_min)
    print("")

    return all_keys, all_samples


def write_result(keys: list, samples: np.ndarray, fpath: str) -> str:
    """Writes the MCMC samples to a CSV file.

    Parameters
    ------------
    - keys: list, The list of parameter and observable keys corresponding to the samples.
    - samples: np.ndarray, The MCMC samples for the parameters and extra keys.
    - fpath: str, The file path where the CSV should be saved.

    Returns
    ------------
    - fpath: str, The file path where the CSV was saved.
    """

    global gr_glo, obs_glo, name_glo

    print(f"Writing samples to '{fpath}'")

    # convert back to original units
    samples_dimen = []
    for i, k in enumerate(keys):
        samples_dimen.append(redimen(samples[:, i], k))
    samples_dimen = np.array(samples_dimen).T

    # construct dataframe and save to csv
    df = pd.DataFrame(samples_dimen, columns=keys)

    # header information
    header = f"Samples from MCMC retrieval of {name_glo}. Len={samples.shape[0]}"

    # Write file
    with open(fpath, "w") as hdl:
        hdl.write(f"# {header} \n")
        df.to_csv(hdl, sep=",", index=False)


    print("    done")
    return fpath


def write_truth(fpath: str) -> str:
    """Write observables (truths) to a CSV file, with uncertainties.

    Parameters
    ------------
    - fpath: str, The file path where the CSV should be saved.

    ------------
    - fpath: str, The file path where the CSV was saved.
    """

    global gr_glo, obs_glo, name_glo

    print(f"Writing truths to '{fpath}'")


    # construct dataframe and save to csv
    data = []
    for k in obs_glo.keys():
        obs_val, obs_err = deepcopy(obs_glo[k])

        if varprops[k].log:
            obs_val = 10 ** obs_val
        obs_val = redimen(obs_val, k)
        obs_val = f"{obs_val:g}"

        if obs_err == '<':
            obs_err_plu = ">value"
            obs_err_min = "-"

        elif obs_err == '>':
            obs_err_plu = "<value"
            obs_err_min = "-"

        else:
            if varprops[k].log:
                obs_err = 10 ** obs_err
            obs_err = redimen(obs_err, k)
            if np.isscalar(obs_err):
                obs_err_plu = f"{obs_err:g}"
                obs_err_min = obs_err_plu
            else:
                obs_err_plu = f"{obs_err[0]:g}"
                obs_err_min = f"{obs_err[1]:g}"

        data.append((k, obs_val, obs_err_plu, obs_err_min))
    df = pd.DataFrame(data, columns=["key", "value", "plus", "minus"])

    # header information
    header = f"Truths used for MCMC retrieval of {name_glo}"

    # Write file
    with open(fpath, "w") as hdl:
        hdl.write(f"# {header} \n")
        df.to_csv(hdl, sep=",", index=False)

    print("    done")
    return fpath


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
        if varprops[k].log:
            ax.set_yscale("log")

        if k in obs_glo.keys():
            if varprops[k].log:
                tru = 10 ** obs_glo[k][0]
            else:
                tru = obs_glo[k][0]
            ax.axhline(y=tru, color=truth_color)

    axes[-1].set_xlabel("Step Number")

    fig.tight_layout()
    if save:
        print(f"    Saving plot to '{save}'")
        fig.savefig(save, bbox_inches="tight", dpi=DPI)
    if show:
        print("    Showing plot GUI")
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_corner(keys: list, samples: np.ndarray, save: str = None, show: bool = False):

    global gr_glo, obs_glo, name_glo

    print(f"Plot retrieval corner from {samples.shape[0]} samples")

    axes_truths = []
    axes_scale = []
    axes_labels = []
    axes_range = []
    for i, k in enumerate(keys):
        ax_min, ax_max = np.amin(samples[:, i]), np.amax(samples[:, i])
        try:
            if varprops[k].log:
                axes_truths.append(10 ** obs_glo[k][0])
            else:
                axes_truths.append(obs_glo[k][0])
            ax_min = min(ax_min, axes_truths[i])
            ax_max = max(ax_max, axes_truths[i])
        except KeyError:
            axes_truths.append(None)

        axes_range.append([ax_min / 1.15, ax_max * 1.15])

        axes_scale.append("log" if varprops[k].log else "linear")
        axes_labels.append(varprops[k].label_short)

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
        label_kwargs={"fontsize": 9},
        labelpad=0.4,
        color=samples_color,
        axes_scale=axes_scale,
        range=axes_range,
        truths=axes_truths,
        truth_color=truth_color,
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
    text += f"Thinned to {samples.shape[0]} samples\n"
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
        fig.savefig(save, bbox_inches="tight", dpi=DPI)
    if show:
        print("    Showing plot GUI")
        plt.show()
    else:
        plt.close(fig)

    return fig
