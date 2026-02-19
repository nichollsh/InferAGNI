from __future__ import annotations

import os
from copy import deepcopy

import numpy as np
import pytest

from inferagni import retrieve, util
from inferagni.grid import Grid


@pytest.mark.integration
def test_retrieve_likelihood(tmp_path):
    # Load a grid (skip heavy emissions/profiles)
    gr = Grid(emits=False, profs=False)

    # Convert bounds for log variables to log10 (as run() does)
    for i, k in enumerate(gr.input_keys):
        if util.varprops[k].log:
            gr.bounds[i] = np.log10(gr.bounds[i])

    # pick an observable from the grid to act as 'truth'
    obs_val = float(gr.data["r_phot"].iat[0])
    obs_err = max(1e-6, abs(obs_val) * 0.01)
    obs = {"_name": "test_planet", "r_phot": [obs_val, obs_err]}

    # Initialise interpolators
    gr.interp_init("r_phot", reinit=True)

    # Set module globals (what log_likelihood expects)
    retrieve.gr_glo = deepcopy(gr)
    retrieve.obs_glo = {k: deepcopy(v) for k, v in obs.items() if not k.startswith("_")}

    # construct a theta within prior bounds (gr)
    theta = np.zeros(len(gr.input_keys), dtype=gr._dtype)
    for i in range(len(theta)):
        lo, hi = gr.bounds[i]
        theta[i] = 0.5 * (lo + hi)

    # sanity checks: prior, likelihood, posterior
    lp = retrieve.log_prior(theta)
    assert np.isfinite(lp)

    ll = retrieve.log_likelihood(theta)
    assert np.isfinite(ll)
    assert np.isfinite(retrieve.log_probability(theta))
    assert np.isclose(retrieve.log_probability(theta), lp + ll)

    # Test write_csv uses globals and returns a file path
    keys = list(gr.input_keys)[:3]
    samples = np.tile(theta[: len(keys)], (5, 1))
    out = retrieve.write_csv(keys, samples, str(tmp_path / "samples.csv"))
    assert os.path.isfile(out)


@pytest.mark.integration
def test_retrieve_run(tmp_path):
    # lightweight grid without heavy emits/profs
    gr = Grid(emits=False, profs=False)

    # prepare a trivial observable taken from the grid
    obs_val = float(gr.data["r_phot"].iat[0])
    obs_err = max(1e-6, abs(obs_val) * 0.01)
    obs = {"_name": "run_test", "r_phot": [obs_val, obs_err]}

    # choose a minimal but valid number of walkers (>= 2*ndim)
    n_dim = len(gr.input_keys)
    n_walkers = max(2 * n_dim, 4)

    # Run the real retrieval with very few steps so it completes quickly
    keys, samples = retrieve.run(
        gr,
        obs,
        n_steps=100,
        n_walkers=n_walkers,
        n_procs=1,
        n_burn=5,
        thin=2,
        extra_keys=[],
    )

    # Basic assertions on outputs
    assert isinstance(keys, list)
    assert isinstance(samples, np.ndarray)
    assert samples.shape[1] == len(keys)

    # verify that output includes input keys
    for k in gr.input_keys:
        assert k in keys

    # cleanup: write csv to tmp_path
    out = retrieve.write_csv(keys[:3], samples[:, :3], str(tmp_path / "run_samples.csv"))
    assert os.path.exists(out)
    with open(out, "r") as h:
        txt = h.read()
    assert "Samples from MCMC retrieval" in txt

    # Test plotting
    figpath = str(tmp_path / "chain.pdf")
    fig = retrieve.plot_chain(samples, show=False, save=figpath)
    assert fig is not None
    assert os.path.isfile(figpath)

    # corner plot
    figpath = str(tmp_path / "corner.pdf")
    fig2 = retrieve.plot_corner(keys, samples, show=False, save=figpath)
    assert fig2 is not None
    assert os.path.isfile(figpath)
