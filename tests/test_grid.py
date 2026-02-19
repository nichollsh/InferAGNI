from __future__ import annotations


import numpy as np
import pytest

from inferagni.grid import Grid

@pytest.mark.unit
def test_grid_basic_getpoints_and_interp():

    # initialise without heavy I/O for emits/profs
    gr = Grid(emits=False, profs=False)

    pts = gr.get_points()
    assert isinstance(pts, list)
    assert len(pts) == len(gr.input_keys)

    # initialise an interpolator and evaluate at the first available grid point
    gr.interp_init(vkey="r_phot", reinit=True)

    # construct a location dict with the first value on each axis
    loc = {k: pts[i][0] for i, k in enumerate(gr.input_keys)}
    val = gr.interp_eval(loc, vkey="r_phot")

    assert isinstance(val, float)

    assert not np.isnan(val)
