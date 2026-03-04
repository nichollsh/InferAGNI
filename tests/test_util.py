from __future__ import annotations

import numpy as np
import pytest

from inferagni import util


@pytest.mark.unit
def test_calc_scaleheight_basic():
    t = 300.0
    mu = 0.02897
    g = 9.81
    got = util.calc_scaleheight(t, mu, g)
    expect = util.Rgas * t / (mu * g)
    assert np.isclose(got, expect)


@pytest.mark.unit
def test_getclose_and_latexify():
    arr = [0.0, 1.0, 2.0]
    assert util.getclose(arr, 1.2) == 1.0
    assert util.latexify("H2O") == "H$_2$O"


@pytest.mark.unit
def test_nondimen_redimen_roundtrip_and_errors():
    key = "r_phot"
    arr = [1.0, 2.0, 3.0]
    u = util.nondimen(arr, key)
    r = util.redimen(u, key)
    assert np.allclose(r, np.array(arr))

    with pytest.raises(KeyError):
        util.nondimen([1.0], "no_such_key")
