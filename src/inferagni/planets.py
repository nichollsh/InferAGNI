from __future__ import annotations

import os

os.environ["EXOATLAS_DATA"] = os.path.join(os.path.dirname(__file__), "exoatlas-data")

import exoatlas as ea

# https://zkbt.github.io/exoatlas/quickstart
# Trigger download and filter
solarsys = ea.SolarSystem()

exoplanets = ea.Exoplanets()
exoplanets = exoplanets[exoplanets.radius() > 0]


def get_obs(name: str) -> dict:

    if name in exoplanets.name():
        pl = exoplanets[name]
        print(f"Found observations for exoplanet '{name}'")

    elif name in solarsys.name():
        pl = solarsys[name]
        print(f"Found observations for solar system planet '{name}'")

    else:
        print(f"Planet '{name}' not found in database")
        return None

    obs_pl = {"_name": name}
    for key, lk in (
        ("r_phot", "radius"),
        ("mass_tot", "mass"),
        ("Teff", "stellar_teff"),
        ("instellation", "relative_insolation"),
    ):
        val = [getattr(pl, lk)(), getattr(pl, lk + "_uncertainty")()]
        obs_pl[key] = [float(v.value[0]) for v in val]

    for k, v in obs_pl.items():
        if k[0] == "_":
            continue
        print(f"    {k:16s}: {v[0]:10g} ± {v[1]:<10g}")
    return obs_pl
