from __future__ import annotations

import exoatlas as ea

# https://zkbt.github.io/exoatlas/quickstart
# Trigger download and filter
solarsys = ea.SolarSystem()

exoplanets = ea.Exoplanets()
exoplanets = exoplanets[exoplanets.radius() > 0]

def get_obs(name:str) -> dict:

    if name in exoplanets.name():
        pl = exoplanets[name]
        print(f"Getting data for exoplanet '{name}'")

    elif name in solarsys.name():
        pl = solarsys[name]
        print(f"Getting data for solar system planet '{name}'")

    else:
        print(f"Planet '{name}' not found in database")
        return None

    obs_pl = dict() #{"_name":name}
    for key,lk in (
        ("r_phot",          "radius"),
        ("mass_tot",        "mass"),
        ("Teff",            "stellar_teff"),
        ("instellation",    "relative_insolation")
        ):
        val = [getattr(pl,lk)(),getattr(pl,lk+"_uncertainty")()]
        obs_pl[key] = [float(v.value[0]) for v in val]

    print("    " + str(obs_pl))
    return obs_pl
