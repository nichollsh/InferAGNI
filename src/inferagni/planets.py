from __future__ import annotations

import builtins
import importlib
import io
import os
from contextlib import redirect_stdout

os.environ["EXOATLAS_DATA"] = os.path.join(os.path.dirname(__file__), "exoatlas-data")


def _import_exoatlas_silently() -> bool:
    """Import exoatlas while suppressing third-party print() spam."""

    original_print = builtins.print
    stdout_buffer = io.StringIO()
    try:
        builtins.print = lambda *args, **kwargs: None
        with redirect_stdout(stdout_buffer):
            global ea
            ea = importlib.import_module("exoatlas")

            global solarsys
            solarsys = ea.SolarSystem()

            global exoplanets
            exoplanets = ea.Exoplanets()
            exoplanets = exoplanets[exoplanets.radius() > 0]
    finally:
        builtins.print = original_print

    return (ea is not None) and (solarsys is not None) and (exoplanets is not None)

# Trigger download and filter
ea = None
exoplanets = None
solarsys = None
_import_exoatlas_silently()


def list_planets(flat: bool = False, quiet: bool = True) -> list[str]:
    """List the names of all planets in the databases.

    Grouped by planetary system, with solar system planets first.

    Arguments
    ------------
    - flat: bool, Return a flat list of planet names. Otherwise, group by system.
    - quiet: bool, Suppress printing of planet names.

    Returns
    ------------
    - names: list of str, The names of all planets in the databases.
    """

    names = []

    # Loop through solar system planets
    sysnames = [str(pl.name()[0]) for pl in solarsys]
    if not quiet:
        print("Solar System planets:")
        print("    " + ", ".join(sysnames))

    # Store solar system planets
    if not flat:
        names.append(sysnames)
    else:
        names.extend(sysnames)

    # Loop through exoplanets
    if not quiet:
        print("\nExoplanets:")
    sys = "UNSET"
    sysnames = []
    for pl in exoplanets:
        name = str(pl.name()[0])

        # don't group by system if requested
        if flat:
            names.append(name)
            if not quiet:
                print(name)

        # group by system
        else:
            # same system
            if name[:-2] == sys:
                sysnames.append(name)
            # new system
            else:
                if sysnames:
                    names.extend(sysnames)
                    if not quiet:
                        print("    " + ", ".join(sysnames))
                sysnames = [name]
                sys = name[:-2]

    return names


def get_sys(name: str, quiet: bool = False) -> dict:
    """Get the observed parameters of all planets in a named system.

    Arguments
    ------------
    - name: str, The name of the system (e.g. "TRAPPIST-1").
    - quiet: bool, If True, suppress printing of observation details.

    Returns
    ------------
    - obs_sys: dict, The observed parameters of all planets in the system, with uncertainties.
    """

    name = name.replace("_", " ")
    obs_sys = {"_name": name.replace(" ", "_")}

    # solar system
    if name.lower() in ("sun", "solar", "solarsystem"):
        for pl in solarsys:
            if not quiet:
                print(" ")
            obs_sys[str(pl.name()[0])] = get_obs(str(pl.name()[0]), quiet=quiet)
        return obs_sys

    # check for planets alphabetically
    for pl in "bcdefghijklmnopqrstuvwxyz":
        if not quiet:
            print(" ")
        plname = name + " " + pl
        obs_pl = get_obs(plname, quiet=quiet)
        if len(obs_pl) > 1:
            obs_sys[pl] = obs_pl
        else:
            break

    return obs_sys


def get_obs(name: str, quiet: bool = False) -> dict:
    """Get the measured parameters of a planet, by name

    Arguments
    ------------
    - name: str, The name of the planet.
    - quiet: bool, If True, suppress printing of observation details.

    Returns
    ------------
    - obs_pl: dict, The observed parameters of the planet, with uncertainties.
    """

    name = name.replace("_", " ")
    obs_pl = {"_name": name.replace(" ", "_")}

    # check databases
    if name in exoplanets.name():
        pl = exoplanets[name]
        quiet or print(f"Measurements of exoplanet {name}")

    elif name in solarsys.name():
        pl = solarsys[name]
        quiet or print(f"Measurements of Solar System planet {name}")

    else:
        print(f"Planet '{name}' not found in databases")
        return obs_pl

    # get parameters
    for key, lk in (
        ("r_phot", "radius"),
        ("mass_tot", "mass"),
        ("Teff", "stellar_teff"),
        ("instellation", "relative_insolation"),
    ):
        val = [getattr(pl, lk)(), getattr(pl, lk + "_uncertainty")()]
        obs_pl[key] = [float(v.value[0]) for v in val]

    # print info
    for k, v in obs_pl.items():
        if k[0] == "_":
            continue
        quiet or print(f"    {k:16s}: {v[0]:10g} ± {v[1]:<10g}")

    return obs_pl
