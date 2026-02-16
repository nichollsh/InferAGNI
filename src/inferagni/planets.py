from __future__ import annotations

import exoatlas as ea
import astropy.units as units

# https://zkbt.github.io/exoatlas/quickstart
# Trigger download and filter
solarsys = ea.SolarSystem()

exoplanets = ea.Exoplanets()
exoplanets = exoplanets[exoplanets.radius() > 0]

