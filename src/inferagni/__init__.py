from __future__ import annotations

__version__ = "26.02.12"

import os

os.environ["EXOATLAS_DATA"] = os.path.join(os.path.dirname(__file__), "exoatlas-data")

import inferagni.cli as cli
import inferagni.grid as grid
import inferagni.plot as plot
import inferagni.retrieve as retrieve
import inferagni.util as util

__all__ = ["util", "grid", "cli", "plot", "retrieve"]
