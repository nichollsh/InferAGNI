from __future__ import annotations

import os
from copy import deepcopy
from textwrap import TextWrapper

import numpy as np
import pandas as pd

from .data import DEFAULT_GRID, check_grid_name
from .util import calc_scaleheight, nondimen, print_sep_min, varprops


class Grid:
    def __init__(
        self,
        gridname: str | None = None,
        emits: bool = True,
        profs: bool = True,
        succ_mode: list | str = "none",
    ):

        # Settings
        self._dtype = np.float32
        self._encoding = "utf-8"
        self._log_clip = np.finfo(self._dtype).max / 100

        # Modes for checking strictness about gridpoint convergence
        #     'succ': only consider points with succ > 0 (most strict)
        #     'fmed': only consider points with flux_loss_med < flux_loss_crit
        #     'none': consider all points (least strict)
        if isinstance(succ_mode, str):
            succ_mode = [succ_mode]
        self._succ_mode = set([s.lower() for s in succ_mode])
        self._flux_loss_crit = 1.0  # W/m^2

        # scalar data (stored as scaled values)
        self._df_points = None  # Dataframe of the grid points (input)
        self._df_results = None  # Dataframe of the results (output)
        self.bounds = None  # Boundcles on the input axes
        self.data = None  # Merged dataframe

        # emission spectra
        self.emits_wl = None  # wavelengths
        self.emits_fl = None  # fluxes

        # atmosphere profiles
        self.profs = None  # load from NetCDF

        print("Loading data from disk...")
        if not gridname:
            gridname = DEFAULT_GRID
        elif not check_grid_name(gridname):
            return

        data_dir = os.path.join(os.path.dirname(__file__), "data", gridname)
        data_dir = os.path.abspath(data_dir)
        print(f"    Source: {data_dir}")

        # check folder exists
        if not os.path.isdir(data_dir):
            print("    Grid data folder not found")
            print("    Please run 'inferagni update' to download the required data")
            print("    Alternatively, run 'inferagni --help' to view available commands")

        # load data
        self._load_scalars(data_dir)  # load from CSVs
        if emits:
            self._load_emits(data_dir)  # load from CSV
        if profs:
            self._load_profs(data_dir)  # load from NetCDF

        print(print_sep_min)
        print("")

        # -------------------------
        # interpolators for the whole grid
        self._interp_method = "linear"  # “linear”, “nearest”,   spline methods: “slinear”, “cubic”, “quintic” and “pchip”.
        self._interp_logscaled = dict()  # Whether log-varying quantities return log-scaled values (if True) or dimensionalised values (if False)
        self._interp = dict()  # Instantiated later

    def _load_scalars(self, data_dir: str):
        """Load grid scalars data from CSV files in the specified directory.

        Parameters
        -----------
        - data_dir : str, The directory containing the CSV files.
        """

        print("Loading grid of scalar quantities")

        # Read the grid point definition file
        self._df_points = pd.read_csv(
            os.path.join(data_dir, "gridpoints.csv"),
            sep=",",
            dtype=self._dtype,
            encoding=self._encoding,
        )

        # Read the consolidated results file
        self._df_results = pd.read_csv(
            os.path.join(data_dir, "consolidated_table.csv"),
            sep=",",
            dtype=self._dtype,
            encoding=self._encoding,
        )

        # Derive some observables
        self._df_results["H_phot"] = calc_scaleheight(
            self._df_results["t_phot"], self._df_results["μ_phot"], self._df_results["g_phot"]
        )
        for k in self._df_results.keys():
            if k.startswith("vmr_"):
                self._df_results["log_" + k] = np.log10(
                    np.clip(self._df_results[k].values, 1 / self._log_clip, 1)
                )

        if "Kzz_max" in self._df_results.columns:
            self._df_results["log_Kzz_max"] = np.log10(
                np.clip(self._df_results["Kzz_max"].values, 1 / self._log_clip, self._log_clip)
            )

        # Merge the dataframes on index
        self.data = pd.merge(self._df_points, self._df_results, on="index")

        # Ensure all quantities have real values
        for k in self.data.columns:
            amax = 1e30
            if varprops[k].log:
                amin = 1e-30
            else:
                amin = -amax
            self.data.loc[:, k] = np.clip(self.data[k].values, amin, amax)[:]

        # Calculate grid size
        print(f"    Grid size: {len(self.data)} points")

        # Define input and output variables
        self.input_keys = list(self._df_points.keys())
        self.output_keys = list(self._df_results.keys())

        # Remove unused keys
        for k in ("index", "worker"):
            for v in (self.input_keys, self.output_keys):
                if k in v:
                    v.remove(k)

        # Re-order the input dimensions
        #   Determine how frequently each input column changes in the CSV row order.
        #   A column that changes more often is the faster-varying (innermost) axis.
        #   Sort keys by change frequency (slowest-varying first). Keep sort stable.
        change_counts = {}
        for k in list(self.input_keys):
            arr = self._df_points[k].values
            if arr.size < 2:
                change_counts[k] = 0
            else:
                change_counts[k] = int(np.sum(arr[1:] != arr[:-1]))
        self.input_keys.sort(key=lambda kk: change_counts.get(kk, 0))

        # Store the bounds on each dimension
        self.bounds = np.array(
            [(self.data[k].min(), self.data[k].max()) for k in self.input_keys]
        )

        # Print info
        print("    Input vars:")
        for i, k in enumerate(self.input_keys):
            print(f"      {k:16s}: range [{self.bounds[i][0]:<10g}, {self.bounds[i][1]:<10g}]")
        wrapper = TextWrapper(width=45, initial_indent=" " * 6, subsequent_indent=" " * 6)
        print("    Output vars: ")
        print(wrapper.fill(", ".join(self.output_keys)))

    def _load_emits(self, data_dir: str):
        """Read fluxes table"""

        print("Loading emission spectra")
        emit_dat = np.loadtxt(
            os.path.join(data_dir, "consolidated_emits.csv"),
            dtype=float,
            delimiter=",",
            converters=lambda x: 0 if x == "index" else float(x),
            encoding=self._encoding,
        )
        self.emits_wl = np.array(emit_dat[0, 1:], copy=True, dtype=self._dtype)
        self.emits_fl = np.array(emit_dat[1:, 1:], copy=True, dtype=self._dtype)
        print("    done")

    def _load_profs(self, data_dir: str):
        """Read atmosphere profiles"""

        import netCDF4 as nc

        print("Loading atmosphere profiles")

        ds = nc.Dataset(os.path.join(data_dir, "consolidated_profs.nc"))

        self.profs = dict()
        self.profs["t"] = np.array(ds["t"][:, :], copy=True, dtype=self._dtype)
        self.profs["p"] = np.array(ds["p"][:, :], copy=True, dtype=self._dtype)
        self.profs["r"] = np.array(ds["r"][:, :], copy=True, dtype=self._dtype)

        ds.close()

        print("    done")

    def show_inputs(self):
        """Print the unique values of each input variable in the grid."""
        for key in self.input_keys:
            print(f"{key:16s}\n\t- {np.unique(self.data[key].values)}")

    def get_points(self):
        """Get the values of the grid axes"""
        points = []
        for key in self.input_keys:
            points.append(pd.unique(self._df_points[key].values))
        return points

    def interp_2d(self, zkey=None, controls=None, resolution=100, method="linear"):
        """Interpolate a 2D grid of radii as a function of mass and `zkey`.

        Must provide control variables in order to define the 2-D slice through N-D space.

        Parameters
        -----------
        - zkey : str, The name of the variable to interpolate
        - controls : dict, A dictionary of control variables to filter the data by.
        - resolution : int, The number of points to interpolate along each axis (mass and radius).
        - method : str, The interpolation method to use. Options are 'linear', 'nearest', and 'cubic'.

        Returns
        -----------
        - itp_x : 2D array, The x-coordinates of the interpolated grid (mass).
        - itp_y : 2D array, The y-coordinates of the interpolated grid (radius).
        - itp_z : 2D array, The interpolated z-values on the grid (the variable specified by zkey).
        """

        from scipy.interpolate import griddata

        # copy dataframe
        subdata = deepcopy(self.data)
        subdata = subdata[subdata["succ"] > 0.0]
        # subdata = subdata[(subdata['r_bound'] < 0.0) | (subdata['r_bound'] > subdata['r_phot'])]

        # check that the number of control variables is correct
        controls_req = len(self.input_keys) - 2  # -1 for zkey, -1 for mass (xkey)
        if len(controls) > controls_req:
            raise ValueError(
                f"Too many control variables. Got {len(controls)}, expected {controls_req}"
            )
        if len(controls) < controls_req:
            raise ValueError(
                f"Too few control variables. Got {len(controls)}, expected {controls_req}"
            )

        # crop data by control variables
        for c in controls.keys():
            if c == zkey:
                raise KeyError(f"Z-value {zkey} cannot also be a control variable")
            if c in subdata.columns:
                print(f"Filter by {c} = {controls[c]}")
                subdata = subdata[np.isclose(subdata[c], controls[c])]
            if len(subdata) < 1:
                print("No data remaining! \n")
                return False

        if zkey and zkey not in subdata.keys():
            raise KeyError(f"Z-value {zkey} not found in dataframe")

        print(f"Number of points: {len(subdata)}")

        # Flatten data
        points = np.array(
            [
                subdata["mass_tot"].values * varprops["mass_tot"].scale,
                subdata["r_phot"].values * varprops["r_phot"].scale,
            ]
        ).T
        values = np.array(subdata[zkey].values * varprops[zkey].scale)

        # Get range on x,y keys
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        print(f"x range: {x_min:.2f} - {x_max:.2f}")  # mass
        print(f"y range: {y_min:.2f} - {y_max:.2f}")  # radius

        # Target grid to interpolate to
        itp_x, itp_y = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
            indexing="ij",
        )

        # Do the interpolation
        itp_z = griddata(points, values, (itp_x, itp_y), method=method)

        # Return the grid for plotting
        return itp_x, itp_y, itp_z

    def interp_init(
        self,
        vkey: str = "r_phot",
        reinit: bool = False,
        method: str | None = None,
        logscale: bool = True,
    ):
        """Instantiate a regular-grid interpolator of `vkey` the whole parameter space.

        The interpolator is constructed on non-dimensionalised and log-scaled values (if applicable).
        Use `interp_eval` to evaluate the non-dimensionalised value.

        Parameters
        -----------
        - vkey : str, The name of the variable to interpolate (e.g. "r_phot").
        - reinit : bool, If True, re-initialise the interpolator even if it already exists.
        - method : str | None, The interpolation method to use. Uses default method if None
        - logscale : bool, If True, log-scale the variable if applicable.
        """

        from scipy.interpolate import RegularGridInterpolator

        # Check if already initialised
        if (vkey in self._interp) and (self._interp[vkey]) and not reinit:
            print(f"Interpolator already initialised on {vkey}")
            return

        # method
        if not method:
            method = self._interp_method

        # organise parameters
        xyz = []
        print(f"Creating {method} interpolator on {vkey}")
        grid_points = self.get_points()
        for i, gp in enumerate(grid_points):
            k = self.input_keys[i]

            # scale data
            xx = np.array(gp, copy=True, dtype=self._dtype)
            xx = nondimen(gp, k)

            # store
            xyz.append(xx)

            # print("\t" + k)
            # print(f"\t\t{xx}")

        # meshgrid the axes
        xyz_g = np.meshgrid(*xyz, indexing="ij")

        # check that data has the interpolant
        if vkey not in self.data.columns:
            raise KeyError(f"Cannot find {vkey} in input dataset. Typo?")

        # arrange value to be interpolated
        v = nondimen(self.data[vkey].values, vkey)
        v = np.array(v, copy=True, dtype=self._dtype)
        v_logscaled = varprops[vkey].log and logscale
        if v_logscaled:
            v = np.log10(np.clip(v, 1 / self._log_clip, self._log_clip))
        v_g = np.reshape(v, xyz_g[0].shape)
        # print(f"    min {np.amin(v_g)}, max {np.amax(v_g)}")

        # instantiate regular-grid interpolator
        # print("    Creating interpolator")
        self._interp[vkey] = RegularGridInterpolator(
            xyz, v_g, fill_value=None, bounds_error=False, method=method
        )
        self._interp_logscaled[vkey] = v_logscaled

        print("    Interpolator ready")

    def interp_eval(self, loc: dict | list, vkey: str = "r_phot", method: str | None = None):
        """Evaulate the interpolator at a single location in parameter space.

        Returns non-dimensionalised value, but removes the log-scaling where appropriate.

        Parameters
        ------------
        - loc : dict or list, The location to evaluate the `vkey` at.
        - vkey : str, The variable to interpolate.
        - method : str | None, The interpolation method to use. Uses default method if None.
        """

        # Check interpolate is setup
        if (vkey not in self._interp) or (not self._interp[vkey]):
            raise RuntimeError(f"Cannot interpolate; interpolator on '{vkey}' not initialised!")

        # parse method
        if method:
            eval_method = method
        else:
            eval_method = self._interp_method

        # parse location
        if isinstance(loc, (list, np.ndarray)):
            eval_loc = list(loc)
        elif isinstance(loc, dict):
            eval_loc = [loc[k] for k in self.input_keys]
        else:
            raise ValueError(f"Location must be a dict or an array, got {type(loc)}")

        # scale inputs
        for i, k in enumerate(self.input_keys):
            eval_loc[i] = nondimen(eval_loc[i], k)

        # evaluate non-dimensionalised value
        val = self._dtype(self._interp[vkey](eval_loc, method=eval_method)[0])

        # remove log-scaling
        if self._interp_logscaled[vkey]:
            val = 10**val

        # Evaluate and return the non-dimensionalised value
        return self._dtype(val)
