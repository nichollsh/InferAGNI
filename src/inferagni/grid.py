from __future__ import annotations

import pandas as pd
import os
import numpy as np
from copy import deepcopy

from .const import *

class Grid:
    def __init__(self, data_dir):

        # scalar data
        self.data, self.input_keys, self.output_keys = self._load_from_dir(data_dir)

        # profiles data
        self.profs = None # TODO

        # emission data
        self.emits = None # TODO

        # interpolator
        self.interp = None # Instantiated later


    def _load_from_dir(self, data_dir):
        """Load grid data from CSV files in the specified directory.

        Parameters
        -----------
        - data_dir : str The directory containing the CSV files.

        """

        print("Loading data from CSV files...")
        print(f"Data directory: {data_dir}")

        # Read the grid point definition file
        gridpoints_df = pd.read_csv(os.path.join(data_dir, "nogit_points.csv"), sep=',')

        # Read the consolidated results file
        results_df = pd.read_csv(os.path.join(data_dir, "nogit_table.csv"), sep=',')

        # Merge the dataframes on index
        data = pd.merge(gridpoints_df, results_df, on="index")

        # Calculate grid size
        gridsize = len(data)
        print(f"Grid size: {gridsize}")

        # Define input and output variables
        input_keys  = list(gridpoints_df.keys())
        output_keys = list(results_df.keys())
        print(f"Input vars:  {input_keys}")
        print(f"Output vars: {output_keys}")

        # Remove redundant variables
        for k in ("index","worker"):
            for v in (input_keys,output_keys):
                if k in v:
                    v.remove(k)

        print("Loaded data")
        return data, input_keys, output_keys

    def show_inputs(self):
        """Print the unique values of each input variable in the grid."""
        unique = {}
        for key in self.input_keys:
            unique[key] = np.unique(self.data[key].values)
            print(f"{key:12s}\n\t- {unique[key]}")
        return unique

    def interpolate_2d(self,
                    zkey=None,controls=None,
                    resolution=100, method='linear'):
        """Interpolate a 2D grid of z-values as a function of mass and radius.

        Parameters
        -----------
        - zkey : str The name of the variable to interpolate
        - controls : dict A dictionary of control variables to filter the data by.
        - resolution : int The number of points to interpolate along each axis (mass and radius).
        - method : str The interpolation method to use. Options are 'linear', 'nearest', and 'cubic'.

        Returns
        -----------
        - itp_x : 2D array The x-coordinates of the interpolated grid (mass).
        - itp_y : 2D array The y-coordinates of the interpolated grid (radius).
        - itp_z : 2D array The interpolated z-values on the grid (the variable specified by zkey).
        """


        from scipy.interpolate import griddata

        # copy dataframe
        subdata = deepcopy(self.data)
        subdata = subdata[subdata['succ'] > 0.0]
        # subdata = subdata[(subdata['r_bound'] < 0.0) | (subdata['r_bound'] > subdata['r_phot'])]

        # check that the number of control variables is correct
        controls_req = len(self.input_keys) - 2 # -1 for zkey, -1 for mass (xkey)
        if len(controls) > controls_req:
            raise ValueError(f"Too many control variables. Got {len(controls)}, expected {controls_req}")
        if len(controls) < controls_req:
            raise ValueError(f"Too few control variables. Got {len(controls)}, expected {controls_req}")

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

        if zkey and not (zkey in subdata.keys()):
            raise KeyError(f"Z-value {zkey} not found in dataframe")

        print(f"Number of points: {len(subdata)}")

        # Flatten data
        points = np.array([subdata["mass_tot"].values*units["r_phot"][0],
                           subdata["r_phot"].values*units["r_phot"][0]
                           ]).T
        values = np.array(subdata[zkey].values*units[zkey][0])

        # Get range on x,y keys
        x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
        y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
        print(f"x range: {x_min:.2f} - {x_max:.2f}") # mass
        print(f"y range: {y_min:.2f} - {y_max:.2f}") # radius

        # Target grid to interpolate to
        itp_x, itp_y = np.meshgrid(np.linspace(x_min, x_max, resolution),
                                     np.linspace(y_min, y_max, resolution),
                                     indexing='ij'
                                     )


        # Do the interpolation
        itp_z = griddata(points, values, (itp_x, itp_y), method=method)

        # Return the grid for plotting
        return itp_x, itp_y, itp_z


    def interp_init(self, vkey:str="r_phot"):
        """Instantiate an interpolator on the grid data.

        Parameters
        -----------
        - vkey : str The name of the variable to interpolate (e.g. "r_phot").
        """

        from scipy.interpolate import LinearNDInterpolator

        # gather data
        xyz = np.array([self.data[k].values for k in self.input_keys[2:]]).T
        v = np.array(self.data[vkey].values)

        # instantiate interpolator
        self.interp = LinearNDInterpolator(xyz, v)

