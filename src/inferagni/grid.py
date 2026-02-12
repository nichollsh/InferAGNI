from __future__ import annotations

import pandas as pd
import os
import numpy as np
from copy import deepcopy

from .const import *

class Grid:
    def __init__(self, data_dir):
        self.data, self.input_keys, self.output_keys = self._load_from_dir(data_dir)

    def _load_from_dir(self, data_dir):

        print("Loading data from CSV files...")
        print(f"Data directory: {data_dir}")

        # Read the grid point definition file
        gridpoints_df = pd.read_csv(os.path.join(data_dir, "nogit_points.csv"), sep=',')

        # Read the consolidated results file
        results_df = pd.read_csv(os.path.join(data_dir, "nogit_results.csv"), sep=',')

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
        unique = {}
        for key in self.input_keys:
            unique[key] = np.unique(self.data[key].values)
            print(f"{key:12s}\n\t- {unique[key]}")
        return unique

    def griddata_2d(self,
                    zkey=None,controls=None,
                    resolution=100, method='linear'):

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
        points = np.array([subdata["mass_tot"].values, subdata["r_phot"].values/R_earth]).T
        values = np.array(subdata[zkey].values)

        # Get range on x,y keys
        x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
        y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
        print(f"x range: {x_min:.2f} - {x_max:.2f}") # mass
        print(f"y range: {y_min:.2f} - {y_max:.2f}") # radius

        # Target grid to interpolate to
        grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, resolution),
                                     np.linspace(y_min, y_max, resolution),
                                     indexing='ij'
                                     )


        # Do the interpolation
        grid_z = griddata(points, values, (grid_x, grid_y), method=method)

        # Return the grid for plotting
        return grid_x, grid_y, grid_z
