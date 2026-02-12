from __future__ import annotations

import pandas as pd
import os

def load_from_dir(data_dir):

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
    input_vars  = list(gridpoints_df.keys())
    output_vars = list(results_df.keys())
    print(f"Input vars:  {input_vars}")
    print(f"Output vars: {output_vars}")

    # Remove redundant variables
    for k in ("index","worker"):
        for v in (input_vars,output_vars):
            if k in v:
                v.remove(k)

    print("Loaded data")
    return data, input_vars, output_vars

