#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 17:36:14 2023

## Script Breakdown:

### `polynomial_regression.py`:

**Description**: This script performs polynomial regression on MVPA results. It loads, processes, analyzes, and visualizes the data.

**Dependencies**: Ensure you have the following Python libraries installed:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - statsmodels

**Workflow**:

1. **Data Loading**: Data is read from `res_long_format.csv` in the `./data/` directory.
   
2. **Data Processing**:
   - Filtering specific comparisons and ROIs using `filter_data`.
   - Calculating mean accuracy for each ROI, comparison, and subject using `calculate_mean_acc`.
   - Renaming comparisons to be more descriptive using `rename_comparisons`.
   - Ordering and sorting data using `order_and_sort_data`.

3. **Data Analysis**:
   - A mixed linear model is fit using the `add_predicted_accuracy` function, predicting accuracy based on ROI and polynomial (quadratic) changes in ROI.
   - Outputs of this analysis, including coefficients, are saved in the format `stats_[comparison].csv`.

4. **Visualization**: 
   - The data is visualized in subplots based on comparison types using the `plot_data` function.
   - The produced figure is saved as `fig_classify_foveal_rois.pdf`.

**Usage**:

To execute the script, navigate to the `./scripts/PPA/` directory and run:

python polynomial_regression.py

@author: costantino_ai

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def filter_data(df):
    """
    Filters the dataframe for the specified comparisons and ROIs.
    
    Parameters:
        df (DataFrame): The original dataframe.
    
    Returns:
        DataFrame: Filtered dataframe.
    """
    # Define the comparisons and ROIs to include
    comparisons = [
        "('face', 'vehicle')",
        "('female', 'male')",
        "('bike', 'car')",
        "('bike', 'car', 'female', 'male')",
    ]
    rois = [
        "Foveal 0.5°",
        "Foveal 1°",
        "Foveal 1.5°",
        "Foveal 2°",
        "Foveal 2.5°",
        "Foveal 3°",
        "Foveal 3.5°",
        "Foveal 4°"
        ]
    
    # Filter the data
    df_filtered = df[df["comparison"].isin(comparisons)]
    df_filtered = df_filtered[df_filtered["roi"].isin(rois)]
    # Extract the number between space and "°" and convert to float
    df_filtered["roi"] = df_filtered["roi"].str.extract(r' (\d+(\.\d+)?)°')[0].astype(float)

    return df_filtered


def calculate_mean_acc(df):
    """
    Calculate the mean accuracy for each ROI, comparison, and subject.
    
    Parameters:
        df (DataFrame): Input dataframe.
        
    Returns:
        DataFrame: DataFrame with mean accuracies.
    """
    return df.groupby(["roi", "comparison", "sub"])["acc"].mean().reset_index()


def add_predicted_accuracy(df):
    """
    Fits a mixed linear model and adds the predicted accuracy to the dataframe.
    
    Parameters:
        df (DataFrame): The input dataframe.
        
    Returns:
        DataFrame: DataFrame with the predicted accuracy column added.
    """
    df["roi"] = df["roi"].astype(float)
    df["roi_2"] = df["roi"] ** 2
    model = smf.mixedlm("acc ~ roi + roi_2", data=df, groups=df["sub"]).fit()
    print(model.summary())
    df["acc_pred"] = model.predict()
    model.summary().tables[1].to_csv("./res/MVPA/POLY-stats_" + df["comparison"].unique()[0] + ".csv")
    return df


def rename_comparisons(df):
    """
    Renames the 'comparison' column values to more descriptive names.
    
    Parameters:
        df (DataFrame): The input dataframe.
        
    Returns:
        DataFrame: DataFrame with renamed 'comparison' column.
    """
    renaming_dict = {
        "'face', 'vehicle'": "Category (Face vs. Vehicle)",
        "'female', 'male'": "Within-category (Female vs. Male)",
        "'bike', 'car'": "Within-category (Bikes vs. Car)",
        "'bike', 'car', 'female', 'male'": "Sub-category (Bikes vs. Car vs Female vs. Male)",
    }
    
    df["comparison"] = df["comparison"].replace(renaming_dict)
    return df


def order_and_sort_data(df):
    """
    Orders the dataframe based on custom sorting order.
    
    Parameters:
        df (DataFrame): The input dataframe.
        
    Returns:
        DataFrame: Ordered and sorted dataframe.
    """
    sorting_order = {
        "Category (Face vs. Vehicle)": 1,
        "Within-category, faces (Female vs. Male)": 2,
        "Within-category, faces (Bikes vs. Car)": 3,
        "Sub-category (Bikes vs. Car vs Female vs. Male)": 4,
    }
    
    df["sorting_order"] = df["comparison"].map(sorting_order)
    df_sorted = df.sort_values(by="sorting_order")
    
    return df_sorted

import re
def plot_data(df):
    """
    Plots the data in subplots based on comparison types.
    
    Parameters:
        df (DataFrame): The input dataframe.
    """
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 5))
    ax = ax.flatten()
    ax_label = ["A", "B", "C", "D"]

    for i, comp in enumerate(df["comparison"].unique()):
        
        pattern = r"'(.*?)'"
        matches = re.findall(pattern, comp)
        title = f'{matches[0].capitalize()} vs. {matches[1].capitalize()}'
        
        axx = ax[i]
        sns.lineplot(
            data=df[df["comparison"] == comp],
            x="roi",
            y="acc",
            errorbar="se",
            err_style="bars",
            marker="o",
            ax=axx,
        )
        sns.lineplot(
            data=df[df["comparison"] == comp],
            x="roi",
            y="acc_pred",
            marker="o",
            ax=axx,
        )
        axx.set_xticks(np.sort(df["roi"].unique()))
        axx.set_xticklabels(
            [
                r"$0.5\degree$",
                r"$0.1\degree$",
                r"$1.5\degree$",
                r"$2.0\degree$",
                r"$2.5\degree$",
                r"$3.0\degree$",
                r"$3.5\degree$",
                r"$4.0\degree$",
            ],
            rotation=0,
        )
        axx.set_xlabel("")
        axx.set_ylabel("Accuracy")
        axx.set_title(title, fontsize=10)
        axx.annotate(ax_label[i], xy=(-0.1, 1.1), xycoords="axes fraction", fontsize=12)
    plt.tight_layout()
    plt.savefig("./res/MVPA/POLY-fig_classify_foveal_rois.pdf")
    plt.close()


if __name__ == '__main__':
    os.makedirs("./res/PPI", exist_ok=True)
    # Load the data
    data = pd.read_csv("./res/MVPA/res_long_format.csv")
    
    # Process and analyze the data
    filtered_data = filter_data(data)
    mean_data = calculate_mean_acc(filtered_data)
    with_predictions = mean_data.groupby(["comparison"], group_keys=False).apply(add_predicted_accuracy)
    renamed_data = rename_comparisons(with_predictions)
    sorted_data = order_and_sort_data(renamed_data)
    
    # Plot the data
    plot_data(sorted_data)
