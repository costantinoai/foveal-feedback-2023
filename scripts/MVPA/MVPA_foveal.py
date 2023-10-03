# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:39:23 2021

@author: 45027900
"""

import os
import glob
import sys
import shutil
import random
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import nibabel as nb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from scipy.stats import ttest_1samp, t, sem
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import warnings

# Ignore specific warnings
warnings.filterwarnings(
    "ignore",
    message="Precision loss occurred in moment calculation due to catastrophic cancellation.",
)
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")


def copy_script(out_dir):
    # Copy the script file into the results folder
    script_file_out = os.path.join(out_dir, os.path.basename(__file__))
    shutil.copy(__file__, script_file_out)


def perform_classification(X, y, runs_idx, roi_name, conditions, subject_id):
    """
    Perform data classification using Support Vector Machines (SVM) and
    cross-validation, printing and returning relevant classification metrics.

    Parameters:
    - X (array-like): Input data, where `n_samples` is the number of samples and
                      `n_features` is the number of features.
    - y (array-like): Target values.
    - runs_idx (array-like): Group labels for the samples.
    - roi_name (str): Name of the Region of Interest.
    - conditions (list): Conditions under which classification is performed.
    - subject_id (str): Identifier of the subject being analyzed.

    Returns:
    - acc (float): Average accuracy of the model.
    - conf_mat (array-like): Confusion matrix of the model.
    """

    # Define SVM kernel type
    kernel = "linear"

    # Establish the number of groups and initialize the GroupKFold
    runs_n = len(np.unique(runs_idx))
    gkf = GroupKFold(n_splits=runs_n)

    # Initialize a NumPy array to store performance at each step
    performance_step = np.zeros(gkf.n_splits)

    # Determine the unique labels and initialize the confusion matrix
    labels_n = len(np.unique(y))
    conf_mat = np.zeros((gkf.n_splits, labels_n, labels_n))

    # Output an overview message
    print(f"### CLASSIFICATION - {subject_id} {conditions} ###")

    # Loop through each split of the data into training and test sets
    for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=runs_idx)):
        # Output the current step and related indices information
        print(f"## {subject_id} ## {roi_name} ## STEP: {i+1} ##")
        print("Indices of train-samples:", train_idx.tolist())
        print("Indices of test-samples:", test_idx.tolist())
        print("... corresponding to the following runs:", runs_idx[test_idx].tolist())

        # Define training and test sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Instantiate and fit the classifier on the training data
        clf = SVC(kernel=kernel, random_state=42).fit(X_train, y_train)

        # Predict labels for test and training sets
        y_hat = clf.predict(X_test)
        y_hat_training = clf.predict(X_train)

        # Store performance metrics and compute the confusion matrix
        performance_step[i] = clf.score(X_test, y_test)
        conf_mat[i] = confusion_matrix(y_test, y_hat)
        
        # Output predictions and performance for the current step
        print(
            f"TRAINING BATCH: \t{list(y_train)} \nTRAINING PREDICTED: {list(y_hat_training)} \nTARGET: \t\t{list(y_test)} \nPREDICTED: \t{list(y_hat)}"
        )
        print(f"TRAINING ACC: {performance_step[i]}, TEST ACC: {round(performance_step[i],2)}, STEP: {i+1}\n")

    # Calculate and output the average accuracy
    acc = np.average(performance_step)
    print(f"TOTAL ACCURACY: {acc}\n\n")

    # Return accuracy and the confusion matrix
    return performance_step, conf_mat,


def initialize_graph_dataframe(params):
    """
    Initialize a DataFrame for graphs.

    Args:
    params (dict): Dictionary containing parameters like 'rois' and 'conditions_list'.

    Returns:
    DataFrame: A pandas DataFrame initialized for graph data.
    """
    columns = ("avg", "std", "comparison", "roi", "cimin", "cimax", "p")
    graph_df = pd.DataFrame(None, columns=columns)
    graph_df["roi"] = params["rois"] * len(params["conditions_list"])
    graph_df["comparison"] = [
        comparison for comparison in params["conditions_list"] for _ in range(len(params["rois"]))
    ]

    return graph_df


def initialize_reports_and_results(params):
    """
    Initialize reports and results structures.

    Args:
    params (dict): Dictionary containing parameters like 'rois' and 'conditions_list'.

    Returns:
    tuple: A tuple containing the reports_all and results dictionaries.
    """
    results = {
        conditions: {
            roi: {
                "acc": [],
                "confmat": [],
                "T-test": {"t": float, "p": float, "avg": float, "std": float, "d": float},
            }
            for roi in params["rois"]
        }
        for conditions in params["conditions_list"]
    }

    return results


def get_beta_subdir(conditions, pipeline_dir):
    """
    Get the sub-directory for beta values based on conditions.

    Args:
    conditions (tuple): A tuple containing condition strings.
    pipeline_dir (str): Base pipeline directory string.

    Returns:
    str: A string indicating the sub-directory for beta values.
    """
    if ("face" in conditions) or ("vehicle" in conditions):
        return pipeline_dir + "_fv"
    else:
        return pipeline_dir + "_all"


def generate_timestamp():
    """
    Generate a formatted timestamp.

    Returns:
    str: A formatted timestamp string.
    """
    ts = str(datetime.now(tz=None)).split(" ")
    ts[1] = ts[1].split(".")[0].split(":")
    ts[1] = ts[1][0] + ts[1][1] + ts[1][2]
    return "_".join(ts)


def setup_logging(params, conditions):
    """
    Setup logging based on parameters.

    Args:
    params (dict): Dictionary containing parameters.
    conditions (tuple): A tuple containing condition strings.

    Returns:
    str: The path to the logfile.
    """
    ts = generate_timestamp()
    logfile = os.path.join(params["out_dir"], ts + "_" + "-".join(list(conditions)) + ".txt")

    print(f"logfile directory: {logfile}")
    if params["log"]:
        os.makedirs(params["out_dir"], exist_ok=True)
        sys.stdout = open(logfile, "w")

    return logfile, ts


def mean_confidence_interval(data, confidence=0.95):
    """
    Compute the mean confidence interval for the given data.

    Parameters:
    - data (array-like): Input data for which the confidence interval is computed.
    - confidence (float, optional): Confidence level. Default is 0.95.

    Returns:
    - m (float): Mean of the data.
    - m - h (float): Lower bound of the confidence interval.
    - m + h (float): Upper bound of the confidence interval.
    - se (float): Standard error of the mean.
    """

    # Convert data to numpy array and compute its length
    a = 1.0 * np.array(data)
    n = len(a)

    # Calculate mean and standard error
    m, se = np.mean(a), sem(a)

    # Compute the margin of error
    h = se * t.ppf((1 + confidence) / 2.0, n - 1)

    return m, m - h, m + h, se


def get_subject_id(sub_id):
    """
    Get a formatted subject ID string.

    Args:
    sub_id (int): Subject number.

    Returns:
    str: A formatted subject ID string.
    """
    return "sub-" + str(sub_id).zfill(2)


def get_mask_files(params, sub):
    """
    Retrieve mask file paths for a given subject.

    Args:
    params (dict): Dictionary containing parameters.
    sub (str): Formatted subject ID.

    Returns:
    dict: A dictionary with ROI names as keys and mask file paths as values.
    """
    return {
        roi: glob.glob(
            os.path.join(params["mask_root"], roi, "Linear", "resampled_nearest", sub + "*FOV*")
        )[0]
        for roi in params["rois"]
    }


def get_beta_dataframes(beta_loc):
    """
    Generate dataframes related to beta values.

    Args:
    beta_loc (str): Path location to beta values.

    Returns:
    DataFrame: A pandas DataFrame with information about beta values.
    """
    # Initialize dataframe
    betas_df = pd.DataFrame(
        None,
        columns=("beta_path", "spm_filename", "condition", "run", "bin", "array"),
    )
    betas_df["beta_path"] = sorted(glob.glob(os.path.join(beta_loc, "beta*.?ii")))

    # Load SPM.mat data
    mat = loadmat(os.path.join(beta_loc, "SPM.mat"))

    # Extract relevant information from the mat data
    matching_indices = [
        beta_id
        for beta_id in range(len(betas_df["beta_path"]))
        if str(mat["SPM"]["Vbeta"][0][0][0][beta_id][0][0])
        == os.path.basename(betas_df["beta_path"][beta_id])
    ]

    betas_df["spm_filename"] = [
        str(mat["SPM"]["Vbeta"][0][0][0][idx][0][0]) for idx in matching_indices
    ]
    betas_df["condition"] = [
        str(mat["SPM"]["Vbeta"][0][0][0][idx][5][0]).split(" ")[-1].split("*")[0]
        for idx in matching_indices
    ]
    betas_df["run"] = [
        str(mat["SPM"]["Vbeta"][0][0][0][idx][5][0]).split(" ")[-2].split("(")[-1][0]
        for idx in matching_indices
    ]
    betas_df["bin"] = [
        str(mat["SPM"]["Vbeta"][0][0][0][idx][5][0]).split(" ")[-1].split("(")[-1][0]
        for idx in matching_indices
    ]
    betas_df["array"] = [np.array(nb.load(beta).get_fdata()) for beta in betas_df["beta_path"]]

    return betas_df


def filter_and_sort_betas(betas_df, conditions):
    """
    Filter beta values based on conditions and sort the dataframe.

    Args:
    betas_df (DataFrame): Original dataframe with beta values.
    conditions (tuple): Conditions to filter by.

    Returns:
    DataFrame: Filtered and sorted DataFrame.
    """
    betas_df = betas_df[betas_df.condition.str.match("|".join(list(conditions)))]
    return betas_df.sort_values(["condition", "run"], axis=0).reset_index(drop=True)


def demean_data_per_run(betas_df):
    """
    Demean the beta data for each run.

    Args:
    betas_df (pd.DataFrame): Dataframe containing the beta data arrays and runs.

    Returns:
    pd.DataFrame: Demeaned beta data per run.
    """
    all_betas_data = np.array([array for array in betas_df["array"]])
    runs = np.unique(betas_df["run"].values)

    for run in runs:
        betas_idx = betas_df["run"] == run
        run_array = all_betas_data[betas_idx.values]
        run_array_avg = np.array([np.mean(run_array, 0) for _ in range(run_array.shape[0])])
        run_array_demeaned = run_array - run_array_avg

        list_id = 0
        for index, beta_idx in enumerate(betas_idx):
            if beta_idx:
                betas_df.iloc[index]["array"] = run_array_demeaned[list_id]
                list_id += 1

    return betas_df


def prepare_dataset_for_classification(betas_df, mask_files, roi_name):
    """
    Prepare and preprocess the dataset for classification.

    Args:
    betas_df (pd.DataFrame): Dataframe containing the beta data arrays.
    mask_files (dict): Dictionary containing ROI names and corresponding mask file paths.

    Returns:
    tuple: Three arrays - X (features), y (labels), runs_idx (run indices).
    """
    mask = nb.load(mask_files[roi_name]).get_fdata() > 0
    betas_masked = [np.nan_to_num(df_array[mask], False, 0.0) for df_array in betas_df["array"]]

    X = betas_masked
    y = list(betas_df["condition"])
    runs_idx = list(betas_df["run"])

    comp = list(zip(X, y, runs_idx))
    random.seed(42)
    random.shuffle(comp)
    X, y, runs_idx = zip(*comp)

    return np.array(X), np.array(y), np.array(runs_idx)


def calculate_ttest_statistics(params, conditions, roi):
    """
    Calculate t-test statistics for a given ROI and conditions.

    Args:
    params (dict): Parameters dictionary containing results.
    conditions (str): Conditions for which t-test is being calculated.
    roi (str): Region of interest.

    Returns:
    dict: A dictionary containing t-test statistics (t, p, avg, std, and d values).
    """
    labels_n = len(conditions)
    
    data = np.array(params["results"]["classification"][conditions][roi]["acc"])
    data = np.average(data, axis=1)
    
    # Perform t-test
    t, p = ttest_1samp(
        data,
        popmean= 1 / labels_n,
        alternative="greater",
    )
    avg, std = np.average(data), np.std(data)

    d = (avg - (1.0 / len(conditions))) / std

    return {"t": t, "p": p, "avg": avg, "std": std, "d": d}


def print_ttest_statistics(stats, roi):
    """
    Print the t-test statistics.

    Args:
    stats (dict): Dictionary containing t-test statistics.
    roi (str): Region of interest.

    Returns:
    None
    """
    print(
        f'ROI {roi}: t = {round(stats["t"], 3)}, p = {round(stats["p"], 3)}, avg = {round(stats["avg"], 3)}, std = {round(stats["std"], 3)}'
    )


def update_graph_dataframe(params, conditions, roi, stats):
    """
    Update the graph dataframe with t-test statistics.

    Args:
    - graph_df (dataframe): df containing various information including graph dataframe.
    - conditions (str): Conditions for which t-test is being calculated.
    - roi (str): Region of interest in which the statistics were computed.
    - stats (dict): Dictionary containing t-test statistics such as p-value, standard deviation, and effect size (d).

    Returns:
    None. The graph dataframe within `params` is updated in place.
    """

    # Filtering rows in the dataframe that match the given conditions and ROI
    condition_roi_filter = (params["results"]["graph_df"]["comparison"] == conditions) & (
        params["results"]["graph_df"]["roi"] == roi
    )

    # Calculating mean and confidence intervals for accuracy values
    avg, cimin, cimax, sem_ = mean_confidence_interval(
        np.mean(params["results"]["classification"][conditions][roi]["acc"], axis=1)
    )

    # Updating the graph dataframe with the calculated statistics
    params["results"]["graph_df"].loc[condition_roi_filter, ["avg", "cimin", "cimax", "sem"]] = (
        avg,
        cimin,
        cimax,
        sem_,
    )
    params["results"]["graph_df"].loc[condition_roi_filter, ["p", "std", "d", "t"]] = (
        stats["p"],
        stats["std"],
        stats["d"],
        stats["t"],
    )

    return params["results"]["graph_df"]


def prepare_data_for_plotting(params):
    """
    Prepare the dataframe for plotting by collating relevant data from the provided parameters.

    Args:
    - params (dict): Dictionary containing conditions, ROIs, results, and other relevant parameters.

    Returns:
    - data_all (DataFrame): A structured dataframe containing columns for comparison, ROI, subject number, accuracy, and ordering.
    """

    # Once all rows are accumulated in the rows_list, convert it into a pandas DataFrame.
    # This approach ensures that we construct the DataFrame only once, making it efficient.
    agg_data = params["results"]["graph_df"]

    # Assign order numbers to each row in the DataFrame based on the comparison type.
    # The 'map' function maps each value in the 'comparison' column to its corresponding order number.
    agg_data["order"] = agg_data["comparison"].apply(get_order)

    # Sort the DataFrame based on the 'roi' and 'order' columns.
    # This ensures that the data is ordered first by the region of interest, and then by the comparison type.
    agg_data = agg_data.sort_values(["roi", "order"], axis=0)

    # Convert 'comparison' column to string
    agg_data["comparison"] = agg_data["comparison"].astype(str)

    # Reset the index of the DataFrame.
    # This ensures that the row indices are consecutive integers, starting from 0.
    agg_data = agg_data.reset_index(drop=True)

    return agg_data


def plot_data(params):
    agg_data = prepare_data_for_plotting(params)
    ax = plot_bars(agg_data)
    plot_error_bars(ax, agg_data)
    annotate_significance(params, ax, agg_data)
    save_and_show(params)


def plot_bars(agg_data):
    """
    Plot data as bar plots using seaborn, focusing on classifier accuracy.

    Parameters:
    - agg_data (DataFrame): The aggregated (mean) dataset containing data points to be plotted.
                            Expected columns include "roi", "acc", and "comparison".

    Returns:
    - ax (matplotlib.axes.Axes): The Axes object of the generated plot.
    """

    # Setting font scale for better readability
    sns.set(font_scale=2)
    # Create a figure with specified dimensions
    fig_dims = (16, 10)
    fig, ax = plt.subplots(figsize=fig_dims)
    # Bar plot of accuracy against ROI, categorized by comparison types
    sns.barplot(x="roi", y="avg", hue="comparison", data=agg_data, ax=ax, errorbar=None)

    # Setting legend with custom descriptions
    legend_labels = [
        "Category (Face vs. Vehicle)",
        "Within-category, faces (Female vs. Male)",
        "Within-category, vehicles (Bike vs. Car)",
        "Sub-category (Bike vs. Car vs. Female vs. Male)",
    ]
    L = plt.legend()
    for idx, label in enumerate(legend_labels):
        L.get_texts()[idx].set_text(label)

    # Adjusting axes limits, labels, title, and adding reference horizontal lines
    plt.ylim((0.2, 1))
    plt.axhline(0.5, alpha=0.8, color="blue", ls="--", zorder=4)  # Blue dashed line at y=0.5
    plt.axhline(0.25, alpha=0.8, color="red", ls="--", zorder=4)  # Red dashed line at y=0.25
    ax.set_ylabel("acc")
    ax.set_xlabel("ROI")
    ax.set_title("Classifier accuracy")

    return ax


def plot_error_bars(ax, data_all):
    """
    Overlay error bars on the given Axes based on the provided data.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object of the plot to which error bars should be added.
    - data_all (DataFrame): The dataset containing data points. Expected columns include "roi", "acc", and "comparison".

    Returns:
    None. The error bars are added to the input `ax` plot.
    """

    # Calculate the x positions for the error bars based on the existing tick positions
    x_pos = np.array(
        list(
            [
                ax.get_xticks()[i] - 0.3,
                ax.get_xticks()[i] - 0.1,
                +ax.get_xticks()[i] + 0.1,
                ax.get_xticks()[i] + 0.3,
            ]
            for i in range(len(ax.get_xticks()))
        )
    ).flatten()

    # Calculate mean y-positions and error bars based on the data
    y_pos = list(data_all["avg"])
    yerr = list(data_all["sem"])

    # Add error bars to the plot
    plt.errorbar(
        x=x_pos, y=y_pos, yerr=yerr, fmt="none", c="black", capsize=5  # , palette="Paired"
    )

# Order the graph dataframe based on the comparison ordering
def get_order(comparison):
    return PLOT_COMPARISONS_ORDER.get(comparison, -1)
    
def annotate_significance(params, ax, data_all):
    """
    Annotate a given plot with significance markers based on the parameters and data provided.

    This function uses a predetermined mapping of comparisons to order the significance annotations.
    For each data point in the plot, based on the p-value, it annotates the graph with stars indicating the level of significance.

    Parameters:
    - params (dict): Dictionary containing graph-related parameters. Must have a key "graph_df" with associated data.
    - ax (matplotlib.axes.Axes): The Axes object of the plot where annotations should be added.
    - data_all (DataFrame): The dataset containing data points to be plotted and their significance levels.

    Returns:
    None. The plot (`ax`) is modified in-place with added annotations.
    """

    # Order the graph dataframe based on the comparison ordering
    params["results"]["graph_df"]["order"] = params["results"]["graph_df"]["comparison"].apply(
        get_order
    )
    params["results"]["graph_df"] = (
        params["results"]["graph_df"].sort_values(["roi", "order"], axis=0).reset_index(drop=True)
    )

    # Determine x positions for annotations
    x_pos = np.array(
        list(
            [
                ax.get_xticks()[i] - 0.3,
                ax.get_xticks()[i] - 0.1,
                +ax.get_xticks()[i] + 0.1,
                ax.get_xticks()[i] + 0.3,
            ]
            for i in range(len(ax.get_xticks()))
        )
    ).flatten()

    # Calculate mean y-positions and error bars based on the data
    y_pos = list(data_all["avg"])
    yerr = list(data_all["sem"])

    # Loop through p-values to determine the significance annotation
    for i, p in enumerate(params["results"]["graph_df"]["p"]):
        # Determine the annotation string based on significance levels
        if p < 0.001:
            displaystring = r"***"
        elif p < 0.01:
            displaystring = r"**"
        elif p < 0.05:
            displaystring = r"*"
        else:
            displaystring = r""

        # Determine the y-position for the annotation
        height = yerr[i] + y_pos[i] + 0.05
        if height > 1:
            height = 1

        # Add the annotation to the plot
        plt.text(x_pos[i], height, displaystring, ha="center", va="center", size=20)


def save_and_show(params):
    """Save the figure and display it."""
    if params["log"]:
        plt.savefig(os.path.join(params["out_dir"], "all_sem.png"))
        print(f"STEP: figures saved in {params['out_dir']}")
    plt.show()


def save_results_to_file(params):
    """
    Save results to a pickle file if logging is enabled.

    Given various datasets and conditions, this function constructs a filename from these inputs,
    serializes the data into a pickle file, and stores it in the specified output directory.

    Returns:
    None. If log is True, it saves a pickle file with the combined data in the specified directory.
    """

    # Only execute the following if logging is enabled
    if params["log"]:
        # Construct the filename using various inputs for clarity and traceability
        pkl_file = os.path.join(
            params["out_dir"], params["ts"] + "_" + "-".join(list(conditions)) + ".pkl"
        )

        # Serialize and save the data into the constructed pickle file
        with open(pkl_file, "wb") as fp:
            pickle.dump(params, fp)

        # Print the location of the saved file for the user
        print(f"STEP: results dictionary saved as {pkl_file}")


def print_stats_for_rois(params):
    """
    Print detailed statistical information for each ROI (Region of Interest) based on various conditions.

    For each ROI in the given list, this function iterates over the conditions, retrieves the associated
    statistics, and formats them in a readable manner for display.

    Parameters:
    - roi_list (list of str): List of Regions of Interest (ROIs) for which statistics need to be printed.
    - results (dict): Nested dictionary containing statistical results.
                     Structure: {condition: {roi: {statistic_method: {statistic_name: value}}}}.
    - conditions_list (list of str): List of conditions to consider when retrieving statistics from the results.

    Returns:
    None. Prints the statistics for each ROI based on the specified conditions.
    """

    # Iterate over each ROI to display its statistics
    for roi in params["rois"]:
        print(f"\n{roi}")  # Print the ROI name

        # Iterate over the specified conditions to retrieve and display the statistics for the current ROI
        for comparison in params["conditions_list"]:
            # Filter the dataframe based on the current comparison and ROI
            filtered_df = params["results"]["graph_df"][
                (params["results"]["graph_df"]["comparison"] == comparison)
                & (params["results"]["graph_df"]["roi"] == roi)
            ]

            # Fetch values
            t_val = round(
                filtered_df.iloc[0]["t"], 2
            )  # assuming 'd' column corresponds to "T-test"["t"]
            p_val = filtered_df.iloc[0]["p"]
            d_val = round(filtered_df.iloc[0]["d"], 2)

            # Convert the p-value to a formatted string for better readability
            p_string = get_p_string(p_val)

            # Get comparison details such as name and bar color
            comp_name, bar_color = get_comparison_info(comparison)

            # Print the formatted statistical result for the current ROI and condition
            print(f"{comp_name}, {bar_color} bar: $t(23) = {t_val}, p {p_string}, d = {d_val}$;")


def get_p_string(p_val):
    """
    Convert p-value into a formatted string.
    """
    if p_val >= 0.05:
        return "= n.s."
    elif 0.01 <= p_val < 0.05:
        return "< .05"
    elif 0.001 <= p_val < 0.01:
        return "< .01"
    elif 0.0001 <= p_val < 0.001:
        return "< .001"
    elif p_val < 0.0001:
        return "< .0001"
    else:
        raise Exception("p is not recognized")


def get_comparison_info(comparison):
    """
    Returns a descriptive name and associated bar color for the given comparison.
    """
    comp_dict = {
        ("face", "vehicle"): ("Face vs. Vehicle", "blue"),
        ("female", "male"): ("Female vs. Male", "orange"),
        ("bike", "car"): ("Bike vs. Car", "green"),
        ("bike", "car", "female", "male"): ("Bike vs. Car vs. Female vs. Male", "red"),
    }
    return comp_dict.get(comparison, ("Unknown", "gray"))


def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("\t" * indent + f'"{str(key)}": ')
            print_dict(value, indent + 1)
        else:
            print("\t" * indent + f'"{str(key)}": "{str(value)}"')


def env_check():
    print("Numpy Version:", np.__version__)
    print("Pandas Version:", pd.__version__)
    print("nibabel Version:", nb.__version__)
    print("Seaborn Version:", sns.__version__)
    print(
        "sklearn Version:",
        GroupKFold.__module__.split(".")[0] + " " + GroupKFold.__module__.split(".")[1],
    )  # Assuming the version is consistent across sklearn's modules


def set_seed(seed=42):
    random.seed(seed)  # Seed for Python's random module
    np.random.seed(seed)  # Seed for NumPy

def extract_data_to_dataframe(params):
    """
    Extract data from nested dictionaries and create a pandas DataFrame.
    
    Given a nested dictionary structure from `params` with keys ['results']['classification'], 
    this function processes the data and transforms it into a long format DataFrame 
    with columns: 'sub', 'roi', 'comparison', 'fold', and 'acc'.
    
    If `log` parameter in `params` is set to True, the function will save the dataframe as 
    a CSV file in the directory specified by `out_root` with the filename `res_long_format_FOVEAL.csv`.
    
    Parameters:
    - params (dict): A dictionary containing the nested structure with classification results.
                     Expected structure:
                     {
                         'results': {
                             'classification': {
                                 'comparison1': {
                                     'roi1': [[acc11, acc12, ...], [acc21, acc22, ...], ...],
                                     ...
                                 },
                                 ...
                             }
                         }
                     }
    
    Returns:
    - pd.DataFrame: A long format dataframe with columns 'sub', 'roi', 'comparison', 'fold', and 'acc'.
    """
    
    rows = []

    # Extract the classification results from the params dictionary
    results = params['results']['classification']

    # Iterate over each comparison in the results
    for comparison in results.keys():
        comparison_dict = results[comparison]
        
        # Iterate over each region of interest (roi) for the current comparison
        for roi in comparison_dict.keys():
            roi_dict = comparison_dict[roi]['acc']
            
            # Iterate over each id in the roi dictionary
            for id_ in range(len(roi_dict)):
                sub_id = id_ + 2  # Compute the sub_id by adding 2 to the current id_
                sub_row = roi_dict[id_]
                
                # Iterate over each fold for the current id in roi dictionary
                for fold_n in range(len(sub_row)):
                    fold_acc = sub_row[fold_n]  # Extract the fold accuracy for the current fold
                    rows.append([sub_id, roi, comparison, fold_n, fold_acc])  # Append data to rows list

    # Convert the rows list to a pandas DataFrame
    results_df_long = pd.DataFrame(data=rows, columns=['sub', 'roi', 'comparison', 'fold', 'acc'])
    
    # Check if log parameter is True
    if params['log'] == True:
        # Construct the complete file path for the CSV file
        file_path = os.path.join(params['out_dir'], 'res_long_format_FOVEAL.csv')
        
        # Save the dataframe as CSV
        results_df_long.to_csv(file_path, index=False)
    
    return results_df_long

# Define the new root directory
derivatives_bids = "/media/costantino_ai/Samsung_T5/2exp_fMRI/Exp/Data/Data/fmri/BIDS/derivatives"

# Parameters
params = {
    "out_root": r"./res/MVPA",
    "beta_root": os.path.join(derivatives_bids, "SPM"),
    "mask_root": os.path.join(
        derivatives_bids, "masks", "original", "original_realignedMNI", "fov"
    ),
    "pipeline_dir": "RSA_blocks_1_lev_6HMP",
    "conditions_list": [
        ("face", "vehicle"),
        ("female", "male"),
        ("bike", "car"),
        ("bike", "car", "female", "male"),
    ],
    "conditions_all": ("bike", "car", "female", "male", "rest"),
    "log": True,  # Set this to True if you want to log results,
    "seed": 42,
    "results": {},
}

params["rois"] = [
    os.path.split(path)[-1] for path in glob.glob(os.path.join(params["mask_root"], "?.?"))
]
params["out_dir"] = os.path.join(params["out_root"], "foveal")

# Define a dictionary that maps comparison conditions to specific order numbers.
# This is useful for determining the order in which data points should be plotted.
PLOT_COMPARISONS_ORDER = {
    ("face", "vehicle"): 0,
    ("female", "male"): 1,
    ("bike", "car"): 2,
    ("bike", "car", "female", "male"): 3,
}

if params["log"]:
    # Create the output folder
    os.makedirs(params["out_dir"], exist_ok=True)

    # Copy the current script to the output folder
    copy_script(params["out_dir"])

# Initialize graphs dataframe
params["results"]["graph_df"] = initialize_graph_dataframe(params)
params["results"]["classification"] = initialize_reports_and_results(params)

for conditions in params["conditions_list"]:
    # Get paths of beta images used for classification
    beta_subdir = get_beta_subdir(conditions, params["pipeline_dir"])

    # Setup logging for current comparison
    logfile, params["ts"] = setup_logging(params, conditions)
    
    # Print env check and parameters
    set_seed(params["seed"])
    env_check()
    print_dict(params)

    print(f"START: {datetime.now()} - Logging = {params['log']}")
    print(f"STEP: pipeline_dir = {params['pipeline_dir']}")

    # SELECT BETAS
    for sub_id in range(2, 26):
        sub = get_subject_id(sub_id)
        beta_loc = os.path.join(params["beta_root"], beta_subdir, sub)

        print(f"STEP: {sub} - starting classification for {conditions}")
        print("STEP: loading masks...", end="\r")
        mask_files = get_mask_files(params, sub)
        print("done!")

        print("STEP: generating beta mapping dataframe...", end="\r")
        betas_df = get_beta_dataframes(beta_loc)
        betas_df = filter_and_sort_betas(betas_df, conditions)
        print("done!")

        print("STEP: removing average pattern of all the conditions to classify...", end="\r")
        betas_df = demean_data_per_run(betas_df)
        print("done!")

        for roi_name in mask_files.keys():
            
            print("STEP: preparing dataset for classification (zeroing NaNs, shuffling)...", end="\r")
            X, y, runs_idx = prepare_dataset_for_classification(betas_df, mask_files, roi_name)
            print("done!")

            acc_long, conf_mat_long = perform_classification(X, y, runs_idx, roi_name, conditions, sub)
            
            params["results"]["classification"][conditions][roi_name]["acc"].append(acc_long)
            params["results"]["classification"][conditions][roi_name]["confmat"].append(conf_mat_long)

    # PERFORM T-TEST
    for roi in params["rois"]:
        ttest_stats = calculate_ttest_statistics(params, conditions, roi)
        print_ttest_statistics(ttest_stats, roi)
        params["results"]["graph_df"] = update_graph_dataframe(params, conditions, roi, ttest_stats)

print("STEP: preparing and plotting figures...", end="\r")

# PLOT DATA
plot_data(params)

# SAVE LONG FORMAT CSV
long_df = extract_data_to_dataframe(params)            

# SAVE PICKLE FILE
save_results_to_file(params)

## PRINT STATS
print_stats_for_rois(params)
