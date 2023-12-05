#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:39:13 2023

Script for preprocessing and analyzing fMRI data.

## Script Overview:

This script loads and processes fMRI data, conducts statistical analyses, and saves the results. It performs the following tasks:

1. **Loading ROI Images**: Loads region of interest (ROI) images for a given subject from specified directories.

2. **Loading Functional Images**: Loads functional MRI (fMRI) images for a subject and run.

3. **Loading and Processing Events**: Loads and processes event data for the subject and run.

4. **Creating Design Matrix**: Creates a first-level design matrix for the fMRI data using Nilearn.

5. **Applying PCA and Extracting Signal**: Applies Principal Component Analysis (PCA) to the data and extracts signals using binary masks.

6. **Performing PPI**.

6. **Saving Results**: Saves the final DataFrame to a CSV file.

## Dependencies:

- numpy
- pandas
- nibabel
- nilearn
- sklearn
- statsmodels
- scipy

Run the `main()` function to execute the entire data processing and analysis pipeline. Make sure to specify the appropriate directory paths and subjects before running the script.

"""
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import make_first_level_design_matrix
from sklearn.decomposition import PCA
import os
import scipy.io
import warnings
import concurrent.futures

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_roi_images(subject, dir_rois):
    """
    Load Region of Interest (ROI) images for the given subject.

    Parameters:
    - subject (str): Identifier for the subject.
    - dir_rois (dict): Directory paths for the ROIs.

    Returns:
    - dict: Numpy arrays for each of the ROIs.
    """
    rois = {}
    paths = {
        "fov": glob.glob(os.path.join(dir_rois, "sub-" + subject, "*label-FOV+20_roi.nii"))[0],
        "per": glob.glob(os.path.join(dir_rois, "sub-" + subject, "*label-PER_roi.nii"))[0],
        "opp": glob.glob(os.path.join(dir_rois, "sub-" + subject, "*label-OPP_roi.nii"))[0],
        "ffa": glob.glob(os.path.join(dir_rois, "sub-" + subject, "*label-FFA_roi.nii"))[0],
        "loc": glob.glob(os.path.join(dir_rois, "sub-" + subject, "*label-LOC_roi.nii"))[0],
        "a1": glob.glob(os.path.join(dir_rois, "sub-"+subject, '*label-A1_roi.nii'))[0]
    }

    for roi_name, path in paths.items():
        data = nib.load(path).get_fdata()
        data = np.reshape(data, (-1, 1)).flatten().T
        data[data != 0] = 1  # Ensure binary masks
        rois[roi_name] = data

    return rois


def load_functional_images(subject, run, dir_func):
    """
    Load functional images for the given subject and run.

    Parameters:
    - subject (str): Identifier for the subject.
    - run (int): Run number.
    - dir_func (str): Directory path for the functional images.

    Returns:
    - np.array: Functional image data.
    """
    path = os.path.join(
        dir_func,
        f"sub-{subject}_task-exp_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    if os.path.exists(path):
        return nib.load(path).get_fdata()
    else:
        return None


def load_and_process_events(subject, run, dir_events):
    """
    Load and process event data for the given subject and run.

    Parameters:
    - subject (str): Identifier for the subject.
    - run (int): Run number.
    - dir_events (str): Directory path for the events data.

    Returns:
    - pd.DataFrame: Processed event data.
    """
    path = os.path.join(
        dir_events,
        f"sub-{subject}_task-exp_run-run-{run}_desc-bike+car+female+male_SPMmulticondition.mat",
    )
    events_data = scipy.io.loadmat(path)
    ev_names, ev_onsets, ev_durations = (
        events_data["names"],
        events_data["onsets"],
        events_data["durations"],
    )

    events = []
    for i in range(ev_names[0, :].shape[0]):
        names = [ev_names[0, i][0]] * ev_onsets[0, i][0].shape[0]
        onsets = ev_onsets[0, i][0]
        durations = ev_durations[0, i][0]
        events.append(pd.DataFrame({"trial_type": names, "onset": onsets, "duration": durations}))

    events = pd.concat(events).sort_values("onset").reset_index(drop=True)
    events["trial_type"] = "y_p"  # Set all to 'task'
    return events


def make_design_matrix(events, motion, frame_times, hrf_model="glover"):
    """
    Create a first level design matrix using nilearn.

    Parameters:
    - events (pd.DataFrame): Event data.
    - motion (np.array): Motion data.
    - frame_times (np.array): Frame times for the scans.
    - hrf_model (str, optional): Hemodynamic Response Function model. Default is 'glover'.

    Returns:
    - pd.DataFrame: Design matrix.
    """
    add_reg_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    return make_first_level_design_matrix(
        frame_times,
        events,
        drift_model="polynomial",
        drift_order=3,
        add_regs=motion,
        add_reg_names=add_reg_names,
        hrf_model=hrf_model,
    )


def apply_pca_and_extract_signal(data, mask):
    """
    Apply PCA to the data and extract the signal using the mask.

    Parameters:
    - data (np.array): Data to which PCA is applied.
    - mask (np.array): Mask used for extraction.

    Returns:
    - np.array: Extracted signal after PCA.
    - float: Explained variance ratio by the PCA component.
    """
    # Mask the data
    masked_data = data.copy()
    masked_data[:, mask == 0] = 0

    # Reduce dimension by removing masked columns
    idx = np.argwhere(np.all(masked_data != 0, axis=0))[:, 0]
    masked_data = masked_data[:, idx]

    # Apply PCA
    pca = PCA(n_components=1, whiten=True)
    pca.fit(masked_data)
    transformed_data = pca.transform(masked_data).flatten()

    return transformed_data, pca.explained_variance_ratio_[0]


def process_run(sub, run, bids_root, dir_rois, sub_fmriprep_dir, rois):
    
        print(f"STEP: Processing subject {sub}, run {run}\n")
    
        # Load functional images (and skip iteration if no func is found)
        func_img_data = load_functional_images(sub, run, sub_fmriprep_dir)

        if func_img_data is None:
            print(f"WARNING: No functional image found for {sub}, run {run}.. SKIPPING.")
            pass

        n_scans = func_img_data.shape[-1]  # Number of time frames in the functional data
        # Reshape to (n_samples, n_features) where a sample
        # is a TR and a feature is a voxel.
        func_img = np.reshape(
            func_img_data, (-1, n_scans)
        ).T  
        # print(f"\tLoaded functional data for subject {sub}, run {run} with {n_scans} scans")

        # Load events
        sub_bids_dir = os.path.join(bids_root, "sub-" + sub, "func")
        events = load_and_process_events(sub, run, sub_bids_dir)
        # print(f"\tLoaded event data for subject {sub}, run {run}")

        # Load motion confounds
        motion = np.loadtxt(
            os.path.join(
                sub_fmriprep_dir, f"sub-{sub}_task-exp_run-{run}_desc-6HMP_regressors.txt"
            )
        )
        # print(f"\tLoaded motion confounds for subject {sub}, run {run}")

        # Verify that the motion data's length matches the number of time frames in the functional data
        assert (
            motion.shape[0] == n_scans
        ), "Mismatch between number of time frames in the functional data and motion regressors."

        # Create design matrix
        tr = 2.0
        frame_times = np.arange(n_scans) * tr
        X = make_design_matrix(events, motion, frame_times)
        # print(f"\tCreated design matrix for subject {sub}, run {run}")

        # Apply PCA and extract signals
        for roi_name, mask in rois.items():
            signal, exp_var = apply_pca_and_extract_signal(func_img, mask)
            X[f"y_{roi_name}"] = signal
            X[f"exp_var_{roi_name}"] = exp_var
            # print(f"\tApplied PCA for ROI {roi_name} in subject {sub}, run {run}")

        X["sub"] = sub
        X["run"] = run

        # PPI terms
        ppi_mult = 2 * X["y_p"] - 1
        for roi_name in rois:
            X[f"y_ppi_{roi_name}"] = ppi_mult * X[f"y_{roi_name}"]
            # print(f"\tCalculated PPI term for ROI {roi_name} in subject {sub}, run {run}")

        X["TR"] = X.index // 2
        
        print(f"DONE: Finished processing subject {sub}, run {run}\n")
        # print("-" * 40)
        
        return X

if __name__ == '__main__':

    # Run the main function
    # bids_root = "/data/projects/fov/data/BIDS"
    bids_root = "../data/BIDS"
    dir_rois = os.path.join(bids_root, "derivatives", "rois")
    fmriprep_dir = os.path.join(bids_root, "derivatives", "fMRIprep")
    out_dir = "../res/PPI/"
    os.makedirs(out_dir, exist_ok=True)

    # List of subjects
    subjects = [str(i).zfill(2) for i in range(2, 26)]
    # subjects = [str(i).zfill(2) for i in range(2,4)]

    # TODO: these two subs throw an error for some reason
    subjects = [x for x in subjects if x not in ['09', '12']]

    X_rec = []

    for sub in subjects:
        # Load ROIs
        rois = load_roi_images(sub, dir_rois)
        print("=" * 40)

        # Identify directories and files
        expression = '*_task-exp_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
        sub_fmriprep_dir = os.path.join(fmriprep_dir, "sub-" + sub, "func")
        
        # Total number of runs
        runs_n = len(glob.glob(os.path.join(sub_fmriprep_dir, expression)))
        
        # Using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Collect all future objects
            futures = [executor.submit(process_run, 
                                       sub,
                                       run, 
                                       bids_root,
                                       dir_rois,
                                       sub_fmriprep_dir,
                                       rois) for run in range(1, runs_n + 1)]
            
            # As each future completes, append its result to X_rec
            for future in concurrent.futures.as_completed(futures):
                X_rec.append(future.result())
        

    # Post processing
    X = pd.concat(X_rec).groupby(["sub", "run", "TR"]).mean().reset_index()

    # Save to file
    filename_out = os.path.join(out_dir, "ppi_results.csv")
    X.to_csv(filename_out, index=False)
    print(f"Saved the final DataFrame to {filename_out}")
