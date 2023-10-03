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

6. **OLS Analysis**: Conducts Ordinary Least Squares (OLS) analysis with specified regressors and prints the results.

7. **Main Function**: Orchestrates the entire data processing and analysis pipeline.

8. **Post Processing**: Further processes the data and calculates PPI terms.

9. **Saving Results**: Saves the final DataFrame to a CSV file.

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

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import make_first_level_design_matrix
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
import os
import scipy.io
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
        'fov': os.path.join(dir_rois['fov'], f'sub-{subject}_FOV_original_realignedT1MNI_resampled.nii'),
        'per_inv': os.path.join(dir_rois['peripheral'], f'sub-{subject}_space-T1w_ROI_peripheral_inverted_realigned_resampled-to-bold.nii'),
        'per_norm': os.path.join(dir_rois['peripheral'], f'sub-{subject}_space-T1w_ROI_peripheral_normal_realigned_resampled-to-bold.nii'),
        'ffa': os.path.join(dir_rois['ffa_loc'], f'sub-{subject}/FFA.nii'),
        'loc': os.path.join(dir_rois['ffa_loc'], f'sub-{subject}/LOC.nii'),
        'a1': os.path.join(dir_rois['a1'], f'sub-{subject}_brodmann_MNIresampled.nii')
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
    path = os.path.join(dir_func, f'sub-{subject}_task-exp_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    return nib.load(path).get_fdata()


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
    path = os.path.join(dir_events, f'sub-{subject}_run-{run}_BLOCKS.mat')
    events_data = scipy.io.loadmat(path)
    ev_names, ev_onsets, ev_durations = events_data['names'], events_data['onsets'], events_data['durations']

    events = []
    for i in range(ev_names[0, :].shape[0]):
        names = [ev_names[0, i][0]] * ev_onsets[0, i][0].shape[0]
        onsets = ev_onsets[0, i][0]
        durations = ev_durations[0, i][0]
        events.append(pd.DataFrame({
            'trial_type': names,
            'onset': onsets,
            'duration': durations
        }))
        
    events = pd.concat(events).sort_values('onset').reset_index(drop=True)
    events['trial_type'] = 'y_p'  # Set all to 'task'
    return events


def make_design_matrix(events, motion, frame_times, hrf_model='glover'):
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
    add_reg_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    return make_first_level_design_matrix(frame_times, events, drift_model='polynomial', drift_order=3, 
                                          add_regs=motion, add_reg_names=add_reg_names, hrf_model=hrf_model)


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


def main():
    # Directories
    dir_confounds = './data/confounds'
    dir_events = './data/events'
    dir_rois = {
        'a1': './data/rois/A1',
        'fov': './data/rois/fov',
        'peripheral': './data/rois/peripheral',
        'ffa_loc': './data/rois/ffa_loc',
    }
    dir_func = './data/func'
    
    # List of subjects
    subjects = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']

    X_rec = []

    for sub in subjects:
        # Load ROIs
        rois = load_roi_images(sub, dir_rois)
        
        for run in range(1, 6):
            print(f"Processing subject {sub}, run {run}")

            # Load functional images
            func_img_data = load_functional_images(sub, run, dir_func)
            n_scans = func_img_data.shape[-1]  # Number of time frames in the functional data
            func_img = np.reshape(func_img_data, (-1, n_scans)).T  # Reshape to (n_samples, n_features)
            print(f"Loaded functional data for subject {sub}, run {run} with {n_scans} scans")

            # Load events
            events = load_and_process_events(sub, run, dir_events)
            print(f"Loaded event data for subject {sub}, run {run}")
            
            # Load motion confounds
            motion = np.loadtxt(os.path.join(dir_confounds, f'sub-{sub}_pipeline-6HMP_run-{run}.txt'))
            print(f"Loaded motion confounds for subject {sub}, run {run}")
            
            # Verify that the motion data's length matches the number of time frames in the functional data
            assert motion.shape[0] == n_scans, "Mismatch between number of time frames in the functional data and motion regressors."
            
            # Create design matrix
            tr = 2.0
            frame_times = np.arange(n_scans) * tr
            X = make_design_matrix(events, motion, frame_times)
            print(f"Created design matrix for subject {sub}, run {run}")

            # Apply PCA and extract signals
            for roi_name, mask in rois.items():
                signal, exp_var = apply_pca_and_extract_signal(func_img, mask)
                X[f'y_{roi_name}'] = signal
                X[f'exp_var_{roi_name}'] = exp_var
                print(f"Applied PCA for ROI {roi_name} in subject {sub}, run {run}")

            X['sub'] = sub
            X['run'] = run
            
            # PPI terms
            ppi_mult = 2 * X['y_p'] - 1
            for roi_name in rois:
                X[f'y_ppi_{roi_name}'] = ppi_mult * X[f'y_{roi_name}']
                print(f"Calculated PPI term for ROI {roi_name} in subject {sub}, run {run}")
            
            X['TR'] = X.index // 2
            
            # Append to the record
            X_rec.append(X)
            print(f"Finished processing subject {sub}, run {run}")

    # Post processing
    X = pd.concat(X_rec).groupby(['sub','run', 'TR']).mean().reset_index()
    sub_even = [subject for subject in subjects if int(subject) % 2 == 0]
    X.loc[np.isin(X['sub'], sub_even), ['y_per_norm', 'y_per_inv']] = X.loc[np.isin(X['sub'], sub_even), ['y_per_inv', 'y_per_norm']]
    
    # Save to file
    filename_out = 'X.csv'
    X.to_csv(filename_out, index=False)
    print(f"Saved the final DataFrame to {filename_out}")

# Run the main function
main()



