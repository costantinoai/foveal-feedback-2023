# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:39:23 2021

@author: 45027900
"""

import os
import glob
import sys
import random
import pickle
import time
from datetime import datetime
import copy
import pandas as pd
import numpy as np
import nibabel as nb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, ParameterGrid, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from scipy.stats import ttest_1samp, t, sem
from scipy.io import loadmat
import matplotlib

# wait_hours = 3
# ctime = time.ctime()
# print(f'START: {ctime}, sleeping for {str(wait_hours)} hours...')
# for half in range(int(wait_hours*2)):
#     print(f'\t...{((wait_hours*2)-half)*30} minutes left')
#     time.sleep(60*30)
# print('...done!')

log = False
grid_search = False
feature_sel = False

if grid_search == False:

    def classify(X_r, y_r, runs_idx, roi_name=str):
        kernel = "linear"
        runs_n = len(np.unique(runs_idx))
        gkf = GroupKFold(n_splits=runs_n)
        performance_step = np.zeros(gkf.n_splits)
        labels_n = len(np.unique(y_r))
        labels_sorted = sorted(np.unique(y_r))
        conf_mat = np.zeros((gkf.n_splits, labels_n, labels_n))
        i = 0
        if feature_sel == True:
            X_r = StandardScaler().fit_transform(X_r)
        for train_idx, test_idx in gkf.split(X=X_r, y=y_r, groups=runs_idx):
            print(f"## {sub} ## {roi_name} ## STEP: {i+1} ##")
            print("Indices of train-samples: %r" % train_idx.tolist())
            print("Indices of test-samples: %r" % test_idx.tolist())
            print(
                "... which correspond to following runs: %r"
                % runs_idx[test_idx].tolist()
            )
            # make test train sets
            X_train = X_r[train_idx]
            X_test = X_r[test_idx]
            y_train = y_r[train_idx]
            y_test = y_r[test_idx]

            if feature_sel == True:
                X_scaler = SelectKBest(f_classif, k=10)
                # fit your transformer on the train-set
                X_scaler_fitted = X_scaler.fit(X_train, y_train)
                X_train = X_scaler_fitted.transform(
                    X_train
                )  # transform your train-set;
                X_test = X_scaler_fitted.transform(X_test)  # transform your test-set;
                idx = X_scaler_fitted.get_support(indices=True)  # get selected features
            # fit
            # (fit your model on the transformed train-set;)
            clf = SVC(kernel=kernel, random_state=42).fit(X_train, y_train)
            # predict (cross-validate to the transformed test-set;)
            y_hat = clf.predict(X_test)
            y_hat_training = clf.predict(X_train)
            performance_step[i] = clf.score(X_test, y_test)
            train_p = clf.score(X_train, y_train)
            conf_mat[i] = confusion_matrix(y_test, y_hat)

            # generate reports for each class (confusion matrix and scores in a dict for each class in a run)
            if feature_sel == True:
                print(f"Indices: {idx}")
            print(
                f"TRAINING BATCH: \t{list(y_train)} \nTRAINING PREDICTED: {list(y_hat_training)} \nTARGET: \t\t{list(y_test)} \nPREDICTED: \t{list(y_hat)}"
            )
            print(
                f"TRAINING ACC: {train_p}, TEST ACC: {round(performance_step[i],2)}, STEP: {i+1}\n"
            )
            i += 1
        acc = np.average(performance_step)
        print(f"TOTAL ACCURACY: {acc}\n\n")
        return acc, conf_mat

else:

    def classify(X, y, runs_idx, roi_name=str):
        # Initialise test-train split (leave-one-run-out)
        start = time.time()
        runs_n = len(np.unique(runs_idx))
        roi_acc = []
        roi_clf = []
        if feature_sel == True:
            X = StandardScaler().fit_transform(X)
        for run_id in range(runs_n):
            X_in = X[runs_idx.astype(int) != run_id + 1]
            y_in = y[runs_idx.astype(int) != run_id + 1]
            X_out = X[runs_idx.astype(int) == run_id + 1]
            y_out = y[runs_idx.astype(int) == run_id + 1]

            gkf = GroupKFold(n_splits=runs_n - 1)
            # Create parameters grid (choose best performing parameters)
            c_values = list(np.geomspace(1e10, 1e-10, 21))
            # gamma_values = list(np.geomspace(1e4, 1e-4, 9))
            # grid_dict = [{'kernel': [['linear']], 'C':[c_values]}, {
            #     'kernel': [['rbf']], 'C':[c_values], 'gamma':[gamma_values]}]
            grid_dict = [{"kernel": [["linear"]], "C": [c_values]}]
            grid = ParameterGrid(grid_dict)
            if feature_sel == True:
                # feature selection
                X_scaler = SelectKBest(f_classif, k=20)
                # fit your transformer on the train-set
                X_scaler_fitted = X_scaler.fit(X_in, y_in)
                X_in = X_scaler_fitted.transform(X_in)  # transform your train-set;
                X_out = X_scaler_fitted.transform(X_out)  # transform your test-set;
                idx = X_scaler_fitted.get_support(indices=True)  # get selected features
            # Initialise and fit classifier
            svc = SVC(max_iter=1e6, random_state=42)
            clf = GridSearchCV(
                estimator=svc,
                param_grid=grid,
                scoring="accuracy",
                cv=gkf,
                refit=True,
                n_jobs=-1,
                error_score="raise",
                return_train_score=True,
            )
            clf.fit(X=X_in, y=y_in, groups=runs_idx[runs_idx.astype(int) != run_id + 1])
            # Get and print results for best params set
            # params = clf.best_params_
            # test_acc = clf.best_score_
            # train_acc = clf.cv_results_['mean_train_score'][clf.best_index_]

            outer_clf = clf.best_estimator_.score(X_out, y_out)
            roi_acc.append(outer_clf)
            roi_clf.append(clf.best_estimator_)
        test_acc = np.average(roi_acc)
        end = time.time()
        print(f"ROI: {roi_name}")
        # print(f'Best parameters: {params}')
        # print(f'Training accuracy: {train_acc}\nTest accuracy: {test_acc}\nTime: {round(end-start, 2)} sec')
        print(f"Test accuracy: {test_acc}\nTime: {round(end-start, 2)} sec")

        return test_acc, roi_clf


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h, se


# SET PARAMETERS AND FOLDERS
sns.set(font_scale=1.3)
per_radius = 1.0
fov_radius = 4.0

out_root = r"D:\2exp_fMRI\Exp\Data\Data\fmri\BIDS\derivatives\MVPA"
beta_root = r"D:\2exp_fMRI\Exp\Data\Data\fmri\BIDS\derivatives\SPM"
parcel_root = (
    r"D:\2exp_fMRI\Exp\Data\Data\fmri\BIDS\derivatives\parcels\nii\final_parcels"
)
mask_root = r"D:\2exp_fMRI\Exp\Data\Data\fmri\BIDS\derivatives\masks\ok\MNI_realigned_resampled_nii"
rois = ["Foveal", "Peripheral", "Opposite", "FFA", "LOC"]

pipeline_dirs = ["RSA_blocks_1_lev_6HMP"]

# conditions_list = [('face', 'vehicle'), ('female', 'male'), ('bike', 'car'), ('bike', 'female'), ('bike', 'male'),
#                    ('car', 'female'), ('car', 'male'), ('bike', 'car', 'female', 'male')]
conditions_list = [
    ("face", "vehicle"),
    ("female", "male"),
    ("bike", "car"),
    ("bike", "car", "female", "male"),
]
conditions_all = "bike", "car", "female", "male", "rest"

pipeline_dir = pipeline_dirs[0]

out_dir = os.path.join(out_root, "MVPA_blocks_6HMP_norest")
if log == True:
    os.makedirs(out_dir, exist_ok=True)
# init graphs dataframe
graph_df = pd.DataFrame(
    None, columns=("avg", "std", "comparison", "roi", "cimin", "cimax", "p")
)
graph_df["roi"] = rois * len(conditions_list)
graph_df["comparison"] = [
    comparison for comparison in conditions_list for i in range(len(rois))
]

reports_all = {
    roi: {
        comparison: np.zeros((len(range(2, 26)), 5, len(comparison), len(comparison)))
        for comparison in conditions_list
    }
    for roi in rois
}
# initialise results dictionary
results = {
    conditions: {
        roi: {
            "Accuracy": [],
            "Reports": [],
            "T-test": {"t": float, "p": float, "avg": float, "std": float, "d": float},
        }
        for roi in rois
    }
    for conditions in conditions_list
}

for conditions in conditions_list:
    # point to the right folder
    if ("face" in conditions) or ("vehicle" in conditions):
        beta_subdir = pipeline_dir + "_fv"
    else:
        beta_subdir = pipeline_dir + "_all"
    # generate timestamp and start logging
    ts = str(datetime.now(tz=None)).split(" ")
    ts[1] = ts[1].split(".")[0].split(":")
    ts[1] = ts[1][0] + ts[1][1] + ts[1][2]
    ts = "_".join(ts)
    if log == True:
        logfile = os.path.join(
            out_dir,
            os.path.basename(sys.argv[0]).split(".")[0]
            + "_"
            + "-".join(list(conditions))
            + "_"
            + pipeline_dir
            + "_"
            + ts
            + ".txt",
        )
        print(f"logfile directory: {logfile}")
        sys.stdout = open(logfile, "w")
    print(
        f"START: {datetime.now()} - Logging = {log}, Gridsearch = {grid_search}, Feature selection: {feature_sel}"
    )
    print(
        f"PARAMETERS:\n\tper_radius = {per_radius}\n\tfov_radius = {fov_radius}\n\tout_root = {out_root}\n\tbeta_root = {beta_root}\n\tparcel_root = {parcel_root}\n\tmask_root = {mask_root}\n\trois = {rois}\n\tconditions_list = {conditions_list}\n\tpipeline_dirs = {pipeline_dirs}"
    )
    print(f"STEP: pipeline_dir = {pipeline_dir}")
    # SELECT BETAS
    for sub_id in range(2, 26):
        sub = "sub-" + str(sub_id).zfill(2)
        beta_loc = os.path.join(beta_root, beta_subdir, sub)
        print(f"STEP: {sub} - starting classification for {conditions}")
        print("STEP: loading masks...", end="\r")
        # define masks locations
        mask_files = {roi: "" for roi in rois}

        mask_files["Foveal"] = glob.glob(os.path.join(mask_root, sub + "*foveal*"))[0]
        if sub_id % 2 == 0:
            mask_files["Peripheral"] = glob.glob(
                os.path.join(mask_root, sub + "*inverted*")
            )[0]
            mask_files["Opposite"] = glob.glob(
                os.path.join(mask_root, sub + "*normal*")
            )[0]
        else:
            mask_files["Peripheral"] = glob.glob(
                os.path.join(mask_root, sub + "*normal*")
            )[0]
            mask_files["Opposite"] = glob.glob(
                os.path.join(mask_root, sub + "*inverted*")
            )[0]
        mask_files["FFA"] = glob.glob(os.path.join(parcel_root, sub, "FFA.nii"))[0]
        mask_files["LOC"] = glob.glob(os.path.join(parcel_root, sub, "LOC.nii"))[0]
        print("done!")

        # generate beta mapping dataframe
        print("STEP: generating beta mapping dataframe...", end="\r")
        betas_df = pd.DataFrame(
            None,
            columns=("beta_path", "spm_filename", "condition", "run", "bin", "array"),
        )
        betas_df["beta_path"] = sorted(glob.glob(os.path.join(beta_loc, "beta*.?ii")))

        mat = loadmat(os.path.join(beta_loc, "SPM.mat"))
        beta_names = [
            str(mat["SPM"]["Vbeta"][0][0][0][beta_id][0][0])
            for beta_id in range(len(betas_df["beta_path"]))
            if str(mat["SPM"]["Vbeta"][0][0][0][beta_id][0][0])
            == os.path.basename(betas_df["beta_path"][beta_id])
        ]
        beta_classes = [
            str(mat["SPM"]["Vbeta"][0][0][0][beta_id][5][0])
            .split(" ")[-1]
            .split("*")[0]
            for beta_id in range(len(betas_df["beta_path"]))
            if str(mat["SPM"]["Vbeta"][0][0][0][beta_id][0][0])
            == os.path.basename(betas_df["beta_path"][beta_id])
        ]
        beta_runs = [
            str(mat["SPM"]["Vbeta"][0][0][0][beta_id][5][0])
            .split(" ")[-2]
            .split("(")[-1][0]
            for beta_id in range(len(betas_df["beta_path"]))
            if str(mat["SPM"]["Vbeta"][0][0][0][beta_id][0][0])
            == os.path.basename(betas_df["beta_path"][beta_id])
        ]
        beta_bins = [
            str(mat["SPM"]["Vbeta"][0][0][0][beta_id][5][0])
            .split(" ")[-1]
            .split("(")[-1][0]
            for beta_id in range(len(betas_df["beta_path"]))
            if str(mat["SPM"]["Vbeta"][0][0][0][beta_id][0][0])
            == os.path.basename(betas_df["beta_path"][beta_id])
        ]

        betas_df["spm_filename"] = beta_names
        betas_df["condition"] = beta_classes
        betas_df["run"] = beta_runs
        betas_df["bin"] = beta_bins
        betas_df["array"] = [
            np.array(nb.load(beta).get_fdata()) for beta in betas_df["beta_path"]
        ]

        betas_df = betas_df[betas_df.condition.str.match("|".join(list(conditions)))]
        betas_df = betas_df.sort_values(["condition", "run"], axis=0).reset_index(
            drop=True
        )
        print("done!")

        # remove the mean value in all conditions tested for each voxel in a run (cocktail-blank removal)
        print(
            "STEP: removing average pattern of all the conditions to classify...",
            end="\r",
        )
        all_betas_data = np.array([array for array in betas_df["array"]])
        runs = np.unique(betas_df["run"].values)
        for run in runs:
            betas_idx = betas_df["run"] == run
            run_array = all_betas_data[betas_idx.values]
            run_array_avg = np.array(
                [np.mean(run_array, 0) for rows in range(run_array.shape[0])]
            )
            run_array_demeaned = run_array - run_array_avg

            list_id = 0
            for index, beta_idx in enumerate(betas_idx):
                if beta_idx:
                    betas_df.iloc[index]["array"] = run_array_demeaned[list_id]
                    list_id += 1
        print("done!")

        print(
            "STEP: preparing dataset for classification (zeroing NaNs, shuffling)...",
            end="\r",
        )
        # mask betas and prepare dataset for classification
        for roi_key in mask_files.keys():
            # import mask and mask data (zero filled)
            # NOTE: here nans are zeroed, but they can be masked
            # this does not make any difference for classification puproses
            mask = nb.load(mask_files[roi_key]).get_fdata() > 0
            betas_masked = [
                np.nan_to_num(df_array[mask], False, 0.0)
                for df_array in betas_df["array"]
            ]
            # betas_masked = [df_array[mask][~np.isnan(df_array[mask])]
            #                 for df_array in betas_df['array']]
            # generate dataset
            X = betas_masked
            y = list(betas_df["condition"])
            runs_idx = list(betas_df["run"])

            # shuffle dataset
            comp = list(zip(X, y, runs_idx))
            random.seed(42)
            random.shuffle(comp)
            X, y, runs_idx = zip(*comp)

            # transform back into array
            X = np.array(X)
            y = np.array(y)
            runs_idx = np.array(runs_idx)
            print("done!")

            # RUN THE CLASSIFIER (LEAVE-ONE-RUN-OUT)
            # accuracy is averaged over 5 runs (cross-validated)
            print(f"### CLASSIFICATION - {sub} {conditions} ###")
            acc, reports_all[roi_key][conditions][sub_id - 2] = classify(
                X, y, runs_idx, roi_name=roi_key
            )
            results[conditions][roi_key]["Accuracy"].append(acc)
    # PERFORM T-TEST
    labels_n = len(conditions)

    for roi in rois:
        (
            results[conditions][roi]["T-test"]["t"],
            results[conditions][roi]["T-test"]["p"],
        ) = ttest_1samp(
            results[conditions][roi]["Accuracy"],
            popmean=1 / labels_n,
            alternative="greater",
        )
        (
            results[conditions][roi]["T-test"]["avg"],
            results[conditions][roi]["T-test"]["std"],
        ) = (
            np.average(results[conditions][roi]["Accuracy"]),
            np.std(results[conditions][roi]["Accuracy"]),
        )
        results[conditions][roi]["T-test"]["d"] = (
            results[conditions][roi]["T-test"]["avg"] - (1.0 / len(conditions))
        ) / results[conditions][roi]["T-test"]["std"]
        print(
            f'ROI {roi}: t = {round(results[conditions][roi]["T-test"]["t"], 3)}, p = {round(results[conditions][roi]["T-test"]["p"], 3)}, avg = {round(results[conditions][roi]["T-test"]["avg"], 3)}, std = {round(results[conditions][roi]["T-test"]["std"], 3)}'
        )
        # pass values to graph dataframe
        (
            graph_df.loc[
                ((graph_df["comparison"] == conditions) & (graph_df["roi"] == roi)),
                "avg",
            ],
            graph_df.loc[
                ((graph_df["comparison"] == conditions) & (graph_df["roi"] == roi)),
                "cimin",
            ],
            graph_df.loc[
                ((graph_df["comparison"] == conditions) & (graph_df["roi"] == roi)),
                "cimax",
            ],
            graph_df.loc[
                ((graph_df["comparison"] == conditions) & (graph_df["roi"] == roi)),
                "sem",
            ],
        ) = mean_confidence_interval(results[conditions][roi]["Accuracy"])
        graph_df.loc[
            ((graph_df["comparison"] == conditions) & (graph_df["roi"] == roi)), "p"
        ] = results[conditions][roi]["T-test"]["p"]
        graph_df.loc[
            ((graph_df["comparison"] == conditions) & (graph_df["roi"] == roi)), "std"
        ] = results[conditions][roi]["T-test"]["std"]
        graph_df.loc[
            ((graph_df["comparison"] == conditions) & (graph_df["roi"] == roi)), "d"
        ] = results[conditions][roi]["T-test"]["d"]
print("STEP: preparing and plotting figures...", end="\r")
# PLOT AND SAVE FIGURES
for roi in graph_df["roi"].unique():
    matplotlib.rcParams.update({"font.size": 220})
    data = graph_df.loc[graph_df["roi"] == roi].reset_index(drop=True)
    labels_tuples = data["comparison"].values
    labels = [" vs. ".join(tup) for tup in labels_tuples]
    try:
        index = list(labels_tuples).index(("bike", "car", "female", "male"))
        labels[index] = "all"
    except:
        pass
    # yerr = np.array([(data['avg'][i] - data['cimin'][i],
    #                   data['cimax'][i] - data['avg'][i]) for i in data.index]).T
    yerr = np.array(data["sem"])
    fig_dims = (15, 12)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.barplot(x="comparison", y="avg", data=data, ci=None, ax=ax)
    plt.errorbar(
        x=list(range(len(data))),
        y=data["avg"],
        yerr=yerr,
        fmt="none",
        c="black",
        capsize=5,
    )
    plt.ylim((0, 1))
    plt.axhline(0.5, alpha=0.8, color="red", ls="--", zorder=4)
    # plt.axhline(0.25, alpha=0.8, color='red', ls='--', zorder=4)
    ax.set_xticklabels(
        labels, rotation=45, horizontalalignment="right", fontweight="light"
    )
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(None)
    if roi == "Foveal":
        ax.set_title(f"ROI: {roi}, 4.0 visual degrees\nClassifier accuracy")
    else:
        ax.set_title(f"ROI: {roi}\nClassifier accuracy")
    # Draw significance annotations
    for i, p in enumerate(data["p"]):
        if p < 0.001:
            displaystring = r"***"
        elif p < 0.01:
            displaystring = r"**"
        elif p < 0.05:
            displaystring = r"*"
        else:
            displaystring = r""
        height = data["sem"][i] + data["avg"][i] + 0.05
        if height > 0.98:
            height = 0.98
        plt.text(
            list(range(len(data)))[i],
            height,
            displaystring,
            ha="center",
            va="center",
            size=20,
        )
    # Save the figure and show
    plt.tight_layout()
    if log == True:
        plt.savefig(os.path.join(out_dir, f"{roi}_sem.png"))
        print("STEP: figures saved in {out_dir}")
    plt.show()
# PLOT AND SAVE FIG (ALL ROIS)
sns.set(font_scale=2)
data = graph_df.loc[
    (graph_df["comparison"] == ("face", "vehicle"))
    | (graph_df["comparison"] == ("bike", "car", "female", "male"))
    | (graph_df["comparison"] == ("female", "male"))
    | (graph_df["comparison"] == ("bike", "car"))
].reset_index(drop=True)
data = data[data["roi"] != "Foveal"]
data["order"] = 0
data["order"][data["comparison"] == ("face", "vehicle")] = 0
data["order"][data["comparison"] == ("female", "male")] = 1
data["order"][data["comparison"] == ("bike", "car")] = 2
data["order"][data["comparison"] == ("bike", "car", "female", "male")] = 3
data = data.sort_values(["roi", "order"], axis=0).reset_index(drop=True)
labels_tuples = data["comparison"].values
fig_dims = (16, 10)
fig, ax = plt.subplots(figsize=fig_dims)
sns.barplot(x="roi", y="avg", hue="comparison", data=data, ax=ax)
L = plt.legend()
L.get_texts()[0].set_text("Category (Face vs. Vehicle)")
L.get_texts()[1].set_text("Within-category, faces (Female vs. Male)")
L.get_texts()[2].set_text("Within-category, vehicles (Bike vs. Car)")
L.get_texts()[3].set_text("Sub-category (Bike vs. Car vs. Female vs. Male)")
# make error bars
x_pos = np.array(
    [
        [
            ax.get_xticks()[i] - 0.3,
            ax.get_xticks()[i] - 0.1,
            +ax.get_xticks()[i] + 0.1,
            ax.get_xticks()[i] + 0.3,
        ]
        for i in range(len(ax.get_xticks()))
    ]
).flatten()
y_pos = data["avg"]
yerr = np.array(data["sem"])
plt.errorbar(
    x=x_pos, y=y_pos, yerr=yerr, fmt="none", c="black", capsize=5, palette="Paired"
)

plt.ylim((0.2, 1))
plt.axhline(0.5, alpha=0.8, color="blue", ls="--", zorder=4)
plt.axhline(0.25, alpha=0.8, color="red", ls="--", zorder=4)
# plt.axhline(0.25, alpha=0.8, color='red', ls='--', zorder=4)
ax.set_ylabel("Accuracy")
ax.set_xlabel("ROI")
ax.set_title("Classifier accuracy")

# Draw significance annotations
for i, p in enumerate(data["p"]):
    if p < 0.001:
        displaystring = r"***"
    elif p < 0.01:
        displaystring = r"**"
    elif p < 0.05:
        displaystring = r"*"
    else:
        displaystring = r""
    height = data["sem"][i] + data["avg"][i] + 0.05
    if height > 1:
        height = 1
    plt.text(x_pos[i], height, displaystring, ha="center", va="center", size=20)
# Save the figure and show
if log == True:
    plt.savefig(os.path.join(out_dir, "all_sem.png"))
    print(f"STEP: figures saved in {out_dir}")
plt.show()

if log == True:
    # SAVE PICKLED CONDITION FILE
    final_res = {"graph_df": graph_df, "results": results}
    pkl_file = os.path.join(
        out_dir,
        os.path.basename(sys.argv[0]).split(".")[0]
        + "_"
        + "-".join(list(conditions))
        + "_"
        + pipeline_dir
        + "_"
        + ts
        + ".pkl",
    )
    with open(pkl_file, "wb") as fp:
        pickle.dump(final_res, fp)
    print("STEP: results dictionary saved as {pkl_file}")
## PLOT CONF MATRICES
sns.set(font_scale=1)

comparisons = [
    ("face", "vehicle"),
    ("female", "male"),
    ("bike", "car"),
    ("bike", "car", "female", "male"),
]
rois = ["Peripheral", "Opposite", "FFA", "LOC"]
num_cols = len(rois)
num_rows = len(comparisons)
fig = plt.figure(figsize=(num_cols * 3, num_rows * 3))
position = 1

for row, comparison in enumerate(comparisons):
    for col, noise in enumerate(rois):
        ax = fig.add_subplot(num_rows, num_cols, position)

        labels = sorted(list(comparison))
        cm = np.sum(reports_all[noise][comparison], (0, 1))
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_fig = ConfusionMatrixDisplay(cm, display_labels=labels)
        if len(comparison) == 2:
            plt.title(
                f"{noise}\n{comparison[0]} vs. {comparison[1]}", fontweight="bold"
            )
        else:
            plt.title(f"Noise: {noise}\nall vs. all", fontweight="bold")
        plt.grid(None)
        cm_fig.plot(
            ax=ax,
            colorbar=False,
            cmap="Blues",
        )

        position += 1
plt.suptitle("Confusion Matrices", fontsize=15)
plt.tight_layout()
plt.show()

## PRINT STATS
rois = ["FFA", "LOC", "Opposite", "Peripheral"]

for roi in rois:
    print("\n")
    if roi == "Peripheral":
        roi_key = "PER"
    elif roi == "Opposite":
        roi_key = "OPP"
    else:
        roi_key = roi
    for comparison in conditions_list:
        t = round(results[comparison][roi]["T-test"]["t"], 2)
        if results[comparison][roi]["T-test"]["p"] >= 0.05:
            p = "= n.s."
        elif (results[comparison][roi]["T-test"]["p"] < 0.05) and (
            results[comparison][roi]["T-test"]["p"] >= 0.01
        ):
            p = "< .05"
        elif (results[comparison][roi]["T-test"]["p"] < 0.01) and (
            results[comparison][roi]["T-test"]["p"] >= 0.001
        ):
            p = "< .01"
        elif (results[comparison][roi]["T-test"]["p"] < 0.001) and (
            results[comparison][roi]["T-test"]["p"] >= 0.0001
        ):
            p = "< .001"
        elif results[comparison][roi]["T-test"]["p"] < 0.0001:
            p = "< .0001"
        else:
            raise Exception("p is not recognized")
        d = round(results[comparison][roi]["T-test"]["d"], 2)
        print(comparison)
        print(f"{roi_key}: t(23) = {t}, p {p}, d = {d}")
