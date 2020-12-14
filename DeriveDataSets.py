import os
import pandas as pd
import numpy as np


def derive_2cat_threshold(data, rain_threshold, with_others):
    new_data = data.copy()
    new_data["roch_rained_today"] = (new_data["p01i"] > rain_threshold) * 1
    if with_others:
        new_data["buf_rained_today"] = (new_data["p01i_buf"] > rain_threshold) * 1
        new_data["syr_rained_today"] = (new_data["p01i_syr"] > rain_threshold) * 1

    new_data["roch_rains_tomorrow"] = np.nan
    for i in range(new_data.shape[0] - 1):
        new_data.loc[i, "roch_rains_tomorrow"] = new_data.loc[i + 1, "roch_rained_today"]

    new_data["roch_rained_today"] = new_data["roch_rained_today"].astype(np.int64)
    if with_others:
        new_data["buf_rained_today"] = new_data["buf_rained_today"].astype(np.int64)
        new_data["syr_rained_today"] = new_data["syr_rained_today"].astype(np.int64)

    new_data = new_data[:-1]
    new_data["roch_rains_tomorrow"] = new_data["roch_rains_tomorrow"].astype(np.int64)

    return new_data


def derive_3cat_threshold(data, with_others):
    new_data = data.copy()
    new_data["roch_rained_today"] = (new_data["p01i"] > 0) * 1
    if with_others:
        new_data["buf_rained_today"] = (new_data["p01i_buf"] > 0) * 1
        new_data["syr_rained_today"] = (new_data["p01i_syr"] > 0) * 1

    for i in range(new_data.shape[0] - 1):
        if new_data.loc[i, "p01i"] > 2:
            new_data.loc[i, "roch_rained_today"] = 2

        if with_others:
            if new_data.loc[i, "p01i_buf"] > 2:
                new_data.loc[i, "buf_rained_today"] = 2

            if new_data.loc[i, "p01i_syr"] > 2:
                new_data.loc[i, "syr_rained_today"] = 2

    new_data["roch_rained_today"] = new_data["roch_rained_today"].astype(np.int64)
    if with_others:
        new_data["buf_rained_today"] = new_data["buf_rained_today"].astype(np.int64)
        new_data["syr_rained_today"] = new_data["syr_rained_today"].astype(np.int64)

    new_data["roch_rains_tomorrow"] = np.nan
    for i in range(new_data.shape[0] - 1):
        new_data.loc[i, "roch_rains_tomorrow"] = new_data.loc[i + 1, "roch_rained_today"]

    new_data = new_data[:-1]
    new_data["roch_rains_tomorrow"] = new_data["roch_rains_tomorrow"].astype(np.int64)

    return new_data


def derive_normalized(data, method, with_others):
    if with_others:
        normalize_cols = ["tmpf", "dwpf", "relh", "drct", "sknt", "alti", "mslp", "vsby",
                          "tmpf_syr", "dwpf_syr", "relh_syr", "drct_syr",
                          "sknt_syr", "alti_syr", "mslp_syr", "vsby_syr",
                          "tmpf_buf", "dwpf_buf", "relh_buf", "drct_buf",
                          "sknt_buf", "alti_buf", "mslp_buf", "vsby_buf"]
    else:
        normalize_cols = ["tmpf", "dwpf", "relh", "drct", "sknt", "alti", "mslp", "vsby"]

    new_data = data.copy()

    if method == "min_max":
        for col in normalize_cols:
            new_data[col] = (new_data[col] - new_data[col].min()) / (new_data[col].max() - new_data[col].min())
    elif method == "z_score":
        for col in normalize_cols:
            new_data[col] = ((new_data[col] - new_data[col].mean()) / new_data[col].std())

    return new_data


def drop_others(data):
    new_data = data.copy()
    new_data.drop(new_data.columns[new_data.columns.str.contains('syr', case=False)], axis=1, inplace=True)
    new_data.drop(new_data.columns[new_data.columns.str.contains('buf', case=False)], axis=1, inplace=True)
    return new_data


og_data_folder = "og_data" + os.path.sep
derived_data_folder = "derived_data" + os.path.sep
rain_thresholds = [0, 0.5, 1, 2]

og_data = pd.read_csv(og_data_folder + "AllAveragedDataNumeric.csv")
without_others_data = drop_others(og_data)
derived_data = {}

# Not Normalized, With Others
derived_data["og_data_2class_0"] = derive_2cat_threshold(og_data, 0, True)
derived_data["og_data_2class_05"] = derive_2cat_threshold(og_data, 0.5, True)
derived_data["og_data_2class_1"] = derive_2cat_threshold(og_data, 1, True)
derived_data["og_data_2class_2"] = derive_2cat_threshold(og_data, 2, True)
derived_data["og_data_3class"] = derive_3cat_threshold(og_data, True)

# Not Normalized, Without Others
derived_data["without_others_2class_0"] = derive_2cat_threshold(without_others_data, 0, False)
derived_data["without_others_2class_05"] = derive_2cat_threshold(without_others_data, 0.5, False)
derived_data["without_others_2class_1"] = derive_2cat_threshold(without_others_data, 1, False)
derived_data["without_others_2class_2"] = derive_2cat_threshold(without_others_data, 2, False)
derived_data["without_others_3class"] = derive_3cat_threshold(without_others_data, False)

# Normalized MinMax, With Others
derived_data["og_data_2class_0_minmax"] = derive_normalized(derived_data["og_data_2class_0"], "min_max", True)
derived_data["og_data_2class_05_minmax"] = derive_normalized(derived_data["og_data_2class_05"], "min_max", True)
derived_data["og_data_2class_1_minmax"] = derive_normalized(derived_data["og_data_2class_1"], "min_max", True)
derived_data["og_data_2class_2_minmax"] = derive_normalized(derived_data["og_data_2class_2"], "min_max", True)
derived_data["og_data_3class_minmax"] = derive_normalized(derived_data["og_data_3class"], "min_max", True)

# Normalized MinMax, Without Others
derived_data["without_others_2class_0_minmax"] = \
    derive_normalized(derived_data["without_others_2class_0"], "min_max", False)
derived_data["without_others_2class_05_minmax"] = \
    derive_normalized(derived_data["without_others_2class_05"], "min_max", False)
derived_data["without_others_2class_1_minmax"] = \
    derive_normalized(derived_data["without_others_2class_1"], "min_max", False)
derived_data["without_others_2class_2_minmax"] = \
    derive_normalized(derived_data["without_others_2class_2"], "min_max", False)
derived_data["without_others_3class_minmax"] = \
    derive_normalized(derived_data["without_others_3class"], "min_max", False)

# Normalized Z-Score, With Others
derived_data["og_data_2class_0_zscore"] = derive_normalized(derived_data["og_data_2class_0"], "z_score", True)
derived_data["og_data_2class_05_zscore"] = derive_normalized(derived_data["og_data_2class_05"], "z_score", True)
derived_data["og_data_2class_1_zscore"] = derive_normalized(derived_data["og_data_2class_1"], "z_score", True)
derived_data["og_data_2class_2_zscore"] = derive_normalized(derived_data["og_data_2class_2"], "z_score", True)
derived_data["og_data_3class_zscore"] = derive_normalized(derived_data["og_data_3class"], "z_score", True)

# Normalized Z-Score, Without Others
derived_data["without_others_2class_0_zscore"] = \
    derive_normalized(derived_data["without_others_2class_0"], "z_score", False)
derived_data["without_others_2class_05_zscore"] = \
    derive_normalized(derived_data["without_others_2class_05"], "z_score", False)
derived_data["without_others_2class_1_zscore"] = \
    derive_normalized(derived_data["without_others_2class_1"], "z_score", False)
derived_data["without_others_2class_2_zscore"] = \
    derive_normalized(derived_data["without_others_2class_2"], "z_score", False)
derived_data["without_others_3class_zscore"] = \
    derive_normalized(derived_data["without_others_3class"], "z_score", False)

# Clear extraneous unnamed cols
for key in derived_data:
    derived_data[key].drop(
        derived_data[key].columns[
            derived_data[key].columns.str.contains('unnamed', case=False)
        ], axis=1, inplace=True
    )
    derived_data[key].drop(
        derived_data[key].columns[
            derived_data[key].columns.str.contains('p01i', case=False)
        ], axis=1, inplace=True
    )
    derived_data[key].drop("day", axis=1, inplace=True)

# Save
for key in derived_data:
    derived_data[key].to_csv(derived_data_folder + key + ".csv")
