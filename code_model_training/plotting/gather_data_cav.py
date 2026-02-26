import pandas as pd
import os

# --------------------
# Config (constant for this file)
# --------------------
DATASET_NAME = "10x"
MODEL_NAME = "EffNet-100"
PARAMS = 5873817    # update per model

RAW_CSV = "/home/puneet/mk/code_model_training/modelResults/10x_Effnet_pretain_false_224/result_vit_2026-01-01_20-49-02/eval_results/fold_wise_results_complete_biomarkers.csv"
OUT_CSV = "all_models_per_fold_metrics.csv"

# Load raw metrics CSV
df = pd.read_csv(RAW_CSV)

# Keep only rows we care about
wanted_metrics = [
    "l1_error_mean_mean",
    "pearson_mean_cellwise_mean",
    "spearman_mean_cellwise_mean"
]

df = df[df["metric"].isin(wanted_metrics)]

# Melt to long format
long_df = df.melt(
    id_vars=["fold", "metric"],
    var_name="group",
    value_name="value"
)

# Keep only ALL group
long_df = long_df[long_df["group"] == "ALL"]

# Pivot to tidy format (one row per fold)
tidy_df = long_df.pivot_table(
    index=["fold"],
    columns="metric",
    values="value"
).reset_index()

# Rename columns nicely
tidy_df = tidy_df.rename(columns={
    "l1_error_mean_mean": "l1_mean",
    "pearson_mean_cellwise_mean": "pearson_cell_mean",
    "spearman_mean_cellwise_mean": "spearman_cell_mean"
})

# Add metadata
tidy_df["dataset"] = DATASET_NAME
tidy_df["model"] = MODEL_NAME
tidy_df["params"] = PARAMS

# Reorder columns
cols = [
    "dataset", "model", "params", "fold",
    "l1_mean", "pearson_cell_mean", "spearman_cell_mean"
]
tidy_df = tidy_df[cols]

# --------------------
# Append to master CSV
# --------------------
if os.path.exists(OUT_CSV):
    tidy_df.to_csv(OUT_CSV, mode="a", header=False, index=False)
else:
    tidy_df.to_csv(OUT_CSV, index=False)

print(f"Appended per-fold results to {OUT_CSV}")
print(tidy_df)