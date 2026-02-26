import pandas as pd
import numpy as np
import argparse

# ======== CONFIG ========
# Path to your single combined CSV file

# /home/sukrit/Desktop/AmitResearch/code/modelResults/ImputerResults_seqFISH_1/result_2025-11-07_11-25-56
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help='Include the filepath')

args = parser.parse_args()


csv_path = f"{args.path}/metrics_result.csv"   # ?? change this to your actual file path

# ======== LOAD DATA ========
df = pd.read_csv(csv_path)

# Filter only validation phase
df_val = df[df["phase"].str.lower() == "val"]

# For each fold, find the row with maximum spearman_mean_genewise
best_per_fold = df_val.loc[df_val.groupby("fold")["spearman_mean_genewise"].idxmax()]

print(best_per_fold.iloc[:,:11])

# ======== COMPUTE MEAN STD ========
metrics = ["pearson_mean_cell_wise","pearson_mean_genewise", "spearman_mean_cell_wise","spearman_mean_genewise", "l1_error_mean", "l2_errors_mean"]
summary = {}

for m in metrics:
    mean_val = best_per_fold[m].mean()
    std_val = best_per_fold[m].std()
    summary[m] = f"{mean_val:.4f} +/- {std_val:.4f}"


# ======== DISPLAY RESULTS ========
# print("\n Best validation Spearman per fold  Summary:")
for metric, value in summary.items():
    print(f"{metric}\t{value}")
