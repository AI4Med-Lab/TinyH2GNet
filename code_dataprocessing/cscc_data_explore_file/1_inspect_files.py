import gzip
import pandas as pd

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
expr_path = "/home/puneet/mk/data/GSE144240_RAW_scCC/GSM4284316_P2_ST_rep1_stdata.tsv.gz"
spot_path = "/home/puneet/mk/data/GSE144240_RAW_scCC/GSM4284316_spot_data-selection-P2_ST_rep1.tsv.gz"

# ------------------------------------------------------------
# Load Expression Matrix
# ------------------------------------------------------------
print(" Checking Expression Data File:")
with gzip.open(expr_path, "rt") as f:
    # read just a few lines to inspect structure
    head_lines = [next(f) for _ in range(1)]
print("First 5 lines:")
for line in head_lines:
    print(line.strip())
print("-" * 80)

try:
    expr_df = pd.read_csv(expr_path, sep="\t", index_col=0)
    print(f"Expression Data loaded successfully.")
    print(f"Shape: {expr_df.shape}")
    print(f"Index (first 5): {expr_df.index[:5].tolist()}")
    print(f"Columns (first 5): {expr_df.columns[:5].tolist()}")
except Exception as e:
    print(f" Could not fully load expression data: {e}")
print("=" * 120)

# ------------------------------------------------------------
# Load Spot Metadata
# ------------------------------------------------------------
print(" Checking Spot Data File:")
with gzip.open(spot_path, "rt") as f:
    head_lines = [next(f) for _ in range(10)]
print("First 10 lines:")
for line in head_lines:
    print(line.strip())
print("-" * 80)

try:
    spot_df = pd.read_csv(spot_path, sep="\t")
    print(f" Spot Data loaded successfully.")
    print(f"Shape: {spot_df.shape}")
    print(f"Columns: {spot_df.columns.tolist()}")
    print("First few rows:")
    print(spot_df.head(5))

    # check for columns that might define scaling
    possible_scale_cols = [c for c in spot_df.columns if "pixel" in c.lower() or "scale" in c.lower()]
    if possible_scale_cols:
        print(f"\n Found possible scaling-related columns: {possible_scale_cols}")
    else:
        print("\nℹ No scaling columns found — coordinates may need rescaling manually.")
except Exception as e:
    print(f" Could not fully load spot data: {e}")

print("=" * 120)
