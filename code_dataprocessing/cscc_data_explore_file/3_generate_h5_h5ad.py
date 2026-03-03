import os
import gzip
import h5py
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import scanpy as sc
import argparse

# Disable PIL large image warning
Image.MAX_IMAGE_PIXELS = None

# ------------------------------------------------------------
# Argument configuration
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Generate .h5 and .h5ad for one or multiple samples by index")

parser.add_argument("--base_dir", required=True, help="Base directory containing GSM files")
parser.add_argument("--sample_index", type=int, nargs="*", help="Sample number(s) to process (1-based index). If omitted, runs all.")
parser.add_argument("--patch_size", type=int, default=210, help="Patch size in pixels (default=210)")
parser.add_argument("--out_dir", default="output_h5", help="Directory to save output files")

args = parser.parse_args()

BASE_DIR = args.base_dir
PATCH_SIZE = args.patch_size
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Identify samples
# ------------------------------------------------------------
all_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith(".gz")])
sample_ids = sorted(set(
    f.split("_")[0] + "_P" + f.split("P")[1].split("_")[0]
    for f in all_files
    if "spot_data-selection" in f
))

# sample_ids = sorted(set([f.split("_")[0] for f in all_files]))

print(f" Found {len(sample_ids)} total samples in {BASE_DIR}")
for i, sid in enumerate(sample_ids, 1):
    print(f"  {i:2d}. {sid}")

# If sample_index not given, process all samples
if args.sample_index:
    selected_indices = args.sample_index
    print(f"\n🔹 Running only for user-selected samples: {selected_indices}")
else:
    selected_indices = list(range(1, len(sample_ids) + 1))
    print("\n🔹 No --sample_index provided → running for ALL samples")

print("=" * 90)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def load_image_gz(path):
    with gzip.open(path, "rb") as f:
        img = Image.open(BytesIO(f.read())).convert("RGB")
    return img

def load_tsv_gz(path):
    return pd.read_csv(path, sep="\t")

def extract_patch(image, center_x, center_y, patch_size):
    half = patch_size // 2
    left = int(center_x - half)
    upper = int(center_y - half)
    right = int(center_x + half)
    lower = int(center_y + half)
    patch = image.crop((left, upper, right, lower))
    return np.array(patch)

# ------------------------------------------------------------
# Loop through selected samples
# ------------------------------------------------------------
for idx in selected_indices:
    if idx < 1 or idx > len(sample_ids):
        print(f" Skipping invalid index {idx} (valid range: 1-{len(sample_ids)})")
        continue

    SAMPLE_ID = sample_ids[idx - 1]
    BASE_ID, PATIENT_ID = SAMPLE_ID.split("_")

    print(f"\nProcessing sample {idx}/{len(sample_ids)} → {SAMPLE_ID}")
    print("-" * 90)

    # Locate input files
    expr_file = next(
        f for f in all_files
        if BASE_ID in f and "stdata" in f and PATIENT_ID in f
    )

    coord_file = next(
        f for f in all_files
        if BASE_ID in f and "spot_data-selection" in f and PATIENT_ID in f
    )

    img_file = next(
        f for f in all_files
        if BASE_ID in f and PATIENT_ID in f and f.endswith(".jpg.gz")
    )

    expr_path = os.path.join(BASE_DIR, expr_file)
    coord_path = os.path.join(BASE_DIR, coord_file)
    img_path = os.path.join(BASE_DIR, img_file)

    print(f" Expression:  {expr_file}")
    print(f" Coordinates: {coord_file}")
    print(f" Image:       {img_file}")

    # Load
    expr_df = load_tsv_gz(expr_path)
    spot_df = load_tsv_gz(coord_path)
    image = load_image_gz(img_path)
    print(f" Image size: {image.size}")

    # ------------------------------------------------------------
    # Prepare expression data
    # ------------------------------------------------------------
    expr_df.rename(columns={expr_df.columns[0]: "spot_id"}, inplace=True)
    expr_df["x"] = expr_df["spot_id"].str.split("x").str[0].astype(int)
    expr_df["y"] = expr_df["spot_id"].str.split("x").str[1].astype(int)

    genes = expr_df.columns[1:-2]  # exclude spot_id, x, y
    merged_df = pd.merge(expr_df, spot_df, on=["x", "y"], how="inner")
    print(f" Matched spots: {merged_df.shape[0]}")

    # ------------------------------------------------------------
    # Extract image patches
    # ------------------------------------------------------------
    patches = []
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc=f"Extracting patches for {SAMPLE_ID}"):
        cx, cy = row["pixel_x"], row["pixel_y"]
        patch = extract_patch(image, cx, cy, PATCH_SIZE)
        patches.append(patch)

    patches = np.stack(patches)
    print(f" Extracted patches shape: {patches.shape} (N, H, W, C)")

    # ------------------------------------------------------------
    # Save H5 (patches + coords)
    # ------------------------------------------------------------
    h5_path = os.path.join(OUT_DIR, f"{SAMPLE_ID}_patches.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("patches", data=patches, compression="gzip")
        f.create_dataset("pixel_x", data=merged_df["pixel_x"].values)
        f.create_dataset("pixel_y", data=merged_df["pixel_y"].values)
        f.create_dataset("x", data=merged_df["x"].values)
        f.create_dataset("y", data=merged_df["y"].values)
    print(f"  Saved patches → {h5_path}")

    # ------------------------------------------------------------
    # Save H5AD (expression + coordinates)
    # ------------------------------------------------------------
    expr_matrix = merged_df[genes].to_numpy(dtype=np.float32)
    adata = sc.AnnData(X=expr_matrix)
    adata.obs["x"] = merged_df["x"].values
    adata.obs["y"] = merged_df["y"].values
    adata.obs["pixel_x"] = merged_df["pixel_x"].values
    adata.obs["pixel_y"] = merged_df["pixel_y"].values
    adata.var_names = genes

    h5ad_path = os.path.join(OUT_DIR, f"{SAMPLE_ID}_spots.h5ad")
    adata.write_h5ad(h5ad_path)
    print(f"  Saved expression + spatial metadata → {h5ad_path}")

print("\n All selected samples processed successfully!")
