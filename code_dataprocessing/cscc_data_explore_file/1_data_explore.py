import os
import gzip
import argparse
import pandas as pd
from PIL import Image
from io import BytesIO
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Disable PIL decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None

# ------------------------------------------------------------
# 1️⃣ Command-line arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compute neighbor distances and save per-sample patch recommendations")
parser.add_argument("--base_dir", required=True, help="Base directory containing GSM files")
parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors to compute distances for")
parser.add_argument("--show_samples", type=int, default=3, help="Number of samples to preview")
parser.add_argument("--out_dir", default="distance_results", help="Directory to save CSV files")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ------------------------------------------------------------
# 2️⃣ Helper functions
# ------------------------------------------------------------
def load_tsv(tsv_path):
    """Load gzipped TSV file"""
    return pd.read_csv(tsv_path, sep="\t")

def load_image(img_path):
    """Load compressed jpg.gz image and return PIL image"""
    with gzip.open(img_path, "rb") as f:
        img = Image.open(BytesIO(f.read()))
    return img.convert("RGB")

def compute_neighbor_distances(coords, n_neighbors=5):
    """Compute nearest neighbor distances (excluding self)"""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    return distances[:, 1:]  # skip self-distance (0)

# ------------------------------------------------------------
# 3️⃣ Explore folder
# ------------------------------------------------------------
all_files = sorted([f for f in os.listdir(args.base_dir) if f.endswith(".gz")])
sample_ids = sorted(set([f.split("_")[0] for f in all_files]))

print(f" Found {len(sample_ids)} unique samples in {args.base_dir}")
print("=" * 100)

for sid in sample_ids[: args.show_samples]:
    files = [os.path.join(args.base_dir, f) for f in all_files if f.startswith(sid)]
    print(f"\n Sample: {sid}")
    print("-" * 100)

    # Locate files
    coord_file = next((f for f in files if "spot_data-selection" in f), None)
    img_file = next((f for f in files if f.endswith(".jpg.gz")), None)

    if not coord_file or not img_file:
        print("  Missing coordinate or image file, skipping.")
        continue

    # Load
    coords_df = load_tsv(coord_file)
    image = load_image(img_file)
    width, height = image.size

    print(f"  Image size: {width} × {height}")

    # ------------------------------------------------------------
    # Compute neighbor distances in pixel space
    # ------------------------------------------------------------
    if {"pixel_x", "pixel_y"}.issubset(coords_df.columns):
        xy_pixels = coords_df[["pixel_x", "pixel_y"]].values
        pixel_dists = compute_neighbor_distances(xy_pixels, n_neighbors=args.n_neighbors)

        nearest_d = pixel_dists[:, 0]  # distance to closest neighbor
        mean_d = np.mean(nearest_d)
        std_d = np.std(nearest_d)
        recommended_patch = int(mean_d)

        print(f" Computed pixel-space neighbor distances")
        print(f"   Mean nearest-neighbor distance: {mean_d:.2f} px ± {std_d:.2f}")
        print(f"   Recommended patch size (no overlap): {recommended_patch} × {recommended_patch} pixels")
        print(f"   Example first 5 distances: {np.round(nearest_d[:5], 2)}")

        # ------------------------------------------------------------
        # Save distances and patch size to CSV
        # ------------------------------------------------------------
        out_df = coords_df.copy()
        out_df["nearest_neighbor_distance"] = nearest_d
        out_df["recommended_patch_px"] = recommended_patch
        out_path = os.path.join(args.out_dir, f"{sid}_neighbor_distances.csv")
        out_df.to_csv(out_path, index=False)

        print(f"  Saved neighbor distances and patch info to: {out_path}")

    else:
        print("  pixel_x and pixel_y columns not found, skipping pixel-space computation.")

print("\n Exploration complete.")
print("=" * 100)
print("Tip: Use 'recommended_patch_px' for non-overlapping patch extraction.")
