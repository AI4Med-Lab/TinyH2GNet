import os
import gzip
import argparse
import pandas as pd
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Disable PIL’s large image warning for huge histology files
Image.MAX_IMAGE_PIXELS = None

# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Overlay spot coordinates (pixel_x, pixel_y) on histology image")
parser.add_argument("--base_dir", required=True, help="Folder containing image and spot TSV files")
parser.add_argument("--sample_id", required=True, help="Sample ID prefix (e.g., GSM4284316)")
parser.add_argument("--out_dir", default="overlay_output", help="Folder to save overlay image")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def load_image_gz(path):
    """Load .jpg.gz histology image"""
    with gzip.open(path, "rb") as f:
        img = Image.open(BytesIO(f.read())).convert("RGB")
    return img

def load_spots_tsv(path):
    """Load spot coordinates TSV"""
    return pd.read_csv(path, sep="\t")

# ------------------------------------------------------------
# Locate files
# ------------------------------------------------------------
sid = args.sample_id
files = [f for f in os.listdir(args.base_dir) if f.startswith(sid)]
img_file = next(f for f in files if f.endswith(".jpg.gz"))
spot_file = next(f for f in files if "spot_data-selection" in f)

img_path = os.path.join(args.base_dir, img_file)
spot_path = os.path.join(args.base_dir, spot_file)

print(f" Image: {img_path}")
print(f" Spots: {spot_path}")

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
image = load_image_gz(img_path)
spots = load_spots_tsv(spot_path)

# Ensure required columns exist
required_cols = {"pixel_x", "pixel_y"}
if not required_cols.issubset(spots.columns):
    raise ValueError(f"Missing columns {required_cols - set(spots.columns)} in {spot_path}")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.scatter(spots["pixel_x"], spots["pixel_y"], s=10, c="red", alpha=0.6)
plt.gca().invert_yaxis()  # flip y-axis to match image origin (0,0 at top-left)
plt.title(f"{sid} — Spots overlayed on histology (pixel coordinates)")
plt.axis("off")

# ------------------------------------------------------------
# Save output
# ------------------------------------------------------------
save_path = os.path.join(args.out_dir, f"{sid}_overlay_pixels.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f" Saved overlay visualization: {save_path}")