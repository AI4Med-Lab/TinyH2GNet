import os
import argparse
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import logging

# ======================================================
# CONFIG
# ======================================================
BASE_PATH = "/home/puneet/mk/data/4751624_her2st/code/her2st-master/data"
CNT_DIR = os.path.join(BASE_PATH, "ST-cnts")
SPOT_DIR = os.path.join(BASE_PATH, "ST-spotfiles")
LOG_PATH = "/home/puneet/mk/code/her2st_data_explore_file/log_results"

os.makedirs(LOG_PATH, exist_ok=True)

# Logging setup
logging.basicConfig(
    filename=os.path.join(LOG_PATH, "spot_distance_analysis.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ======================================================
# ARGUMENTS
# ======================================================
parser = argparse.ArgumentParser(description="Compute nearest neighbor distances for HER2-ST samples.")
parser.add_argument(
    "--sample_ids",
    nargs="+",
    required=True,
    help="List of sample IDs to process (e.g., --sample_ids A1 A2 A3)"
)
args = parser.parse_args()

# ======================================================
# MAIN LOOP
# ======================================================
for SAMPLE_ID in args.sample_ids:
    logging.info("=" * 70)
    logging.info(f"Processing sample: {SAMPLE_ID}")
    logging.info("=" * 70)

    CNT_PATH = os.path.join(CNT_DIR, f"{SAMPLE_ID}.tsv.gz")
    SPOT_PATH = os.path.join(SPOT_DIR, f"{SAMPLE_ID}_selection.tsv")

    if not os.path.exists(SPOT_PATH):
        logging.warning(f" Spot file missing for sample {SAMPLE_ID}")
        continue

    try:
        # ----------------------------------------------------------
        # Load spot coordinate file
        # ----------------------------------------------------------
        spots = pd.read_csv(SPOT_PATH, sep="\t")
        logging.info(f"Loaded spot file for {SAMPLE_ID}: {spots.shape[0]} entries")

        if not {"pixel_x", "pixel_y"}.issubset(spots.columns):
            logging.error(f" Missing pixel_x/pixel_y columns for sample {SAMPLE_ID}")
            continue

        # ----------------------------------------------------------
        # Compute pairwise distance matrix
        # ----------------------------------------------------------
        coords = spots[["pixel_x", "pixel_y"]].values
        dist_mat = distance_matrix(coords, coords)  # shape = (n_spots, n_spots)

        # ----------------------------------------------------------
        # For each spot, get distances to its 5 nearest neighbors
        # ----------------------------------------------------------
        # Sort each row (distance to other spots)
        sorted_dists = np.sort(dist_mat, axis=1)
        # Exclude the first column (distance to itself = 0)
        nearest5 = sorted_dists[:, 1:6]

        # ----------------------------------------------------------
        # Store results
        # ----------------------------------------------------------
        dist_df = pd.DataFrame({
            "spot_index": np.arange(len(spots)),
            "x": spots["x"],
            "y": spots["y"],
            "pixel_x": spots["pixel_x"],
            "pixel_y": spots["pixel_y"],
            "dist_1": nearest5[:, 0],
            "dist_2": nearest5[:, 1],
            "dist_3": nearest5[:, 2],
            "dist_4": nearest5[:, 3],
            "dist_5": nearest5[:, 4],
            "mean_5nn": nearest5.mean(axis=1)
        })

        # ----------------------------------------------------------
        # Save CSV
        # ----------------------------------------------------------
        out_csv = os.path.join(LOG_PATH, f"{SAMPLE_ID}_nearest_neighbors.csv")
        dist_df.to_csv(out_csv, index=False)
        logging.info(f" Saved nearest-neighbor distances to: {out_csv}")

        # Log average distance stats
        mean_dist = dist_df["mean_5nn"].mean()
        min_dist = dist_df["mean_5nn"].min()
        max_dist = dist_df["mean_5nn"].max()
        logging.info(f"Average 5NN distance: {mean_dist:.2f} px (min={min_dist:.2f}, max={max_dist:.2f})")

    except Exception as e:
        logging.exception(f"Error processing {SAMPLE_ID}: {e}")

logging.info(" Distance computation complete for all samples.")
print(f" All distance CSVs saved under: {LOG_PATH}")
