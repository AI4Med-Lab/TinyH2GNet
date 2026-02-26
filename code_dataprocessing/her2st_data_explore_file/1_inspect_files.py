import os
import pandas as pd
import argparse
import logging

# ======================================================
# CONFIGURATION
# ======================================================
BASE_PATH = "/home/puneet/mk/data/4751624_her2st/code/her2st-master/data"
LOG_PATH = "/home/puneet/mk/code/her2st_data_explore_file/log_results"
CNT_DIR = os.path.join(BASE_PATH, "ST-cnts")
SPOT_DIR = os.path.join(BASE_PATH, "ST-spotfiles")
LOG_FILE = os.path.join(LOG_PATH, "explore_ST.log")

# ======================================================
# LOGGER SETUP
# ======================================================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logging.getLogger("").addHandler(console)

# ======================================================
# ARGUMENT PARSER
# ======================================================
parser = argparse.ArgumentParser(description="Explore HER2-ST spatial data and log spot matching info.")
parser.add_argument(
    "--sample_ids",
    nargs="+",
    type=str,
    default=None,
    help="List of sample IDs to explore (e.g., --sample_ids A1 A2 A3). If not given, explores all samples."
)
args = parser.parse_args()

# ======================================================
# DETECT AVAILABLE SAMPLES
# ======================================================
all_samples = sorted([
    f.replace(".tsv.gz", "")
    for f in os.listdir(CNT_DIR)
    if f.endswith(".tsv.gz")
])

logging.info(f"Total available samples: {len(all_samples)}")
logging.info(f"Sample IDs: {', '.join(all_samples)}")

if args.sample_ids:
    selected_samples = [s for s in args.sample_ids if s in all_samples]
    missing = [s for s in args.sample_ids if s not in all_samples]
    if missing:
        logging.warning(f"Samples not found in ST-cnts folder: {', '.join(missing)}")
else:
    selected_samples = all_samples

logging.info(f"Samples selected for exploration: {', '.join(selected_samples)}")

# ======================================================
# MAIN EXPLORATION LOOP
# ======================================================
for SAMPLE_ID in selected_samples:
    logging.info("=" * 80)
    logging.info(f"Exploring sample: {SAMPLE_ID}")
    logging.info("=" * 80)

    CNT_PATH = os.path.join(CNT_DIR, f"{SAMPLE_ID}.tsv.gz")
    SPOT_PATH = os.path.join(SPOT_DIR, f"{SAMPLE_ID}_selection.tsv")

    if not os.path.exists(CNT_PATH):
        logging.error(f"Missing count file: {CNT_PATH}")
        continue
    if not os.path.exists(SPOT_PATH):
        logging.error(f"Missing spot file: {SPOT_PATH}")
        continue

    try:
        # --- Load count matrix ---
        cnts = pd.read_csv(CNT_PATH, sep="\t", index_col=0)
        logging.info(f"Expression matrix shape: {cnts.shape[0]} spots × {cnts.shape[1]} genes")

        # --- Load spatial spot file ---
        spots = pd.read_csv(SPOT_PATH, sep="\t")
        logging.info(f"Spot file shape: {spots.shape[0]} entries | Columns: {list(spots.columns)}")

        # --- Extract coordinates from expression file ---
        spot_coords = cnts.index.str.extract(r'(?P<x>\d+)x(?P<y>\d+)').astype(int)
        spot_coords.index = cnts.index

        # --- Merge and match ---
        merged = pd.merge(spot_coords, spots, on=["x", "y"], how="inner")

        total_expr = len(spot_coords)
        total_spots = len(spots)
        matched = len(merged)
        missing = total_expr - matched

        logging.info(f"Spots in expression file: {total_expr}")
        logging.info(f"Spots in coordinate file: {total_spots}")
        logging.info(f"Matching spots: {matched}")
        logging.info(f"Missing spots: {missing}")
        if missing == 0:
            logging.info("All expression spots found in spot file.")
        else:
            logging.warning(f" {missing} expression spots missing from spot file.")

        # --- Show sample matched entries ---
        logging.info("Example matched entries:")
        logging.info("\n" + str(merged.head().to_string()))

    except Exception as e:
        logging.exception(f"Error processing {SAMPLE_ID}: {e}")

# ======================================================
# DONE
# ======================================================
logging.info("=" * 80)
logging.info(" Exploration complete for all selected samples.")
logging.info("=" * 80)

print(" Exploration complete. Detailed log saved to:", LOG_FILE)