#!/usr/bin/env python3
"""
Generate H5 and H5AD files for Xenium H&E + Spatial Transcriptomics Data

Outputs:
1️ cells_patches.h5  → (N, H, W, C) array + coordinates
2️ cells_expression.h5ad → gene expression + spatial coordinates
"""

import os
import h5py
import zarr
import tifffile
import json
import argparse
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm

# ---------------------------------------------------------------------
#   Argument Parser
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate .h5 and .h5ad for Xenium dataset")
    parser.add_argument("--cells_zarr", required=True, help="Path to cells.zarr.zip")
    parser.add_argument("--input_files", required=True, help="Path to directory containing experiment.xenium")
    parser.add_argument("--he_image", required=True, help="Path to H&E image (.ome.tif)")
    parser.add_argument("--gene_expr", required=True, help="Path to cell_feature_matrix.h5 or .h5ad")
    parser.add_argument("--alignment_csv", required=True, help="Path to H&E alignment CSV file")
    parser.add_argument("--output_dir", required=True, help="Directory to save .h5 and .h5ad files")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size in pixels (default: 128)")
    parser.add_argument("--log_file", default="xenium_processing.log", help="Log file path")
    return parser.parse_args()


# ---------------------------------------------------------------------
#  Logging setup
# ---------------------------------------------------------------------
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.info("=== Xenium H&E Patch + Expression Generator Initialized ===")


# ---------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------
def load_transforms(cells_zarr, align_csv, experiment_json):
    """Load alignment, mask scale, and compose final µm→pixel transform."""
    T_he = pd.read_csv(align_csv, header=None).values
    T_inv = np.linalg.inv(T_he)
    mask_homog = np.array(cells_zarr["masks/homogeneous_transform"])
    scale_from_mask = mask_homog[0, 0]

    with open(experiment_json) as f:
        meta = json.load(f)
    pixel_size_um = meta.get("pixel_size", 0.2125)

    S = scale_from_mask
    T_full = T_inv @ np.array([[S, 0, 0], [0, S, 0], [0, 0, 1]])

    logging.info(f"Affine (H&E→Xenium):\n{T_he}")
    logging.info(f"Inverse (Xenium→H&E):\n{T_inv}")
    logging.info(f"Composed transform (µm→H&E pixels):\n{T_full}")
    logging.info(f"Scale from mask: {S:.4f} px/µm, Pixel size: {pixel_size_um:.4f} µm/px")
    return T_full, pixel_size_um


def extract_patch(img, x, y, size):
    """Extract centered patch around (x, y)."""
    half = size // 2
    H, W = img.shape[:2]
    x1, y1 = int(x - half), int(y - half)
    x2, y2 = int(x + half), int(y + half)
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None
    return img[y1:y2, x1:x2, :]


def read_xenium_expression(path):
    """Read Xenium/10x-style .h5 or .h5ad expression matrix safely (auto-detects orientation and fixes shape mismatch)."""
    import scipy.sparse as sp

    logging.info(f"Loading gene expression from: {path}")
    if path.endswith(".h5ad"):
        adata = sc.read(path)
        logging.info(f"Loaded existing .h5ad: {adata.n_obs} cells × {adata.n_vars} genes")
        return adata

    with h5py.File(path, "r") as f:
        grp = f["matrix"]
        shape = grp["shape"][:]
        data = grp["data"][:]
        indices = grp["indices"][:]
        indptr = grp["indptr"][:]
        n1, n2 = shape
        logging.info(f"Detected matrix shape: {n1} × {n2}")
        logging.info(f"CSR components: data={len(data)}, indptr={len(indptr)}")

        # --- Auto-detect correct orientation ---
        # --- Auto-detect correct orientation ---
        if len(indptr) == n1 + 1:
            # Typical case: already cells × genes
            logging.info("Orientation: cells × genes (direct)")
            X = sp.csr_matrix((data, indices, indptr), shape=(n1, n2))
            n_cells, n_genes = n1, n2
        elif len(indptr) == n2 + 1:
            # Our case: genes × cells — we build with swapped shape
            logging.info("Orientation: genes × cells (swapping shape without .T)")
            X = sp.csr_matrix((data, indices, indptr), shape=(n2, n1))
            n_cells, n_genes = n2, n1
        else:
            raise ValueError(
                f"Invalid CSR structure: indptr={len(indptr)} incompatible with shape={shape}"
            )

        # --- Decode metadata ---
        barcodes = [b.decode("utf-8") for b in grp["barcodes"][:]]
        features = grp["features"]
        gene_names = [g.decode("utf-8") for g in features["name"][:]]
        gene_ids = [g.decode("utf-8") for g in features["id"][:]]
        feature_types = [g.decode("utf-8") for g in features["feature_type"][:]]
        genome = [g.decode("utf-8") for g in features["genome"][:]]

    # --- Construct AnnData ---
    adata = sc.AnnData(X=X)
    adata.obs_names = barcodes
    adata.var_names = gene_names
    adata.var["gene_ids"] = gene_ids
    adata.var["feature_types"] = feature_types
    adata.var["genome"] = genome

    logging.info(f" Final AnnData: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata

# ---------------------------------------------------------------------
#   Main Processing
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    setup_logging(args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("=== Loading Xenium Centroids and Transforms ===")
    cells_z = zarr.open(args.cells_zarr, mode="r")
    cell_summary = np.array(cells_z["cell_summary"])
    cell_ids = np.array(cells_z["cell_id"])
    x_um, y_um = cell_summary[:, 0], cell_summary[:, 1]
    logging.info(f"Loaded {len(x_um)} centroids")

    experiment_json = os.path.join(args.input_files, "experiment.xenium")
    T_full, pixel_size_um = load_transforms(cells_z, args.alignment_csv, experiment_json)

    coords_um = np.vstack([x_um, y_um, np.ones_like(x_um)])
    coords_px = T_full @ coords_um
    x_px, y_px = coords_px[0], coords_px[1]

    # -----------------------------------------------------------------
    # Load H&E Image
    # -----------------------------------------------------------------
    logging.info("Loading H&E image...")
    img = tifffile.imread(args.he_image)
    if img.ndim == 3 and img.shape[0] in (3, 4):
        img = np.transpose(img[:3], (1, 2, 0))
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        img = img[:, :, :3]
    elif img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    else:
        img = np.max(img, axis=0)
        img = np.stack([img] * 3, axis=-1)

    H, W = img.shape[:2]
    valid = (x_px >= 0) & (x_px < W) & (y_px >= 0) & (y_px < H)
    x_px, y_px, cell_ids = x_px[valid], y_px[valid], cell_ids[valid]
    logging.info(f"Valid centroids: {len(x_px)} / {len(valid)} within bounds")

    # -----------------------------------------------------------------
    # Generate H&E patches
    # -----------------------------------------------------------------
    patch_size = args.patch_size
    logging.info(f"Extracting {len(x_px)} patches of size {patch_size}×{patch_size}")
    patches = []
    for cx, cy in tqdm(zip(x_px, y_px), total=len(x_px), desc="Extracting patches"):
        p = extract_patch(img, cx, cy, patch_size)
        if p is not None:
            patches.append(p)
    patches = np.array(patches, dtype=np.uint8)

    h5_out = os.path.join(args.output_dir, "cells_patches.h5")
    with h5py.File(h5_out, "w") as hf:
        hf.create_dataset("patches", data=patches, compression="gzip")
        hf.create_dataset("pixel_x", data=x_px)
        hf.create_dataset("pixel_y", data=y_px)
        hf.create_dataset("cell_ids", data=cell_ids)
        hf.create_dataset("coords_px", data=np.vstack([x_px, y_px]).T)

    logging.info(f"Saved patches to {h5_out} with shape {patches.shape}")

    # -----------------------------------------------------------------
    # Generate H5AD (expression + spatial coordinates)
    # -----------------------------------------------------------------
    logging.info("Loading and aligning gene expression matrix...")
    adata = read_xenium_expression(args.gene_expr)
    obs_ids = np.array(adata.obs_names, dtype=str)

    coord_map = {str(cid): (x, y) for cid, x, y in zip(cell_ids, x_px, y_px)}
    adata.obs["x_pixel"] = [coord_map.get(cid, (np.nan, np.nan))[0] for cid in obs_ids]
    adata.obs["y_pixel"] = [coord_map.get(cid, (np.nan, np.nan))[1] for cid in obs_ids]
    adata.obs["x_um"] = adata.obs["x_pixel"] * pixel_size_um
    adata.obs["y_um"] = adata.obs["y_pixel"] * pixel_size_um
    adata.obs["cell_id"] = obs_ids

    adata.obsm["spatial"] = np.vstack([adata.obs["x_pixel"], adata.obs["y_pixel"]]).T
    adata.uns["spatial_metadata"] = {
        "pixel_size_um": float(pixel_size_um),
        "patch_size_px": args.patch_size,
        "transform_matrix": T_full.tolist(),
        "alignment_csv": args.alignment_csv,
    }

    h5ad_out = os.path.join(args.output_dir, "cells_expression.h5ad")
    adata.write(h5ad_out)
    logging.info(f" Saved H5AD with {adata.n_obs} cells × {adata.n_vars} genes to {h5ad_out}")

    logging.info(" Processing complete for both .h5 and .h5ad outputs.")


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()