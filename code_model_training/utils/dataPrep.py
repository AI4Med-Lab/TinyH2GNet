import h5py
import anndata as ad
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import ConcatDataset


import torch
from torch.utils.data import Dataset
import h5py
import scanpy as sc
import numpy as np
import pandas as pd

def log1p_normalization(arr, scale_factor=1000000):
    """Apply log1p normalization to the given array."""
    eps = 1e-8  # small constant to avoid divide-by-zero
    return np.log1p((arr / (np.sum(arr, axis=1, keepdims=True) + eps)) * scale_factor)


class PatchDataset(Dataset):
    def __init__(self, gene_path, img_path, gene_names=None, transform=None, log_norm=True, scale_factor=1000000):
        self.img_path = img_path
        self.gene_path = gene_path
        self.gene_names = gene_names
        self.transform = transform
        self.log_norm = log_norm
        self.scale_factor = scale_factor

        print("Initializing PatchDataset...")
        print(f"Image file: {self.img_path}")
        print(f"Gene file:  {self.gene_path}")


        # === Load image data from HDF5 ===
        with h5py.File(self.img_path, "r") as f:
            # check for possible keys like "patches" or "img"
            if "patches" in f:
                self.image = f["patches"][:]
            elif "img" in f:
                self.image = f["img"][:]
            else:
                raise KeyError(f"No valid image dataset found in {self.img_path}")
        print(f" Loaded images: {self.image.shape}")

        # === Load gene expression data ===
        adata = sc.read_h5ad(self.gene_path)

        # --- If a gene list is provided, subset to those genes ---
        if self.gene_names:
            # gene_list = pd.read_csv(self.gene_csv).iloc[:, 0].tolist()
            gene_list = self.gene_names
            valid_genes = [g for g in gene_list if g in adata.var_names]
            adata = adata[:, valid_genes].copy()
            print(f" Using {len(valid_genes)} valid genes from gene list ({len(gene_list)} total).")

        # --- Convert expression matrix to dense ---
        gene_exp = adata.X
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()

        # --- Apply log1p normalization ---
        if self.log_norm:
            print(" Applying custom log1p normalization...")
            gene_exp = log1p_normalization(gene_exp, scale_factor=self.scale_factor)
            print(f" Applied log1p normalization with scale_factor={self.scale_factor}.")

        self.gene_data = gene_exp
        print(f" Final gene matrix shape: {self.gene_data.shape}")

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx].astype(np.uint8)

        # Apply transformation if provided (resize, normalization, etc.)
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Corresponding gene expression vector
        gene_exp = torch.tensor(self.gene_data[idx], dtype=torch.float32)

        return image, gene_exp
    

# class PatchDataset_2(Dataset):
#     def __init__(self, gene_path, img_path, gene_names=None, transform=None, log_norm=True, scale_factor=1000000):
#         self.img_path = img_path
#         self.gene_path = gene_path
#         self.gene_names = gene_names
#         self.transform = transform
#         self.log_norm = log_norm
#         self.scale_factor = scale_factor

#         print("Initializing PatchDataset...")
#         print(f"Image file: {self.img_path}")
#         print(f"Gene file:  {self.gene_path}")


#         # === Load image data from HDF5 ===
#         with h5py.File(self.img_path, "r") as f:
#             # check for possible keys like "patches" or "img"
#             if "patches" in f:
#                 self.image = f["patches"][:]
#                 self.x = f["x"][:]
#                 self.y = f["y"][:]
#             elif "img" in f:
#                 self.image = f["img"][:]
#             else:
#                 raise KeyError(f"No valid image dataset found in {self.img_path}")
#         print(f" Loaded images: {self.image.shape}")

#         # === Load gene expression data ===
#         adata = sc.read_h5ad(self.gene_path)

#         # --- If a gene list is provided, subset to those genes ---
#         if self.gene_names:
#             # gene_list = pd.read_csv(self.gene_csv).iloc[:, 0].tolist()
#             gene_list = self.gene_names
#             valid_genes = [g for g in gene_list if g in adata.var_names]
#             adata = adata[:, valid_genes].copy()
#             print(f" Using {len(valid_genes)} valid genes from gene list ({len(gene_list)} total).")

#         # --- Convert expression matrix to dense ---
#         gene_exp = adata.X
#         if not isinstance(gene_exp, np.ndarray):
#             gene_exp = gene_exp.toarray()

#         # --- Apply log1p normalization ---
#         if self.log_norm:
#             print(" Applying custom log1p normalization...")
#             gene_exp = log1p_normalization(gene_exp, scale_factor=self.scale_factor)
#             print(f" Applied log1p normalization with scale_factor={self.scale_factor}.")

#         self.gene_data = gene_exp
#         print(f" Final gene matrix shape: {self.gene_data.shape}")

#     def __len__(self):
#         return len(self.image)

#     def __getitem__(self, idx):
#         image = self.image[idx].astype(np.uint8)
#         x = self.x[idx].astype(np.uint8)
#         y = self.y[idx].astype(np.uint8)

#         # Apply transformation if provided (resize, normalization, etc.)
#         if self.transform:
#             image = self.transform(image)
#         else:
#             image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

#         # Corresponding gene expression vector
#         gene_exp = torch.tensor(self.gene_data[idx], dtype=torch.float32)
#         x = torch.tensor(self.x[idx], dtype=torch.float32)
#         y = torch.tensor(self.y[idx], dtype=torch.float32)

#         return image, gene_exp, x, y