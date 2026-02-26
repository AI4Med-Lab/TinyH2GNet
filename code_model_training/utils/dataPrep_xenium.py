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
    """
    Apply log1p normalization to rows of arr:
      normalized_row = log1p( (row / row.sum()) * scale_factor )

    Handles rows with sum 0 by leaving them zero (avoids NaN).
    """
    arr = np.asarray(arr, dtype=np.float64)
    lib = arr.sum(axis=1, keepdims=True)
    # avoid divide by zero
    lib[lib == 0] = 1.0
    scaled = (arr / lib) * float(scale_factor)
    return np.log1p(scaled)

class PatchDataset(Dataset):
    def __init__(self, gene_path, img_path, gene_names=None, transform=None):
        self.img_path = img_path
        self.gene_path = gene_path
        self.gene_names = gene_names
        self.transform = transform

        print("Initializing PatchDataset...")
        print(f"Image file: {self.img_path}")
        print(f"Gene file:  {self.gene_path}")

        # === Load images ===
        with h5py.File(self.img_path, "r") as f:
            if "patches" in f:
                self.image = f["patches"][:]
            elif "img" in f:
                self.image = f["img"][:]
            else:
                raise KeyError(f"No valid image dataset found in {self.img_path}")
        print(f" Loaded images: {self.image.shape}")

        # === Load gene expression ===
        adata = sc.read_h5ad(self.gene_path)

        # subset genes if needed
        if gene_names:
            valid_genes = [g for g in gene_names if g in adata.var_names]
            adata = adata[:, valid_genes].copy()

        gene_exp = adata.X
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()

        # Store raw counts only
        self.gene_data = gene_exp.astype(np.float32)
        print(f" Raw gene matrix: {self.gene_data.shape}")

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = self.image[idx].astype(np.uint8)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        gene_raw = torch.tensor(self.gene_data[idx], dtype=torch.float32)

        return img, gene_raw


class FoldDataset(Dataset):
    def __init__(self, image_data, gene_data, indices, transform=None):
        """
        image_data: numpy array of shape (N, H, W, 3)
        gene_data: numpy array of shape (N, G)
        indices: list or array of sample indices for this fold
        transform: torchvision transforms
        """
        self.image_data = image_data
        self.gene_data = gene_data
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        img = self.image_data[real_idx].astype(np.uint8)
        genes = torch.tensor(self.gene_data[real_idx], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return img, genes

