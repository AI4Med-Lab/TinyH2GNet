import h5py
from torch.utils.data import Dataset
import torch
import scanpy as sc
import numpy as np

def log1p_normalization(arr, scale_factor=1000000):
    eps = 1e-8
    return np.log1p((arr / (np.sum(arr, axis=1, keepdims=True) + eps)) * scale_factor)


class PatchDataset(Dataset):
    def __init__(
        self,
        gene_path,
        img_path,
        gene_names=None,
        transform=None,
        log_norm=True,
        scale_factor=1000000
    ):
        self.img_path = img_path
        self.gene_path = gene_path
        self.gene_names = gene_names
        self.transform = transform
        self.log_norm = log_norm
        self.scale_factor = scale_factor

        print(f"Loading image H5: {self.img_path}")

        with h5py.File(self.img_path, "r") as f:
            self.images = f["patches"][:]      # (N,H,W,3)
            self.pixel_x = f["pixel_x"][:].astype(np.float32)
            self.pixel_y = f["pixel_y"][:].astype(np.float32)

        print("Images:", self.images.shape)
        print("pixel_x range:", self.pixel_x.min(), "→", self.pixel_x.max())
        print("pixel_y range:", self.pixel_y.min(), "→", self.pixel_y.max())

        # ============================================================
        # 🔥 PER-SLIDE COORDINATE NORMALIZATION (CRITICAL)
        # ============================================================
        ### NEW
        self.x_mean = self.pixel_x.mean()
        self.x_std  = self.pixel_x.std() + 1e-6
        self.y_mean = self.pixel_y.mean()
        self.y_std  = self.pixel_y.std() + 1e-6

        self.pixel_x = (self.pixel_x - self.x_mean) / self.x_std
        self.pixel_y = (self.pixel_y - self.y_mean) / self.y_std

        print(
            f"Normalized coords: "
            f"x μ={self.x_mean:.2f}, σ={self.x_std:.2f} | "
            f"y μ={self.y_mean:.2f}, σ={self.y_std:.2f}"
        )
        # ============================================================

        # --- Load gene expression ---
        adata = sc.read_h5ad(self.gene_path)

        if self.gene_names:
            valid_genes = [g for g in self.gene_names if g in adata.var_names]
            adata = adata[:, valid_genes].copy()
            print(f"Using {len(valid_genes)} genes")

        gene_exp = adata.X
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()

        if self.log_norm:
            gene_exp = log1p_normalization(gene_exp, scale_factor=self.scale_factor)

        self.gene_data = gene_exp.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.uint8)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        gene_exp = torch.tensor(self.gene_data[idx], dtype=torch.float32)

        # (x,y) normalized per-slide
        coord = torch.tensor(
            [self.pixel_x[idx], self.pixel_y[idx]],
            dtype=torch.float32
        )

        return img, coord, gene_exp


def compute_gene_stats(dataloader):
    all_genes = []

    for _, _, gene_exp in dataloader:
        all_genes.append(gene_exp)

    all_genes = torch.cat(all_genes, dim=0)  # (N, G)

    mean = all_genes.mean(dim=0)
    std = all_genes.std(dim=0)

    std[std < 1e-6] = 1.0  # prevent divide-by-zero

    return mean, std
