import h5py
import scanpy as sc
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def log1p_normalization_train_val_separate(
    raw_genes,
    train_idx,
    val_idx,
    scale_factor=1_000_000,
):
    """
    Fold-safe normalization (TRAIN & VAL SEPARATE):
    - train normalized using train-only statistics
    - val normalized using val-only statistics
    - NO shared scale
    """

    # -------------------------
    # TRAIN normalization
    # -------------------------
    train_genes = raw_genes[train_idx]

    train_lib = train_genes.sum(axis=1, keepdims=True)
    train_lib[train_lib == 0] = 1.0

    train_scale = scale_factor / train_lib.mean()

    train_norm = np.log1p(
        (train_genes / train_lib) * train_scale
    ).astype(np.float32)

    # -------------------------
    # VAL normalization
    # -------------------------
    val_genes = raw_genes[val_idx]

    val_lib = val_genes.sum(axis=1, keepdims=True)
    val_lib[val_lib == 0] = 1.0

    val_scale = scale_factor / val_lib.mean()

    val_norm = np.log1p(
        (val_genes / val_lib) * val_scale
    ).astype(np.float32)

    # -------------------------
    # Repack into full arrays
    # -------------------------
    train_norm_full = np.zeros_like(raw_genes, dtype=np.float32)
    val_norm_full   = np.zeros_like(raw_genes, dtype=np.float32)

    train_norm_full[train_idx] = train_norm
    val_norm_full[val_idx]     = val_norm

    return train_norm_full, val_norm_full


def load_xenium_raw_data(h5_img_path, h5ad_gene_path, gene_names=None):
    """
    Loads full Xenium tissue (single slide)
    Returns raw_images, raw_genes, pixel_x, pixel_y
    """

    # -----------------------------
    # Load image + coords
    # -----------------------------
    # -----------------------------
    # Load image + coords
    # -----------------------------
    with h5py.File(h5_img_path, "r") as f:
        images = f["patches"][:]        # (N, H, W, 3)
        pixel_x = f["pixel_x"][:].astype(np.float32)
        pixel_y = f["pixel_y"][:].astype(np.float32)

    print(f"Loaded patches: {images.shape}")
    print(f"pixel_x range: {pixel_x.min():.1f} → {pixel_x.max():.1f}")
    print(f"pixel_y range: {pixel_y.min():.1f} → {pixel_y.max():.1f}")

    # -----------------------------
    # GLOBAL coordinate normalization
    # -----------------------------
    x_mean = pixel_x.mean()
    x_std  = pixel_x.std() + 1e-6
    y_mean = pixel_y.mean()
    y_std  = pixel_y.std() + 1e-6

    pixel_x = (pixel_x - x_mean) / x_std
    pixel_y = (pixel_y - y_mean) / y_std

    print(
        f"[Global coords normalized] "
        f"x μ={x_mean:.2f}, σ={x_std:.2f} | "
        f"y μ={y_mean:.2f}, σ={y_std:.2f}"
    )

    # -----------------------------
    # Load gene expression
    # -----------------------------
    adata = sc.read_h5ad(h5ad_gene_path)

    if gene_names is not None:
        valid_genes = [g for g in gene_names if g in adata.var_names]
        adata = adata[:, valid_genes].copy()
        print(f"Using {len(valid_genes)} genes")

    gene_exp = adata.X
    if not isinstance(gene_exp, np.ndarray):
        gene_exp = gene_exp.toarray()

    print(f"Gene matrix: {gene_exp.shape}")

    return (
        images.astype(np.uint8),
        gene_exp.astype(np.float32),
        pixel_x,   # already normalized
        pixel_y    # already normalized
    )
    

class XeniumLoCIGENDataset(Dataset):
    def __init__(
        self,
        images,
        genes,
        pixel_x,
        pixel_y,
        indices,
        transform=None,
    ):
        self.images = images
        self.genes = genes
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        img = self.images[i]
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0

        gene = torch.tensor(self.genes[i], dtype=torch.float32)

        coord = torch.tensor(
            [self.pixel_x[i], self.pixel_y[i]],
            dtype=torch.float32
        )

        return img, coord, gene
