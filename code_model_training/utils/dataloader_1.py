import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms as T  

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import scanpy as sc
import anndata
import h5py
import numpy as np
from PIL import Image


def extract_center_patch(img, patch_size=16):
    """
    Extracts the central patch and zeroes out everything else.
    
    Args:
        img (np.ndarray): Input image of shape (H, W, C).
        patch_size (int): Size of the central square patch.
    
    Returns:
        np.ndarray: Image of same shape with only central patch retained, rest zero.
    """
    h, w, c = img.shape
    cy, cx = h // 2, w // 2
    half = patch_size // 2

    # Compute bounds of the central patch
    top = max(cy - half, 0)
    bottom = min(cy + half, h)
    left = max(cx - half, 0)
    right = min(cx + half, w)

    # Create a zero image
    masked_img = np.zeros_like(img)
    masked_img[top:bottom, left:right] = img[top:bottom, left:right]

    return masked_img

class ImageDataset(Dataset):
    def __init__(self, h5ad_path,h5_path, transform=None, patch_size=16):
        # Load HDF5 image data
        self.h5 = h5py.File(h5_path, 'r')
        self.imgs = self.h5['img']  # shape [N, H, W, C] or [N, C, H, W]

        # Load gene expression
        self.adata = anndata.read_h5ad(h5ad_path)
        self.expr = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
        # self.expr = self.expr[:5]
        
        # Sanity check
        assert self.imgs.shape[0] == self.expr.shape[0], "Mismatched samples"

        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Resize((224, 224))
        ])

        self.patch_size = patch_size

    def __len__(self):
        return len(self.expr)

    def __getitem__(self, idx):
        img = self.imgs[idx]  # shape: HWC or CHW
        if isinstance(img, torch.Tensor):  # ensure it's numpy for transform
            img = img.numpy()
        if img.shape[0] <= 4:  # CHW ? HWC
            img = np.transpose(img, (1, 2, 0))
        
        # Apply transform (will convert to CHW + normalize, etc.)
        # img = extract_center_patch(img, patch_size=self.patch_size)
        img = self.transform(img)
        
        genes = torch.tensor(self.expr[idx]).float()
        return img, genes