"""

"""


import sys

sys.path.append("/home/sukrit/image_2_gene/code_model_training/models")

sys.path.append("/home/sukrit/image_2_gene/code_model_training/utils")

import os
import json
import torch
import numpy as np 
import pandas as pd 
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from compute_metrics_evaluation import compute_metrics
from models import STNet, EfficientNet, EfficientNetB4GeneRegressor, Custom_VGG16, HisToGene, TCGN, EfficientNet_GeneCaptionContrastive
from torchvision import transforms
import h5py
import anndata as ad
from sklearn.model_selection import train_test_split, KFold, GroupKFold
import timm
from tqdm import tqdm
import scanpy as sc
import gc

def set_seed(seed=123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(123)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128


params = {
    "dataset_name": "cscc",
    "model": "EfficientNet",
    "model_directory": "/home/sukrit/image_2_gene/code_model_training/modelResults/cscc_EfficientNet_true_new_data_split_patients_wise/result_2026-02-03_17-23-36",
    "dataset_path": "/home/sukrit/image_2_gene/data/cscc_dataset/224",
    "genes": "/home/sukrit/image_2_gene/code_model_training/gene_outputs",
    "hvg20": "/home/sukrit/image_2_gene/code_data_preprocessing/hvg/hvg_lists/cscc_hvg_top_20.csv",
    "hvg50": "/home/sukrit/image_2_gene/code_data_preprocessing/hvg/hvg_lists/cscc_hvg_top_50.csv",
    "embed_dim": 512,
    "nhead": 8,
    "dim_feedforward": 1024,
    "batch_size": 128,
    "device": "cuda",
    "scale_factor": 1000000
}



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



def evaluate(**params):

    dataset = params["dataset_name"]
    dataset_path = params["dataset_path"]

    if params["dataset_name"] == "cscc":
        gsm_samples = [ 'GSM4284316_P2','GSM4284317_P2','GSM4284318_P2',
                        'GSM4284319_P5','GSM4284320_P5','GSM4284321_P5',
                        'GSM4284322_P9','GSM4284323_P9','GSM4284324_P9',
                        'GSM4284325_P10','GSM4284326_P10','GSM4284327_P10']
        
    if params["dataset_name"] == "hist2st_845":
        gsm_samples = [
            '23209_C1', '23209_C2', '23209_D1', '23268_C1', '23268_C2', '23268_D1', '23269_C1', '23269_C2',
            '23269_D1', '23270_D2', '23270_E1', '23270_E2', '23272_D2', '23272_E1', '23272_E2', '23277_D2',
            '23277_E1', '23277_E2', '23287_C1', '23287_C2', '23287_D1', '23288_D2', '23288_E1', '23288_E2',
            '23377_C1', '23377_C2', '23377_D1', '23450_D2', '23450_E1', '23450_E2', '23506_C1', '23506_C2',
            '23506_D1', '23508_D2', '23508_E1', '23508_E2', '23567_D2', '23567_E1', '23567_E2', '23803_D2',
            '23803_E1', '23803_E2', '23810_D2', '23810_E1', '23810_E2', '23895_C1', '23895_C2', '23895_D1',
            '23901_C2', '23901_D1', '23903_C1', '23903_C2', '23903_D1', '23944_D2', '23944_E1', '23944_E2',
            '24044_D2', '24044_E1', '24044_E2', '24105_C1', '24105_C2', '24105_D1', '24220_D2', '24220_E1',
            '24220_E2', '24223_D2', '24223_E1', '24223_E2'
        ]

    if params["dataset_name"] == "her2":
        gsm_samples =  [
            "A1", "A2", "A3", "A4", "A5", "A6",
            "B1", "B2", "B3", "B4", "B5", "B6",
            "C1", "C2", "C3", "C4", "C5", "C6",
            "D1", "D2", "D3", "D4", "D5", "D6",
            "E1", "E2", "E3",
            "F1", "F2", "F3",
            "G1", "G2", "G3",
            "H1", "H2", "H3"
        ]

    # -------------------------
    # File paths
    # -------------------------

    def get_patient_id(sample, dataset_name):
        if dataset_name == "cscc":
            # GSM4284316_P2 → P2
            return sample.split("_")[1]

        elif dataset_name == "hist2st_845":
            # 23209_C1 → 23209
            return sample.split("_")[0]

        elif dataset_name == "her2":
            # A1 → A
            return sample[0]

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    if params["dataset_name"] == "cscc":

        geneset = pd.read_csv("geneAnalysisData/cscc_gene_to_cancerdis.csv")
        patch_paths = [os.path.join(dataset_path, f"{sample}_patches.h5") for sample in gsm_samples]
        adata_paths = [os.path.join(dataset_path, f"{sample}_spots.h5ad") for sample in gsm_samples]

    if params["dataset_name"] == "her2":
        patch_paths = [os.path.join(dataset_path, f"{sample}.h5") for sample in gsm_samples]
        adata_paths = [os.path.join(dataset_path, f"{sample}.h5ad") for sample in gsm_samples]

    if params["dataset_name"] == "hist2st_845":
        patch_paths = [os.path.join(dataset_path, f"{sample}.h5") for sample in gsm_samples]
        adata_paths = [os.path.join(dataset_path, f"{sample}.h5ad") for sample in gsm_samples]


    # --- Check file existence ---
    for p, a in zip(patch_paths, adata_paths):
        if not os.path.exists(p):
            print(f"Missing patch file: {p}")
        if not os.path.exists(a):
            print(f"Missing spot file: {a}")

    # --- Patient grouping (for leakage-free CV) ---
    patient_ids = [get_patient_id(s, params["dataset_name"]) for s in gsm_samples]
    unique_patients = sorted(set(patient_ids))

    print(f"Total patients: {len(unique_patients)} → {unique_patients}")

    # --- Gene CSV file ---
    genes_file = os.path.join(params['genes'], f"{params['dataset_name']}.npy")
    print("+++++++++++++++++++++++++++++++",genes_file)
    gene_names = np.load(genes_file, allow_pickle=True).tolist()
    print(f" Using gene list from: {gene_names}")
    if not os.path.exists(genes_file):
        # raise FileNotFoundError(f"Gene list not found: {genes_file}")
        print(genes_file)
        print(gene_names)

    # --- Optional image transforms ---
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # --- K-Fold setup ---
    k = params.get("k_folds", 5)
    n_patients = len(unique_patients)

    if k > n_patients:
        print(
            f"k_folds ({k}) > number of patients ({n_patients}). "
            f"Setting k = {n_patients}"
        )
        k = n_patients

    kf = GroupKFold(n_splits=k, shuffle=True, random_state=42)
    fold_models = []


    if params["dataset_name"] == "cscc":
        marker_genes = ["EGR3","KRT16","PI3","TP53","NOTCH1","CDKN2A","HIF1A","MMP3","CCNA2","CCNB2","UBE2C"]
    else:
        marker_genes = ["BRCA1","BRCA2","BARD1","BRIP1","PALB2","RAD51","RAD54L","XRCC3","ERBB2","HER2","ESR1","ER1","PGR","GATA3","PIK3CA","TP53","PPM1D","RB1CC1","HMMR","NQO2","SLC22A18","PTEN","EGFR","KIT","NOTCH1","NOTCH4","FZD7","LRP6","FGFR1","CCND1"]
 


    # -----------------------------
    # Load genes & HVGs
    # -----------------------------
    gene_names = np.load(
        os.path.join(params["genes"], f"{dataset}.npy"),
        allow_pickle=True
    ).tolist()

    num_genes = len(gene_names)


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    datasets = [
        PatchDataset(
            gene_path=a,
            img_path=p,
            gene_names=gene_names,
            transform=transform,
            log_norm=True,
            scale_factor=1000000
        )
        for a, p in zip(adata_paths, patch_paths)
    ]

    full_dataset = ConcatDataset(datasets)

    model_name = params["model"]

    # num_genes = y_true.shape[1]
    if model_name == "EfficientNet":
        model = EfficientNet(num_genes=num_genes)
    elif model_name == "STNet":
        model = STNet(num_genes=num_genes)
    elif model_name == "Custom_VGG16":
        model = Custom_VGG16(num_genes=num_genes)
    elif model_name == "EfficientNet_GeneCaptionContrastive":
        model = EfficientNet_GeneCaptionContrastive(num_genes=num_genes)

    elif model_name == "HisToGene":
            model = HisToGene(patch_size=16, n_layers= 8, n_genes=num_genes)
            
    elif model_name == "TCGN":

        # Set up model parameters
        tcgn_kwargs = dict(
            img_size= 224,
            in_chans=3,
            num_classes=num_genes,
            embed_dims=[52, 104, 208, 416],
            stem_channel=16,
            fc_dim=1280,
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[3.6, 3.6, 3.6, 3.6],
            qkv_bias=True,
            qk_scale=None,
            representation_size=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            hybrid_backbone=None,
            norm_layer=None,
            depths=[2, 2, 10, 2],
            qk_ratio=1,
            sr_ratios=[8, 4, 2, 1],
            dp=0.1
        )

        # Instantiate the model
        model = TCGN(**tcgn_kwargs)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # kf = KFold(n_splits=k, shuffle=True, random_state=42)

    model_1 = model
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(gsm_samples, groups=patient_ids)):
        train_patients = sorted(set(patient_ids[i] for i in train_idx))
        val_patients   = sorted(set(patient_ids[i] for i in val_idx))

        print(f"Train patients: {len(train_patients)}, {train_patients}")
        print(f"Val patients: {len(val_patients)},  {val_patients}")

        val_patch_files   = [patch_paths[i] for i in val_idx]
        val_adata_files   = [adata_paths[i] for i in val_idx]

        print(f"\n===== Fold {fold_idx+1} =====")

        val_set = Subset(full_dataset, val_idx)

        model_path = os.path.join(
            params["model_directory"],
            f"fold_{fold_idx+1}",
            "best_model.pth"
        )

        model = model_1
        checkpoint = torch.load(model_path)

        model.load_state_dict(checkpoint)

        model.to(DEVICE)

        model.eval()

        loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

        preds, gts = [], []

        results = []

        with torch.no_grad():
            for imgs, genes in tqdm(loader, desc="Predicting"):

                imgs = imgs.to(DEVICE)
                genes = genes.to(DEVICE)


                out = model(imgs)
                if isinstance(out, tuple):
                    out = out[0]

                preds.append(out.cpu().numpy())
                gts.append(genes.cpu().numpy())

            preds, gts =  np.vstack(preds), np.vstack(gts)

            gs_idx = [i for i, gene in enumerate(gene_names) if gene in marker_genes]


            print(f"gs_idx: {gs_idx}")

            result = compute_metrics(gts[:, gs_idx], preds[:, gs_idx])

            results.append(result)

            print(f"results: {results} \n")


            torch.cuda.empty_cache()
            gc.collect()

            print("================================================================")


    res_df = pd.DataFrame(results)

    mean_vals = res_df.mean()
    std_vals  = res_df.std()

    summary = res_df.agg(['mean', 'std']).T
    print(summary)



if __name__ == "__main__":
    evaluate(**params)
