import sys
sys.path.append('/home/sukrit/image_2_gene/code_model_training/models')
sys.path.append('/home/sukrit/image_2_gene/code_model_training/utils')

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import KFold
import random
from utils.data_prep_cell_info import PatchDataset
from compute_metrics import compute_metrics
from compute_metric_cell_type import masked_kl_divergence, masked_spearman_celltype, masked_l1_celltype 
from models import STNet, EfficientNet, EfficientNetB4GeneRegressor, Custom_VGG16, HisToGene, TCGN, EfficientNet_GeneCaptionContrastive
from torchvision import transforms
from models_celltype import EfficientNet_CellType_CrossAttention
# =========================================================
# GLOBALS
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128


# =========================================================
# Utilities
# =========================================================
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_name, model_path, num_genes,  num_sc_genes=None, num_celltypes=None, params=None):
    if model_name == "EfficientNet":
        model = EfficientNet(num_genes=num_genes)
    elif model_name == "STNet":
        model = STNet(num_genes=num_genes)
    elif model_name == "Custom_VGG16":
        model = Custom_VGG16(num_genes=num_genes)
    elif model_name == "EfficientNet_GeneCaptionContrastive":
        model = EfficientNet_GeneCaptionContrastive(num_genes=num_genes)
    elif model_name == "EfficientNet_CellType_CrossAttention":
            assert num_sc_genes is not None, "num_sc_genes is required"
            assert num_celltypes is not None, "num_celltypes is required"
            
            model = EfficientNet_CellType_CrossAttention(
                num_sc_genes=num_sc_genes,
                num_st_genes=num_genes,
                num_celltypes=num_celltypes,
                embed_dim=256,
                num_heads= 4,
                pretrained=True
            )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    return model

def predict(model, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    preds, gts = [], []
    prop_preds, prop_gts, token_masks = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):

            if len(batch) == 4:
                images, st_gene, prop_gt, token_mask = batch
            else:
                raise ValueError("Dataset must return (image, gene, prop, mask)")

            images = images.to(DEVICE)
            st_gene = st_gene.to(DEVICE)
            prop_gt = prop_gt.to(DEVICE)
            token_mask = token_mask.to(DEVICE)

            out = model(images)

            if isinstance(out, tuple):
                y_pred, prop_pred, _ = out
            else:
                y_pred = out
                prop_pred = None

            preds.append(y_pred.detach().cpu())
            gts.append(st_gene.detach().cpu())

            if prop_pred is not None:
                prop_preds.append(prop_pred.detach().cpu())
                prop_gts.append(prop_gt.detach().cpu())
                token_masks.append(token_mask.detach().cpu())

    return (
        torch.cat(preds, dim=0),
        torch.cat(gts, dim=0),
        torch.cat(prop_preds, dim=0) if prop_preds else None,
        torch.cat(prop_gts, dim=0) if prop_gts else None,
        torch.cat(token_masks, dim=0) if token_masks else None,
    )


def compute_topk_metrics(y_true, y_pred, percent):
    """
    Compute metrics for top-K% genes based on Pearson correlation.
    """
    from scipy.stats import pearsonr

    num_genes = y_true.shape[1]
    k = max(1, int((percent / 100) * num_genes))

    # Compute gene-wise Pearson
    pearsons = []
    for i in range(num_genes):
        try:
            r, _ = pearsonr(y_true[:, i], y_pred[:, i])
            pearsons.append(r if not np.isnan(r) else 0)
        except:
            pearsons.append(0)

    pearsons = np.array(pearsons)
    top_idx = np.argsort(pearsons)[-k:]

    return compute_metrics(
        y_true[:, top_idx],
        y_pred[:, top_idx]
    )

# =========================================================
# MAIN EVALUATION
# =========================================================
def evaluate(params):

    set_seed(123)

    dataset = params["dataset_name"]
    dataset_path = params["dataset_path"]

    if dataset == "cscc":
        gsm_samples = [
            'GSM4284320','GSM4284323','GSM4284322','GSM4284317',
            'GSM4284327','GSM4284316','GSM4284326','GSM4284325',
            'GSM4284324','GSM4284321','GSM4284319','GSM4284318'
        ]
    elif dataset == "her2":
        gsm_samples = ["A1","A2","A3","A4","A5","A6","B1","B2","B3","B4","B5","B6",
                       "C1","C2","C3","C4","C5","C6","D1","D2","D3","D4","D5","D6",
                       "E1","E2","E3","F1","F2","F3","G1","G2","G3","H1","H2","H3"]
    elif dataset == "hist2st_845":
        gsm_samples = ['23209_C1', '23209_C2', '23209_D1', '23268_C1', '23268_C2', '23268_D1', '23269_C1', '23269_C2',
            '23269_D1', '23270_D2', '23270_E1', '23270_E2', '23272_D2', '23272_E1', '23272_E2', '23277_D2',
            '23277_E1', '23277_E2', '23287_C1', '23287_C2', '23287_D1', '23288_D2', '23288_E1', '23288_E2',
            '23377_C1', '23377_C2', '23377_D1', '23450_D2', '23450_E1', '23450_E2', '23506_C1', '23506_C2',
            '23506_D1', '23508_D2', '23508_E1', '23508_E2', '23567_D2', '23567_E1', '23567_E2', '23803_D2',
            '23803_E1', '23803_E2', '23810_D2', '23810_E1', '23810_E2', '23895_C1', '23895_C2', '23895_D1',
            '23901_C2', '23901_D1', '23903_C1', '23903_C2', '23903_D1', '23944_D2', '23944_E1', '23944_E2',
            '24044_D2', '24044_E1', '24044_E2', '24105_C1', '24105_C2', '24105_D1', '24220_D2', '24220_E1',
            '24220_E2', '24223_D2', '24223_E1', '24223_E2']  # (same list as you use)

    # -------------------------
    # File paths
    # -------------------------

    if params["dataset_name"] == "cscc":
        patch_paths = [os.path.join(dataset_path, f"{sample}_patches.h5") for sample in gsm_samples]
        adata_paths = [os.path.join(dataset_path, f"{sample}_spots.h5ad") for sample in gsm_samples]
        celltype_prop_paths = [os.path.join(dataset_path,"spot_to_celltypes",f"{sample}_celltypes.csv") for sample in gsm_samples]
    if params["dataset_name"] == "her2":
        patch_paths = [os.path.join(dataset_path, f"{sample}.h5") for sample in gsm_samples]
        adata_paths = [os.path.join(dataset_path, f"{sample}.h5ad") for sample in gsm_samples]
    if params["dataset_name"] == "hist2st_845":
        patch_paths = [os.path.join(dataset_path, f"{sample}.h5") for sample in gsm_samples]
        adata_paths = [os.path.join(dataset_path, f"{sample}.h5ad") for sample in gsm_samples]

    # -----------------------------
    # Load genes & HVGs
    # -----------------------------
    gene_names = np.load(
        os.path.join(params["genes"], f"{dataset}.npy"),
        allow_pickle=True
    ).tolist()

    hvg20 = pd.read_csv(params["hvg20"])["gene"].tolist()
    hvg50 = pd.read_csv(params["hvg50"])["gene"].tolist()

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    hvg20_idx = [gene_to_idx[g] for g in hvg20 if g in gene_to_idx]
    hvg50_idx = [gene_to_idx[g] for g in hvg50 if g in gene_to_idx]

    # -----------------------------
    # Dataset
    # -----------------------------
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    datasets = [
        PatchDataset(
            img_path=img_p,
            gene_path=adata_p,
            mode="val",
            proportion_csv=prop_p,
            gene_names=gene_names,
            transform=transform,
            log_norm=True,
            scale_factor=params.get("scale_factor", 1000000)
        )
        for img_p, adata_p, prop_p in zip(
            patch_paths,
            adata_paths,
            celltype_prop_paths
        )
    ]

    full_dataset = ConcatDataset(datasets)

    # -----------------------------
    # K-FOLD EVALUATION
    # -----------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (_, val_idx) in enumerate(kf.split(range(len(full_dataset)))):

        print(f"\n===== Fold {fold+1} =====")

        val_set = Subset(full_dataset, val_idx)

        model_path = os.path.join(
            params["model_directory"],
            f"fold_{fold+1}",
            "best_model.pth"
        )

        gene_df = pd.read_csv(params["celltype_gene_csv"]).set_index("cell_type")
        num_celltypes = gene_df.shape[0]
        num_sc_genes  = gene_df.shape[1]

        model = load_model(
            params["model"],
            model_path,
            num_genes=len(gene_names),
            num_sc_genes=num_sc_genes,
            num_celltypes=num_celltypes,
        )

        y_pred, y_true, prop_pred, prop_gt, token_mask = predict(model, val_set)

        y_true = y_true.cpu().numpy()
        y_pred= y_pred.cpu().numpy()

        # ---------- Metrics ----------
        metrics_all = compute_metrics(y_true, y_pred, gene_names)

        metrics_hvg20 = compute_metrics(
            y_true[:, hvg20_idx],
            y_pred[:, hvg20_idx],
            [gene_names[i] for i in hvg20_idx]
        )

        metrics_hvg50 = compute_metrics(
            y_true[:, hvg50_idx],
            y_pred[:, hvg50_idx],
            [gene_names[i] for i in hvg50_idx]
        )

        # NEW: percentile-based metrics
        metrics_top1  = compute_topk_metrics(y_true, y_pred, 1)
        metrics_top5  = compute_topk_metrics(y_true, y_pred, 5)
        metrics_top10 = compute_topk_metrics(y_true, y_pred, 10)

        # -------------------------
        # Cell-type metrics
        # -------------------------
        celltype_metrics = {
            "celltype_kl": masked_kl_divergence(prop_pred, prop_gt, token_mask),
            "celltype_l1": masked_l1_celltype(prop_pred, prop_gt, token_mask),
            "celltype_spearman": masked_spearman_celltype(prop_pred, prop_gt, token_mask),
        }

        fold_results.append({
            "fold": fold + 1,
            "all": metrics_all,
            "hvg20": metrics_hvg20,
            "hvg50": metrics_hvg50,
            "top1": metrics_top1,
            "top5": metrics_top5,
            "top10": metrics_top10,
            "CELLTYPE": celltype_metrics
        })

    # -----------------------------
    # Aggregate Results
    # -----------------------------
    def aggregate_with_std(key):
        metrics = fold_results[0][key].keys()
        
        out = {}
        for m in metrics:
            values = [f[key][m] for f in fold_results]
            out[f"{m}_mean"] = float(np.mean(values))
            out[f"{m}_std"]  = float(np.std(values))
        
        return out


    final_results = {
        "ALL": aggregate_with_std("all"),
        "HVG_20": aggregate_with_std("hvg20"),
        "HVG_50": aggregate_with_std("hvg50"),
        "TOP_1%": aggregate_with_std("top1"),
        "TOP_5%": aggregate_with_std("top5"),
        "TOP_10%": aggregate_with_std("top10"),
        "CELLTYPE": aggregate_with_std("CELLTYPE"),
    }

    # ===============================
    # Convert to wide table format
    # ===============================
    all_metric_names = set()

    for group in final_results:
        all_metric_names.update(final_results[group].keys())

    all_metric_names = sorted(all_metric_names)

    rows = []

    for metric_name in all_metric_names:
        row = {"metric": metric_name}

        for group in ["ALL", "HVG_20", "HVG_50", "TOP_1%", "TOP_5%", "TOP_10%", "CELLTYPE"]:
            row[group] = final_results.get(group, {}).get(metric_name, None)
        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(f"{dataset}_{params['model']}_final_results_complete.csv", index=False)
    print(" Saved final_results_complete.csv")

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    with open("parameters_evaluation_cell_type.json") as f:
        params = json.load(f)

    evaluate(params)
