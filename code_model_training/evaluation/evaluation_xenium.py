import sys
sys.path.append('/home/puneet/mk/code_model_training/models')
sys.path.append('/home/puneet/mk/code_model_training/utils')

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import KFold
from utils.dataPrep_xenium import PatchDataset, FoldDataset, log1p_normalization

from compute_metrics_evaluation import compute_metrics
from models import STNet, EfficientNet, EfficientNetB4GeneRegressor, Custom_VGG16, HisToGene, TCGN, EfficientNet_GeneCaptionContrastive
from torchvision import transforms
from model_eff_net_versions import EfficientNetTinyStudent
import ast
from time import time

from scipy.stats import ConstantInputWarning
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# GLOBALS
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128


# =========================================================
# Utilities
# =========================================================
def set_seed(seed=123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_model(model_name, model_path, num_genes):
    if model_name == "EfficientNet":
        model = EfficientNet(num_genes=num_genes)
        print(f"Effnet initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    elif model_name == "STNet":
        model = STNet(num_genes=num_genes)
        print(f"STNet initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    elif model_name == "VGG":
        model = Custom_VGG16(num_genes=num_genes)
        print(f"VGG initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    elif model_name == "EfficientNet_GeneCaptionContrastive":
        model = EfficientNet_GeneCaptionContrastive(num_genes=num_genes)

    elif model_name == "HisToGene":
        model = HisToGene(patch_size=16, n_layers= 8, n_genes=num_genes)
        print(f"HisToGene initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        
        
    elif params["model"] == "EfficientNetTinyStudent":
            print("EfficientNetTinyStudent (tiny) for gene expression prediction.")
            phi = params.get("phi", -7.0)
            model = EfficientNetTinyStudent(num_genes=num_genes, phi=phi)
            
            print(f"EfficientNetTinyStudent initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        
            
    elif model_name == "TCGN":
        
        print("Initializing TCGN model...")

        # Set up model parameters
        tcgn_kwargs = dict(
            img_size=params.get("image_height", 224),
            in_chans=params.get("image_channels", 3),
            num_classes=num_genes,
            embed_dims=params.get("embed_dims", [52, 104, 208, 416]),
            stem_channel=params.get("stem_channel", 16),
            fc_dim=params.get("fc_dim", 1280),
            num_heads=params.get("num_heads", [1, 2, 4, 8]),
            mlp_ratios=params.get("mlp_ratios", [3.6, 3.6, 3.6, 3.6]),
            qkv_bias=True,
            qk_scale=None,
            representation_size=None,
            drop_rate=params.get("drop_rate", 0.0),
            attn_drop_rate=params.get("attn_drop_rate", 0.0),
            drop_path_rate=params.get("drop_path_rate", 0.0),
            hybrid_backbone=None,
            norm_layer=None,
            depths=params.get("depths", [2, 2, 10, 2]),
            qk_ratio=params.get("qk_ratio", 1),
            sr_ratios=params.get("sr_ratios", [8, 4, 2, 1]),
            dp=params.get("dropout", 0.1)
        )

        # Instantiate the model
        model = TCGN(**tcgn_kwargs)
        print(f"TCGN initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        
        
        
        # model = TCGN(num_classes=num_genes)

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

    with torch.no_grad():
        for imgs, genes in tqdm(loader, desc="Predicting"):
            imgs = imgs.to(DEVICE)
            genes = genes.to(DEVICE)

            out = model(imgs)
            if isinstance(out, tuple):
                out = out[0]

            out = torch.where(out < 0, torch.tensor(0.0, device=out.device),out)
            
            preds.append(out.cpu().numpy())
            gts.append(genes.cpu().numpy())

    return np.vstack(preds), np.vstack(gts)

def load_disease_genes(csv_path):
    """
    Reads disease gene CSV.
    Column: 'Gene'
    Each row contains a single gene symbol (e.g., TP53, BRCA1).
    Returns a sorted list of UNIQUE uppercase gene names.
    """
    df = pd.read_csv(csv_path)

    genes = (
        df["Gene"]
        .dropna()                # remove NaNs
        .astype(str)             # ensure string
        .str.strip()             # trim spaces
        .str.upper()             # case-insensitive match
        .unique()                # remove duplicates
        .tolist()
    )

    return sorted(genes)


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
    
    FOLD_TIMINGS = []
    
    # =========================================================
    # GLOBALS
    # =========================================================
    global DEVICE, BATCH_SIZE

    # Device from params
    device_str = params.get("device", "cuda")

    if device_str == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif device_str == "cpu":
        DEVICE = torch.device("cpu")
    else:
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        DEVICE = torch.device("cpu")

    # Batch size from params
    BATCH_SIZE = params.get("batch_size", 128)

    print(f"[INFO] Using device = {DEVICE}")
    print(f"[INFO] Using batch size = {BATCH_SIZE}")
    
    # -----------------------------
    # Dataset
    # -----------------------------

    patch_file = f"{params['dataset_path']}/cells_patches.h5"
    adata_file = f"{params['dataset_path']}/cells_expression.h5ad"

    # Load AnnData
    adata = sc.read_h5ad(adata_file)

    # Gene names from Xenium (Ensembl or symbols, whatever your model was trained on)
    gene_names = adata.var_names.to_numpy().tolist()

    print(f"Loaded {len(gene_names)} genes from Xenium adata")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    single_dataset = PatchDataset(
        gene_path=adata_file,
        img_path=patch_file,
        gene_names=None,     # IMPORTANT: PatchDataset should read gene names from adata
        transform=transform
    )
    
    # Load raw arrays
    raw_images = single_dataset.image           # (N, H, W, C)
    raw_genes  = single_dataset.gene_data       # (N, G)


    # -----------------------------
    # Load genes & HVGs
    # -----------------------------

    hvg20 = pd.read_csv(params["hvg20"])["gene"].tolist()
    hvg50 = pd.read_csv(params["hvg50"])["gene"].tolist()

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    hvg20_idx = [gene_to_idx[g] for g in hvg20 if g in gene_to_idx]
    hvg50_idx = [gene_to_idx[g] for g in hvg50 if g in gene_to_idx]

    # Case-insensitive mapping
    gene_to_idx_ci = {g.upper(): i for i, g in enumerate(gene_names)}

    # -----------------------------
    # Match disease genes with dataset genes

    # Original disease genes (already upper-case from loader)
    disease_genes_all = load_disease_genes(params["disease_gene_csv"])
    print(f"Loaded {len(disease_genes_all)} disease genes from CSV")
    

    # -----------------------------
    # Default case: gene_names already symbols
    # -----------------------------
    disease_genes_matched = []
    disease_gene_indices = []
    disease_genes_missing = []

    for g in disease_genes_all:
        g_uc = g.upper()
        if g_uc in gene_to_idx_ci:
            disease_genes_matched.append(g_uc)              # symbol
            disease_gene_indices.append(gene_to_idx_ci[g_uc])
        else:
            disease_genes_missing.append(g)

    print(f"Matched disease genes with dataset: {len(disease_genes_matched)}")
    print("Matched disease genes:")
    print(disease_genes_matched)

    # -----------------------------
    # Print missing genes
    # -----------------------------
    if disease_genes_missing:
        print(f"\nDisease genes NOT found in dataset ({len(disease_genes_missing)}):")
        for g in disease_genes_missing:
            print(g)

    disease_genes = disease_genes_matched

    # -----------------------------
    # K-FOLD EVALUATION
    # -----------------------------
   
    k = params.get("k_folds", 5)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(single_dataset)))):

        print(f"\n===== Fold {fold_idx+1}/{k} =====")
        print(f"Train spots: {len(train_idx)}, Val spots: {len(val_idx)}")
        
        # ---------- Normalize only validation rows ----------
        val_norm_full = np.zeros_like(raw_genes, dtype=np.float32)

        val_norm = log1p_normalization(
            raw_genes[val_idx],
            scale_factor=params.get("scale_factor", 1000000)
        )

        val_norm_full[val_idx] = val_norm

        # ---------- Build FoldDataset ----------
        val_dataset = FoldDataset(
            image_data=raw_images,
            gene_data=val_norm_full,   # ✔ full matrix with only val normalized
            indices=val_idx,
            transform=transform
        )

        print(f"Val cells: {len(val_dataset)}")

        # val_set = Subset(single_dataset, val_idx)

        model_path = os.path.join(
            params["model_directory"],
            f"fold_{fold_idx+1}",
            "best_model.pth"
        )

        model = load_model(
            params["model"],
            model_path,
            num_genes=len(gene_names)
        )
        
        # ======================
        # Timing + GPU memory
        # ======================
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t_start = time()

        y_pred, y_true = predict(model, val_dataset)
        
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t_end = time()

        fold_time = t_end - t_start
        num_spots = len(val_dataset)
        throughput = num_spots / fold_time

        if DEVICE.type == "cuda":
            peak_mem_alloc = torch.cuda.max_memory_allocated() / (1024**2)   # MB
            peak_mem_reserved = torch.cuda.max_memory_reserved() / (1024**2)
        else:
            peak_mem_alloc = None
            peak_mem_reserved = None

        print(f"[Fold {fold_idx+1}] Inference time: {fold_time:.2f}s | "
              f"Throughput: {throughput:.2f} spots/sec | "
              f"Peak GPU mem: {peak_mem_alloc:.1f} MB")

        FOLD_TIMINGS.append({
            "fold": fold_idx + 1,
            "num_val_spots": num_spots,
            "total_inference_time_sec": fold_time,
            "throughput_spots_per_sec": throughput,
            "peak_gpu_mem_allocated_mb": peak_mem_alloc,
            "peak_gpu_mem_reserved_mb": peak_mem_reserved,
            "batch_size": BATCH_SIZE
        })

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


        # ============================
        # Disease gene metrics (fold)
        # ============================
        disease_metrics_fold = {}

        valid_disease_indices = [
            gene_to_idx_ci[g] for g in disease_genes if g in gene_to_idx_ci
        ]

        # ---- Disease ALL genes (single column)
        if len(valid_disease_indices) > 0:
            disease_metrics_fold["DISEASE_ALL_GENES"] = compute_metrics(
                y_true[:, valid_disease_indices],
                y_pred[:, valid_disease_indices],
                [gene_names[i] for i in valid_disease_indices]
            )

        # ---- Per-gene metrics
        for gene in disease_genes:
            if gene not in gene_to_idx_ci:
                continue

            idx = gene_to_idx_ci[gene]

            disease_metrics_fold[gene] = compute_metrics(
                y_true[:, idx:idx+1],
                y_pred[:, idx:idx+1],
                genes=[gene]
            )


        fold_results.append({
            "fold": fold_idx + 1,
            "all": metrics_all,
            "hvg20": metrics_hvg20,
            "hvg50": metrics_hvg50,
            "top1": metrics_top1,
            "top5": metrics_top5,
            "top10": metrics_top10,
            "DISEASE": disease_metrics_fold
        })


        print(f"fold_results: {fold_results}")

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

    def aggregate_disease_metrics(gene_key):
        metrics = fold_results[0]["DISEASE"][gene_key].keys()
        out = {}

        for m in metrics:
            vals = [
                f["DISEASE"][gene_key][m]
                for f in fold_results
                if gene_key in f["DISEASE"]
            ]
            out[f"{m}_mean"] = round(float(np.mean(vals)), 4)
            out[f"{m}_std"]  = round(float(np.std(vals)), 4)

        return out


    final_results = {
        "ALL": aggregate_with_std("all"),
        "HVG_20": aggregate_with_std("hvg20"),
        "HVG_50": aggregate_with_std("hvg50"),
        "TOP_1%": aggregate_with_std("top1"),
        "TOP_5%": aggregate_with_std("top5"),
        "TOP_10%": aggregate_with_std("top10")
    }

    # -----------------------------
    # Add disease columns
    # -----------------------------
    final_results["DISEASE_ALL_GENES"] = aggregate_disease_metrics("DISEASE_ALL_GENES")

    for gene in disease_genes:
        if gene in gene_to_idx_ci:
            final_results[gene] = aggregate_disease_metrics(gene)


    # ===============================
    # Save fold-wise wide table CSV (Xenium)
    # ===============================
    fold_rows = []

    # ===============================
    # Convert to wide table format
    # ===============================
    groups = [
        "ALL", "HVG_20", "HVG_50",
        "TOP_1%", "TOP_5%", "TOP_10%",
        "DISEASE_ALL_GENES"
    ] + [g for g in disease_genes if g in gene_to_idx_ci]

    for fold_dict in fold_results:
        fold_id = fold_dict["fold"]

        for metric_name in fold_dict["all"].keys():
            row = {
                "fold": fold_id,
                "metric": f"{metric_name}_mean"
            }

            # Core groups
            row["ALL"] = fold_dict["all"][metric_name]
            row["HVG_20"] = fold_dict["hvg20"][metric_name]
            row["HVG_50"] = fold_dict["hvg50"][metric_name]
            row["TOP_1%"] = fold_dict["top1"][metric_name]
            row["TOP_5%"] = fold_dict["top5"][metric_name]
            row["TOP_10%"] = fold_dict["top10"][metric_name]

            # Disease ALL
            if "DISEASE_ALL_GENES" in fold_dict["DISEASE"]:
                row["DISEASE_ALL_GENES"] = fold_dict["DISEASE"]["DISEASE_ALL_GENES"][metric_name]
            else:
                row["DISEASE_ALL_GENES"] = None

            # Per-disease gene columns
            for gene in disease_genes:
                if gene in fold_dict["DISEASE"]:
                    row[gene] = fold_dict["DISEASE"][gene][metric_name]
                else:
                    row[gene] = None

            fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)

    fold_csv_path = os.path.join(save_root, "fold_wise_results_complete_biomarkers.csv")
    fold_df.to_csv(fold_csv_path, index=False)

    print(f"[INFO] Xenium fold-wise results saved to: {fold_csv_path}")


    rows = []

    for metric_name in final_results["ALL"].keys():
        row = {"metric": metric_name}

        for group in groups:
            row[group] = final_results[group][metric_name]

        rows.append(row)

    df = pd.DataFrame(rows)

    final_csv_path = os.path.join(save_root, "final_results_complete_biomarkers.csv")
    df.to_csv(final_csv_path, index=False)
    print(f"Saved biomarkers results to: {final_csv_path}")
    
    
    # ===============================
    # Save runtime + GPU memory stats
    # ===============================
    timing_df = pd.DataFrame(FOLD_TIMINGS)
    timing_csv_path = os.path.join(save_root, "fold_runtime_gpu.csv")
    timing_df.to_csv(timing_csv_path, index=False)

    summary = {
        "mean_inference_time_sec": timing_df["total_inference_time_sec"].mean(),
        "std_inference_time_sec": timing_df["total_inference_time_sec"].std(),
        "mean_throughput_spots_per_sec": timing_df["throughput_spots_per_sec"].mean(),
        "std_throughput_spots_per_sec": timing_df["throughput_spots_per_sec"].std(),
        "mean_peak_gpu_mem_alloc_mb": timing_df["peak_gpu_mem_allocated_mb"].mean(),
        "std_peak_gpu_mem_alloc_mb": timing_df["peak_gpu_mem_allocated_mb"].std(),
    }

    summary_df = pd.DataFrame([summary])
    summary_csv_path = os.path.join(save_root, "fold_runtime_gpu_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"Runtime CSV saved: {timing_csv_path}")
    print(f"Runtime summary CSV saved: {summary_csv_path}")
# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    with open("parameters_evaluation.json") as f:
        params = json.load(f)
        
    save_root = os.path.join(
        params["model_directory"],
        "eval_results"
    )
    os.makedirs(save_root, exist_ok=True)

    evaluate(params)
