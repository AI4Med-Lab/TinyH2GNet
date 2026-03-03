import sys
sys.path.append('code_model_training/models')
sys.path.append('code_model_training/utils')

import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import KFold, GroupKFold

from utils.dataPrep import PatchDataset
from compute_metrics_evaluation import compute_metrics
from models import STNet, EfficientNet, EfficientNetB4GeneRegressor, Custom_VGG16, HisToGene, TCGN, EfficientNet_GeneCaptionContrastive
from torchvision import transforms
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
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = params.get("batch_size", 128)


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
        print(f"EfficientNet initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
        
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
    
    elif model_name == "TCGN":
        print("Initializing TCGN model...")

        # # Set up model parameters
        # tcgn_kwargs = dict(
        #     img_size=params.get("image_height", 224),
        #     in_chans=params.get("image_channels", 3),
        #     num_classes=num_genes,
        #     embed_dims=params.get("embed_dims", [52, 104, 208, 416]),
        #     stem_channel=params.get("stem_channel", 16),
        #     fc_dim=params.get("fc_dim", 1280),
        #     num_heads=params.get("num_heads", [1, 2, 4, 8]),
        #     mlp_ratios=params.get("mlp_ratios", [3.6, 3.6, 3.6, 3.6]),
        #     qkv_bias=True,
        #     qk_scale=None,
        #     representation_size=None,
        #     drop_rate=params.get("drop_rate", 0.0),
        #     attn_drop_rate=params.get("attn_drop_rate", 0.0),
        #     drop_path_rate=params.get("drop_path_rate", 0.0),
        #     hybrid_backbone=None,
        #     norm_layer=None,
        #     depths=params.get("depths", [2, 2, 10, 2]),
        #     qk_ratio=params.get("qk_ratio", 1),
        #     sr_ratios=params.get("sr_ratios", [8, 4, 2, 1]),
        #     dp=params.get("dropout", 0.1)
        # )

        # # Instantiate the model
        # model = TCGN(**tcgn_kwargs)
        model = TCGN(num_classes=num_genes)
        print(f"TCGN initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    
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
    
    FOLD_METRICS = []
    
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

    dataset = params["dataset_name"]
    dataset_path = params["dataset_path"]

    if dataset == "cscc":
        gsm_samples = [ 'GSM4284316_P2','GSM4284317_P2','GSM4284318_P2',
                        'GSM4284319_P5','GSM4284320_P5','GSM4284321_P5',
                        'GSM4284322_P9','GSM4284323_P9','GSM4284324_P9',
                        'GSM4284325_P10','GSM4284326_P10','GSM4284327_P10']
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

    # Case-insensitive mapping
    gene_to_idx_ci = {g.upper(): i for i, g in enumerate(gene_names)}

    # -----------------------------
    # Match disease genes with dataset genes

    # Original disease genes (already upper-case from loader)
    disease_genes_all = load_disease_genes(params["disease_gene_csv"])
    print(f"Loaded {len(disease_genes_all)} disease genes from CSV")

    # -----------------------------
    # Match with dataset genes (case-insensitive)
    # -----------------------------
    # -----------------------------
    # SPECIAL CASE: hist2st_845 uses Ensembl IDs
    # -----------------------------
    if params["dataset_name"] == "hist2st_845":

        mapping_csv = "/home/sukrit/image_2_gene/code_data_preprocessing/hvg/hvg_lists/his2st_geneid_to_gene_all.csv"
        df_map = pd.read_csv(mapping_csv)

        df_map["query"] = df_map["query"].astype(str).str.strip()           # Ensembl
        df_map["symbol"] = df_map["symbol"].astype(str).str.strip().str.upper()  # Gene symbol

        # symbol → ensembl
        symbol2ensembl = {}
        for _, row in df_map.iterrows():
            sym = row["symbol"]
            ens = row["query"]
            if sym not in symbol2ensembl:   # keep first mapping
                symbol2ensembl[sym] = ens

        disease_genes_matched = []        # Ensembl IDs
        disease_gene_indices = []
        disease_genes_missing = []
        disease_gene_symbol_map = {}      # Ensembl → Symbol (for reporting)

        for g in disease_genes_all:  # TP53, BRCA1, etc.
            g_uc = g.upper()

            if g_uc in symbol2ensembl:
                ens_id = symbol2ensembl[g_uc]

                if ens_id in gene_to_idx:
                    disease_genes_matched.append(ens_id)
                    disease_gene_indices.append(gene_to_idx[ens_id])
                    disease_gene_symbol_map[ens_id] = g_uc   # keep symbol for later reporting
                else:
                    disease_genes_missing.append(f"{g_uc} (mapped to {ens_id}, not in dataset)")
            else:
                disease_genes_missing.append(f"{g_uc} (no Ensembl mapping)")

        print(f"Matched disease genes with dataset (hist2st_845): {len(disease_genes_matched)}")
        print("Matched disease genes (Symbol → Ensembl):")
        for ens in disease_genes_matched:
            print(f"{disease_gene_symbol_map[ens]} → {ens}")

    else:
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

    # Use Ensembl IDs for hist2st_845, symbols otherwise
    disease_genes = disease_genes_matched

    # -----------------------------
    # K-FOLD EVALUATION
    # -----------------------------
    patient_ids = [get_patient_id(s, params["dataset_name"]) for s in gsm_samples]
    unique_patients = sorted(set(patient_ids))

   
    k = params.get("k_folds", 5)
    n_patients = len(unique_patients)

    if k > n_patients:
        k = n_patients

    kf = GroupKFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []

    for fold, (_, val_idx) in enumerate(kf.split(gsm_samples, groups=patient_ids)):

        print(f"\n===== Fold {fold+1} =====")
        
        val_samples = [gsm_samples[i] for i in val_idx]

        print("Val slides:", val_samples)
        
        val_patients   = sorted(set(patient_ids[i] for i in val_idx))

        print(f"Val patients: {len(val_patients)},  {val_patients}")
        
        val_patch_files   = [patch_paths[i] for i in val_idx]
        val_adata_files   = [adata_paths[i] for i in val_idx]

        print(f"Fold {fold+1}: Val={len(val_patch_files)}")
        
        val_datasets_list = [
            PatchDataset(
                gene_path=adata_p,
                img_path=img_p,
                gene_names=gene_names,
                transform=None,
                log_norm=True,
                scale_factor=params.get("scale_factor", 1000000)
            )
            for adata_p, img_p in zip(val_adata_files, val_patch_files)
        ]  

        val_set = ConcatDataset(val_datasets_list)
        
        print("Val spots:", len(val_set))

        model_path = os.path.join(
            params["model_directory"],
            f"fold_{fold+1}",
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

        y_pred, y_true = predict(model, val_set)
        
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t_end = time()

        fold_time = t_end - t_start
        num_spots = len(val_set)
        throughput = num_spots / fold_time

        if DEVICE.type == "cuda":
            peak_mem_alloc = torch.cuda.max_memory_allocated() / (1024**2)   # MB
            peak_mem_reserved = torch.cuda.max_memory_reserved() / (1024**2)
        else:
            peak_mem_alloc = None
            peak_mem_reserved = None

        print(f"[Fold {fold+1}] Inference time: {fold_time:.2f}s | "
              f"Throughput: {throughput:.2f} spots/sec | "
              f"Peak GPU mem: {peak_mem_alloc:.1f} MB")

        FOLD_TIMINGS.append({
            "fold": fold + 1,
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
            "fold": fold + 1,
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
            col_name = gene
            if params["dataset_name"] == "hist2st_845":
                col_name = disease_gene_symbol_map.get(gene, gene)  # TP53 instead of ENSG...

            final_results[col_name] = aggregate_disease_metrics(gene)

    # ===============================
    # Save fold-wise wide table CSV
    # ===============================
    fold_rows = []

    # ===============================
    # Convert to wide table format
    # ===============================
    groups = [
        "ALL", "HVG_20", "HVG_50",
        "TOP_1%", "TOP_5%", "TOP_10%",
        "DISEASE_ALL_GENES"
    ]

    for gene in disease_genes:
        if gene in gene_to_idx_ci:
            if params["dataset_name"] == "hist2st_845":
                groups.append(disease_gene_symbol_map.get(gene, gene))
            else:
                groups.append(gene)
                
                
    for fold_dict in fold_results:
        fold_id = fold_dict["fold"]

        # metrics available in ALL group
        for metric_name in fold_dict["all"].keys():
            row = {
                "fold": fold_id,
                "metric": f"{metric_name}_mean"  # keep same naming style
            }

            # ALL / HVG / TOP-K
            row["ALL"] = fold_dict["all"][metric_name]
            row["HVG_20"] = fold_dict["hvg20"][metric_name]
            row["HVG_50"] = fold_dict["hvg50"][metric_name]
            row["TOP_1%"] = fold_dict["top1"][metric_name]
            row["TOP_5%"] = fold_dict["top5"][metric_name]
            row["TOP_10%"] = fold_dict["top10"][metric_name]

            # Disease all genes
            if "DISEASE_ALL_GENES" in fold_dict["DISEASE"]:
                row["DISEASE_ALL_GENES"] = fold_dict["DISEASE"]["DISEASE_ALL_GENES"][metric_name]
            else:
                row["DISEASE_ALL_GENES"] = None

            # Per-gene disease columns
            for gene in disease_genes:
                if gene in fold_dict["DISEASE"]:
                    col_name = gene
                    if params["dataset_name"] == "hist2st_845":
                        col_name = disease_gene_symbol_map.get(gene, gene)

                    row[col_name] = fold_dict["DISEASE"][gene][metric_name]

            fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)

    fold_csv_path = os.path.join(save_root, "fold_wise_results_complete_biomarkers.csv")
    fold_df.to_csv(fold_csv_path, index=False)

    print(f"[INFO] Fold-wise wide CSV saved to: {fold_csv_path}")

    # groups = [
    #     "ALL", "HVG_20", "HVG_50",
    #     "TOP_1%", "TOP_5%", "TOP_10%",
    #     "DISEASE_ALL_GENES"
    # ] + [g for g in disease_genes if g in gene_to_idx_ci]


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
    with open("parameters_evaluation_hist.json") as f:
        params = json.load(f)

    save_root = os.path.join(
        params["model_directory"],
        "eval_results"
    )
    os.makedirs(save_root, exist_ok=True)

    evaluate(params)
