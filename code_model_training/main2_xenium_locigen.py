import sys
sys.path.append('/home/puneet/mk/code_model_training/models')
sys.path.append('/home/puneet/mk/code_model_training/utils')

import warnings
warnings.filterwarnings("ignore")

import gc
import csv
import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from datetime import datetime
import argparse
import json
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from torch.nn import DataParallel
from sklearn.model_selection import KFold
from utils.dataPrep_xenium_locigen import XeniumLoCIGENDataset , load_xenium_raw_data, log1p_normalization_train_val_separate
from compute_metrics import compute_metrics, spearmanrr
from setup_logger import setup_logging 
from set_deterministic_seed import set_deterministic_seed
from models import STNet, EfficientNet, EfficientNetB4GeneRegressor, Custom_VGG16, HisToGene, TCGN
from torch.utils.data import ConcatDataset
from lociGen import LoCIGENv2
from locigen_dual import LoCIGEN
from utils.loss_lociGene import locigen_loss, locigen_loss_v2_1
from proposedModels import ImageGeneCrossTransformer
from sklearn.model_selection import KFold
from proposedModels2 import ImageToGeneTransformer

def main():
     
    g, seed_worker = set_deterministic_seed(123)   
     
    parser = argparse.ArgumentParser(description="Script that uses a JSON configuration file")
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {args.config}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON file at {args.config}")
        return
    
    # g, seed_worker = set_deterministic_seed(params.get("seed", 123))
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_dir = os.path.join(params['result_path'], f"result_vit_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)

    json_file = os.path.join(model_save_dir, f"experiment_info.json")
    with open(json_file, "w") as exp:
        json.dump(params, exp, indent=4)

    log_file = os.path.join(model_save_dir, "results_log.txt")
    logger = setup_logging(log_file)
    logger.info("Logging setup complete.")
    logger.info(f"Experiment information saved to the path: {json_file}")

    metrics_csv_path = os.path.join(model_save_dir, "metrics_result.csv")

    # Write CSV headers (only once before the loop starts)
    # We'll append per-fold & per-epoch rows later
    with open(metrics_csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'fold', 'epoch', 'phase', 'custom_loss', 'pearson_mean_cell_wise', 'spearman_mean_cell_wise', 'pearson_mean_genewise', 
            'spearman_mean_genewise','l1_error_mean', 'l2_errors_mean', 'r2_scores_mean', 'pearson_std',
            'l2_error_q1', 'l2_error_q2', 'l2_error_q3',
            'r2_score_q1', 'r2_score_q2', 'r2_score_q3', 'mape_mean', 'mape_std', 'rmse_mean', 'rmse_std'
        ])
        writer.writeheader()

    # --- Training routine (mostly unchanged) ---
    def train_regression(train_datasets, val_datasets, fold_idx,g, seed_worker,
                                          lambda_reg=params.get("lambda_reg", 0.0),
                                          learning_rate=params['learning_rate'],
                                          epochs=params['epochs'],
                                          batch_size=params['batch_size'],
                                          weight_decay=params['weight_decay']):

        models = {}

        logger.info(f"train_datasets length:  {len(train_datasets)}")
        train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True,  num_workers=0, persistent_workers=False, worker_init_fn=seed_worker, generator=g)
        val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False,  num_workers=0, persistent_workers=False, worker_init_fn=seed_worker, generator=g)

        logger.info(f"Number of train batches: {len(train_dataloader)}")
        logger.info("Initializing model...")

        if params['dataset_name'] == "cscc":
            num_genes = 171
        elif params['dataset_name'] == "her2":
            num_genes = 785
        elif params['dataset_name'] == "hist2st_845":
            num_genes = 845
        elif params['dataset_name'] == "10X_HumandBreastCancer":
            num_genes = 541
        else:
            raise ValueError(f"Unknown dataset name: {params['dataset_name']}")

        if params["model"] == "STNet":
            pretrained = params.get("pretrained", True)
            model = STNet(num_genes=num_genes, pretrained=pretrained)
        
        elif params["model"] == "VGG":
            pretrained = params.get("pretrained", False)
            model = Custom_VGG16(num_genes=num_genes, pretrained=pretrained)
            
        elif params["model"] == "Effnet":
            pretrained = params.get("pretrained", True)
            model = EfficientNet(num_genes=num_genes, pretrained=pretrained) 
            
        elif params["model"] == "HisToGene":
            model = HisToGene(patch_size=16, n_layers= 8, n_genes=num_genes)
            
        elif params["model"] == "TCGN":            
            logger.info("Initializing TCGN model...")

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
            logger.info(f"TCGN initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")


        elif params["model"] == "ImageGeneCrossTransformer":
            # --- Get model hyperparameters ---
            embed_dim = params.get("embed_dim", 512)
            nhead = params.get("nhead", 4)
            dim_feedforward = params.get("dim_feedforward", 1024)
            dropout = params.get("dropout", 0.1)
            pretrained = params.get("pretrained", False)
            H = params.get("image_height", 224)
            W = params.get("image_width", 224)
            image_channels = params.get("image_channels", 3)

            # --- Logging configuration ---
            logger.info("Initializing ImageGeneCrossTransformer with:")
            logger.info(f"- Image size: {H}x{W}")
            logger.info(f"- Embedding dimension (d_model): {embed_dim}")
            logger.info(f"- Attention heads: {nhead}")
            logger.info(f"- Feedforward dim: {dim_feedforward}")
            logger.info(f"- Dropout: {dropout}")
            logger.info(f"- Pretrained EfficientNet: {pretrained}")
            logger.info(f"- Number of genes: {num_genes}")
            logger.info(f"- Image channels: {image_channels}")

            # --- Model initialization ---
            model = ImageGeneCrossTransformer(
                num_genes=num_genes,
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pretrained=pretrained,
            )

        elif params["model"] == "ImageToGeneTransformer":
            # --- Get model hyperparameters ---
            embed_dim = params.get("embed_dim", 768)
            nhead = params.get("nhead", 8)
            encoder_layers = params.get("encoder_layers", 6)
            decoder_layers = params.get("decoder_layers", 4)
            dim_feedforward = params.get("dim_feedforward", 2048)
            dropout = params.get("dropout", 0.1)
            patch_size = params.get("patch_size", 16)
            activation = params.get("activation", "relu")
            layer_norm_eps = params.get("layer_norm_eps", 1e-5)
            batch_first = params.get("batch_first", True)
            freeze_encoder = params.get("freeze_encoder", False)
            H = params.get("image_height", 128)
            W = params.get("image_width", 128)
            image_channels = params.get("image_channels", 3)

            # --- Logging configuration ---
            logger.info("Initializing ImageToGeneTransformer with:")
            logger.info(f"- Image size: {H}x{W}")
            logger.info(f"- Embedding dimension (d_model): {embed_dim}")
            logger.info(f"- Encoder layers: {encoder_layers}")
            logger.info(f"- Decoder layers: {decoder_layers}")
            logger.info(f"- Attention heads: {nhead}")
            logger.info(f"- Feedforward dim: {dim_feedforward}")
            logger.info(f"- Dropout: {dropout}")
            logger.info(f"- Patch size: {patch_size}")
            logger.info(f"- Activation: {activation}")
            logger.info(f"- Layer norm eps: {layer_norm_eps}")
            logger.info(f"- Batch first: {batch_first}")
            logger.info(f"- Freeze encoder: {freeze_encoder}")
            logger.info(f"- Number of genes: {num_genes}")
            logger.info(f"- Image channels: {image_channels}")

            # --- Model initialization ---
            model = ImageToGeneTransformer(
                H=H,
                W=W,
                C=image_channels,
                num_genes=num_genes,
                d_model=embed_dim,
                encoder_layers=encoder_layers,
                decoder_layers=decoder_layers,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                freeze_encoder=freeze_encoder,
                patch_size=patch_size,
            )
        elif params["model"] == "LoCIGEN":
            logger.info("Initializing LoCIGEN model...")
            pretrained = params.get("pretrained", True)
            # model = LoCIGENv2(num_genes=num_genes, pretrained=pretrained, dropout=0.5)
            model = LoCIGEN( num_genes=num_genes, mode="xenium", pretrained=pretrained, dropout=0.3)
        
        else:
            logger.error('No model found with name: %s', params["model"])
            raise ValueError("No model found")

        logger.info("Model\n%s", model)


        # Device / GPU logic (robust). Always set a device (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model.to(device)

        # Optimizer
        if params.get("optimizer", "Adam").lower() == "adam":
            optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)
        elif params.get("optimizer", "sgd").lower() == "sgd":
            optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif params.get("optimizer", "Adamw").lower() == "adamw":   
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            logger.error("Optimizer not defined!!! Defaulting to Adam")
            optimizer = Adam(model.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=params.get('lr_patience', 5),
            factor=params.get('lr_factor', 0.5),
            min_lr=params.get('min_lr', 1e-7),
            verbose=True
        )

        train_losses = []
        val_losses = []

        best_spearman = -np.inf
        patience_counter = 0
        patience = params.get('early_stopping_patience', 15)

        # Training loop
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            all_y_true = []
            all_y_pred = []
            pearson_train_corr = []
            spearman_train_corr = []

            for batch_idx, (images, coords, targets) in enumerate(train_dataloader):
                # Move data to device
                images = images.to(device)             # shape: (B, 3, 224, 224)
                coords = coords.to(device)
                y_true = targets.to(device)            # shape: (B, num_genes)
                
                
                # === Sanity check: Detect NaNs or Infs in input data ===
                if not torch.isfinite(images).all():
                    logger.warning(f"[Epoch {epoch+1} | Batch {batch_idx}] NaN/Inf detected in input images. Skipping batch.")
                    continue
                if not torch.isfinite(y_true).all():
                    logger.warning(f"[Epoch {epoch+1} | Batch {batch_idx}] NaN/Inf detected in ground truth. Skipping batch.")
                    continue


                if params["model"] in ["ImageGeneCrossTransformer", "ImageToGeneTransformer"]:
                    y_pred,_ = model(images, gene_values=y_true)  # Pass gene values during training
                elif params["model"] == "LoCIGEN":
                    out = model(images, coords)
                    y_pred = out["pred"]                
                else:
                    y_pred = model(images)                 # shape: (B, num_genes)


                y_pred = torch.where(y_pred < 0, torch.tensor(0.0, device=y_pred.device), y_pred)
                
                # === Sanity check: Detect NaNs/Infs in predictions ===
                if not torch.isfinite(y_pred).all():
                    logger.warning(f"[Epoch {epoch+1} | Batch {batch_idx}] NaN/Inf detected in model output. Zeroing out predictions.")
                    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)


                print(f"image : {images.shape}, y_true: {y_true.shape}, y_pred: {y_pred.shape}")
                
                if batch_idx == 0:
                    print("\n===================== DEBUG: Epoch {} =====================".format(epoch + 1))
                    print("Sample y_true values (first sample, first 10 genes):")
                    print(y_true[0, :10].detach().cpu().numpy())
                    print("Sample y_pred values (first sample, first 10 genes):")
                    print(y_pred[0, :10].detach().cpu().numpy())

                    # Compute basic stats
                    y_true_np = y_true.detach().cpu().numpy()
                    y_pred_np = y_pred.detach().cpu().numpy()

                    print(f"y_true  -> mean={np.nanmean(y_true_np):.4f}, std={np.nanstd(y_true_np):.4f}, "
                        f"min={np.nanmin(y_true_np):.4f}, max={np.nanmax(y_true_np):.4f}")

                    print(f"y_pred  -> mean={np.nanmean(y_pred_np):.4f}, std={np.nanstd(y_pred_np):.4f}, "
                        f"min={np.nanmin(y_pred_np):.4f}, max={np.nanmax(y_pred_np):.4f}")

                # choose loss
                # loss = None
                loss_fn_name = params.get("loss_fn", "")
                if loss_fn_name == "mse" or loss_fn_name == "MSELoss":
                    mse_criterion = torch.nn.MSELoss()
                    loss = mse_criterion(y_pred, y_true)
                elif loss_fn_name == "locigen_loss":
                    loss = locigen_loss(out, y_true)
                else:
                    logger.error("Loss Function not defined!!!")
                    raise ValueError("Loss Function not defined")

                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                gc.collect()

                epoch_train_loss += float(loss.item())
                all_y_true.append(y_true.detach().cpu().numpy())
                all_y_pred.append(y_pred.detach().cpu().numpy())

                try:
                    batch_pearson = pearsonr(y_pred.detach().cpu().numpy().flatten(), y_true.detach().cpu().numpy().flatten())[0]
                    print(f"Batch {batch_idx} Pearson correlation: {batch_pearson:.4f}")
                except Exception as e:
                    print("Exception+++++++++++++++++++++++++++++++++++++++++++++ ",{e})
                    batch_pearson = 0.0
                pearson_train_corr.append(float(batch_pearson.mean().item()))

                # Spearman (use your function that returns vector)
                try:
                    batch_spearman = spearmanrr(y_pred, y_true)
                    spearman_train_corr.append(batch_spearman.mean().item())

                except Exception as e:
                    print(f"code is in Exception++++++++++++++++++++++++++++++++ {e}")
                    spearman_train_corr.append(0.0)

                if batch_idx % 10 == 0:
                    logger.info(f"Fold {fold_idx+1} Train Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Pearson: {pearson_train_corr[-1]:.4f}, Spearman: {spearman_train_corr[-1]:.4f}")

            # epoch-level aggregation
            all_y_true = np.vstack(all_y_true) if len(all_y_true) > 0 else np.array([])
            all_y_pred = np.vstack(all_y_pred) if len(all_y_pred) > 0 else np.array([])

            print(f"all_y_true: {type(all_y_true)}")
            print(f"all_y_pred: {type(all_y_pred)}")

            print(f"========================= {np.mean((all_y_pred - all_y_true) ** 2)}")
            results_train = compute_metrics(all_y_true, all_y_pred) if all_y_true.size else {}

            avg_train_loss = epoch_train_loss / max(1, len(train_dataloader))
            logger.info(f"Fold {fold_idx+1} Train Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Pearson Mean: {np.mean(pearson_train_corr):.4f}, Spearman Mean: {np.mean(spearman_train_corr):.4f}")
            if results_train:
                logger.info(f"Training Metrics: {results_train}")

            # Write train row to CSV
            with open(metrics_csv_path, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    'fold', 'epoch', 'phase', 'custom_loss', 'pearson_mean_cell_wise', 'spearman_mean_cell_wise', 'pearson_mean_genewise', 'spearman_mean_genewise',
                    'l1_error_mean', 'l2_errors_mean', 'r2_scores_mean', 'pearson_std',
                    'l2_error_q1', 'l2_error_q2', 'l2_error_q3',
                    'r2_score_q1', 'r2_score_q2', 'r2_score_q3', 'mape_mean', 'mape_std', 'rmse_mean', 'rmse_std'
                ])
                row = {'fold': fold_idx + 1, 'epoch': epoch + 1, 'phase': 'train', 'custom_loss': f"{avg_train_loss:.4f}", 'pearson_mean_cell_wise': f"{np.mean(pearson_train_corr):.4f}", 'spearman_mean_cell_wise': f"{np.mean(spearman_train_corr):.4f}"}
                # add other metrics if present
                if results_train:
                    for k in results_train:
                        if k in writer.fieldnames:
                            row[k] = results_train[k]
                writer.writerow(row)

            train_losses.append(avg_train_loss)
            
            del loss, y_pred, y_true, images, all_y_true, all_y_pred
            torch.cuda.empty_cache()
            gc.collect()

            # Validation
            model.eval()
            epoch_val_loss = 0.0
            val_y_true = []
            val_y_pred = []
            spearman_val_corr = []
            pearson_val_corr = []

            with torch.no_grad():
                for batch_idx, (images, coords, targets) in enumerate(val_dataloader):
                    # Move images and targets to GPU
                    images = images.to(device)                          # shape: (B, 3, 224, 224)
                    coords = coords.to(device)
                    y_true = targets.to(device)                         # shape: (B, num_genes)

                    # --- Forward pass through model ---
                    if params["model"] in ["ImageGeneCrossTransformer", "ImageToGeneTransformer"]:
                        y_pred,_ = model(images)
                    elif params["model"] == "LoCIGEN":
                        out = model(images, coords)
                        y_pred = out["pred"]
                    else:
                        y_pred = model(images)    
                        y_pred = torch.where(y_pred < 0, torch.tensor(0.0, device=y_pred.device), y_pred)

                    loss_fn_name = params.get("loss_fn", "")
                    if loss_fn_name == "mse" or loss_fn_name == "MSELoss":
                        mse_criterion = torch.nn.MSELoss()
                        loss = mse_criterion(y_pred, y_true)
                    elif loss_fn_name == "locigen_loss":
                        loss = locigen_loss(out, y_true)
                    else:
                        logger.error("Loss Function not defined!!!")
                        raise ValueError("Loss Function not defined")

                    epoch_val_loss += float(loss.item())
                    val_y_true.append(y_true.detach().cpu().numpy())
                    val_y_pred.append(y_pred.detach().cpu().numpy())

                    try:
                        batch_pearson = pearsonr(y_pred.detach().cpu().numpy().flatten(), y_true.detach().cpu().numpy().flatten())[0]
                    except Exception:
                        batch_pearson = 0.0
                    pearson_val_corr.append(float(batch_pearson.mean().item()))

                    try:
                        batch_spearman = spearmanrr(y_pred, y_true)
                        spearman_val_corr.append(float(batch_spearman.mean().item()))
                    except Exception:
                        spearman_val_corr.append(0.0)

                    if batch_idx % 10 == 0:
                        logger.info(f"Fold {fold_idx+1} Val Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Pearson: {pearson_val_corr[-1]:.4f}, Spearman: {spearman_val_corr[-1]:.4f}")

                val_y_true = np.vstack(val_y_true) if len(val_y_true) > 0 else np.array([])
                val_y_pred = np.vstack(val_y_pred) if len(val_y_pred) > 0 else np.array([])

                # Save scatter of first up-to-25 samples (if present)
                if val_y_true.size:
                    n_display = min(25, val_y_true.shape[0])
                    val_y_true_25 = val_y_true[:n_display, :]
                    val_y_pred_25 = val_y_pred[:n_display, :]

                    # Create grid (square)
                    grid_sz = int(np.ceil(np.sqrt(n_display)))
                    fig, axes = plt.subplots(nrows=grid_sz, ncols=grid_sz, figsize=(4 * grid_sz, 4 * grid_sz))
                    axes = np.atleast_2d(axes)
                    for idx in range(n_display):
                        r = idx // grid_sz
                        c = idx % grid_sz
                        axes[r, c].scatter(val_y_true_25[idx], val_y_pred_25[idx], alpha=0.6)
                        min_val = min(val_y_true_25[idx].min(), val_y_pred_25[idx].min())
                        max_val = max(val_y_true_25[idx].max(), val_y_pred_25[idx].max())
                        axes[r, c].plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red')
                        axes[r, c].set_title(f"Sample {idx+1}")
                    plt.tight_layout()
                    scatter_save_path = os.path.join(model_save_dir, f"fold_{fold_idx+1}", f"scatter_plot_25_samples_epoch_{epoch+1}.png")
                    os.makedirs(os.path.dirname(scatter_save_path), exist_ok=True)
                    plt.savefig(scatter_save_path)
                    plt.close(fig)

                results_val = compute_metrics(val_y_true, val_y_pred) if val_y_true.size else {}


                # Memory cleanup per validation batch
                del loss, y_pred, y_true, images
                torch.cuda.empty_cache()
                gc.collect()


                avg_val_loss = epoch_val_loss / max(1, len(val_dataloader))
                logger.info(f"Fold {fold_idx+1} Val Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss:.4f}, Pearson Mean: {np.mean(pearson_val_corr):.4f}, Spearman Mean: {np.mean(spearman_val_corr):.4f}")
                if results_val:
                    logger.info(f"Validation Metrics: {results_val}")

                # Write val row to CSV
                with open(metrics_csv_path, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=[
                        'fold', 'epoch', 'phase', 'custom_loss', 'pearson_mean_cell_wise', 'spearman_mean_cell_wise', 'pearson_mean_genewise', 'spearman_mean_genewise',
                        'l1_error_mean', 'l2_errors_mean', 'r2_scores_mean', 'pearson_std',
                        'l2_error_q1', 'l2_error_q2', 'l2_error_q3',
                        'r2_score_q1', 'r2_score_q2', 'r2_score_q3', 'mape_mean', 'mape_std', 'rmse_mean', 'rmse_std'
                    ])
                    row = {'fold': fold_idx + 1, 'epoch': epoch + 1, 'phase': 'val', 'custom_loss': f"{avg_val_loss:.4f}", 'pearson_mean_cell_wise': f"{np.mean(pearson_val_corr):.4f}", 'spearman_mean_cell_wise':  f"{np.mean(spearman_val_corr):.4f}"}
                    if results_val:
                        for k in results_val:
                            if k in writer.fieldnames:
                                row[k] = results_val[k]
                    writer.writerow(row)

                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)

                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate for epoch {epoch + 1}: {current_lr}")

                # Early stopping based on spearman_mean_cell_wise if present
                current_spearman = float(results_val.get('spearman_mean_genewise', np.mean(spearman_val_corr) if spearman_val_corr else -np.inf))
                if current_spearman > best_spearman:
                    best_spearman = current_spearman
                    patience_counter = 0
                    # Save best
                    best_model_path = os.path.join(model_save_dir, f"fold_{fold_idx+1}", "best_model.pth")
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"Saved best model to {best_model_path}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement in spearman genewise. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    logger.info("Early stopping triggered. Breaking training loop.")
                    break

                # plot losses
                try:
                    plt.figure(figsize=(8, 5))
                    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o")
                    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title(f"Fold {fold_idx+1} Loss")
                    plt.legend()
                    plt.grid(True)
                    graph_save_path = os.path.join(model_save_dir, f"fold_{fold_idx+1}", f"loss_plot_epoch.png")
                    plt.savefig(graph_save_path)
                    plt.close('all')  # Close all figures to prevent memory leaks
                except Exception as e:
                    logger.warning(f"Could not plot losses: {e}")

            # Save snapshot per epoch
            snapshot_path = os.path.join(model_save_dir, f"fold_{fold_idx+1}", f"epoch_model.pth")
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            torch.save(model.state_dict(), snapshot_path)

        models["complete_model"] = model
        return models

    # --- Define paths --- 
    patch_file = f"{params['dataset_path']}/cells_patches.h5" 
    adata_file = f"{params['dataset_path']}/cells_expression.h5ad"

    # --- Optional image transforms ---
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    images, genes, pixel_x, pixel_y = load_xenium_raw_data(
        h5_img_path=patch_file,
        h5ad_gene_path=adata_file,
        gene_names=None
    )

    logger.info(f"Loaded Xenium data: images={images.shape}, genes={genes.shape}")
    # --- K-Fold setup ---
    k = params.get("k_folds", 5)

    kf = KFold(n_splits= k, shuffle=True, random_state=42)
    fold_models = []
        
    # results_path_fold = os.path.join(params["result_path"], "result_vit_2025-11-22_23-27-42")  
    # # find all existing fold_* folders
    # completed_folds = {
    #     int(name.split("_")[1]) - 1
    #     for name in os.listdir(results_path_fold)
    #     if name.startswith("fold_")
    # }
    
    # start_fold = 1
    # --- K-Fold Loop ---
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(images)):
        # if fold_idx in completed_folds:
        #     logger.info(f"Skipping Fold {fold_idx+1} (already completed)")
        #     continue
        # if fold_idx < start_fold:
        #     logger.info(f"Skipping Fold {fold_idx+1} (already completed)")
        #     continue
        
    
        logger.info(f"\n===== Starting Fold {fold_idx+1} =====")

        # ---------- Create FULL gene matrix for this fold ----------
        train_genes_full, val_genes_full = log1p_normalization_train_val_separate(
            raw_genes=genes,
            train_idx=train_idx,
            val_idx=val_idx,
            scale_factor=params.get("scale_factor", 1000000)
        )

        # --------------------------------------------------
        # Build datasets (coords INCLUDED)
        # --------------------------------------------------
        train_dataset = XeniumLoCIGENDataset(
            images=images,
            genes=train_genes_full,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            indices=train_idx,
            transform=transform
        )

        val_dataset = XeniumLoCIGENDataset(
            images=images,
            genes=val_genes_full,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            indices=val_idx,
            transform=transform
        )


        logger.info(f"Fold {fold_idx+1}: Train={len(train_dataset)}, Val={len(val_dataset)}")

        # ---------- Train ----------
        fold_model = train_regression(
            train_dataset, val_dataset, fold_idx,
            g=g,
            seed_worker=seed_worker,
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            batch_size=params["batch_size"]
        )

        fold_models.append(fold_model)
        logger.info(f"===== Completed Fold {fold_idx+1} =====")

    # --- Save summary ---
    summary_file = os.path.join(model_save_dir, "folds_summary.txt")
    with open(summary_file, "w") as sf:
        sf.write(f"Completed {k} folds. Models saved in {model_save_dir}\n")

    logger.info("All folds completed successfully.")


if __name__ == "__main__":
    main()
