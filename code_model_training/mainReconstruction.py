'''
her2:  /home/puneet/mk/data/her2st_dataset/224

cscc: /home/puneet/mk/data/cscc_dataset/224

model:   ImageGeneCrossTransformerRoPE

'''

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
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from datetime import datetime
import argparse
import json
import pandas as pd
from scipy import sparse
from pytorch_msssim import ssim as torch_ssim
from torch.nn import DataParallel
from sklearn.model_selection import KFold
from utils.dataPrep import PatchDataset
from compute_metrics import compute_metrics
from torchvision import transforms
from setup_logger import setup_logging
from set_deterministic_seed import set_deterministic_seed
from models import STNet, EfficientNet, EfficientNetB4GeneRegressor, Custom_VGG16, HisToGene, TCGN
from torch.utils.data import ConcatDataset
from proposedModels import ImageGeneCrossTransformer
from sklearn.model_selection import KFold
from proposedModels2 import ImageToGeneTransformer
from reconsModels import ResNet50UNet, SMPResNet50UNet, SMPDenseNet121UNet
from utils.compute_image_metrices import compute_image_metrics_per_sample, compute_batch_ssim, aggregate_image_epoch_metrics



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

    with open(metrics_csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'fold', 'epoch', 'phase', 'custom_loss', 'ssim_mean', 'ssim_std', 'l1_error_mean', 'l2_errors_mean', 'r2_scores_mean',
            'l2_error_q1', 'l2_error_q2', 'l2_error_q3',
            'r2_score_q1', 'r2_score_q2', 'r2_score_q3', 'mape_mean', 'mape_std', 'rmse_mean', 'rmse_std'
        ])
        writer.writeheader()

    # --- Training routine (mostly unchanged) ---
    def train_regression(train_datasets, val_datasets, fold_idx,
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
        else:
            raise ValueError(f"Unknown dataset name: {params['dataset_name']}")

        # image channels (for reconstruction models)
        image_channels = params.get("image_channels", 3)


        
        # === Reconstruction / image->image UNet models ===
        if params["model"] == "SMPResNet50UNet":
            pretrained = params.get("pretrained", True)
            encoder_weights = 'imagenet' if pretrained else None
            model = SMPResNet50UNet(in_channels=image_channels, out_channels=image_channels, encoder_weights=encoder_weights)
        elif params["model"] == "SMPDenseNet121UNet":
            pretrained = params.get("pretrained", True)
            encoder_weights = 'imagenet' if pretrained else None
            model = SMPDenseNet121UNet(in_channels=image_channels, out_channels=image_channels, encoder_weights=encoder_weights)

        elif params["model"] == "ResNet50UNet":
            # local ResNet50 UNet implementation (from reconsModels)
            model = ResNet50UNet(out_channels=image_channels)
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

        best_val_loss = np.inf
        patience_counter = 0
        patience = params.get('early_stopping_patience', 15)

        # Training loop
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0.0
            all_y_true = []
            all_y_pred = []
            ssim_train_list = []
            recon_train_l1 = []
            recon_train_l2 = []
            recon_train_rmse = []
            recon_train_mape = []
            recon_train_r2 = []

            for batch_idx, batch in enumerate(train_dataloader):
                # Support dataloaders that return either (images, targets) or images only
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, targets = batch
                else:
                    images = batch
                    targets = None

                # Move data to device
                images = images.to(device)             # shape: (B, C, H, W)
                is_recon = params["model"] in ["SMPResNet50UNet", "ResNet50UNet", "SMPDenseNet121UNet"]

                # For reconstruction tasks, target is the input image itself
                if is_recon:
                    y_true = images
                else:
                    if targets is None:
                        logger.warning(f"Batch {batch_idx} missing targets but model expects targets. Skipping batch.")
                        continue
                    y_true = targets.to(device)

                # === Sanity check: Detect NaNs or Infs in input data ===
                if not torch.isfinite(images).all():
                    logger.warning(f"[Epoch {epoch+1} | Batch {batch_idx}] NaN/Inf detected in input images. Skipping batch.")
                    continue
                if not torch.isfinite(y_true).all():
                    logger.warning(f"[Epoch {epoch+1} | Batch {batch_idx}] NaN/Inf detected in ground truth. Skipping batch.")
                    continue

                # Forward pass
                if params["model"] in ["ImageGeneCrossTransformer", "ImageToGeneTransformer"]:
                    y_pred,_ = model(images, gene_values=y_true)  # Pass gene values during training
                else:
                    y_pred = model(images)

                # Clip outputs appropriately
                if is_recon:
                    y_pred = torch.clamp(y_pred, 0.0, 1.0)
                else:
                    y_pred = torch.where(y_pred < 0, torch.tensor(0.0, device=y_pred.device), y_pred)

                # === Sanity check: Detect NaNs/Infs in predictions ===
                if not torch.isfinite(y_pred).all():
                    logger.warning(f"[Epoch {epoch+1} | Batch {batch_idx}] NaN/Inf detected in model output. Zeroing out predictions.")
                    y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

                print(f"image : {images.shape}, y_true: {y_true.shape}, y_pred: {y_pred.shape}")

                if batch_idx == 0:
                    print("\n===================== DEBUG: Epoch {} =====================".format(epoch + 1))
                    if is_recon:
                        # show basic stats for images
                        y_true_np = y_true.detach().cpu().numpy()
                        y_pred_np = y_pred.detach().cpu().numpy()
                        print(f"Image y_true  -> mean={np.nanmean(y_true_np):.4f}, std={np.nanstd(y_true_np):.4f}, min={np.nanmin(y_true_np):.4f}, max={np.nanmax(y_true_np):.4f}")
                        print(f"Image y_pred  -> mean={np.nanmean(y_pred_np):.4f}, std={np.nanstd(y_pred_np):.4f}, min={np.nanmin(y_pred_np):.4f}, max={np.nanmax(y_pred_np):.4f}")
                    else:
                        print("Sample y_true values (first sample, first 10 genes):")
                        print(y_true[0, :10].detach().cpu().numpy())
                        print("Sample y_pred values (first sample, first 10 genes):")
                        print(y_pred[0, :10].detach().cpu().numpy())

                # choose loss
                loss_fn_name = params.get("loss_fn", "")
                if loss_fn_name == "mse" or loss_fn_name == "MSELoss":
                    mse_criterion = torch.nn.MSELoss()
                    loss = mse_criterion(y_pred, y_true)
                else:
                    logger.error("Loss Function not defined!!!")
                    raise ValueError("Loss Function not defined")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                gc.collect()

                epoch_train_loss += float(loss.item())

                # Metrics collection
                if not is_recon:
                    all_y_true.append(y_true.detach().cpu().numpy())
                    all_y_pred.append(y_pred.detach().cpu().numpy())
                    # For gene-prediction models we rely on compute_metrics later; skip pearson/spearman
                    last_ssim = 0.0
                else:
                    # For reconstruction models compute SSIM per-batch
                    last_ssim, last_ssim_std = compute_batch_ssim(y_true, y_pred)
                    ssim_train_list.append(last_ssim)

                    # compute per-sample metrics and store for epoch-level aggregation
                    per_sample = compute_image_metrics_per_sample(y_true, y_pred)
                    recon_train_l1.append(per_sample['l1'])
                    recon_train_l2.append(per_sample['l2'])
                    recon_train_rmse.append(per_sample['rmse'])
                    recon_train_mape.append(per_sample['mape'])
                    recon_train_r2.append(per_sample['r2'])

                if batch_idx % 10 == 0:
                    ssim_log = f"SSIM: {ssim_train_list[-1]:.4f}" if len(ssim_train_list) else "SSIM: N/A"
                    logger.info(f"Fold {fold_idx+1} Train Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, {ssim_log}")

            # epoch-level aggregation
            all_y_true = np.vstack(all_y_true) if len(all_y_true) > 0 else np.array([])
            all_y_pred = np.vstack(all_y_pred) if len(all_y_pred) > 0 else np.array([])

            print(f"all_y_true: {type(all_y_true)}")
            print(f"all_y_pred: {type(all_y_pred)}")

            print(f"========================= {np.mean((all_y_pred - all_y_true) ** 2)}")
            results_train = compute_metrics(all_y_true, all_y_pred) if all_y_true.size else {}

            avg_train_loss = epoch_train_loss / max(1, len(train_dataloader))
            # Log SSIM mean for reconstruction models, otherwise skip
            if len(ssim_train_list):
                logger.info(f"Fold {fold_idx+1} Train Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, SSIM Mean: {np.mean(ssim_train_list):.4f}")
            else:
                logger.info(f"Fold {fold_idx+1} Train Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
            if results_train:
                logger.info(f"Training Metrics: {results_train}")

            # Write train row to CSV
            with open(metrics_csv_path, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    'fold', 'epoch', 'phase', 'custom_loss', 'ssim_mean', 'ssim_std', 'l1_error_mean', 'l2_errors_mean', 'r2_scores_mean',
                    'l2_error_q1', 'l2_error_q2', 'l2_error_q3',
                    'r2_score_q1', 'r2_score_q2', 'r2_score_q3', 'mape_mean', 'mape_std', 'rmse_mean', 'rmse_std'
                ])
                ssim_mean = np.mean(ssim_train_list) if len(ssim_train_list) else 0.0
                ssim_std = np.std(ssim_train_list) if len(ssim_train_list) else 0.0

                # Aggregate recon per-sample metrics across batches (optimized)
                agg = aggregate_image_epoch_metrics(recon_train_l1, recon_train_l2, recon_train_r2, recon_train_mape, recon_train_rmse)

                row = {
                    'fold': fold_idx + 1,
                    'epoch': epoch + 1,
                    'phase': 'train',
                    'custom_loss': f"{avg_train_loss:.4f}",
                    'ssim_mean': f"{ssim_mean:.4f}",
                    'ssim_std': f"{ssim_std:.4f}",
                    'l1_error_mean': f"{agg['l1_mean']:.6f}",
                    'l2_errors_mean': f"{agg['l2_mean']:.6f}",
                    'r2_scores_mean': f"{agg['r2_mean']:.6f}",
                    'l2_error_q1': f"{agg['l2_q1']:.6f}",
                    'l2_error_q2': f"{agg['l2_q2']:.6f}",
                    'l2_error_q3': f"{agg['l2_q3']:.6f}",
                    'r2_score_q1': f"{agg['r2_q1']:.6f}",
                    'r2_score_q2': f"{agg['r2_q2']:.6f}",
                    'r2_score_q3': f"{agg['r2_q3']:.6f}",
                    'mape_mean': f"{agg['mape_mean']:.6f}",
                    'mape_std': f"{agg['mape_std']:.6f}",
                    'rmse_mean': f"{agg['rmse_mean']:.6f}",
                    'rmse_std': f"{agg['rmse_std']:.6f}"
                }

                # add other metrics if present (e.g., gene metrics)
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
            ssim_val_list = []
            recon_val_l1 = []
            recon_val_l2 = []
            recon_val_rmse = []
            recon_val_mape = []
            recon_val_r2 = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    # Support dataloaders that return either (images, targets) or images only
                    if isinstance(batch, (list, tuple)) and len(batch) == 2:
                        images, targets = batch
                    else:
                        images = batch
                        targets = None

                    # Move images and targets to GPU
                    images = images.to(device)                          # shape: (B, C, H, W)
                    is_recon = params["model"] in ["SMPResNet50UNet", "ResNet50UNet", "SMPDenseNet121UNet"]

                    if is_recon:
                        y_true = images
                    else:
                        if targets is None:
                            logger.warning(f"Validation batch {batch_idx} missing targets but model expects targets. Skipping batch.")
                            continue
                        y_true = targets.to(device)

                    # --- Forward pass through model ---
                    if params["model"] in ["ImageGeneCrossTransformer", "ImageToGeneTransformer"]:
                        y_pred,_ = model(images)
                    else:
                        y_pred = model(images)

                    if is_recon:
                        y_pred = torch.clamp(y_pred, 0.0, 1.0)
                    else:
                        y_pred = torch.where(y_pred < 0, torch.tensor(0.0, device=y_pred.device), y_pred)

                    loss_fn_name = params.get("loss_fn", "")
                    if loss_fn_name == "mse" or loss_fn_name == "MSELoss":
                        mse_criterion = torch.nn.MSELoss()
                        loss = mse_criterion(y_pred, y_true)
                    else:
                        logger.error("Loss Function not defined!!!")
                        raise ValueError("Loss Function not defined")

                    epoch_val_loss += float(loss.item())

                    if not is_recon:
                        val_y_true.append(y_true.detach().cpu().numpy())
                        val_y_pred.append(y_pred.detach().cpu().numpy())
                    else:
                        last_ssim, last_ssim_std = compute_batch_ssim(y_true, y_pred)
                        ssim_val_list.append(last_ssim)

                        # compute per-sample metrics for validation reconstruction
                        per_sample_val = compute_image_metrics_per_sample(y_true, y_pred)
                        recon_val_l1.append(per_sample_val['l1'])
                        recon_val_l2.append(per_sample_val['l2'])
                        recon_val_rmse.append(per_sample_val['rmse'])
                        recon_val_mape.append(per_sample_val['mape'])
                        recon_val_r2.append(per_sample_val['r2'])

                    if batch_idx % 10 == 0:
                        ssim_log = f"SSIM: {ssim_val_list[-1]:.4f}" if len(ssim_val_list) else "SSIM: N/A"
                        logger.info(f"Fold {fold_idx+1} Val Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, {ssim_log}")

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
                if len(ssim_val_list):
                    logger.info(f"Fold {fold_idx+1} Val Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss:.4f}, SSIM Mean: {np.mean(ssim_val_list):.4f}")
                else:
                    logger.info(f"Fold {fold_idx+1} Val Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss:.4f}")
                if results_val:
                    logger.info(f"Validation Metrics: {results_val}")

                # Write val row to CSV
                with open(metrics_csv_path, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=[
                        'fold', 'epoch', 'phase', 'custom_loss', 'ssim_mean', 'ssim_std', 'l1_error_mean', 'l2_errors_mean', 'r2_scores_mean',
                        'l2_error_q1', 'l2_error_q2', 'l2_error_q3',
                        'r2_score_q1', 'r2_score_q2', 'r2_score_q3', 'mape_mean', 'mape_std', 'rmse_mean', 'rmse_std'
                    ])
                    ssim_mean_val = np.mean(ssim_val_list) if len(ssim_val_list) else 0.0
                    ssim_std_val = np.std(ssim_val_list) if len(ssim_val_list) else 0.0

                    # Aggregate recon per-sample validation metrics across batches (optimized)
                    agg_val = aggregate_image_epoch_metrics(recon_val_l1, recon_val_l2, recon_val_r2, recon_val_mape, recon_val_rmse)

                    row = {
                        'fold': fold_idx + 1,
                        'epoch': epoch + 1,
                        'phase': 'val',
                        'custom_loss': f"{avg_val_loss:.4f}",
                        'ssim_mean': f"{ssim_mean_val:.4f}",
                        'ssim_std': f"{ssim_std_val:.4f}",
                        'l1_error_mean': f"{agg_val['l1_mean']:.6f}",
                        'l2_errors_mean': f"{agg_val['l2_mean']:.6f}",
                        'r2_scores_mean': f"{agg_val['r2_mean']:.6f}",
                        'l2_error_q1': f"{agg_val['l2_q1']:.6f}",
                        'l2_error_q2': f"{agg_val['l2_q2']:.6f}",
                        'l2_error_q3': f"{agg_val['l2_q3']:.6f}",
                        'r2_score_q1': f"{agg_val['r2_q1']:.6f}",
                        'r2_score_q2': f"{agg_val['r2_q2']:.6f}",
                        'r2_score_q3': f"{agg_val['r2_q3']:.6f}",
                        'mape_mean': f"{agg_val['mape_mean']:.6f}",
                        'mape_std': f"{agg_val['mape_std']:.6f}",
                        'rmse_mean': f"{agg_val['rmse_mean']:.6f}",
                        'rmse_std': f"{agg_val['rmse_std']:.6f}"
                    }

                    if results_val:
                        for k in results_val:
                            if k in writer.fieldnames:
                                row[k] = results_val[k]
                    writer.writerow(row)

                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)

                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate for epoch {epoch + 1}: {current_lr}")

                # Early stopping based on validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best
                    best_model_path = os.path.join(model_save_dir, f"fold_{fold_idx+1}", "best_model.pth")
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"Saved best model to {best_model_path}")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement in val loss. Patience: {patience_counter}/{patience}")

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

    if params["dataset_name"] == "cscc":
        gsm_samples = [
            'GSM4284320', 'GSM4284323', 'GSM4284322',
            'GSM4284317', 'GSM4284327', 'GSM4284316',
            'GSM4284326', 'GSM4284325', 'GSM4284324',
            'GSM4284321', 'GSM4284319', 'GSM4284318'
        ]
        
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
        # gsm_samples =  [
        #     "A1", "A2", "A3", "A4", "A5", "A6"
        # ]      

    # --- Define paths ---
    dataset_path = params.get("dataset_path", "")

    if params["dataset_name"] == "cscc":
        patch_paths = [os.path.join(dataset_path, f"{sample}_patches.h5") for sample in gsm_samples]
        adata_paths = [os.path.join(dataset_path, f"{sample}_spots.h5ad") for sample in gsm_samples]
    if params["dataset_name"] == "her2":
        patch_paths = [os.path.join(dataset_path, f"{sample}.h5") for sample in gsm_samples]
        adata_paths = [os.path.join(dataset_path, f"{sample}.h5ad") for sample in gsm_samples]
    if params["dataset_name"] == "hist2st_845":
        patch_paths = [os.path.join(dataset_path, f"{sample}.h5") for sample in gsm_samples]
        adata_paths = [os.path.join(dataset_path, f"{sample}.h5ad") for sample in gsm_samples]
    logger.info(f"Total samples: {len(gsm_samples)}")
    for p, a in zip(patch_paths, adata_paths):
        logger.info(f"  - {os.path.basename(p)} ↔ {os.path.basename(a)}")

    # --- Check file existence ---
    for p, a in zip(patch_paths, adata_paths):
        if not os.path.exists(p):
            logger.warning(f"Missing patch file: {p}")
        if not os.path.exists(a):
            logger.warning(f"Missing spot file: {a}")

    # --- Gene CSV file ---
    genes_file = os.path.join(params['genes'], f"{params['dataset_name']}.npy")
    print("+++++++++++++++++++++++++++++++",genes_file)
    gene_names = np.load(genes_file, allow_pickle=True).tolist()
    logger.info(f" Using gene list from: {gene_names}")
    if not os.path.exists(genes_file):
        # raise FileNotFoundError(f"Gene list not found: {genes_file}")
        print(genes_file)
        print(gene_names)

    # --- Optional image transforms ---
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # --- K-Fold setup ---
    n_samples = len(gsm_samples)
    k = params.get("k_folds", 5)
    if k > n_samples:
        logger.warning("k_folds > number of samples. Setting k = number of samples.")
        k = n_samples


    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_models = []

    # --- K-Fold Loop ---
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(n_samples))):
        logger.info(f"\n===== Starting Fold {fold_idx+1}/{k} =====")

        train_patch_files = [patch_paths[i] for i in train_idx]
        train_adata_files = [adata_paths[i] for i in train_idx]
        val_patch_files   = [patch_paths[i] for i in val_idx]
        val_adata_files   = [adata_paths[i] for i in val_idx]

        logger.info(f"Fold {fold_idx+1}: Train={len(train_patch_files)}, Val={len(val_patch_files)}")
        train_datasets_list = [
            PatchDataset(
                gene_path=adata_p,
                img_path=img_p,
                gene_names=gene_names,
                transform=transform,
                log_norm=True,
                scale_factor=params.get("scale_factor", 1000000)
            )
            for adata_p, img_p in zip(train_adata_files, train_patch_files)
        ]

        val_datasets_list = [
            PatchDataset(
                gene_path=adata_p,
                img_path=img_p,
                gene_names=gene_names,
                transform=transform,
                log_norm=True,
                scale_factor=params.get("scale_factor", 1000000)
            )
            for adata_p, img_p in zip(val_adata_files, val_patch_files)
        ]

        # --- Combine datasets ---

        train_dataset_concat = ConcatDataset(train_datasets_list)
        val_dataset_concat   = ConcatDataset(val_datasets_list)

        logger.info(f"Fold {fold_idx+1}: Train={len(train_dataset_concat)}, Val={len(val_dataset_concat)}")

        # --- Create fold directory ---
        fold_dir = os.path.join(model_save_dir, f"fold_{fold_idx+1}")
        os.makedirs(fold_dir, exist_ok=True)

        # --- Train model for this fold ---
        models = train_regression(
            train_dataset_concat,
            val_dataset_concat,
            fold_idx,
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            batch_size=params["batch_size"]
        )

        fold_models.append(models)
        logger.info(f"===== Completed Fold {fold_idx+1}/{k} =====")

    # --- Save summary ---
    summary_file = os.path.join(model_save_dir, "folds_summary.txt")
    with open(summary_file, "w") as sf:
        sf.write(f"Completed {k} folds. Models saved in {model_save_dir}\n")

    logger.info("All folds completed successfully.")


if __name__ == "__main__":
    main()
