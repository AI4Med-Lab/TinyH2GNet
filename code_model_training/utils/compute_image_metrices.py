"""Image metrics helpers.

This module provides `compute_image_metrics_per_sample` which accepts
PyTorch tensors of shape (B,C,H,W) and returns per-sample numpy arrays
for common image-quality statistics.
"""

import torch
import numpy as np
from pytorch_msssim import ssim as torch_ssim
import logging

logger = logging.getLogger(__name__)


def compute_image_metrics_per_sample(y_true_tensor, y_pred_tensor, eps=1e-8):
    """Return per-sample image metrics as numpy arrays: l1, l2, r2, mape, rmse.
    Inputs are tensors (B,C,H,W).
    """
    # move to cpu numpy
    with torch.no_grad():
        yt = y_true_tensor.detach().cpu()
        yp = y_pred_tensor.detach().cpu()
        B = yt.shape[0]

        # per-sample L1 and L2 (mean over pixels)
        absdiff = (yp - yt).abs()
        l1_per = absdiff.mean(dim=(1, 2, 3)).numpy()
        l2_per = ((yp - yt) ** 2).mean(dim=(1, 2, 3)).numpy()
        rmse_per = np.sqrt(l2_per)

        # MAPE per-sample (use element-wise division with eps)
        mape_per = ((absdiff) / (yt.abs() + eps)).mean(dim=(1, 2, 3)).numpy()

        # R2 per-sample: 1 - SS_res/SS_tot (flattened)
        t_flat = yt.view(B, -1).numpy()
        p_flat = yp.view(B, -1).numpy()
        ss_res = np.sum((t_flat - p_flat) ** 2, axis=1)
        t_mean = np.mean(t_flat, axis=1, keepdims=True)
        ss_tot = np.sum((t_flat - t_mean) ** 2, axis=1)
        r2_per = 1.0 - (ss_res / (ss_tot + eps))
        r2_per[ss_tot == 0] = 0.0

    return {
        'l1': l1_per,
        'l2': l2_per,
        'rmse': rmse_per,
        'mape': mape_per,
        'r2': r2_per,
    }


def compute_batch_ssim(y_true_tensor, y_pred_tensor):
    """Compute mean SSIM over a batch using `pytorch_msssim.ssim`.
    Expects tensors in (B,C,H,W) with values in [0,1]. Returns (mean, std).
    """
    try:
        # ensure float tensors
        yt = y_true_tensor.detach().float()
        yp = y_pred_tensor.detach().float()

        # if batch dim missing, add
        if yt.dim() == 3:
            yt = yt.unsqueeze(0)
            yp = yp.unsqueeze(0)

        # compute per-sample SSIM (size_average=False -> returns tensor of shape (N,))
        vals = torch_ssim(yp, yt, data_range=1.0, size_average=False)
        vals_np = vals.detach().cpu().numpy().astype(float)
        return float(vals_np.mean()), float(vals_np.std())
    except Exception as e:
        try:
            logger.warning(f"SSIM computation failed: {e}")
        except Exception:
            print(f"SSIM computation failed: {e}")
        return 0.0, 0.0


def aggregate_image_epoch_metrics(l1_list, l2_list, r2_list, mape_list, rmse_list):
    """Aggregate per-batch per-sample metric arrays into epoch-level stats.

    Each input is a list of 1D numpy arrays (per-batch per-sample results).
    Returns a dict with floats ready for CSV insertion.
    """
    if len(l2_list):
        l1_all = np.concatenate(l1_list)
        l2_all = np.concatenate(l2_list)
        r2_all = np.concatenate(r2_list)
        mape_all = np.concatenate(mape_list)
        rmse_all = np.concatenate(rmse_list)

        l1_mean = float(np.mean(l1_all))
        l2_mean = float(np.mean(l2_all))
        r2_mean = float(np.mean(r2_all))
        mape_mean = float(np.mean(mape_all))
        mape_std = float(np.std(mape_all))
        rmse_mean = float(np.mean(rmse_all))
        rmse_std = float(np.std(rmse_all))

        l2_q1, l2_q2, l2_q3 = np.percentile(l2_all, [25, 50, 75]).tolist()
        r2_q1, r2_q2, r2_q3 = np.percentile(r2_all, [25, 50, 75]).tolist()
    else:
        l1_mean = l2_mean = r2_mean = mape_mean = mape_std = rmse_mean = rmse_std = 0.0
        l2_q1 = l2_q2 = l2_q3 = r2_q1 = r2_q2 = r2_q3 = 0.0

    return {
        'l1_mean': l1_mean,
        'l2_mean': l2_mean,
        'r2_mean': r2_mean,
        'mape_mean': mape_mean,
        'mape_std': mape_std,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
        'l2_q1': l2_q1,
        'l2_q2': l2_q2,
        'l2_q3': l2_q3,
        'r2_q1': r2_q1,
        'r2_q2': r2_q2,
        'r2_q3': r2_q3,
    }