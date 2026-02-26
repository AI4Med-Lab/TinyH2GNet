import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
# Spearman (batch-wise)
# Uses YOUR spearmanrr function
from compute_metrics import spearmanrr

def locigen_loss(
    out: dict,
    y_true: torch.Tensor,
    lambda_spearman: float = 0.3,
    use_uncertainty: bool = False, ):
    """
    LoCIGEN-v2 loss

    Args:
        out: model output dict
        y_true: (B, G)
        lambda_spearman: weight for Spearman term
        use_uncertainty: enable heteroscedastic loss

    Returns:
        scalar loss
    """

    y_pred = out["pred"]

    # Base MSE
    if use_uncertainty:
        log_var = out["log_var"]
        mse = torch.mean(
            torch.exp(-log_var) * (y_pred - y_true) ** 2 + log_var
        )
    else:
        mse = F.mse_loss(y_pred, y_true)
 
    spearman = spearmanrr(y_pred, y_true)

    loss = mse - lambda_spearman * spearman
    return loss

def locigen_loss_v2_1(
    out,
    y_true,
    gene_mean,
    gene_std,
    lambda_spearman=0.1,
    eps=1e-6
):
    """
    LoCIGEN v2.1 loss:
    - Gene-wise normalized MSE
    - Variance-aware gene weighting
    - Batch-level Spearman regularization
    """

    y_pred = out["pred"]          # (B, G)
    log_var = out["log_var"]      # (B, G)

    # -----------------------------
    # 1. Normalize gene expression
    # -----------------------------
    gene_std = gene_std.clamp(min=eps)

    y_true_norm = (y_true - gene_mean) / gene_std
    y_pred_norm = (y_pred - gene_mean) / gene_std

    # -----------------------------
    # 2. Variance-aware weights
    # -----------------------------
    gene_weights = 1.0 / (gene_std ** 2)
    gene_weights = gene_weights / gene_weights.mean()   # normalize

    # -----------------------------
    # 3. Heteroscedastic MSE
    # -----------------------------
    mse = (y_pred_norm - y_true_norm) ** 2
    weighted_mse = gene_weights * mse

    hetero_loss = 0.5 * (
        torch.exp(-log_var) * weighted_mse + log_var
    )
    mse_loss = hetero_loss.mean()

    # -----------------------------
    # 4. Spearman loss (BATCH-WISE)
    # -----------------------------
    # IMPORTANT: pass full (B,G) tensor
    spearman_loss = 1.0 - spearmanrr(
        y_pred_norm,
        y_true_norm,
        regularization="l2",
        regularization_strength=1.0
    )

    # -----------------------------
    # 5. Total loss
    # -----------------------------
    total_loss = mse_loss + lambda_spearman * spearman_loss

    return total_loss