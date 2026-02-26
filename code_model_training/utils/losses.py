import warnings
import h5py
import csv
import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import ConcatDataset
from scipy.stats import pearsonr, ConstantInputWarning
from datetime import datetime
import logging
import argparse
import json
import pandas as pd
from scipy import sparse
import torchsort
from scipy.spatial.distance import cdist
from compute_metrics import spearmanrr

torch.manual_seed(42)


def custom_loss_l1_l2_pred_cosine(y_true, y_pred, neighbors_gene_expression, lambda_reg=0.1):
    mse = MSELoss()(y_pred, y_true)
    
    # Ensure neighbors have the correct shape: (batch_size, k_neighbors, output_size)
    batch_size = y_pred.size(0)

    # Flatten the neighbors for cosine similarity calculation
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    
    l1_loss = torch.mean(torch.abs(y_pred))  # L1 regularization to encourage sparsity
    l2_loss = torch.mean(torch.pow(y_pred, exponent=2))  #L2 regularized to encourage elsatic net

    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

    return mse + lambda_reg * l1_loss + lambda_reg * l2_loss + lambda_reg * mean_cosine_dist


def custom_loss_l1_pred_cosine(y_true, y_pred, neighbors_gene_expression, lambda_reg=0.1, lambda_l1=0.2):
    mse = MSELoss()(y_pred, y_true)
    
    # Ensure neighbors have the correct shape: (batch_size, k_neighbors, output_size)
    batch_size = y_pred.size(0)

    # Flatten the neighbors for cosine similarity calculation
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    
    l1_loss = torch.mean(torch.abs(y_pred))  # L1 regularization to encourage sparsity

    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

    return mse + lambda_l1 * l1_loss + lambda_reg * mean_cosine_dist

def custom_loss_pred_cosine(y_true, y_pred, neighbors_gene_expression, lambda_reg=0.1):
    mse = MSELoss()(y_pred, y_true)
    
    # Ensure neighbors have the correct shape: (batch_size, k_neighbors, output_size)
    batch_size = y_pred.size(0)

    # Flatten the neighbors for cosine similarity calculation
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    
    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

    return mse + lambda_reg * mean_cosine_dist

def custom_loss_spearman(y_true, y_pred, neighbors_gene_expression, model, lambda_l1=0.1, lambda_l2=0.1, lambda_reg=0.1):
    mse = MSELoss()(y_pred, y_true)

    # Ensure neighbors have the correct shape: (batch_size, k_neighbors, output_size)
    batch_size = y_pred.size(0)

    # Flatten the neighbors for similarity calculation
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))

    # Calculate L1 and L2 regularization terms for predictions
    l1_loss_pred = torch.mean(torch.abs(y_pred))  # L1 regularization to encourage sparsity
    # l2_loss_pred = torch.mean(torch.pow(y_pred, 2))  # L2 regularization to encourage smoothness

    # Combine L1 and L2 with their respective lambda values
    l1_loss = lambda_l1 * l1_loss_pred
    l2_loss = lambda_l2 * l2_loss_weights

    # Calculate Spearman rank correlation instead of cosine similarity
    spearman_corr = torch.stack([
        spearmanrr(y_pred[i].unsqueeze(0), neighbors_flattened[i], regularization_strength=1.0)
        for i in range(batch_size)
    ])
    mean_spearman_corr = 1 - torch.mean(spearman_corr)  # Mean Spearman distance


    # Final loss
    return mse + l1_loss + l2_loss + lambda_reg * mean_spearman_corr


def custom_loss_l1_pred_l2_weights_cosine(y_true, y_pred, neighbors_gene_expression, model, lambda_l1=0.1, lambda_l2=0.1, lambda_reg=0.1):
    mse = MSELoss()(y_pred, y_true)
    
    # Ensure neighbors have the correct shape: (batch_size, k_neighbors, output_size)
    batch_size = y_pred.size(0)

    # Flatten the neighbors for cosine similarity calculation
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    
    l1_loss = lambda_l1 * torch.mean(torch.abs(y_pred))  # L1 regularization to encourage sparsity
    l2_loss = lambda_l2 * sum(
        torch.sum(param ** 2) 
        for name, param in model.named_parameters() 
        if param.requires_grad and "bias" not in name
    )

    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

    return mse + l1_loss + l2_loss + lambda_reg * mean_cosine_dist

def custom_loss_l1_spearman(y_true, y_pred,  lambda_l1=0.1,  lambda_reg=0.1):
    mse = MSELoss()(y_pred, y_true)
    # Ensure neighbors have the correct shape: (batch_size, k_neighbors, output_size)
    batch_size = y_pred.size(0)
    # Flatten the neighbors for cosine similarity calculation
    # neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    
    l1_loss = torch.mean(torch.abs(y_pred))  # L1 regularization to encourage sparsity
    spearman_loss = 1-spearmanrr(y_pred, y_true, regularization_strength=1.0)


    return mse + lambda_l1 * l1_loss + spearman_loss


def custom_loss_l1_spearman_cosine(y_true, y_pred, neighbors_gene_expression, lambda_l1=0.1,  lambda_reg=0.1):
    mse = MSELoss()(y_pred, y_true)
    # Ensure neighbors have the correct shape: (batch_size, k_neighbors, output_size)
    batch_size = y_pred.size(0)
    # Flatten the neighbors for cosine similarity calculation
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))

    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance
    
    l1_loss = torch.mean(torch.abs(y_pred))  # L1 regularization to encourage sparsity
    spearman_loss = 1-spearmanrr(y_pred, y_true, regularization_strength=1.0)


    return mse + lambda_l1 * l1_loss + lambda_l1 * spearman_loss + lambda_l1 * mean_cosine_dist



def custom_loss_weighted_non_zeros(y_true, y_pred, neighbors_gene_expression, lambda_reg=0.1, lambda_zero=5):
    """
    Custom loss function to predict ground truth zeros as zeros and non-zeros accurately.
    Combines MSE, cosine similarity regularization, and zero-penalty regularization.

    Parameters:
    -----------
    y_true : torch.Tensor
        Ground truth tensor of shape (batch_size, output_size).
    y_pred : torch.Tensor
        Predicted tensor of shape (batch_size, output_size).
    neighbors_gene_expression : torch.Tensor
        Tensor of neighboring gene expressions (batch_size, k_neighbors, output_size).
    lambda_reg : float
        Weight for cosine similarity regularization.
    lambda_zero : float
        Weight for zero-penalty regularization.
    """
    # Mean Squared Error (MSE) Loss
    mse = torch.nn.functional.mse_loss(y_pred, y_true)

    # Zero-Penalty Regularization: Penalize non-zero predictions for zero ground truth
    zero_mask = (y_true == 0).float()  # Mask for ground truth zeros
    zero_penalty = torch.mean((y_pred * zero_mask)**2)  # Penalize non-zero predictions for zero ground truth

    # Weighted MSE for Non-Zero Targets
    non_zero_mask = (y_true != 0).float()  # Mask for ground truth non-zeros
    weighted_mse = torch.mean(non_zero_mask * (y_pred - y_true)**2)

    # Cosine Similarity Regularization
    batch_size = y_pred.size(0)
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

    
    return weighted_mse + lambda_zero * zero_penalty + lambda_reg * mean_cosine_dist

def custom_loss_weighted_penalty_non_zeros(y_true, y_pred, neighbors_gene_expression, lambda_reg=0.1, lambda_zero=1.0, lambda_non_zero=1.0):
    """
    Custom loss function to predict ground truth zeros as zeros and non-zeros accurately.

    Parameters:
    -----------
    y_true : torch.Tensor
        Ground truth tensor of shape (batch_size, output_size).
    y_pred : torch.Tensor
        Predicted tensor of shape (batch_size, output_size).
    neighbors_gene_expression : torch.Tensor
        Tensor of neighboring gene expressions (batch_size, k_neighbors, output_size).
    lambda_reg : float
        Weight for cosine similarity regularization.
    lambda_zero : float
        Weight for zero-penalty regularization.
    lambda_non_zero : float
        Weight for non-zero penalty regularization.
    """
    # Zero-Penalty Regularization: Penalize non-zero predictions for zero ground truth
    zero_mask = (y_true == 0).float()  # Mask for ground truth zeros
    zero_penalty = torch.mean((y_pred * zero_mask) ** 2)  # Penalize non-zero predictions for zero ground truth

    # Non-Zero Penalty Regularization: Penalize deviation from true values for non-zero ground truth
    non_zero_mask = (y_true != 0).float()  # Mask for ground truth non-zeros
    non_zero_penalty = torch.mean((y_pred - y_true) ** 2 * non_zero_mask)

    # Cosine Similarity Regularization
    batch_size = y_pred.size(0)
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

    # Final Loss Combining Zero, Non-Zero, and Cosine Regularization
    loss = (
        lambda_zero * zero_penalty
        + lambda_non_zero * non_zero_penalty
        + lambda_reg * mean_cosine_dist
    )
    return loss

def custom_loss_pred_zeros(y_true, y_pred, neighbors_gene_expression, lambda_reg=0.1, lambda_zero=2.0, lambda_bce=0.5):
    """
    Custom loss function to handle imbalanced zeros and non-zeros in predictions.

    Parameters:
    -----------
    y_true : torch.Tensor
        Ground truth tensor of shape (batch_size, output_size).
    y_pred : torch.Tensor
        Predicted tensor of shape (batch_size, output_size).
    neighbors_gene_expression : torch.Tensor
        Tensor of neighboring gene expressions (batch_size, k_neighbors, output_size).
    lambda_reg : float
        Weight for cosine similarity regularization.
    lambda_zero : float
        Weight for zero-penalty regularization.
    lambda_bce : float
        Weight for BCE loss for zero prediction accuracy.
    """
    # Mean Squared Error (MSE) Loss for all predictions
    mse = torch.nn.functional.mse_loss(y_pred, y_true)

    # Zero-Penalty Regularization: Penalize non-zero predictions for zero ground truth
    zero_mask = (y_true == 0).float()
    zero_penalty = torch.mean((y_pred * zero_mask) ** 2)

    # Weighted MSE for Non-Zero Targets
    non_zero_mask = (y_true != 0).float()
    weighted_mse = torch.mean(non_zero_mask * (y_pred - y_true) ** 2)

    # Cosine Similarity Regularization
    batch_size = y_pred.size(0)
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)

    # Binary Cross-Entropy Loss for Zero Prediction
    zero_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, zero_mask)

    # Total Loss
    total_loss = (
        weighted_mse
        + lambda_zero * zero_penalty
        + lambda_bce * zero_bce_loss
        + lambda_reg * mean_cosine_dist
    )

    return total_loss


def custom_loss_weighted_non_zeros_l1(y_true, y_pred, neighbors_gene_expression, lambda_reg=0.1, lambda_zero=2,lambda_non_zero=0.1):
    """
    Custom loss function to predict ground truth zeros as zeros and non-zeros accurately.
    Combines MSE, cosine similarity regularization, and zero-penalty regularization.

    Parameters:
    -----------
    y_true : torch.Tensor
        Ground truth tensor of shape (batch_size, output_size).
    y_pred : torch.Tensor
        Predicted tensor of shape (batch_size, output_size).
    neighbors_gene_expression : torch.Tensor
        Tensor of neighboring gene expressions (batch_size, k_neighbors, output_size).
    lambda_reg : float
        Weight for cosine similarity regularization.
    lambda_zero : float
        Weight for zero-penalty regularization.
    """
    # Mean Squared Error (MSE) Loss
    mse = torch.nn.functional.mse_loss(y_pred, y_true)

    # Zero-Penalty Regularization: Penalize non-zero predictions for zero ground truth
    zero_mask = (y_true == 0).float()  # Mask for ground truth zeros

    zero_penalty = torch.mean(zero_mask* torch.abs(y_pred))  # Penalize non-zero predictions for zero ground truth

    # Weighted MSE for Non-Zero Targets
    non_zero_mask = (y_true != 0).float()  # Mask for ground truth non-zeros
    non_zero_penalty = torch.mean(non_zero_mask * torch.abs(y_pred))

    # Cosine Similarity Regularization
    batch_size = y_pred.size(0)
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

    
    return mse + non_zero_penalty * lambda_non_zero + lambda_zero * zero_penalty + lambda_reg * mean_cosine_dist

def custom_loss_weighted_penalty(y_true, y_pred, neighbors_gene_expression, lambda_reg=0.1, lambda_zero=5,lambda_non_zero=.6):
    """
    Custom loss function to predict ground truth zeros as zeros and non-zeros accurately.
    Combines MSE, cosine similarity regularization, and zero-penalty regularization.

    Parameters:
    -----------
    y_true : torch.Tensor
        Ground truth tensor of shape (batch_size, output_size).
    y_pred : torch.Tensor
        Predicted tensor of shape (batch_size, output_size).
    neighbors_gene_expression : torch.Tensor
        Tensor of neighboring gene expressions (batch_size, k_neighbors, output_size).
    lambda_reg : float
        Weight for cosine similarity regularization.
    lambda_zero : float
        Weight for zero-penalty regularization.
    """
    # Mean Squared Error (MSE) Loss
    #mse = torch.nn.functional.mse_loss(y_pred, y_true)

    # Zero-Penalty Regularization: Penalize non-zero predictions for zero ground truth
    zero_mask = (y_true == 0).float()  # Mask for ground truth zeros
    zero_penalty = torch.mean((y_pred * zero_mask)**2)  # Penalize non-zero predictions for zero ground truth

    # Weighted MSE for Non-Zero Targets
    non_zero_mask = (y_true != 0).float()  # Mask for ground truth non-zeros
    non_zero_penalty = torch.mean(non_zero_mask * (y_pred - y_true)**2)

    # Cosine Similarity Regularization
    batch_size = y_pred.size(0)
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

    
    return non_zero_penalty * lambda_non_zero + lambda_zero * zero_penalty + lambda_reg * mean_cosine_dist


# def custom_loss_dynamic_lambda_v5(
#     y_true, y_pred, neighbors_gene_expression, 
#     base_lambda_mse=4.0, base_lambda_zero=5.0, base_lambda_cosine=1.0
# ):
#     """
#     Custom loss function with dynamic adjustment of lambda values (no max caps).

#     Parameters:
#         y_true : torch.Tensor
#             Ground truth tensor of shape (batch_size, output_size).
#         y_pred : torch.Tensor
#             Predicted tensor of shape (batch_size, output_size).
#         neighbors_gene_expression : torch.Tensor
#             Tensor of neighboring gene expressions (batch_size, k_neighbors, output_size).
#         base_lambda_mse : float
#             Base weight for MSE loss term.
#         base_lambda_zero : float
#             Base weight for the zero penalty.
#         base_lambda_cosine : float
#             Base weight for cosine similarity regularization.
#     """
#     # Masks for zeros and non-zeros
#     zero_mask = (y_true == 0).float()  # Mask for zero ground truth
#     non_zero_mask = (y_true != 0).float()  # Mask for non-zero ground truth

#     # 1. Mean Squared Error (MSE) Loss for all targets
#     mse_loss = torch.nn.functional.mse_loss(y_pred, y_true)

#     # 2. Zero Penalty (L1 regularization on zero targets only)
#     zero_penalty = torch.mean(zero_mask * torch.abs(y_pred))  # L1 for zero targets

#     # 3. Cosine Similarity Regularization
#     batch_size = y_pred.size(0)
#     neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
#     cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
#     mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

#     # 4. Dynamic Lambda Adjustments
#     dynamic_lambda_mse = base_lambda_mse / (mse_loss + 1e-8)
#     dynamic_lambda_zero = base_lambda_zero / (zero_penalty + 1e-8)
#     dynamic_lambda_cosine = base_lambda_cosine / (mean_cosine_dist + 1e-8)
#     # 5. Normalize Lambda Values
#     total_lambda = dynamic_lambda_mse + dynamic_lambda_zero + dynamic_lambda_cosine
#     dynamic_lambda_mse /= total_lambda
#     dynamic_lambda_zero /= total_lambda
#     dynamic_lambda_cosine /= total_lambda
#     print(f"dynamic_lambda_mse: {dynamic_lambda_mse}, dynamic_lambda_zero: {dynamic_lambda_zero}, dynamic_lambda_cosine: {dynamic_lambda_cosine}")
#     print(f"zero Penalty: {zero_penalty}, mean_cosine_dist: {mean_cosine_dist}, mse_loss: {mse_loss}")

#     # 6. Combined Loss
#     loss = (
#         dynamic_lambda_mse * mse_loss  # MSE with dynamic lambda
#         + dynamic_lambda_zero * zero_penalty  # Zero penalty with dynamic lambda
#         + dynamic_lambda_cosine * mean_cosine_dist  # Cosine similarity regularization with dynamic lambda
#     )

#     return loss


def custom_loss_dynamic_lambda_v5(
    y_true, y_pred, neighbors_gene_expression, 
    base_lambda_mse=0.3, base_lambda_zero=0.6, base_lambda_cosine=0.1
):
    """
    Custom loss function with dynamic adjustment of lambda values (no max caps).

    Parameters:
        y_true : torch.Tensor
            Ground truth tensor of shape (batch_size, output_size).
        y_pred : torch.Tensor
            Predicted tensor of shape (batch_size, output_size).
        neighbors_gene_expression : torch.Tensor
            Tensor of neighboring gene expressions (batch_size, k_neighbors, output_size).
        base_lambda_mse : float
            Base weight for MSE loss term.
        base_lambda_zero : float
            Base weight for the zero penalty.
        base_lambda_cosine : float
            Base weight for cosine similarity regularization.
    """
    # Masks for zeros and non-zeros
    zero_mask = (y_true == 0).float()  # Mask for zero ground truth
    non_zero_mask = (y_true != 0).float()  # Mask for non-zero ground truth

    # 1. Mean Squared Error (MSE) Loss for all targets
    mse_loss = torch.nn.functional.mse_loss(y_pred, y_true)

    # 2. Zero Penalty (L1 regularization on zero targets only)
    zero_penalty = torch.mean(zero_mask * torch.abs(y_pred))  # L1 for zero targets

    # 3. Cosine Similarity Regularization
    batch_size = y_pred.size(0)
    neighbors_flattened = neighbors_gene_expression.view(batch_size, -1, neighbors_gene_expression.size(-1))
    cosine_sim = torch.nn.functional.cosine_similarity(y_pred.unsqueeze(1), neighbors_flattened, dim=-1)
    mean_cosine_dist = torch.mean(1 - cosine_sim)  # Mean cosine distance

    # 4. Dynamic Lambda Adjustments
    # dynamic_lambda_mse = base_lambda_mse / (mse_loss + 1e-8)
    # dynamic_lambda_zero = base_lambda_zero / (zero_penalty + 1e-8)
    # dynamic_lambda_cosine = base_lambda_cosine / (mean_cosine_dist + 1e-8)
    # # 5. Normalize Lambda Values
    # total_lambda = dynamic_lambda_mse + dynamic_lambda_zero + dynamic_lambda_cosine
    # dynamic_lambda_mse /= total_lambda
    # dynamic_lambda_zero /= total_lambda
    # dynamic_lambda_cosine /= total_lambda
    # print(f"dynamic_lambda_mse: {dynamic_lambda_mse}, dynamic_lambda_zero: {dynamic_lambda_zero}, dynamic_lambda_cosine: {dynamic_lambda_cosine}")
    print(f"zero Penalty: {zero_penalty}, mean_cosine_dist: {mean_cosine_dist}, mse_loss: {mse_loss}")

    # 6. Combined Loss
    loss = (
        base_lambda_mse * mse_loss  # MSE with dynamic lambda
        + base_lambda_zero * zero_penalty  # Zero penalty with dynamic lambda
        + base_lambda_cosine * mean_cosine_dist  # Cosine similarity regularization with dynamic lambda
    )

    return loss


# def custom_spearmanr_old(pred_vals, target_vals, alpha=2.0, **kw):
#     """
#     Computes a weighted Spearman’s rank correlation.
#     - Uses `torchsort.soft_rank()` for differentiability.
#     - Applies standard deviation normalization instead of L2 norm.
#     - Weights non-zero target values more heavily to handle class imbalance.

#     Arguments:
#         pred: (batch_size, feature_dim) - Model predictions
#         target: (batch_size, feature_dim) - Ground truth labels
#         alpha: (float) - Weight factor for non-zero values (default: 2.0)
#         **kw: Parameters for `torchsort.soft_rank()`

#     Returns:
#         Weighted Spearman correlation (1 - correlation, for loss usage)
#     """
#     # Soft rank transformation
#     pred = torchsort.soft_rank(pred_vals, **kw)
#     target = torchsort.soft_rank(target_vals, **kw)

#     # Mean-centering
#     pred_mean = pred.mean(dim=1, keepdim=True)
#     target_mean = target.mean(dim=1, keepdim=True)

#     pred_centered = pred - pred_mean
#     target_centered = target - target_mean

#     # Normalize using standard deviation
#     pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-8  # Prevent division by zero
#     target_std = target_centered.std(dim=1, keepdim=True) + 1e-8

#     pred_normalized = pred_centered / pred_std
#     target_normalized = target_centered / target_std

#     # Compute Spearman correlation
#     spearman_corr = (pred_normalized * target_normalized).sum(dim=0) / pred.shape[0]

#     # Apply non-zero weighting
#     non_zero_mask = (target_vals != 0).float()  # Mask where target is non-zero
#     weights = 1 + (alpha - 1) * non_zero_mask  # Increase weight for non-zero values
#     # print(f"weights::::::::::::::::::::::: {weights.shape} AND spearman_corr:::::::::::::::::::::::::::::::::::::{spearman_corr.shape} ")
#     weighted_corr = (weights * spearman_corr).sum() / weights.sum()  # Weighted correlation

#     return 1 - weighted_corr  # Convert correlation to loss

def custom_spearmanr_old(pred, target, alpha=2.0, **kw):
    """
    Computes a weighted Spearman’s rank correlation.
    - Uses `torchsort.soft_rank()` for differentiability.
    - Applies standard deviation normalization instead of L2 norm.
    - Weights non-zero target values more heavily to handle class imbalance.

    Arguments:
        pred: (batch_size, feature_dim) - Model predictions
        target: (batch_size, feature_dim) - Ground truth labels
        alpha: (float) - Weight factor for non-zero values (default: 2.0)
        **kw: Parameters for `torchsort.soft_rank()`

    Returns:
        Weighted Spearman correlation (1 - correlation, for loss usage)
    """
    # Soft rank transformation
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)

    # Mean-centering
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # Normalize using standard deviation
    pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-8  # Prevent division by zero
    target_std = target_centered.std(dim=1, keepdim=True) + 1e-8

    pred_normalized = pred_centered / pred_std
    target_normalized = target_centered / target_std

    # Compute Spearman correlation
    spearman_corr = (pred_normalized * target_normalized).sum(dim=0) / pred.shape[0]

    # Apply non-zero weighting
    non_zero_mask = (target != 0).float()  # Mask where target is non-zero
    weights = 1 + (alpha - 1) * non_zero_mask  # Increase weight for non-zero values
    # print(f"weights::::::::::::::::::::::: {weights.shape} AND spearman_corr:::::::::::::::::::::::::::::::::::::{spearman_corr.shape} ")
    weighted_corr = (weights * spearman_corr).sum() / weights.sum()  # Weighted correlation

    return 1 - weighted_corr  # Convert correlation to loss

# def custom_spearmanr(pred_vals, target_vals, alpha=2.0, **kw):
#     """
#     Computes a weighted Spearman’s rank correlation.
#     - Uses `torchsort.soft_rank()` for differentiability.
#     - Applies standard deviation normalization instead of L2 norm.
#     - Weights non-zero target values more heavily to handle class imbalance.

#     Arguments:
#         pred: (batch_size, feature_dim) - Model predictions
#         target: (batch_size, feature_dim) - Ground truth labels
#         alpha: (float) - Weight factor for non-zero values (default: 2.0)
#         **kw: Parameters for `torchsort.soft_rank()`

#     Returns:
#         Weighted Spearman correlation (1 - correlation, for loss usage)
#     """
#     # Soft rank transformation
#     pred = torchsort.soft_rank(pred_vals, **kw) #batch size, gene expression vector size for example, 1024, 460
#     target = torchsort.soft_rank(target_vals, **kw)

#     # Mean-centering
#     pred_mean = pred.mean(dim=1, keepdim=True)
#     target_mean = target.mean(dim=1, keepdim=True)

#     pred_centered = pred - pred_mean
#     target_centered = target - target_mean

#     # Normalize using standard deviation
#     pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-8  # Prevent division by zero
#     target_std = target_centered.std(dim=1, keepdim=True) + 1e-8

#     # pred_normalized = pred_centered 
#     # target_normalized = target_centered 

#     # Compute Spearman correlation

#     # print('pred_centered', pred_centered.shape, 'target_centered', target_centered.shape, 'pred_std', pred_std.shape, 'target_std', target_std.shape)
#     spearman_corr = torch.div((pred_centered * target_centered).sum(dim=1), (pred_std*target_std).squeeze())

#     # print('(pred_centered * target_centered).sum(dim=1)', (pred_centered * target_centered).sum(dim=1).shape)
#     # Apply non-zero weighting
#     non_zero_mask = (target_vals != 0).float()  # Mask where target is non-zero
#     # print('non_zero_mask zeros', torch.sum(non_zero_mask))
#     weights = 1 + (alpha - 1) * non_zero_mask  # Increase weight for non-zero values
#     # print(f"weights::::::::::::::::::::::: {weights.shape} AND spearman_corr:::::::::::::::::::::::::::::::::::::{spearman_corr.shape} ")
#     weighted_corr = (weights.T * spearman_corr).sum() / weights.sum()  # Weighted correlation

#     return 1 - weighted_corr  # Convert correlation to loss


# def custom_spearmanr_with_mse(pred, target, alpha=2.0, **kw):
#     """
#     Computes a weighted Spearman’s rank correlation.
#     - Uses `torchsort.soft_rank()` for differentiability.
#     - Applies standard deviation normalization instead of L2 norm.
#     - Weights non-zero target values more heavily to handle class imbalance.

#     Arguments:
#         pred: (batch_size, feature_dim) - Model predictions
#         target: (batch_size, feature_dim) - Ground truth labels
#         alpha: (float) - Weight factor for non-zero values (default: 2.0)
#         **kw: Parameters for `torchsort.soft_rank()`

#     Returns:
#         Weighted Spearman correlation (1 - correlation, for loss usage)
#     """
#     # Soft rank transformation
#     mse = MSELoss()(pred, target)
#     pred = torchsort.soft_rank(pred, **kw)
#     target = torchsort.soft_rank(target, **kw)

#     # Mean-centering
#     pred_mean = pred.mean(dim=1, keepdim=True)
#     target_mean = target.mean(dim=1, keepdim=True)

#     pred_centered = pred - pred_mean
#     target_centered = target - target_mean

#     # Normalize using standard deviation
#     pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-8  # Prevent division by zero
#     target_std = target_centered.std(dim=1, keepdim=True) + 1e-8

#     pred_normalized = pred_centered / pred_std
#     target_normalized = target_centered / target_std

#     # Compute Spearman correlation
#     spearman_corr = (pred_normalized * target_normalized).sum(dim=0) / pred.shape[0]

#     # Apply non-zero weighting
#     non_zero_mask = (target != 0).float()  # Mask where target is non-zero
#     weights = 1 + (alpha - 1) * non_zero_mask  # Increase weight for non-zero values
#     print(f"weights::::::::::::::::::::::: {weights.shape} AND spearman_corr:::::::::::::::::::::::::::::::::::::{spearman_corr.shape} ")
#     weighted_corr = (weights * spearman_corr).sum() / weights.sum()  # Weighted correlation

#     return mse + 1 - weighted_corr  # Convert correlation to loss

def custom_spearmanr(pred, target, alpha=2.0, **kw):
    """
    Computes a weighted Spearman’s rank correlation.
    - Uses `torchsort.soft_rank()` for differentiability.
    - Applies standard deviation normalization instead of L2 norm.
    - Weights non-zero target values more heavily to handle class imbalance.

    Arguments:
        pred: (batch_size, feature_dim) - Model predictions
        target: (batch_size, feature_dim) - Ground truth labels
        alpha: (float) - Weight factor for non-zero values (default: 2.0)
        **kw: Parameters for `torchsort.soft_rank()`

    Returns:
        Weighted Spearman correlation (1 - correlation, for loss usage)
    """
    # Soft rank transformation
    # mse = MSELoss()(pred, target)
    pred_r = torchsort.soft_rank(pred, **kw)
    target_r = torchsort.soft_rank(target, **kw)

    # Mean-centering
    pred_mean =  pred_r.mean(dim=0, keepdim=True)
    target_mean = target_r.mean(dim=0, keepdim=True)

    pred_centered = pred_r - pred_mean
    target_centered = target_r - target_mean

    # Normalize using standard deviation
    pred_std = pred_centered.std(dim=0, keepdim=True) + 1e-8  # Prevent division by zero
    target_std = target_centered.std(dim=0, keepdim=True) + 1e-8

    pred_normalized = pred_centered / pred_std
    target_normalized = target_centered / target_std

    # Compute Spearman correlation
    spearman_corr = (pred_normalized * target_normalized).sum(dim=0) / pred.shape[0]

    # Apply non-zero weighting
    non_zero_mask = (target != 0).float()  # Mask where target is non-zero
    weights = 1 + (alpha - 1) * (non_zero_mask.sum(dim=0)/target.shape[0])  # Increase weight for non-zero values
    print(f"weights::::::::::::::::::::::: {weights.shape} AND spearman_corr:::::::::::::::::::::::::::::::::::::{spearman_corr.shape} ")
    weighted_corr = (weights * spearman_corr).sum() / weights.sum()  # Weighted correlation

    return 1 - weighted_corr  # Convert correlation to loss

def custom_spearmanr_with_mse(pred, target, alpha=2.0, **kw):
    """
    Computes a weighted Spearman’s rank correlation.
    - Uses `torchsort.soft_rank()` for differentiability.
    - Applies standard deviation normalization instead of L2 norm.
    - Weights non-zero target values more heavily to handle class imbalance.

    Arguments:
        pred: (batch_size, feature_dim) - Model predictions
        target: (batch_size, feature_dim) - Ground truth labels
        alpha: (float) - Weight factor for non-zero values (default: 2.0)
        **kw: Parameters for `torchsort.soft_rank()`

    Returns:
        Weighted Spearman correlation (1 - correlation, for loss usage)
    """
    # Soft rank transformation
    mse = MSELoss()(pred, target)
    pred_r = torchsort.soft_rank(pred, **kw)
    target_r = torchsort.soft_rank(target, **kw)

    # Mean-centering
    pred_mean =  pred_r.mean(dim=0, keepdim=True)
    target_mean = target_r.mean(dim=0, keepdim=True)

    pred_centered = pred_r - pred_mean
    target_centered = target_r - target_mean

    # Normalize using standard deviation
    pred_std = pred_centered.std(dim=0, keepdim=True) + 1e-8  # Prevent division by zero
    target_std = target_centered.std(dim=0, keepdim=True) + 1e-8

    pred_normalized = pred_centered / pred_std
    target_normalized = target_centered / target_std

    # Compute Spearman correlation
    spearman_corr = (pred_normalized * target_normalized).sum(dim=0) / pred.shape[0]

    # Apply non-zero weighting
    non_zero_mask = (target != 0).float()  # Mask where target is non-zero
    weights = 1 + (alpha - 1) * (non_zero_mask.sum(dim=0)/target.shape[0])  # Increase weight for non-zero values
    print(f"weights::::::::::::::::::::::: {weights.shape} AND spearman_corr:::::::::::::::::::::::::::::::::::::{spearman_corr.shape} ")
    weighted_corr = (weights * spearman_corr).sum() / weights.sum()  # Weighted correlation

    return mse + (1 - weighted_corr)  # Convert correlation to loss



def custom_spearmanr_with_mse_l1(pred, target, alpha=2.0, lambda_l1=0.1, **kw):
    """
    Computes a weighted Spearman’s rank correlation.
    - Uses `torchsort.soft_rank()` for differentiability.
    - Applies standard deviation normalization instead of L2 norm.
    - Weights non-zero target values more heavily to handle class imbalance.

    Arguments:
        pred: (batch_size, feature_dim) - Model predictions
        target: (batch_size, feature_dim) - Ground truth labels
        alpha: (float) - Weight factor for non-zero values (default: 2.0)
        **kw: Parameters for `torchsort.soft_rank()`

    Returns:
        Weighted Spearman correlation (1 - correlation, for loss usage)
    """
    # Soft rank transformation
    mse = MSELoss()(pred, target)
    pred_r = torchsort.soft_rank(pred, **kw)
    target_r = torchsort.soft_rank(target, **kw)

    # Mean-centering
    pred_mean =  pred_r.mean(dim=0, keepdim=True)
    target_mean = target_r.mean(dim=0, keepdim=True)

    pred_centered = pred_r - pred_mean
    target_centered = target_r - target_mean

    # Normalize using standard deviation
    pred_std = pred_centered.std(dim=0, keepdim=True) + 1e-8  # Prevent division by zero
    target_std = target_centered.std(dim=0, keepdim=True) + 1e-8

    pred_normalized = pred_centered / pred_std
    target_normalized = target_centered / target_std

    # Compute Spearman correlation
    spearman_corr = (pred_normalized * target_normalized).sum(dim=0) / pred.shape[0]

    l1_loss = torch.mean(torch.abs(pred))  # L1 regularization to encourage sparsity

    # Apply non-zero weighting
    non_zero_mask = (target != 0).float()  # Mask where target is non-zero
    weights = 1 + (alpha - 1) * (non_zero_mask.sum(dim=0)/target.shape[0])  # Increase weight for non-zero values
    print(f"weights::::::::::::::::::::::: {weights.shape} AND spearman_corr:::::::::::::::::::::::::::::::::::::{spearman_corr.shape} ")
    weighted_corr = (weights * spearman_corr).sum() / weights.sum()  # Weighted correlation

    return mse + (1 - weighted_corr) + (lambda_l1 * l1_loss) # Convert correlation to loss



def spearmanrr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return 1 - (pred * target).sum()


def spearmanrr_feature_wise(pred, target, **kw):
    pred_rank = torchsort.soft_rank(pred, **kw)
    target_rank = torchsort.soft_rank(target, **kw)

    # Normalize each feature separately
    pred_rank = pred_rank - pred_rank.mean(dim=0, keepdim=True)
    pred_rank = pred_rank / (pred_rank.norm(dim=0, keepdim=True) + 1e-6)  # Avoid division by zero

    target_rank = target_rank - target_rank.mean(dim=0, keepdim=True)
    target_rank = target_rank / (target_rank.norm(dim=0, keepdim=True) + 1e-6)

    # Compute Spearman correlation per feature (reduce across batch dimension)
    spearman_corr = (pred_rank * target_rank).sum(dim=0)  # (460,)

    # Return the mean correlation across all features
    return 1-spearman_corr.mean()  # Scalar loss value



def custom_spearmanr_non_zero_mask(pred, target, alpha=2.0, **kw):
    """
    Computes a weighted Spearman’s rank correlation.
    - Uses `torchsort.soft_rank()` for differentiability.
    - Applies standard deviation normalization instead of L2 norm.
    - Weights non-zero target values more heavily to handle class imbalance.

    Arguments:
        pred: (batch_size, feature_dim) - Model predictions
        target: (batch_size, feature_dim) - Ground truth labels
        alpha: (float) - Weight factor for non-zero values (default: 2.0)
        **kw: Parameters for `torchsort.soft_rank()`

    Returns:
        Weighted Spearman correlation (1 - correlation, for loss usage)
    """
    non_zero_mask = (target != 0).float()  # Mask where target is non-zero
    # Soft rank transformation
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)

    # Mean-centering
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # Normalize using standard deviation
    pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-8  # Prevent division by zero
    target_std = target_centered.std(dim=1, keepdim=True) + 1e-8

    pred_normalized = pred_centered / pred_std
    target_normalized = target_centered / target_std

    # Compute Spearman correlation
    spearman_corr = (pred_normalized * target_normalized).sum(dim=0) / pred.shape[0]

    # Apply non-zero weighting
    weights = 1 + (alpha - 1) * non_zero_mask  # Increase weight for non-zero values
    # print(f"weights::::::::::::::::::::::: {weights.shape} AND spearman_corr:::::::::::::::::::::::::::::::::::::{spearman_corr.shape} ")
    weighted_corr = (weights * spearman_corr).sum() / weights.sum()  # Weighted correlation

    return 1 - weighted_corr  # Convert correlation to loss



def custom_spearmanr_old2(pred, target, alpha=2.0, **kw):
    """
    Computes a weighted Spearman’s rank correlation.
    - Uses `torchsort.soft_rank()` for differentiability.
    - Applies standard deviation normalization instead of L2 norm.
    - Weights non-zero target values more heavily to handle class imbalance.

    Arguments:
        pred: (batch_size, feature_dim) - Model predictions
        target: (batch_size, feature_dim) - Ground truth labels
        alpha: (float) - Weight factor for non-zero values (default: 2.0)
        **kw: Parameters for `torchsort.soft_rank()`

    Returns:
        Weighted Spearman correlation (1 - correlation, for loss usage)
    """
    # Soft rank transformation
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)

    # Mean-centering
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # Normalize using standard deviation
    pred_std = pred_centered.std(dim=1, keepdim=True) + 1e-8  # Prevent division by zero
    target_std = target_centered.std(dim=1, keepdim=True) + 1e-8

    pred_normalized = pred_centered / pred_std
    target_normalized = target_centered / target_std

    # Compute Spearman correlation
    spearman_corr = (pred_normalized * target_normalized).sum(dim=0) / pred.shape[0]

    # Apply non-zero weighting
    # non_zero_mask = (target != 0).float()  # Mask where target is non-zero
    # weights = 1 + (alpha - 1) * non_zero_mask  # Increase weight for non-zero values
    # # print(f"weights::::::::::::::::::::::: {weights.shape} AND spearman_corr:::::::::::::::::::::::::::::::::::::{spearman_corr.shape} ")
    # weighted_corr = (weights * spearman_corr).sum() / weights.sum()  # Weighted correlation

    return 1 - spearman_corr.mean()  # Convert correlation to loss

##############################Contrastive_loss###############################################################

def contrastive_loss(image_features, ground_truth_features, temperature=1.0):
    """Compute symmetric contrastive loss between image embeddings and ground truth embeddings."""
    # Normalize image embeddings (always required)
    image_embeddings = F.normalize(image_features, p=2, dim=-1)

    # Normalize ground truth embeddings only if necessary
    ground_truth_embeddings = F.normalize(ground_truth_features, p=2, dim=-1)

    # Compute pairwise cosine similarities
    logits = torch.matmul(image_embeddings, ground_truth_embeddings.T) * torch.exp(torch.tensor(temperature))

    # Ground truth labels (correct pairs along the diagonal)
    labels = torch.arange(logits.shape[0], device=logits.device)

    # Symmetric cross-entropy loss
    loss_i = F.cross_entropy(logits, labels)  # Image to GT loss
    loss_t = F.cross_entropy(logits.T, labels)  # GT to Image loss
    
    return (loss_i + loss_t) / 2  # Final loss

##############################New_set_06_june_25###############################################################


def mse_spearman_loss(pred, target, mse_weight=1.0, spearman_weight=1.0, **torchsort_kwargs):
    # MSE loss
    mse = F.mse_loss(pred, target)

    # Spearman loss
    pred_rank = torchsort.soft_rank(pred, **torchsort_kwargs)
    target_rank = torchsort.soft_rank(target, **torchsort_kwargs)

    pred_rank = pred_rank - pred_rank.mean()
    pred_rank = pred_rank / pred_rank.norm()
    target_rank = target_rank - target_rank.mean()
    target_rank = target_rank / target_rank.norm()

    spearman_corr = (pred_rank * target_rank).sum()
    spearman_loss = 1 - spearman_corr

    # Combined loss
    return mse_weight * mse + spearman_weight * spearman_loss


def wt_mse_spearman_loss(pred, target, mse_weight=0.3, spearman_weight=0.7, **torchsort_kwargs):
    # MSE loss
    mse = F.mse_loss(pred, target)

    # Spearman loss
    pred_rank = torchsort.soft_rank(pred, **torchsort_kwargs)
    target_rank = torchsort.soft_rank(target, **torchsort_kwargs)

    pred_rank = pred_rank - pred_rank.mean()
    pred_rank = pred_rank / pred_rank.norm()
    target_rank = target_rank - target_rank.mean()
    target_rank = target_rank / target_rank.norm()

    spearman_corr = (pred_rank * target_rank).sum()
    spearman_loss = 1 - spearman_corr

    # Combined loss
    return mse_weight * mse + spearman_weight * spearman_loss