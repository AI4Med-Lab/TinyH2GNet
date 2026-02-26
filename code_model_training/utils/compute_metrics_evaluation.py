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
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import ConcatDataset
from scipy.stats import pearsonr, ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)

from scipy.stats import spearmanr as sp
from datetime import datetime
import logging
import argparse
import json
import pandas as pd
from scipy import sparse
import torchsort

def compute_metrics(y_test, preds_all, genes=None):
    """
    Computes metrics L2 errors R2 scores and Pearson correlations for each target/gene.

    :param y_test: Ground truth values (numpy array of shape [n_samples, n_targets]).
    :param preds_all: Predictions for all targets (numpy array of shape [n_samples, n_targets]).
    :param genes: Optional list of gene names corresponding to targets.
    :return: A dictionary containing metrics.
    """

    errors = []
    l1_errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    spearman_corrs = []
    spearman_genes = []
    mape_list = []    # NEW
    rmse_list = []    # NEW


    for i, target in enumerate(range(y_test.shape[1])):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]

        l1_error_value = float(np.mean(np.abs(target_vals - preds)))
        # Compute L2 error
        l2_error = float(np.mean((preds - target_vals) ** 2))
        rmse_value = float(np.sqrt(l2_error)) 
        
        mask = np.abs(target_vals) > 1e-3
        if np.any(mask):
            mape_value = np.mean(np.abs((target_vals[mask] - preds[mask]) / np.abs(target_vals[mask]))) * 100
        else:
            mape_value = 0.0

        # Compute R2 score
        total_variance = np.sum((target_vals - np.mean(target_vals)) ** 2)
        if total_variance == 0:
            r2_score = 0.0
        else:
            r2_score = float(1 - np.sum((target_vals - preds) ** 2) / total_variance)

        if len(target_vals) < 2 or len(preds) < 2:
            pearson_corr = 0.0
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConstantInputWarning)
                try:
                    pearson_corr, _ = pearsonr(target_vals.flatten(), preds.flatten())
                    pearson_corr = pearson_corr if not np.isnan(pearson_corr) else 0.0
                except ConstantInputWarning:
                    pearson_corr = 0.0

        # Compute Spearman correlation
        if len(target_vals) < 2 or len(preds) < 2:
            spearman_corr = 0.0
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConstantInputWarning)
                try:
                    spearman_corr, _ = sp(target_vals.flatten(), preds.flatten())
                    spearman_corr = spearman_corr if not np.isnan(spearman_corr) else 0.0
                except ConstantInputWarning:
                    spearman_corr = 0.0

        errors.append(l2_error)
        l1_errors.append(l1_error_value)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        mape_list.append(mape_value)
        rmse_list.append(rmse_value)

        # Record gene-specific Pearson correlation
        if genes is not None:
            pearson_genes.append({'name': genes[i],'pearson_corr': pearson_corr})
            spearman_genes.append({'name': genes[i], 'spearman_corr': spearman_corr})

    # -----------------------------
    #  Cell-wise metrics
    # -----------------------------

    # -----------------------------
    # Cell-wise / Single-gene handling
    # -----------------------------

    if y_test.shape[1] == 1:
        # TOP-1 case → correlation across cells
        try:
            p = pearsonr(y_test[:, 0], preds_all[:, 0])[0]
            p = 0.0 if np.isnan(p) else p
        except:
            p = 0.0

        try:
            s = sp(y_test[:, 0], preds_all[:, 0])[0]
            s = 0.0 if np.isnan(s) else s
        except:
            s = 0.0

        pearson_cellwise = p
        spearman_cellwise = s

    else:
        # Normal cell-wise case (≥2 genes)
        cell_pearsons = []
        cell_spearmans = []

        for i in range(y_test.shape[0]):
            y_true_cell = y_test[i]
            y_pred_cell = preds_all[i]

            if np.std(y_true_cell) < 1e-6 or np.std(y_pred_cell) < 1e-6:
                continue

            try:
                p = pearsonr(y_true_cell, y_pred_cell)[0]
                if not np.isnan(p):
                    cell_pearsons.append(p)
            except:
                pass

            try:
                s = sp(y_true_cell, y_pred_cell)[0]
                if not np.isnan(s):
                    cell_spearmans.append(s)
            except:
                pass

        pearson_cellwise = np.mean(cell_pearsons) if cell_pearsons else 0.0
        spearman_cellwise = np.mean(cell_spearmans) if cell_spearmans else 0.0



    results = {
        #  NEW — Cell-wise
        "pearson_mean_cellwise": float(f"{np.mean(pearson_cellwise):.4f}"),
        "spearman_mean_cellwise": float(f"{np.mean(spearman_cellwise):.4f}"),

        #  Existing — Gene-wise
        'pearson_mean_genewise': float(f"{np.mean(pearson_corrs): .4f}"),
        'spearman_mean_genewise': float(f"{np.mean(spearman_corrs): .4f}"),
        'l1_error_mean': float(f"{np.mean(l1_errors): .4f}"),
        'l2_errors_mean': float(f"{np.mean(errors):.4f}"),
        'r2_scores_mean': float(f"{np.mean(r2_scores):.4f}"),
        # 'l2_errors': list(errors),
        # 'r2_scores': list(r2_scores),
        # 'pearson_corrs': pearson_genes if genes is not None else list(pearson_corrs),
        'pearson_std': float(f"{np.std(pearson_corrs):.4f}"),
        'l2_error_q1': float(f"{np.percentile(errors, 25): .4f}"),
        'l2_error_q2': float(f"{np.median(errors): .4f}"),
        'l2_error_q3': float(f"{np.percentile(errors, 75): .4f}"),
        'r2_score_q1': float(f"{np.percentile(r2_scores, 25): .4f}"),
        'r2_score_q2': float(f"{np.median(r2_scores): .4f}"),
        'r2_score_q3': float(f"{np.percentile(r2_scores, 75): .4f}"),
        # === New metrics ===
        'mape_mean': float(f"{np.mean(mape_list):.4f}"),
        'mape_std': float(f"{np.std(mape_list):.4f}"),
        'rmse_mean': float(f"{np.mean(rmse_list):.4f}"),
        'rmse_std': float(f"{np.std(rmse_list):.4f}")
    }

    return results


def spearmanrr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()
