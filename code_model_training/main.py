##############################
'''
Dataset path :-  /home/puneet/mk/data/her2st_dataset/
Dataset path :-  /home/puneet/mk/data/cscc_dataset/
'''
##############################




import sys
sys.path.append('/home/puneet/mk/code_model_training/models')
sys.path.append('/home/puneet/mk/code_model_training/utils')

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
from datetime import datetime
import logging
import argparse
import json
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from torch.nn import DataParallel
# from test_transformers import VisionTransformer, GeneExpressionSwinTimm
# from transformerModels import VisionTransformer
# from test_transformers_multi_layers import VisionTransformer

from transformerModels import VisionTransformer
# from models import GraphNN

from losses_2 import CustomLossRgv2, CustomLossRgv3, CustomLossRgv4
from models2 import EncoderDecoderRgv
from regression_models import CNN_regression_model, CNN_regression_reduced_model,CNN_regression_model_three_fc
from losses import contrastive_loss, spearmanrr, custom_spearmanr_old2, spearmanrr_feature_wise, custom_spearmanr_with_mse,custom_loss_dynamic_lambda_v5, custom_spearmanr_old, custom_spearmanr_non_zero_mask, custom_loss_pred_cosine, custom_loss_l1_pred_cosine,custom_loss_l1_spearman, custom_loss_l1_spearman_cosine,custom_loss_weighted_non_zeros, custom_loss_pred_zeros, custom_loss_weighted_non_zeros_l1, custom_loss_weighted_penalty_non_zeros,custom_loss_weighted_penalty, custom_spearmanr
# from get_neighbour_indices import SpatialGeneExpressionDataset, SpatialGeneExpressionDatasetValidationBased
from get_neighbour_indices_2 import SpatialGeneExpressionDataset, SpatialGeneExpressionValidationDataset
from compute_metrics import compute_metrics, spearmanrr
from setup_logger import setup_logging
from data_splits import split_train_validation
torch.manual_seed(123)


# Suppress all warnings
warnings.filterwarnings("ignore")
def main():
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

    params["info"] = "Model for Inflamed data."
    params["neighbours_information"] = "euclidean distence using KNN function"
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_dir = os.path.join(params['result_path'], f"result_vit_{timestamp}")
    os.makedirs(model_save_dir,  exist_ok=True)

    json_file = os.path.join(model_save_dir,f"experiment_info.json")

    with open(json_file, "w") as exp:
        json.dump(params, exp, indent=4)

    
    log_file = os.path.join(model_save_dir, "results_log.txt")
    logger = setup_logging(log_file)
    logger.info("Logging setup complete.")
    logger.info(f"Experiment information saved to the path: {json_file}")


    metrics_csv_path = os.path.join(model_save_dir, "metrics_result.csv")

    # Write CSV headers (only once before the loop starts)
    with open(metrics_csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'epoch','custom_loss','pearson_mean','spearman_mean_cell_wise','spearman_mean_genewise', 'l1_error_mean', 'l2_errors_mean', 'r2_scores_mean', 
            'pearson_std', 'l2_error_q1', 'l2_error_q2', 'l2_error_q3', 'r2_score_q1', 'r2_score_q2', 'r2_score_q3'
        ])
        writer.writeheader()

    def train_regression_with_custom_loss(train_datasets, val_datasets, k_spatial_neighbors=params['k_spatial_neighbors'], k_gene_exp_neighbors=params['k_gene_exp_neighbors'], lambda_reg=params["lambda_reg"], learning_rate=params['learning_rate'], epochs=params['epochs'], batch_size=params['batch_size']):
        models = {}

        # train_datasets, ext_datasets =  split_train_validation(train_datasets, val_split_ratio = params["val_split_ratio"])

        # logger.info(f"train_datasets:  {len(train_datasets)} ext_datasets: {len(ext_datasets)}")
        logger.info(f"train_datasets:  {len(train_datasets)}")
        train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)

        # val_datasets = val_datasets + ext_datasets
        print(f"val_datasets {len(val_datasets)}")
        val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False)

        logger.info(f"Number of batches: {len(train_dataloader)}")
        logger.info("Initializing model...")

        input_height = (params['k_spatial_neighbors'] + params['k_gene_exp_neighbors'] + 1)  # Example input height
        input_width = params['embeddings']  # Example input width
        output_size = 460  # Example output size

        if params["model"] == "CNN_regression_model":
            model = CNN_regression_model(input_height=input_height, input_width=input_width, output_size=output_size)

        elif params["model"] == "EncoderDecoderRgv":
            # Hyperparameters

            input_dim = params["embeddings"]
            hidden_dim = 512
            output_dim = 460  
            batch_size = params["batch_size"]
            learning_rate = 0.0001
            dropout_rate = 0.1
            regularization_strength = 2.0

            model = EncoderDecoderRgv(input_dim, hidden_dim, output_dim, dropout_rate)


        elif params["model"] == "VisionTransformer":
            # input_dim = 1024
            # seq_length = 11
            # num_hiddens = 512
            # num_heads = 2
            # num_blks = 2
            # output_dim = 460

            # seq_length = 11
            # input_dim = 1024
            # hidden_dim = 512
            # num_heads = 32
            # num_blks = 4
            # output_dim = 460
            seq_length = 1
            input_dim = params["embeddings"]
            hidden_dim = 512
            num_heads = 2
            num_blks = 4
            output_dim = 200
            dropout = 0.1
            model = VisionTransformer(input_dim, seq_length,  hidden_dim, num_heads, num_blks, output_dim, dropout)
        # elif params["model"] == "GNN":
        #     model = GraphNN()
        else:
            print('No model found')
        
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs available: {num_gpus}")

        gpu_id = params["device_ids"]
        # Check if GPU 1 and GPU 2 are available
        if num_gpus > 2:  # Ensure you have at least 3 GPUs (indices 0, 1, and 2)
            logger.info("Using GPUs " + str(gpu_id[0]))
            device_ids = params["device_ids"]  # Specify the GPUs to use
            device = torch.device(f"cuda:{device_ids[0]}")  # Set primary device to cuda:1
            print(device)
        else:
            logger.info("GPU 1 and GPU 2 are not available. Please check your system configuration.")
            raise RuntimeError("Insufficient GPUs available.")

        torch.cuda.empty_cache()

        # Wrap model with DataParallel and send to device
        if len(device_ids) > 1:
            logger.info(f"Using {len(device_ids)} GPUs: {device_ids}")
            model = DataParallel(model, device_ids=device_ids)

        model.to(device)  # Move the model to the primary device
        logger.info(f"Model is using device(s): {device_ids}")

        logger.info(f"Model:{model}")

        if params["optimizer"] == "Adam":
            optimizer = Adam(model.parameters(), lr=learning_rate)
        elif params["optimizer"] == "sgd":
            optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            print("Optimizer not defined!!!")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Minimize validation loss
        patience=params['lr_patience'],  # Number of epochs to wait before reducing LR
        factor=params['lr_factor'],  # Factor by which the learning rate will be reduced
        min_lr=params['min_lr'],  # Minimum learning rate
        verbose=True  # Log LR reduction
        )

        train_losses = []
        val_losses = []

        # Initialize variables for early stopping
        # best_val_loss = float("inf")
        best_spearman = -1
        patience_counter = 0
        patience = 15  # You can adjust this value

        # Training loop
        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0
            all_y_true = []
            all_y_pred = []
            pearson_train_corr = []
            spearman_train_corr = []

            count = 0
            for features, target in train_dataloader:
                print("Feature Length", features['stacked_embeddings'].shape)
                stacked_embeddings = torch.tensor(features['stacked_embeddings'], dtype=torch.float32).to(device)
                if params["mean"] is True:
                    stacked_embeddings = stacked_embeddings.mean(1)
                    print("After Feature Length", stacked_embeddings.shape)
                y_true = torch.tensor(target, dtype=torch.float32)[:, 0, :].to(device)
                print("y_true ", y_true.shape)
                if params["subset_gene"] is True:
                    data = np.load("top_k_genes.npz")

                    # top_k_values = data["values"]
                    top_k_indices = data["indices"]
                    top_k_indices =  torch.tensor(top_k_indices,dtype = torch.long)
                    print(top_k_indices.shape)
                    y_true = y_true[:,top_k_indices.view(-1)]

                stacked_embeddings = stacked_embeddings.unsqueeze(1)

                neighbors_gene_expression = target[:, 1:].view(target.shape[0], params['k_spatial_neighbors'] + params['k_gene_exp_neighbors'], target.shape[2]).to(device)

                optimizer.zero_grad()
                y_pred = model(stacked_embeddings)

                print(y_pred.shape)
                # return
                # print("y_pred::::::::::::", y_pred.shape)
                # loss = custom_loss_l1_l2_pred_dot_product(y_true, y_pred, neighbors_gene_expression, model, lambda_l1=lambda_l1, lambda_l2=lambda_l2, lambda_reg=lambda_reg)
                if params["loss_fn"] == "custom_loss_l1_pred_cosine": 
                    loss = custom_loss_l1_pred_cosine(y_true, y_pred, neighbors_gene_expression)
                elif params["loss_fn"] == "custom_loss_l1_spearman_cosine":
                    loss = custom_loss_l1_spearman_cosine(y_true, y_pred, neighbors_gene_expression, lambda_reg=lambda_reg)
                elif params["loss_fn"] == "custom_loss_l1_spearman":
                    loss = custom_loss_l1_spearman(y_true, y_pred, lambda_reg=lambda_reg)
                elif params["loss_fn"] == "custom_loss_weighted_non_zeros_l1":
                    loss = custom_loss_weighted_non_zeros_l1(y_true, y_pred, neighbors_gene_expression, lambda_reg=lambda_reg)
                elif params["loss_fn"] == "custom_loss_weighted_penalty_non_zeros":
                    loss = custom_loss_weighted_penalty_non_zeros(y_true, y_pred, neighbors_gene_expression, lambda_reg=lambda_reg)
                elif params["loss_fn"] == "custom_spearmanr_with_mse":
                    loss = custom_spearmanr_with_mse(y_pred,y_true)
                elif params["loss_fn"] == "custom_spearmanr_old":
                    loss = custom_spearmanr_old(y_pred,y_true)
                elif params["loss_fn"] == "custom_spearmanr_old2":
                    loss = custom_spearmanr_old2(y_pred,y_true)
                elif params["loss_fn"] == "custom_spearmanr_non_zero_mask":
                    loss = custom_spearmanr_non_zero_mask(y_pred,y_true)
                elif params["loss_fn"] == "contrastive_loss":
                    loss = contrastive_loss(y_pred,y_true)
                elif params["loss_fn"] == "CustomLossRgv4":
                    criterion = CustomLossRgv4(l1_lambda=0.1, regularization_strength=1.0)
                    loss = criterion(y_true, y_pred)  
                elif params["loss_fn"] == "CustomLossRgv2":
                    criterion = CustomLossRgv2(l1_lambda=0.1, regularization_strength=1.0)
                    loss = criterion(y_true, y_pred)
                else:
                    print("Loss Function not defined!!!") 

                loss.backward()
                optimizer.step()



                torch.cuda.empty_cache()

                epoch_train_loss += loss.item()
                all_y_true.append(y_true.detach().cpu().numpy())
                all_y_pred.append(y_pred.detach().cpu().numpy())

                # print(y_pred.shape)
                # print(y_true.shape)

                # Calculate Pearson correlation for the batch
                batch_pearson, _ = pearsonr(y_pred.detach().cpu().numpy().flatten(), y_true.detach().cpu().numpy().flatten())

                pearson_train_corr.append(batch_pearson.mean().item())


                # Calculate Spearman correlation for the batch
                batch_spearman = spearmanrr(y_pred, y_true)
                spearman_train_corr.append(batch_spearman.mean().item())

                if count % 10 == 0:
                    logger.info(f"Training Epoch {epoch + 1}/{epochs}, Training Batch: {count}, Training Batch Loss: {loss.item():.4f}, Pearson Correlation: {pearson_train_corr[-1]:.4f} , Spearman Correlation: {spearman_train_corr[-1]:.4f}")
                count += 1

            all_y_true = np.vstack(all_y_true)
            all_y_pred = np.vstack(all_y_pred)
            results_train = compute_metrics(all_y_true, all_y_pred)

            logger.info(f"Training Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_train_loss / len(train_dataloader):.4f}, Pearson Mean: {np.mean(pearson_train_corr):.4f} , Spearman Mean: {np.mean(spearman_train_corr):.4f}")
            logger.info(f"Training Metrics: {results_train}")

            results_train['epoch'] = f"train {epoch + 1}"
            results_train['custom_loss'] = f"{epoch_train_loss / len(train_dataloader):.4f}"
            results_train['spearman_mean_cell_wise'] = f"{np.mean(spearman_train_corr):.4f}"

            train_losses.append(epoch_train_loss / len(train_dataloader))

            # Validation phase
            model.eval()
            epoch_val_loss = 0
            val_y_true = []
            val_y_pred = []
            spearman_val_corr = []
            pearson_val_corr = []

            count = 0
            with torch.no_grad():
                for features, target in val_dataloader:

                    print("Feature Length", features['stacked_embeddings'].shape)
                    # print("target Length", len(target))
                    stacked_embeddings = torch.tensor(features['stacked_embeddings'], dtype=torch.float32).to(device)


                    # y_true = torch.tensor(target, dtype=torch.float32)[:, 0, :].to(device)
                    y_true = torch.tensor(target, dtype=torch.float32).to(device)
                    print("y_true original squeeze ", y_true.squeeze(1).shape)
                    y_true = y_true.squeeze(1) 
                    if  params["subset_gene"] is True:
                        data = np.load("top_k_genes.npz")

                        # top_k_values = data["values"]
                        top_k_indices = data["indices"]
                        top_k_indices =  torch.tensor(top_k_indices, dtype = torch.long)
                        y_true = y_true[:, top_k_indices.view(-1)]
                        print("y_true squeeze after gene selectin ", y_true.shape)

                    if params["mean"] is not True:
                        stacked_embeddings = stacked_embeddings.unsqueeze(1)
                    # else:
                    #     stacked_embeddings = stacked_embeddings.squeeze(1)
                    #     target = target.squeeze(1)


                    print("stacked_embeddings ", stacked_embeddings.shape)
                    print("target shape before slicing: ", target.shape)


                    # neighbors_gene_expression = target[:, 1:].view(target.shape[0], params['k_spatial_neighbors'] + params['k_gene_exp_neighbors'], target.shape[2]).to(device)
                    y_pred = model(stacked_embeddings)
                    y_pred = torch.where(y_pred < 0, torch.tensor(0.0, device=y_pred.device), y_pred)
                    # loss = custom_loss_l1_l2_pred_dot_product(y_true, y_pred, neighbors_gene_expression, model, lambda_l1=lambda_l1, lambda_l2=lambda_l2, lambda_reg=lambda_reg)
                    if params["loss_fn"] == "custom_loss_l1_pred_cosine": 
                        loss = custom_loss_l1_pred_cosine(y_true, y_pred, neighbors_gene_expression)
                    elif params["loss_fn"] == "custom_loss_l1_spearman_cosine":
                        loss = custom_loss_l1_spearman_cosine(y_true, y_pred, neighbors_gene_expression, lambda_reg=lambda_reg)
                    elif params["loss_fn"] == "custom_loss_l1_spearman":
                        loss = custom_loss_l1_spearman(y_true, y_pred, lambda_reg=lambda_reg)
                    elif params["loss_fn"] == "custom_loss_weighted_non_zeros_l1":
                        loss = custom_loss_weighted_non_zeros_l1(y_true, y_pred, neighbors_gene_expression, lambda_reg=lambda_reg)
                    elif params["loss_fn"] == "custom_loss_weighted_penalty_non_zeros":
                        loss = custom_loss_weighted_penalty_non_zeros(y_true, y_pred, neighbors_gene_expression, lambda_reg=lambda_reg)
                    elif params["loss_fn"] == "custom_spearmanr_with_mse":
                        loss = custom_spearmanr_with_mse(y_pred,y_true)
                    elif params["loss_fn"] == "custom_spearmanr_old":
                        loss = custom_spearmanr_old(y_pred,y_true)
                    elif params["loss_fn"] == "custom_spearmanr_old2":
                        loss = custom_spearmanr_old2(y_pred,y_true)
                    elif params["loss_fn"] == "custom_spearmanr_non_zero_mask":
                        loss = custom_spearmanr_non_zero_mask(y_pred,y_true)
                    elif params["loss_fn"] == "contrastive_loss":
                        loss = contrastive_loss(y_pred,y_true)
                    elif params["loss_fn"] == "CustomLossRgv2":
                        criterion = CustomLossRgv2(l1_lambda=0.1, regularization_strength=1.0)
                        loss = criterion(y_true, y_pred)
                    elif params["loss_fn"] == "CustomLossRgv4":
                        criterion = CustomLossRgv4(l1_lambda=0.1, regularization_strength=1.0)
                        loss = criterion(y_true, y_pred)                    
                    else:
                        print("Loss Function not defined!!!")
                    

                    epoch_val_loss += loss.item()
                    val_y_true.append(y_true.detach().cpu().numpy())
                    val_y_pred.append(y_pred.detach().cpu().numpy())


                    # Calculate Pearson correlation for the batch
                    batch_pearson, _ = pearsonr(y_pred.detach().cpu().numpy().flatten(), y_true.detach().cpu().numpy().flatten())
                    pearson_val_corr.append(batch_pearson.mean().item())

                    # Calculate Spearman correlation for the batch
                    batch_spearman = spearmanrr(y_pred, y_true)
                    spearman_val_corr.append(batch_spearman.mean().item())

                    if count % 10 == 0:
                        logger.info(f"Validation Epoch {epoch + 1}/{epochs}, Validation Batch: {count}, Validation Batch Loss: {loss:.4f}, Pearson Correlation: {pearson_val_corr[-1]:.4f} , Spearman Correlation: {spearman_val_corr[-1]:.4f}")
                    count += 1

                val_y_true = np.vstack(val_y_true) if val_y_true else np.array([])
                val_y_pred = np.vstack(val_y_pred) if val_y_pred else np.array([])

                # import random
                # random_samples = random.sample(range(1, batch_size), 5)
                val_y_true_25 = val_y_true[:25, :]  
                val_y_pred_25 = val_y_pred[:25, :]  

                # Create a 5x5 grid of subplots
                fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(25, 25))  # Adjust figsize for better visibility

                for i in range(5):  # Row index
                    for j in range(5):  # Column index
                        sample_index = i * 5 + j  # Calculate the sample index (0 to 24)
                        
                        # Scatter plot for the specific sample
                        axes[i, j].scatter(val_y_true_25[sample_index], val_y_pred_25[sample_index], 
                                           color="blue", alpha=0.6, label="True vs Predicted")
                        
                        # Add a red line for perfect prediction
                        min_val = min(val_y_true_25[sample_index].min(), val_y_pred_25[sample_index].min())
                        max_val = max(val_y_true_25[sample_index].max(), val_y_pred_25[sample_index].max())
                        axes[i, j].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction Line')
                        
                        # Add labels and title
                        axes[i, j].set_xlabel("Ground Truth (460 Features)")
                        axes[i, j].set_ylabel("Predicted Values (460 Features)")
                        axes[i, j].set_title(f"Sample {sample_index + 1}")
                        axes[i, j].grid(True)

                # Add a single legend for the entire figure
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))

                # Adjust layout, save, and show the plot
                plt.tight_layout()
                scatter_save_path = os.path.join(model_save_dir, f"scatter_plot_25_samples_epoch_{epoch+1}.png")
                plt.savefig(scatter_save_path, format="png")

                results_val = compute_metrics(val_y_true, val_y_pred)

                logger.info(f"Validation Epoch {epoch + 1}/{epochs}, Validation Loss: {epoch_val_loss / len(val_dataloader):.4f}, Pearson Mean: {np.mean(pearson_val_corr):.4f} , Spearman Mean: {np.mean(spearman_val_corr):.4f}")
                logger.info(f"Validation Metrics: {results_val}")
                
                results_val['epoch'] = f"val {epoch + 1}"
                results_val['custom_loss'] = f"{epoch_val_loss / len(val_dataloader):.4f}"
                results_val['spearman_mean_cell_wise'] = f"{np.mean(spearman_val_corr):.4f}"

                # Save metrics to CSV
                with open(metrics_csv_path, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=[
                        'epoch', 'custom_loss', 'pearson_mean', 'spearman_mean_cell_wise','spearman_mean_genewise','l1_error_mean', 'l2_errors_mean', 'r2_scores_mean', 
                        'pearson_std', 'l2_error_q1', 'l2_error_q2', 'l2_error_q3',
                        'r2_score_q1', 'r2_score_q2', 'r2_score_q3'
                    ])
                    writer.writerow({key: results_train[key] for key in writer.fieldnames})
                    writer.writerow({key: results_val[key] for key in writer.fieldnames})

                val_losses.append(epoch_val_loss / len(val_dataloader))

                scheduler.step(epoch_val_loss / len(val_dataloader))

                # Log the current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate for epoch {epoch + 1}: {current_lr}")

                # Early stopping logic
                if float(results_val['spearman_mean_cell_wise']) > float(best_spearman):
                    best_spearman = results_val['spearman_mean_cell_wise']
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model.pth"))
                    logger.info("Best model saved.")
                else:
                    patience_counter += 1
                    logger.info(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    logger.info("Early stopping triggered. Training stopped.")
                    break
      
                # Plot losses

                # print("train_losses :::::: ", type(train_losses))
                # print("train_losses :::::: ", train_losses)

                # train_losses = [loss.cpu().item() if isinstance(loss, torch.Tensor) else float(loss) for loss in train_losses]
                # val_losses = [loss.cpu().item() if isinstance(loss, torch.Tensor) else float(loss) for loss in val_losses]
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss", marker="o")
                plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training and Validation Loss")
                plt.legend()
                plt.grid()
                graph_save_path = os.path.join(model_save_dir, "loss_plot.png")
                plt.savefig(graph_save_path)
                plt.close()

            torch.save(model.state_dict(), os.path.join(model_save_dir, f"{epoch+1}_regression_model.pth"))

        models["complete_model"] = model

        return models

    #Paths to your files
    list_ST_name_data = ["UC1_NI", "UC1_I", "UC6_NI", "UC6_I", "UC7_I", "UC9_I", "DC5"]
    # list_ST_name_data = ["UC1_NI"]
    # if params["data"] == "Inflamed":
    #     list_ST_name_data = ['UC1_I', 'UC6_I', 'UC7_I', 'UC9_I']
    # elif params["data"] == "Non-Inflamed":
    #     list_ST_name_data = ['UC1_NI', 'UC6_NI', 'DC5']


    train_paths = [
        (
            f"{params['label_path']}{folder}_train.h5ad",
            f"{params['dataset_path']}{folder}_train_embeddings.h5",
        )
        for folder in list_ST_name_data
    ]

    train_datasets = [SpatialGeneExpressionDataset(adata_path, embedding_file_path,output_dir= params['neighbors_path'] ,is_train=True, k_spatial_neighbors=params['k_spatial_neighbors'], k_gene_exp_neighbors=params['k_gene_exp_neighbors'], add_coords = params["add_coords"]) for  i, (adata_path, embedding_file_path) in enumerate(train_paths)]
    train_datasets = ConcatDataset(train_datasets)
    logger.info(f"Training Data size {len(train_datasets)}")

    # val_paths = [
    # (
    #     f"{params['label_path']}{folder}_train.h5ad",
    #     f"{params['label_path']}{folder}_test.h5ad",
    #     f"{params['dataset_path']}{folder}_train_embeddings.h5",
    #     f"{params['dataset_path']}{folder}_test_embeddings.h5"
    # )
    # for folder in list_ST_name_data
    # ]


    # os.makedirs(params['neighbors_path'], exist_ok=True)

    # val_datasets = [SpatialGeneExpressionDatasetValidationBased(adata_path = adata_path_train, adata_path_val = adata_path_val, embedding_file_path_train = embedding_path_train, embedding_file_path_val = embedding_path_test,
    # output_dir= params['neighbors_path_val'] , k_spatial_neighbors=params['k_spatial_neighbors'], k_gene_exp_neighbors=params['k_gene_exp_neighbors']) for  i, (adata_path_train, adata_path_val, embedding_path_train, embedding_path_test) in enumerate(val_paths)]
    # val_datasets = ConcatDataset(val_datasets)
    # print(f"Validation Data size {len(val_datasets)}")


    val_paths = [
        (
            f"{params['label_path']}{folder}_test.h5ad",
            f"{params['dataset_path']}{folder}_test_embeddings.h5",
        )
        for folder in list_ST_name_data
    ]

    # val_datasets = [SpatialGeneExpressionDataset(adata_path, embedding_file_path, output_dir= params['neighbors_path'] ,is_train=False, k_spatial_neighbors=params['k_spatial_neighbors'], k_gene_exp_neighbors=params['k_gene_exp_neighbors'], add_coords = params["add_coords"]) for  i, (adata_path, embedding_file_path) in enumerate(val_paths)]
    val_datasets = [SpatialGeneExpressionDataset(adata_path, embedding_file_path, output_dir= params['neighbors_path'] ,is_train=False, k_spatial_neighbors=params['k_spatial_neighbors'], k_gene_exp_neighbors=params["val_k_gene_exp_neighbors"], add_coords = params["add_coords"]) for  i, (adata_path, embedding_file_path) in enumerate(val_paths)]
    val_datasets = ConcatDataset(val_datasets)


    logger.info(f"Validation Data size {len(val_datasets)}")


    models = train_regression_with_custom_loss(train_datasets,val_datasets, k_spatial_neighbors=params['k_spatial_neighbors'], k_gene_exp_neighbors=params['k_gene_exp_neighbors'], batch_size=params['batch_size'])


if __name__ == "__main__":
    main()