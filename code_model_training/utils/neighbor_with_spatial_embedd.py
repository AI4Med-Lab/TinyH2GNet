import h5py
import csv
import anndata as ad
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
import os
import matplotlib.pyplot as plt
import joblib
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import ConcatDataset
from scipy.stats import pearsonr, ConstantInputWarning
from scipy import sparse
import logging
import json


class SpatialGeneExpressionAndEmbeddDataset(Dataset):
    def __init__(self, adata_path, embedding_file_path, output_dir, is_train=True,
                 k_spatial_neighbors=10, k_embedding_neighbors=10):
        """
        Dataset for spatial and embedding-based data integration.

        Args:
            adata_path (str): Path to AnnData file containing spatial data.
            embedding_file_path (str): Path to file containing precomputed embeddings.
            output_dir (str): Directory to save the neighbor indices CSV.
            is_train (bool): Whether the dataset is for training or validation.
            k_spatial_neighbors (int): Number of nearest neighbors for spatial coordinates.
            k_embedding_neighbors (int): Number of nearest neighbors for embeddings.
        """
        self.adata_path = adata_path
        self.embedding_file_path = embedding_file_path
        self.output_dir = output_dir
        self.is_train = is_train
        self.k_spatial_neighbors = k_spatial_neighbors
        self.k_embedding_neighbors = k_embedding_neighbors
        self.csv_file = self._get_csv_file_path()
        self.data = []

        dataset_name = os.path.basename(self.adata_path).replace(".h5ad", "")

        self.spatial_knn_model_path = os.path.join(self.output_dir, f"{dataset_name}_knn_spatial.pkl")
        self.embedding_knn_model_path = os.path.join(self.output_dir, f"{dataset_name}_knn_embeddings.pkl")

        # Load AnnData and embeddings once
        
        self.adata = sc.read_h5ad(self.adata_path)
        self.embeddings, self.spatial_coords = self._load_embeddings_and_coords(embedding_file_path)
        
        # Precompute neighbor indices once
        self._prepare_data()

    def _get_csv_file_path(self):
        """
        Generate the CSV file path based on dataset name and split.
        """
        dataset_name = os.path.basename(self.adata_path).replace(".h5ad", "")
        return os.path.join(self.output_dir, f"{dataset_name}_neighbors.csv")

    def _load_embeddings_and_coords(self, embedding_file_path):
        """
        Load embeddings and spatial coordinates from the file.
        """
        with h5py.File(embedding_file_path, 'r') as f:
            embeddings = f['embeddings'][:]
            spatial_coords = f['coords'][:]
        return embeddings, spatial_coords

    def _prepare_data(self):
        """
        Load or generate neighbor indices for the dataset.
        """
        if os.path.exists(self.csv_file):
            # Load precomputed neighbor indices from CSV
            self.data = pd.read_csv(self.csv_file).to_dict(orient="records")
        else:
            # Generate neighbor indices if CSV file doesn't exist
            self._generate_neighbor_indices()

    def _generate_neighbor_indices(self):
        """
        Generate neighbor indices and save them to a CSV file.
        """
        n_cells = self.spatial_coords.shape[0]

        # Finding nearest neighbors for spatial data
        knn_spatial = NearestNeighbors(n_neighbors=self.k_spatial_neighbors + 1, n_jobs=-1).fit(self.spatial_coords)
        dataset_name = os.path.basename(self.adata_path).replace(".h5ad", "")
        os.makedirs(self.output_dir,exist_ok=True)
        joblib.dump(knn_spatial,  self.spatial_knn_model_path)
        _, indices_spatial = knn_spatial.kneighbors(self.spatial_coords)

        #Finding nearest neighbors for embeddings using cosine similarity
        # knn_embeddings = NearestNeighbors(n_neighbors=self.k_embedding_neighbors + 1, metric='cosine', n_jobs=-1).fit(self.embeddings)
        knn_embeddings = NearestNeighbors(n_neighbors=self.k_embedding_neighbors + 1, n_jobs=-1).fit(self.embeddings)
        _, indices_embeddings = knn_embeddings.kneighbors(self.embeddings)
        joblib.dump(knn_embeddings, self.embedding_knn_model_path)

        # Prepare CSV data for neighbor indices
        csv_data = []
        for i in range(n_cells):
            spatial_indices = indices_spatial[i][1:]  # Exclude the first index (self)
            embedding_indices = indices_embeddings[i][1:]  # Exclude the first index (self)

            # Store neighbors in separate columns
            spatial_columns = {f"spatial_n{j+1}": spatial_indices[j] for j in range(self.k_spatial_neighbors)}
            embedding_columns = {f"embedding_n{j+1}": embedding_indices[j] for j in range(self.k_embedding_neighbors)}

            # Combine the main cell index and neighbors in a dictionary
            csv_data.append({
                "main_cell_index": i,
                **spatial_columns,
                **embedding_columns,
            })

        # Save to CSV
        pd.DataFrame(csv_data).to_csv(self.csv_file, index=False)
        self.data = csv_data          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the features (embeddings) and target (gene expression) for a given index.
        """
        main_cell_index = self.data[idx]["main_cell_index"]

        
        # Extract spatial and embedding neighbors from precomputed indices
        spatial_indices = [int(self.data[idx].get(f"spatial_n{n+1}", -1)) for n in range(self.k_spatial_neighbors)]
        embedding_indices = [int(self.data[idx].get(f"embedding_n{n+1}", -1)) for n in range(self.k_embedding_neighbors)]
        
        # Remove invalid indices (-1 indicates missing neighbor data)
        spatial_indices = [i for i in spatial_indices if i >= 0]
        embedding_indices = [i for i in embedding_indices if i >= 0]

        # Extract embeddings
        main_embedding = self.embeddings[main_cell_index]
        spatial_embeddings = self.embeddings[spatial_indices] if spatial_indices else np.empty((0, self.embeddings.shape[1]))
        embedding_neighbors = self.embeddings[embedding_indices] if embedding_indices else np.empty((0, self.embeddings.shape[1]))

        # Extract targets from AnnData
        gene_expression = self.adata.X
        main_target = gene_expression[main_cell_index]
        spatial_targets = gene_expression[spatial_indices] if spatial_indices else np.empty((0, gene_expression.shape[1]))
        embedding_targets = gene_expression[embedding_indices] if embedding_indices else np.empty((0, gene_expression.shape[1]))

        # Handle sparse matrices
        if sparse.issparse(gene_expression):
            main_target = main_target.toarray()
            spatial_targets = spatial_targets.toarray()
            embedding_targets = embedding_targets.toarray()

        # Combine embeddings
        embedding_list = [main_embedding]
        target_list = [main_target]

        if spatial_embeddings.size > 0:
            embedding_list.append(spatial_embeddings)
            target_list.append(spatial_targets)

        if embedding_neighbors.size > 0:
            embedding_list.append(embedding_neighbors)
            target_list.append(embedding_targets)

        # Stack the embeddings and targets
        stacked_embeddings = np.vstack(embedding_list).astype(np.float32)
        stacked_targets = np.vstack(target_list).astype(np.float32)

        # Convert to tensors for PyTorch model
        features = {"stacked_embeddings": torch.tensor(stacked_embeddings)}
        target = torch.tensor(stacked_targets)

        return features, target



##### ======================================================================Validtion data==========================================================

class ValidationNeighborDataset(Dataset):
    def __init__(self, train_knn_spatial, train_knn_embeddings, train_h5_path, train_h5ad_path, output_dir, validation_h5_path, validation_h5ad_path,
                 k_spatial_neighbors=10, k_embedding_neighbors=10):
        """
        Dataset for generating validation data nearest neighbors and their targets.

        Args:
            train_knn_spatial (NearestNeighbors): KNN model trained on spatial coordinates.
            train_knn_embeddings (NearestNeighbors): KNN model trained on embeddings.
            train_h5_path (str): Path to the `.h5` file containing training embeddings.
            train_h5ad_path (str): Path to the `.h5ad` file containing training gene expression data.
            validation_h5_path (str): Path to the validation `.h5` file.
            k_spatial_neighbors (int): Number of nearest neighbors for spatial coordinates.
            k_embedding_neighbors (int): Number of nearest neighbors for embeddings.
        """


        self.output_dir = output_dir
        self.validation_h5ad_path = validation_h5ad_path
        self.train_h5ad_path = train_h5ad_path
        self.knn_spatial = joblib.load(train_knn_spatial) 
        self.knn_embeddings = joblib.load(train_knn_embeddings)
        self.k_spatial_neighbors = k_spatial_neighbors
        self.k_embedding_neighbors = k_embedding_neighbors
        self.csv_file = self._get_csv_file_path()
        self.data = []

        os.makedirs(self.output_dir, exist_ok=True)
        # Load validation spatial coordinates and embeddings
        with h5py.File(validation_h5_path, 'r') as f:
            self.val_coords = f['coords'][:]
            self.val_embeddings = f['embeddings'][:]

        self.val_gene_expression = sc.read_h5ad(validation_h5ad_path).X    
        if sparse.issparse(self.val_gene_expression):
            self.val_gene_expression = self.val_gene_expression.toarray()

        # Load training embeddings from `.h5` file
        with h5py.File(train_h5_path, 'r') as f:
            self.train_embeddings = f['embeddings'][:]

        # Load training gene expression data from `.h5ad` file
        self.train_gene_expression = sc.read_h5ad(train_h5ad_path).X
        if sparse.issparse(self.train_gene_expression):
            self.train_gene_expression = self.train_gene_expression.toarray()

        # Prepare data (load or generate neighbor indices)
        self._prepare_data()
            
    def _get_csv_file_path(self):
        """
        Generate the CSV file path based on dataset name and split.
        """
        dataset_name = os.path.basename(self.validation_h5ad_path).replace(".h5ad", "")
        return os.path.join(self.output_dir, f"{dataset_name}_neighbors.csv")

    def _prepare_data(self):
        """
        Load or generate neighbor indices for the validation dataset.
        """
        if os.path.exists(self.csv_file):
            # Load precomputed neighbor indices from CSV
            self.data = pd.read_csv(self.csv_file).to_dict(orient="records")
        else:
            # Generate neighbor indices if CSV file doesn't exist
            self._generate_neighbor_indices()   

    def _generate_neighbor_indices(self):
        """
        Generate neighbor indices for validation data using pre-trained KNN models
        and save them to a CSV file.
        """

        # Find nearest neighbors for validation data using the training models
        _, indices_spatial = self.knn_spatial.kneighbors(self.val_coords)
        _, indices_embeddings = self.knn_embeddings.kneighbors(self.val_embeddings)

        n_cells = self.val_coords.shape[0]

        # Prepare CSV data for neighbor indices
        csv_data = []
        for i in range(n_cells):
            spatial_indices = indices_spatial[i][:self.k_spatial_neighbors]
            embedding_indices = indices_embeddings[i][:self.k_embedding_neighbors]

            # Store neighbors in separate columns
            spatial_columns = {f"spatial_n{j+1}": spatial_indices[j] for j in range(self.k_spatial_neighbors)}
            embedding_columns = {f"embedding_n{j+1}": embedding_indices[j] for j in range(self.k_embedding_neighbors)}

            # Combine the main cell index and neighbors in a dictionary
            csv_data.append({
                "main_cell_index": i,
                **spatial_columns,
                **embedding_columns,
            })

        # Save to CSV
        pd.DataFrame(csv_data).to_csv(self.csv_file, index=False)
        self.data = csv_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get features (main embedding and neighbors) and targets (gene expression of neighbors).

        Args:
            idx (int): Index of the validation sample.

        Returns:
            dict: Features containing main embedding and neighbors.
            torch.Tensor: Targets for the neighbors.
        """
        # Get main validation embedding
        main_cell_idx = self.data[idx]["main_cell_index"]
        # print(main_cell_idx)
        # print(len(main_cell_idx))

        spatial_neighbors = [self.data[idx][f"spatial_n{j+1}"] for j in range(self.k_spatial_neighbors)]
        embedding_neighbors = [self.data[idx][f"embedding_n{j+1}"] for j in range(self.k_embedding_neighbors)]

        # Get the embeddings and gene expression values for neighbors
        main_embedding = self.val_embeddings[main_cell_idx]
        spatial_neighbor_embeddings = self.train_embeddings[spatial_neighbors]
        embedding_neighbor_embeddings = self.train_embeddings[embedding_neighbors]

        # Get the targets (gene expression) for neighbors from the training data
        main_target = self.val_gene_expression[main_cell_idx]
        spatial_neighbor_targets = self.train_gene_expression[spatial_neighbors]
        embedding_neighbor_targets = self.train_gene_expression[embedding_neighbors]

        # Combine embeddings and targets
        embedding_list = [main_embedding]
        target_list = [main_target]

        if spatial_neighbor_embeddings.size > 0:
            embedding_list.append(spatial_neighbor_embeddings)
            target_list.append(spatial_neighbor_targets)

        if embedding_neighbor_embeddings.size > 0:
            embedding_list.append(embedding_neighbor_embeddings)
            target_list.append(embedding_neighbor_targets)

        # Stack embeddings and targets
        stacked_embeddings = np.vstack(embedding_list).astype(np.float32)
        stacked_targets = np.vstack(target_list).astype(np.float32)
        # print("stacked_targets.shape",stacked_targets.shape)
        # Convert to tensors
        features = {"stacked_embeddings": torch.tensor(stacked_embeddings)}
        target = torch.tensor(stacked_targets)
        # print("target",target.shape)

        return features, target