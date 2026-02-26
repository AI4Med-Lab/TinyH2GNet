import h5py
import csv
import anndata as ad
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import ConcatDataset
from scipy.stats import pearsonr, ConstantInputWarning
from scipy import sparse
import logging
import json
# import torchsort

class SpatialGeneExpressionDataset(Dataset):
    def __init__(self, adata_path, embedding_file_path, output_dir, is_train=True,
                 k_spatial_neighbors=10, k_gene_exp_neighbors=10):
        """
        Dataset for spatial and gene expression data integration.

        Args:
            adata_path (str): Path to AnnData file containing spatial and gene expression data.
            embedding_file_path (str): Path to file containing precomputed embeddings.
            output_dir (str): Directory to save the neighbor indices CSV.
            is_train (bool): Whether the dataset is for training or validation.
            k_spatial_neighbors (int): Number of nearest neighbors for spatial coordinates.
            k_gene_exp_neighbors (int): Number of nearest neighbors for gene expression data.
        """
        self.adata_path = adata_path
        self.embedding_file_path = embedding_file_path
        self.output_dir = output_dir
        self.is_train = is_train
        self.k_spatial_neighbors = k_spatial_neighbors
        self.k_gene_exp_neighbors = k_gene_exp_neighbors
        self.csv_file = self._get_csv_file_path()
        self.data = []

        # Load AnnData and embeddings once
        self.adata = sc.read_h5ad(self.adata_path)
        self.embeddings = self._load_embeddings(embedding_file_path)
        
        # Precompute neighbor indices once
        self._prepare_data()

    def _get_csv_file_path(self):
        """
        Generate the CSV file path based on dataset name and split.
        """
        dataset_name = os.path.basename(self.adata_path).replace(".h5ad", "")
        return os.path.join(self.output_dir, f"{dataset_name}_neighbors.csv")

    def _load_embeddings(self, embedding_file_path):
        """
        Load embeddings from the file.
        """
        with h5py.File(embedding_file_path, 'r') as f:
            return f['embeddings'][:]

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
        spatial_coords = self.adata.obsm['spatial']
        gene_expression = self.adata.X
        n_cells = spatial_coords.shape[0]

        # Finding nearest neighbors for spatial data
        knn_spatial = NearestNeighbors(n_neighbors=self.k_spatial_neighbors + 1, n_jobs=-1).fit(spatial_coords)
        _, indices_spatial = knn_spatial.kneighbors(spatial_coords)

        # Finding nearest neighbors for gene expression data
        knn_expression = NearestNeighbors(n_neighbors=self.k_gene_exp_neighbors + 1, n_jobs=-1).fit(gene_expression)
        _, indices_expression = knn_expression.kneighbors(gene_expression)

        # Prepare CSV data for neighbor indices
        csv_data = []
        for i in range(n_cells):
            spatial_indices = indices_spatial[i][1:]  # Exclude the first index (self)
            expression_indices = indices_expression[i][1:]  # Exclude the first index (self)

            # Store neighbors in separate columns
            spatial_columns = {f"spatial_n{j+1}": spatial_indices[j] for j in range(self.k_spatial_neighbors)}
            expression_columns = {f"gene_expression_n{j+1}": expression_indices[j] for j in range(self.k_gene_exp_neighbors)}

            # Combine the main cell index and neighbors in a dictionary
            csv_data.append({
                "main_cell_index": i,
                **spatial_columns,
                **expression_columns,
            })

        # Save to CSV
        pd.DataFrame(csv_data).to_csv(self.csv_file, index=False)
        self.data = csv_data          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the features (embeddings, gene expression data) and target (gene expression) for a given index.
        """
        main_cell_index = self.data[idx]["main_cell_index"]
        
        # Extract spatial and expression neighbors from precomputed indices
        spatial_indices = [int(self.data[idx].get(f"spatial_n{n+1}", -1)) for n in range(self.k_spatial_neighbors)]
        expression_indices = [int(self.data[idx].get(f"gene_expression_n{n+1}", -1)) for n in range(self.k_gene_exp_neighbors)]
        
        # Remove invalid indices (-1 indicates missing neighbor data)
        spatial_indices = [i for i in spatial_indices if i >= 0]
        expression_indices = [i for i in expression_indices if i >= 0]

        # Extract gene expression matrix (keep it sparse if possible)
        gene_expression = self.adata.X

        main_gene_expression = gene_expression[main_cell_index]
        spatial_gene_expression = gene_expression[spatial_indices]
        expression_gene_expression = gene_expression[expression_indices]

        # Handle sparse matrices
        if sparse.issparse(gene_expression):
            main_gene_expression = main_gene_expression.toarray()
            spatial_gene_expression = spatial_gene_expression.toarray()
            expression_gene_expression = expression_gene_expression.toarray()

        # Extract embeddings
        main_embedding = self.embeddings[main_cell_index]
        spatial_embeddings = self.embeddings[spatial_indices] if spatial_indices else np.empty((0, self.embeddings.shape[1]))
        expression_embeddings = self.embeddings[expression_indices] if expression_indices else np.empty((0, self.embeddings.shape[1]))
        # print("main_embedding:::::::::::::::::::::::::::::::::::::", main_embedding.shape)
        # Combine embeddings and gene expression data
        embedding_list = [main_embedding]
        gene_expression_list = [main_gene_expression]

        if spatial_embeddings.size > 0:
            embedding_list.append(spatial_embeddings)
            gene_expression_list.append(spatial_gene_expression)

        if expression_embeddings.size > 0:
            embedding_list.append(expression_embeddings)
            gene_expression_list.append(expression_gene_expression)

        # Stack the embeddings and gene expressions
        stacked_embeddings = np.vstack(embedding_list).astype(np.float32)

        # print("++++++++++ stacked_embeddings +++++++++", stacked_embeddings.shape)
        stacked_gene_expression = np.vstack(gene_expression_list).astype(np.float32)
        # print(stacked_embeddings.shape)
        # Convert to tensors for PyTorch model
        features = {"stacked_embeddings": torch.tensor(stacked_embeddings)}
        target = torch.tensor(stacked_gene_expression)  # Gene expression is the target

        return features, target



### Second method 

class SpatialGeneExpressionValidationDataset(Dataset):
    def __init__(self, embedding_file_path, output_dir, k_spatial_neighbors=10):
        """
        Dataset for spatial neighbor extraction based on embeddings.

        Args:
            embedding_file_path (str): Path to file containing precomputed embeddings.
            output_dir (str): Directory to save the neighbor indices CSV.
            k_spatial_neighbors (int): Number of nearest neighbors for spatial coordinates.
        """
        self.embedding_file_path = embedding_file_path
        self.output_dir = output_dir
        self.k_spatial_neighbors = k_spatial_neighbors
        self.csv_file = self._get_csv_file_path()
        self.data = []

        # Load embeddings once
        self.embeddings, self.spatial_coords = self._load_embeddings(embedding_file_path)

        # Precompute neighbor indices once
        self._prepare_data()

    def _get_csv_file_path(self):
        """
        Generate the CSV file path based on embedding file name.
        """
        dataset_name = os.path.basename(self.embedding_file_path).replace(".h5", "")
        return os.path.join(self.output_dir, f"{dataset_name}_neighbors.csv")

    def _load_embeddings(self, embedding_file_path):
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
        Generate spatial neighbor indices and save them to a CSV file.
        """
        n_cells = self.spatial_coords.shape[0]

        # Finding nearest neighbors for spatial data using spatial coordinates from embeddings
        knn_spatial = NearestNeighbors(n_neighbors=self.k_spatial_neighbors + 1, n_jobs=-1).fit(self.spatial_coords)
        _, indices_spatial = knn_spatial.kneighbors(self.spatial_coords)

        # Prepare CSV data for neighbor indices
        csv_data = []
        for i in range(n_cells):
            spatial_indices = indices_spatial[i][1:]  # Exclude the first index (self)

            # Store neighbors in separate columns
            spatial_columns = {f"spatial_n{j+1}": spatial_indices[j] for j in range(self.k_spatial_neighbors)}

            # Combine the main cell index and neighbors in a dictionary
            csv_data.append({
                "main_cell_index": i,
                **spatial_columns
            })

        # Save to CSV
        pd.DataFrame(csv_data).to_csv(self.csv_file, index=False)
        self.data = csv_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the features (embeddings) for a given index along with nearest neighbors' embeddings.
        """
        main_cell_index = self.data[idx]["main_cell_index"]

        # Extract spatial neighbors from precomputed indices
        spatial_indices = [int(self.data[idx].get(f"spatial_n{n+1}", -1)) for n in range(self.k_spatial_neighbors)]

        # Remove invalid indices (-1 indicates missing neighbor data)
        spatial_indices = [i for i in spatial_indices if i >= 0]

        # Extract embeddings
        main_embedding = self.embeddings[main_cell_index]
        spatial_embeddings = self.embeddings[spatial_indices] if spatial_indices else np.empty((0, self.embeddings.shape[1]))

        # Combine embeddings
        embedding_list = [main_embedding]

        if spatial_embeddings.size > 0:
            embedding_list.append(spatial_embeddings)

        # Stack the embeddings
        stacked_embeddings = np.vstack(embedding_list).astype(np.float32)

        # Convert to tensors for PyTorch model
        features = {"stacked_embeddings": torch.tensor(stacked_embeddings)}

        return features





### code for 10 spatial + copied main for validation

class SpatialGeneExpressionValidationCopiedDataset(Dataset):
    def __init__(self, embedding_file_path, output_dir, k_spatial_neighbors=10):
        """
        Dataset for spatial neighbor extraction based on embeddings.

        Args:
            embedding_file_path (str): Path to file containing precomputed embeddings.
            output_dir (str): Directory to save the neighbor indices CSV.
            k_spatial_neighbors (int): Number of nearest neighbors for spatial coordinates.
        """
        self.embedding_file_path = embedding_file_path
        self.output_dir = output_dir
        self.k_spatial_neighbors = k_spatial_neighbors
        self.csv_file = self._get_csv_file_path()
        self.data = []

        # Load embeddings once
        self.embeddings, self.spatial_coords = self._load_embeddings(embedding_file_path)

        # Precompute neighbor indices once
        self._prepare_data()

    def _get_csv_file_path(self):
        """
        Generate the CSV file path based on embedding file name.
        """
        dataset_name = os.path.basename(self.embedding_file_path).replace(".h5", "")
        return os.path.join(self.output_dir, f"{dataset_name}_spatial_neighbors.csv")

    def _load_embeddings(self, embedding_file_path):
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
        Generate spatial neighbor indices and save them to a CSV file.
        """
        n_cells = self.spatial_coords.shape[0]

        # Finding nearest neighbors for spatial data using spatial coordinates from embeddings
        knn_spatial = NearestNeighbors(n_neighbors=self.k_spatial_neighbors + 1, n_jobs=-1).fit(self.spatial_coords)
        _, indices_spatial = knn_spatial.kneighbors(self.spatial_coords)

        # Prepare CSV data for neighbor indices
        csv_data = []
        for i in range(n_cells):
            spatial_indices = indices_spatial[i][1:]  # Exclude the first index (self)

            # Store neighbors in separate columns
            spatial_columns = {f"spatial_n{j+1}": spatial_indices[j] for j in range(self.k_spatial_neighbors)}

            # Combine the main cell index and neighbors in a dictionary
            csv_data.append({
                "main_cell_index": i,
                **spatial_columns
            })

        # Save to CSV
        pd.DataFrame(csv_data).to_csv(self.csv_file, index=False)
        self.data = csv_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the features (embeddings) for a given index along with nearest neighbors' embeddings.
        """
        main_cell_index = self.data[idx]["main_cell_index"]

        # Extract spatial neighbors from precomputed indices
        spatial_indices = [int(self.data[idx].get(f"spatial_n{n+1}", -1)) for n in range(self.k_spatial_neighbors)]

        # Remove invalid indices (-1 indicates missing neighbor data)
        spatial_indices = [i for i in spatial_indices if i >= 0]

        # Extract embeddings
        main_embedding = self.embeddings[main_cell_index]
        spatial_embeddings = self.embeddings[spatial_indices] if spatial_indices else np.empty((0, self.embeddings.shape[1]))

        # Combine embeddings
        embedding_list = [main_embedding]

        if spatial_embeddings.size > 0:
            embedding_list.append(spatial_embeddings)

        # Add the main embedding 5 times
        for _ in range(5):
            embedding_list.append(main_embedding)

        # Stack the embeddings
        stacked_embeddings = np.vstack(embedding_list).astype(np.float32)

        # Convert to tensors for PyTorch model
        features = {"stacked_embeddings": torch.tensor(stacked_embeddings)}

        return features



#Copied test using first function

class SpatialGeneExpressionDatasetVal(Dataset):
    def __init__(self, adata_path, embedding_file_path, output_dir, is_train=True,
                 k_spatial_neighbors=10, k_gene_exp_neighbors=10):
        """
        Dataset for spatial and gene expression data integration.

        Args:
            adata_path (str): Path to AnnData file containing spatial and gene expression data.
            embedding_file_path (str): Path to file containing precomputed embeddings.
            output_dir (str): Directory to save the neighbor indices CSV.
            is_train (bool): Whether the dataset is for training or validation.
            k_spatial_neighbors (int): Number of nearest neighbors for spatial coordinates.
            k_gene_exp_neighbors (int): Number of nearest neighbors for gene expression data.
        """
        self.adata_path = adata_path
        self.embedding_file_path = embedding_file_path
        self.output_dir = output_dir
        self.is_train = is_train
        self.k_spatial_neighbors = k_spatial_neighbors
        self.k_gene_exp_neighbors = k_gene_exp_neighbors
        self.csv_file = self._get_csv_file_path()
        self.data = []

        # Load AnnData and embeddings once
        self.adata = sc.read_h5ad(self.adata_path)
        self.embeddings = self._load_embeddings(embedding_file_path)
        
        # Precompute neighbor indices once
        self._prepare_data()

    def _get_csv_file_path(self):
        """
        Generate the CSV file path based on dataset name and split.
        """
        dataset_name = os.path.basename(self.adata_path).replace(".h5ad", "")
        return os.path.join(self.output_dir, f"{dataset_name}_neighbors.csv")

    def _load_embeddings(self, embedding_file_path):
        """
        Load embeddings from the file.
        """
        with h5py.File(embedding_file_path, 'r') as f:
            return f['embeddings'][:]

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
        spatial_coords = self.adata.obsm['spatial']
        gene_expression = self.adata.X
        n_cells = spatial_coords.shape[0]

        # Finding nearest neighbors for spatial data
        knn_spatial = NearestNeighbors(n_neighbors=self.k_spatial_neighbors + 1, n_jobs=-1).fit(spatial_coords)
        _, indices_spatial = knn_spatial.kneighbors(spatial_coords)

        # Finding nearest neighbors for gene expression data
        knn_expression = NearestNeighbors(n_neighbors=self.k_gene_exp_neighbors + 1, n_jobs=-1).fit(gene_expression)
        _, indices_expression = knn_expression.kneighbors(gene_expression)

        # Prepare CSV data for neighbor indices
        csv_data = []
        for i in range(n_cells):
            spatial_indices = indices_spatial[i][1:]  # Exclude the first index (self)
            expression_indices = indices_expression[i][1:]  # Exclude the first index (self)

            # Store neighbors in separate columns
            spatial_columns = {f"spatial_n{j+1}": spatial_indices[j] for j in range(self.k_spatial_neighbors)}
            
            if self.is_train:
                expression_columns = {f"gene_expression_n{j+1}": expression_indices[j] for j in range(self.k_gene_exp_neighbors)}
            else:
                print(f"indices_spatial: {indices_spatial[i][0]}")
                expression_columns = {f"gene_expression_n{j+1}": indices_spatial[i][0] for j in range(self.k_gene_exp_neighbors)}

            # Combine the main cell index and neighbors in a dictionary
            csv_data.append({
                "main_cell_index": i,
                **spatial_columns,
                **expression_columns,
            })

        # Save to CSV
        pd.DataFrame(csv_data).to_csv(self.csv_file, index=False)
        self.data = csv_data          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the features (embeddings, gene expression data) and target (gene expression) for a given index.
        """
        main_cell_index = self.data[idx]["main_cell_index"]
        
        # Extract spatial and expression neighbors from precomputed indices
        spatial_indices = [int(self.data[idx].get(f"spatial_n{n+1}", -1)) for n in range(self.k_spatial_neighbors)]
        expression_indices = [int(self.data[idx].get(f"gene_expression_n{n+1}", -1)) for n in range(self.k_gene_exp_neighbors)]
        
        # Remove invalid indices (-1 indicates missing neighbor data)
        spatial_indices = [i for i in spatial_indices if i >= 0]
        expression_indices = [i for i in expression_indices if i >= 0]

        # Extract gene expression matrix (keep it sparse if possible)
        gene_expression = self.adata.X

        main_gene_expression = gene_expression[main_cell_index]
        spatial_gene_expression = gene_expression[spatial_indices]
        expression_gene_expression = gene_expression[expression_indices]

        # Handle sparse matrices
        if sparse.issparse(gene_expression):
            main_gene_expression = main_gene_expression.toarray()
            spatial_gene_expression = spatial_gene_expression.toarray()
            expression_gene_expression = expression_gene_expression.toarray()

        # Extract embeddings
        main_embedding = self.embeddings[main_cell_index]
        spatial_embeddings = self.embeddings[spatial_indices] if spatial_indices else np.empty((0, self.embeddings.shape[1]))
        expression_embeddings = self.embeddings[expression_indices] if expression_indices else np.empty((0, self.embeddings.shape[1]))

        # Combine embeddings and gene expression data
        embedding_list = [main_embedding]
        gene_expression_list = [main_gene_expression]

        if spatial_embeddings.size > 0:
            embedding_list.append(spatial_embeddings)
            gene_expression_list.append(spatial_gene_expression)

        if expression_embeddings.size > 0:
            embedding_list.append(expression_embeddings)
            gene_expression_list.append(expression_gene_expression)

        # Stack the embeddings and gene expressions
        stacked_embeddings = np.vstack(embedding_list).astype(np.float32)
        stacked_gene_expression = np.vstack(gene_expression_list).astype(np.float32)
        # print(stacked_embeddings.shape)
        # Convert to tensors for PyTorch model
        features = {"stacked_embeddings": torch.tensor(stacked_embeddings)}
        # print(features["stacked_embeddings"].shape)
        target = torch.tensor(stacked_gene_expression)  # Gene expression is the target

        return features, target


###########################Validation Nearest Neighbour from Training data ###########################################################


class SpatialGeneExpressionDatasetValidationBased(Dataset):
    def __init__(self, adata_path, adata_path_val,embedding_file_path_train,embedding_file_path_val, output_dir, 
                 k_spatial_neighbors=10, k_gene_exp_neighbors=10):
        """
        Dataset for spatial and gene expression data integration.

        Args:
            adata_path (str): Path to AnnData file containing spatial and gene expression data.
            embedding_file_path (str): Path to file containing precomputed embeddings.
            output_dir (str): Directory to save the neighbor indices CSV.
            is_train (bool): Whether the dataset is for training or validation.
            k_spatial_neighbors (int): Number of nearest neighbors for spatial coordinates.
            k_gene_exp_neighbors (int): Number of nearest neighbors for gene expression data.
        """
        self.adata_path = adata_path
        self.adata_path_val = adata_path_val
        self.embedding_file_path_train = embedding_file_path_train
        self.embedding_file_path_val = embedding_file_path_val
        self.output_dir = output_dir
        self.k_spatial_neighbors = k_spatial_neighbors
        self.k_gene_exp_neighbors = k_gene_exp_neighbors
        self.csv_file = self._get_csv_file_path()
        self.data = []

        # Load AnnData and embeddings once
        self.adata = sc.read_h5ad(self.adata_path)
        self.adata_val = sc.read_h5ad(self.adata_path_val)
        self.embeddings_train, self.spatial_coords_train = self._load_embeddings(embedding_file_path_train)
        self.embeddings_val, self.spatial_coords_val = self._load_embeddings(embedding_file_path_val)
        
        # Precompute neighbor indices once
        self._prepare_data()

    def _get_csv_file_path(self):
        """
        Generate the CSV file path based on dataset name and split.
        """
        dataset_name = os.path.basename(self.adata_path).replace(".h5ad", "")
        return os.path.join(self.output_dir, f"{dataset_name}_neighbors.csv")

    def _load_embeddings(self, embedding_file_path):
        """
        Load embeddings from the file.
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

        train_gene_expression = self.adata.X

        n_cells = self.spatial_coords_val.shape[0]

        # Finding nearest neighbors for spatial data

        knn_spatial = NearestNeighbors(n_neighbors=self.k_spatial_neighbors + 1, n_jobs=-1).fit(self.spatial_coords_train)
        _, indices_spatial = knn_spatial.kneighbors(self.spatial_coords_val)


        gene_expressions_for_val = train_gene_expression[indices_spatial[:, 1]]

        # Step 3: Find the nearest neighbors in gene expression space
        knn_expression = NearestNeighbors(n_neighbors=self.k_gene_exp_neighbors + 1, n_jobs=-1).fit(train_gene_expression)
        _, indices_expression = knn_expression.kneighbors(gene_expressions_for_val)


        # Prepare CSV data for neighbor indices
        csv_data = []
        for i in range(n_cells):
            spatial_indices = indices_spatial[i][1:]  # Exclude the first index (self)
            expression_indices = indices_expression[i][1:]  # Exclude the first index (self)

            # Store neighbors in separate columns
            spatial_columns = {f"spatial_n{j+1}": spatial_indices[j] for j in range(self.k_spatial_neighbors)}
            expression_columns = {f"gene_expression_n{j+1}": expression_indices[j] for j in range(self.k_gene_exp_neighbors)}

            # Combine the main cell index and neighbors in a dictionary
            csv_data.append({
                "main_cell_index": i,
                **spatial_columns,
                **expression_columns,
            })

        # Save to CSV
        pd.DataFrame(csv_data).to_csv(self.csv_file, index=False)
        self.data = csv_data          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the features (embeddings, gene expression data) and target (gene expression) for a given index.
        """
        main_cell_index = self.data[idx]["main_cell_index"]
        
        # Extract spatial and expression neighbors from precomputed indices
        spatial_indices = [int(self.data[idx].get(f"spatial_n{n+1}", -1)) for n in range(self.k_spatial_neighbors)]
        expression_indices = [int(self.data[idx].get(f"gene_expression_n{n+1}", -1)) for n in range(self.k_gene_exp_neighbors)]
        
        # Remove invalid indices (-1 indicates missing neighbor data)
        spatial_indices = [i for i in spatial_indices if i >= 0]
        expression_indices = [i for i in expression_indices if i >= 0]

        # Extract gene expression matrix (keep it sparse if possible)
        # gene_expression = self.adata.X
        train_gene_expression  = self.adata.X
        val_gene_expression = self.adata_val.X

        main_gene_expression = val_gene_expression[main_cell_index]
        spatial_gene_expression = train_gene_expression[spatial_indices]
        expression_gene_expression = train_gene_expression[expression_indices]

        # Handle sparse matrices
        if sparse.issparse(val_gene_expression):
            main_gene_expression = main_gene_expression.toarray()
            spatial_gene_expression = spatial_gene_expression.toarray()
            expression_gene_expression = expression_gene_expression.toarray()

        # Extract embeddings
        main_embedding = self.embeddings_val[main_cell_index]
        spatial_embeddings = self.embeddings_train[spatial_indices] if spatial_indices else np.empty((0, self.embeddings_train.shape[1]))
        expression_embeddings = self.embeddings_train[expression_indices] if expression_indices else np.empty((0, self.embeddings_train.shape[1]))

        # Combine embeddings and gene expression data
        embedding_list = [main_embedding]
        gene_expression_list = [main_gene_expression]

        if spatial_embeddings.size > 0:
            embedding_list.append(spatial_embeddings)
            gene_expression_list.append(spatial_gene_expression)

        if expression_embeddings.size > 0:
            embedding_list.append(expression_embeddings)
            gene_expression_list.append(expression_gene_expression)

        # Stack the embeddings and gene expressions
        stacked_embeddings = np.vstack(embedding_list).astype(np.float32)
        stacked_gene_expression = np.vstack(gene_expression_list).astype(np.float32)
        # print(stacked_embeddings.shape)
        # Convert to tensors for PyTorch model
        features = {"stacked_embeddings": torch.tensor(stacked_embeddings)}
        # print(features["stacked_embeddings"].shape)
        target = torch.tensor(stacked_gene_expression)  # Gene expression is the target

        return features, target





########### To get patch embeddings ###############

import torch
from torch.utils.data import Dataset
import h5py
import anndata as ad
import os

class SpatialTranscriptomicsDataset(Dataset):
    def __init__(self, dataset_paths, transform=None):
        """
        Args:
            dataset_paths (list of tuples): List of (h5ad_path, h5_path) pairs.
            transform (callable, optional): Transformations to apply to images.
        """
        self.dataset_paths = dataset_paths
        self.transform = transform
        self.data = []

        # Load HDF5 data into memory
        for h5ad_path, h5_path in self.dataset_paths:
            with h5py.File(h5_path, 'r') as hf:
                imgs = hf["img"][:]  # Assuming images are stored under "img"
            adata = ad.read_h5ad(h5ad_path)  # Read .h5ad file
            
            # Store image-label pairs
            for i in range(len(imgs)):
                self.data.append((imgs[i], adata.X[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # Convert image to tensor and normalize
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label



############################################################ integrated 128 pixel away neighbours #############################################################



class SpatialGeneExpressionIntegrated128pxAwayDataset(Dataset):
    def __init__(self, adata_path, embedding_file_path, output_dir, is_train=True,
                 k_spatial_neighbors=10, k_gene_exp_neighbors=10):
        """
        Dataset for spatial and gene expression data integration.

        Args:
            adata_path (str): Path to AnnData file containing spatial and gene expression data.
            embedding_file_path (str): Path to file containing precomputed embeddings.
            output_dir (str): Directory to save the neighbor indices CSV.
            is_train (bool): Whether the dataset is for training or validation.
            k_spatial_neighbors (int): Number of nearest neighbors for spatial coordinates.
            k_gene_exp_neighbors (int): Number of nearest neighbors for gene expression data.
        """
        self.adata_path = adata_path
        self.embedding_file_path = embedding_file_path
        self.output_dir = output_dir
        self.is_train = is_train
        self.k_spatial_neighbors = k_spatial_neighbors
        self.k_gene_exp_neighbors = k_gene_exp_neighbors
        self.csv_file = self._get_csv_file_path()
        self.data = []

        # Load AnnData and embeddings once
        self.adata = sc.read_h5ad(self.adata_path)
        self.embeddings = self._load_embeddings(embedding_file_path)
        
        # Precompute neighbor indices once
        self._prepare_data()

    def _get_csv_file_path(self):
        """
        Generate the CSV file path based on dataset name and split.
        """
        dataset_name = os.path.basename(self.adata_path).replace(".h5ad", "")
        return os.path.join(self.output_dir, f"{dataset_name}_neighbors.csv")

    def _load_embeddings(self, embedding_file_path):
        """
        Load embeddings from the file.
        """
        with h5py.File(embedding_file_path, 'r') as f:
            return f['embeddings'][:]

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
        spatial_coords = self.adata.obsm['spatial']
        gene_expression = self.adata.X
        n_cells = spatial_coords.shape[0]

        # Query many spatial neighbors to allow filtering
        k_large = 10000  # large number to allow filtering after
        if k_large > n_cells:
            k_large = n_cells
        knn_spatial = NearestNeighbors(n_neighbors=k_large, n_jobs=-1).fit(spatial_coords)
        spatial_dists, spatial_indices_all = knn_spatial.kneighbors(spatial_coords)

        # Gene expression neighbors (no distance threshold needed)
        knn_expression = NearestNeighbors(n_neighbors=self.k_gene_exp_neighbors + 1, n_jobs=-1).fit(gene_expression)
        _, indices_expression = knn_expression.kneighbors(gene_expression)

        target_dist = 32


        csv_data = []

        for i in range(n_cells):
            # Exclude self and apply distance threshold
            filtered_spatial = [
                int(spatial_indices_all[i][j])
                for j in range(1, k_large)  # skip index 0 (self)
                if spatial_dists[i][j] >= target_dist
            ]

            # Take only top-k neighbors after filtering
            filtered_spatial = filtered_spatial[:self.k_spatial_neighbors]

            # Pad with -1 if fewer than k neighbors found
            while len(filtered_spatial) < self.k_spatial_neighbors:
                filtered_spatial.append(-1)

            # Exclude self in gene expression neighbors
            expression_indices = [int(idx) for idx in indices_expression[i][1:self.k_gene_exp_neighbors+1]]

            spatial_columns = {f"spatial_n{j+1}": filtered_spatial[j] for j in range(self.k_spatial_neighbors)}
            expression_columns = {f"gene_expression_n{j+1}": expression_indices[j] for j in range(self.k_gene_exp_neighbors)}

            csv_data.append({
                "main_cell_index": i,
                **spatial_columns,
                **expression_columns,
            })

        pd.DataFrame(csv_data).to_csv(self.csv_file, index=False)
        self.data = csv_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the features (embeddings, gene expression data) and target (gene expression) for a given index.
        """
        main_cell_index = self.data[idx]["main_cell_index"]
        
        # Extract spatial and expression neighbors from precomputed indices
        spatial_indices = [int(self.data[idx].get(f"spatial_n{n+1}", -1)) for n in range(self.k_spatial_neighbors)]
        expression_indices = [int(self.data[idx].get(f"gene_expression_n{n+1}", -1)) for n in range(self.k_gene_exp_neighbors)]
        
        # Remove invalid indices (-1 indicates missing neighbor data)
        spatial_indices = [i for i in spatial_indices if i >= 0]
        expression_indices = [i for i in expression_indices if i >= 0]

        # Extract gene expression matrix (keep it sparse if possible)
        gene_expression = self.adata.X

        main_gene_expression = gene_expression[main_cell_index]
        spatial_gene_expression = gene_expression[spatial_indices]
        expression_gene_expression = gene_expression[expression_indices]

        # Handle sparse matrices
        if sparse.issparse(gene_expression):
            main_gene_expression = main_gene_expression.toarray()
            spatial_gene_expression = spatial_gene_expression.toarray()
            expression_gene_expression = expression_gene_expression.toarray()

        # Extract embeddings
        main_embedding = self.embeddings[main_cell_index]
        spatial_embeddings = self.embeddings[spatial_indices] if spatial_indices else np.empty((0, self.embeddings.shape[1]))
        expression_embeddings = self.embeddings[expression_indices] if expression_indices else np.empty((0, self.embeddings.shape[1]))
        # print("main_embedding:::::::::::::::::::::::::::::::::::::", main_embedding.shape)
        # Combine embeddings and gene expression data
        embedding_list = [main_embedding]
        gene_expression_list = [main_gene_expression]

        if spatial_embeddings.size > 0:
            embedding_list.append(spatial_embeddings)
            gene_expression_list.append(spatial_gene_expression)

        if expression_embeddings.size > 0:
            embedding_list.append(expression_embeddings)
            gene_expression_list.append(expression_gene_expression)

        # Stack the embeddings and gene expressions
        stacked_embeddings = np.vstack(embedding_list).astype(np.float32)

        # print("++++++++++ stacked_embeddings +++++++++", stacked_embeddings.shape)
        stacked_gene_expression = np.vstack(gene_expression_list).astype(np.float32)
        # print(stacked_embeddings.shape)
        # Convert to tensors for PyTorch model
        features = {"stacked_embeddings": torch.tensor(stacked_embeddings)}
        target = torch.tensor(stacked_gene_expression)  # Gene expression is the target

        return features, target



####################################################### Code for  taking patches instead of embeddings ##############################################


class PatchDataset(Dataset):
    def __init__(self, gene_path, img_path, transform=None):
        self.img_path =  img_path
        self.gene_path = gene_path
        self.transform = transform
        self.image = []
        self.gene_data = []
        print(" In Patch Data")
        with h5py.File(self.img_path, 'r') as f:
            self.image = f["img"][:]


        print("image: ", self.image.shape)

        gene_exp = sc.read_h5ad(self.gene_path).X
        if not isinstance(gene_exp, np.ndarray):
            gene_exp = gene_exp.toarray()  # Convert sparse to dense
        self.gene_data = gene_exp  # Store gene data



    def __len__(self):
            return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx].astype(np.uint8)  # Ensure dtype for ToPILImage
        if self.transform:
            image = self.transform(image)  # Transformed to (3, 224, 224)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        gene_exp = torch.tensor(self.gene_data[idx], dtype=torch.float32)  # (460,)

        return image, gene_exp



#################################################### code for taking patch and gene expression with labels ##################################


# class PatchDataset(Dataset):
#     def __init__(self, gene_path, img_path, class_file_path= "/home/raghvendra/home3/raghvendra/Crunch_2/Sumit/XG & CAT Boost Annotation/Data/data.csv", transform=None):
#         self.class_file_path = class_file_path
#         self.img_path =  img_path
#         self.gene_path = gene_path
#         self.transform = transform
#         self.image = []
#         self.gene_data = []
#         print(" In Patch Data")
#         with h5py.File(self.img_path, 'r') as f:
#             self.image = f["img"][:]


#         print("image: ", self.image.shape)

#         adata = sc.read_h5ad(self.gene_path)
#         gene_exp = adata.X
#         if not isinstance(gene_exp, np.ndarray):
#             gene_exp = gene_exp.toarray()  # Convert sparse to dense
#         self.gene_data = gene_exp  # Store gene data

#         df = pd.read_csv(self.class_file_path)
#         all_columns = list(df.columns)
#         gene_cols = [col for col in all_columns if col not in ["annotation", "status"]]

#         X_train = df[gene_cols].values
#         class_labels = df["annotation"].values

#         common_genes = [gene for gene in gene_cols if gene in adata.var_names]

#         # Extract aligned features
#         data_1 = df[common_genes].values
#         commonData = adata[:, common_genes].X.toarray() if not isinstance(adata.X, pd.DataFrame) else adata[:, common_genes].X.values

#         knn = KNeighborsClassifier(n_neighbors=5)
#         knn.fit(data_1, class_labels)

#         self.label2id = {label: idx for idx, label in enumerate(sorted(set(self.predicted_labels)))}
#         self.id2label = {v: k for k, v in self.label2id.items()}
#         self.labels = np.array([self.label2id[label] for label in self.predicted_labels])
                



    # def __len__(self):
    #         return len(self.image)

    # def __getitem__(self, idx):
    #     image = self.image[idx].astype(np.uint8)  # Ensure dtype for ToPILImage
    #     if self.transform:
    #         image = self.transform(image)  # Transformed to (3, 224, 224)
    #     else:
    #         image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

    #     gene_exp = torch.tensor(self.gene_data[idx], dtype=torch.float32)  # (460,)
    #     label = self.labels[idx]

    #     return image, gene_exp, label



