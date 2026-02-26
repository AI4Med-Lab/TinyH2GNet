import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import scipy.sparse as sp
import gc
import ctypes
import torchsort
torch.manual_seed(123)
device = "cuda:0"

def spearmanr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()


# Custom dataset loader
class CustomDataset(Dataset):
    def __init__(self, data, train_columns, target_columns):
        # Load the dataset from a file (assuming CSV format)
        self.input_data = torch.tensor(data[:, train_columns], dtype=torch.float32)
        self.output_data = torch.tensor(data[:, target_columns], dtype=torch.float32)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

# Encoder-Decoder model
class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.BatchNorm1d(hidden_dim*4),
            nn.ReLU(),  # Ensures non-negative outputsinput_data
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim*4, output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Custom loss function
class CustomLoss(nn.Module):
    def __init__(self, l1_lambda=0.1):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1_lambda = l1_lambda

    def forward(self, outputs, targets):
        mse_loss = self.mse(outputs, targets)
        l1_loss = torch.mean(torch.abs(outputs))  # L1 regularization to encourage sparsity
        spearman_loss = 1-spearmanr(outputs, targets, regularization_strength=1.0)

        return mse_loss + self.l1_lambda * l1_loss + spearman_loss

# Function to make weights sparse
def make_sparse(model, device, sparsity=0.1):
    for param in model.parameters():
        if len(param.shape) > 1:  # Only apply to weight matrices, not biases
            mask = torch.rand(param.shape).to(device) > sparsity
            param.data *= mask.float()

# Hyperparameters
input_dim = 455
hidden_dim = 128
output_dim = 16303
batch_size = 8192
learning_rate = 0.001
num_epochs = 100
dropout_rate = 0.1

# +
#Load all the necessary data and annotation information
sparse_big_df = sp.load_npz("data/scRNAseq_matrix.npz")
all_genes = pd.read_csv("data/all_scRNA_gene_labels.csv")
train_genes = pd.read_csv("data/train_gene_labels.csv")
annotation_df = pd.read_csv("data/scRNAseq_annotation_info.csv")

#Get all gene names
all_gene_names = all_genes["gene_symbols"].tolist()
train_gene_names = train_genes["gene_symbols"].tolist()

#Get train gene names
subset_train_gene_names = [gene_name for gene_name in train_gene_names if gene_name in all_gene_names]
print("Total train genes:")
#print(len(subset_train_gene_names))

#Get target gene names
target_gene_names = [gene_name for gene_name in all_gene_names if gene_name not in train_gene_names]
print("Total target genes")
print(len(target_gene_names))
print(target_gene_names)

#Convert the sparse matrix to a dense matrix
big_df = sparse_big_df.todense()

#Get ids of train and target columns
train_columns = [all_gene_names.index(gene_name) for gene_name in subset_train_gene_names]
target_columns = [all_gene_names.index(gene_name) for gene_name in target_gene_names]
print(len(train_columns))

# +
#Save the training dataset
#train_X = pd.DataFrame(big_df[:,train_columns])
#train_X.columns = subset_train_gene_names

#final_train_X = pd.concat([train_X, annotation_df[["annotation","status"]]],axis=1)
#final_train_X.head()
#final_train_X.to_csv("data/training_set_with_annotations.csv",header=True,sep="\t")

# +
# Load and split the dataset
dataset = CustomDataset(big_df, train_columns, target_columns)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

del big_df
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0) 

# +
#Create the train test and validation sets and remove the big data matrix and free memory
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

del dataset
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0) 
# -

#Get the train, validation and test loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model, loss function, and optimizer
model = EncoderDecoder(input_dim, hidden_dim, output_dim, dropout_rate)
criterion = CustomLoss(l1_lambda=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)
print(model)

# Training loop
for epoch in range(num_epochs):
    model.train()
    k=0
    for batch_inputs, batch_targets in train_loader:
        if (k % 5)==0 and k>0:
            print("Training loss after iteration "+str(k)+" is: ",str(loss.item()))
        optimizer.zero_grad()
        outputs = model(batch_inputs.to(device))
        loss = criterion(outputs, batch_targets.to(device))
        loss.backward()
        optimizer.step()
        #make_sparse(model,device)  # Apply sparsity after each update
        k+=1
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:
            outputs = model(batch_inputs.to(device))
            val_loss += criterion(outputs, batch_targets.to(device)).item()
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
    
    torch.cuda.empty_cache()

#Code to save the model
torch.save(model,"Models/Encoder_Decoder_spearman_model_"+str(hidden_dim)+"_"+str(learning_rate)+"_"+str(dropout_rate)+".pt")

# Test the model
model.eval()
test_loss = 0
with torch.no_grad():
    for batch_inputs, batch_targets in test_loader:
        outputs = model(batch_inputs.to(device))
        test_loss += criterion(outputs, batch_targets.to(device)).item()
test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

# +
# Check sparsity of outputs
sparsity = 0
with torch.no_grad():
    #sample_input = next(iter(test_loader))[0]
    for batch_inputs, batch_targets in test_loader:
        outputs = model(batch_inputs.to(device))
        sparsity += (outputs == 0).float().mean()
    sparsity /= len(test_loader)
    print(f"Output sparsity: {sparsity.item():.2%}")

del model, train_loader, test_loader, val_loader, train_dataset, val_dataset, test_dataset, criterion
torch.cuda.empty_cache()
gc.collect()