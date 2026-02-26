import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import ConcatDataset
from scipy.stats import pearsonr, ConstantInputWarning
from scipy.spatial.distance import cdist
import torchvision.models as models
# import torchvision.models as efficientnet_b4, vgg16
from torch.nn import DataParallel
from torchvision import models, transforms
import torchvision.transforms as T  

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import math


## for UNI
from timm import create_model
from huggingface_hub import hf_hub_download

class CNN_regression_model(nn.Module):
    def __init__(self, input_height=16, input_width=1024, output_size=460):
        super(CNN_regression_model, self).__init__()

        # Set kernel sizes
        kernel_size_1 = (8, 1)
        kernel_size_2 = (1, 100)
        kernel_size_3 = (3, 3)
        pool_size_1 = (1, 2)  # Max pooling size
        pool_size_2 = (1, 10)
        pool_size_3 = (2, 2)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=kernel_size_1)
        self.pool1 = nn.MaxPool2d(pool_size_1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=kernel_size_2)
        self.pool2 = nn.MaxPool2d(pool_size_2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size_3)
        self.pool3 = nn.MaxPool2d(pool_size_3)
        self.bn3 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=kernel_size_3)

        # Compute output dimensions after convolutions and pooling
        conv1_out_height, conv1_out_width = self.compute_conv_output_size(input_height, input_width, kernel_size_1)
        # print(f"conv1 output size: ({conv1_out_height}, {conv1_out_width})")
        
        pool1_out_height, pool1_out_width = self.compute_pool_output_size(conv1_out_height, conv1_out_width, pool_size_1)
        # print(f"pool1 output size: ({pool1_out_height}, {pool1_out_width})")

        conv2_out_height, conv2_out_width = self.compute_conv_output_size(pool1_out_height, pool1_out_width, kernel_size_2)
        # print(f"conv2 output size: ({conv2_out_height}, {conv2_out_width})")
        
        pool2_out_height, pool2_out_width = self.compute_pool_output_size(conv2_out_height, conv2_out_width, pool_size_2)
        # print(f"pool2 output size: ({pool2_out_height}, {pool2_out_width})")

        conv3_out_height, conv3_out_width = self.compute_conv_output_size(pool2_out_height, pool2_out_width, kernel_size_3)
        # print(f"conv3 output size: ({conv3_out_height}, {conv3_out_width})")

        pool3_out_height, pool3_out_width = self.compute_pool_output_size(conv3_out_height, conv3_out_width, pool_size_3)

        # Dynamically calculate flattened size based on actual output dimensions after conv3
        self.flattened_size = 64 * pool3_out_height * pool3_out_width  # Update flattened size
        # print(f"Flattened size: {self.flattened_size}")

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1024)  # Use calculated flattened_size
        self.bn_fc = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, output_size)

    def compute_conv_output_size(self, input_height, input_width, kernel_size, stride=1, padding=0):
        # Formula to calculate output size after convolution or pooling
        output_height = (input_height - kernel_size[0] + 2 * padding) // stride + 1
        output_width = (input_width - kernel_size[1] + 2 * padding) // stride + 1
        return output_height, output_width

    def compute_pool_output_size(self, input_height, input_width, kernel_size, padding=0):
        # Formula to calculate output size after convolution or pooling with stride=2
        output_height = (input_height - kernel_size[0] + 2 * padding) // kernel_size[0] + 1
        output_width = (input_width - kernel_size[1] + 2 * padding) // kernel_size[1] + 1
        
        return output_height, output_width

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"Shape after conv1: {x.shape}")  # Debugging shape
        
        x = self.pool1(x)
        # print(f"Shape after pool1: {x.shape}")  # Debugging shape

        x = F.relu(self.bn2(self.conv2(x)))
        # print(f"Shape after conv2: {x.shape}")  # Debugging shape
        
        x = self.pool2(x)
        # print(f"Shape after pool2: {x.shape}")  # Debugging shape

        # x = F.relu(self.conv3(x))
        # print(f"Shape after conv3: {x.shape}")  # Debugging shape

        x = F.relu(self.bn3(self.conv3(x)))
        # print(f"Shape after conv2: {x.shape}")  # Debugging shape
        
        x = self.pool3(x)
        # print(f"Shape after pool2: {x.shape}")  # Debugging shape


        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(f"Shape after flatten: {x.shape}")  # Debugging shape

        x = F.relu(self.bn_fc(self.fc1(x)))
        # print(f"Shape after fc1: {x.shape}")  # Debugging shape
        
        x = self.fc2(x)
        return x
        
class RegressionModel(nn.Module):
    def __init__(self, input_size=2048 * 11, output_size=460):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                    # Input flattening layer
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),     # First hidden layer
            nn.ReLU(),               
            nn.Linear(2048, 1024),           # Third hidden layer
            nn.BatchNorm1d(1024),     # First hidden layer
            nn.ReLU(),
            nn.Linear(1024, output_size)      # Output layer
        )

    def forward(self, x):
        return self.model(x)


class CNN_regression_reduced_model(nn.Module):
    def __init__(self, input_height=16, input_width=1024, output_size=460):
        super(CNN_regression_reduced_model, self).__init__()

        # Set kernel sizes
        kernel_size_1 = (8, 1)
        kernel_size_2 = (1, 100)
        kernel_size_3 = (3, 3)
        pool_size_1 = (1, 2)  # Max pooling size
        pool_size_2 = (1, 10)
        pool_size_3 = (2, 2)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=kernel_size_1)
        self.pool1 = nn.MaxPool2d(pool_size_1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size_2)
        self.pool2 = nn.MaxPool2d(pool_size_2)
        self.bn2 = nn.BatchNorm2d(128)

        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size_3)
        # self.pool3 = nn.MaxPool2d(pool_size_3)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=kernel_size_3)

        # Compute output dimensions after convolutions and pooling
        conv1_out_height, conv1_out_width = self.compute_conv_output_size(input_height, input_width, kernel_size_1)
        # print(f"conv1 output size: ({conv1_out_height}, {conv1_out_width})")
        
        pool1_out_height, pool1_out_width = self.compute_pool_output_size(conv1_out_height, conv1_out_width, pool_size_1)
        # print(f"pool1 output size: ({pool1_out_height}, {pool1_out_width})")

        conv2_out_height, conv2_out_width = self.compute_conv_output_size(pool1_out_height, pool1_out_width, kernel_size_2)
        # print(f"conv2 output size: ({conv2_out_height}, {conv2_out_width})")
        
        pool2_out_height, pool2_out_width = self.compute_pool_output_size(conv2_out_height, conv2_out_width, pool_size_2)
        # print(f"pool2 output size: ({pool2_out_height}, {pool2_out_width})")

        # conv3_out_height, conv3_out_width = self.compute_conv_output_size(pool2_out_height, pool2_out_width, kernel_size_3)
        # print(f"conv3 output size: ({conv3_out_height}, {conv3_out_width})")

        # pool3_out_height, pool3_out_width = self.compute_pool_output_size(conv3_out_height, conv3_out_width, pool_size_3)

        # Dynamically calculate flattened size based on actual output dimensions after conv3
        self.flattened_size = 128 * pool2_out_height * pool2_out_width  # Update flattened size
        # print(f"Flattened size: {self.flattened_size}")

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1024)  # Use calculated flattened_size
        self.bn_fc = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, output_size)

    def compute_conv_output_size(self, input_height, input_width, kernel_size, stride=1, padding=0):
        # Formula to calculate output size after convolution or pooling
        output_height = (input_height - kernel_size[0] + 2 * padding) // stride + 1
        output_width = (input_width - kernel_size[1] + 2 * padding) // stride + 1
        return output_height, output_width

    def compute_pool_output_size(self, input_height, input_width, kernel_size, padding=0):
        # Formula to calculate output size after convolution or pooling with stride=2
        output_height = (input_height - kernel_size[0] + 2 * padding) // kernel_size[0] + 1
        output_width = (input_width - kernel_size[1] + 2 * padding) // kernel_size[1] + 1
        
        return output_height, output_width

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"Shape after conv1: {x.shape}")  # Debugging shape
        
        x = self.pool1(x)
        # print(f"Shape after pool1: {x.shape}")  # Debugging shape

        x = F.relu(self.bn2(self.conv2(x)))
        # print(f"Shape after conv2: {x.shape}")  # Debugging shape
        
        x = self.pool2(x)
        # print(f"Shape after pool2: {x.shape}")  # Debugging shape

        # x = F.relu(self.conv3(x))
        # print(f"Shape after conv3: {x.shape}")  # Debugging shape

        # x = F.relu(self.bn3(self.conv3(x)))
        # print(f"Shape after conv2: {x.shape}")  # Debugging shape
        
        # x = self.pool3(x)
        # print(f"Shape after pool2: {x.shape}")  # Debugging shape


        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(f"Shape after flatten: {x.shape}")  # Debugging shape

        x = F.relu(self.bn_fc(self.fc1(x)))
        # print(f"Shape after fc1: {x.shape}")  # Debugging shape
        
        x = self.fc2(x)
        return x


############# New Model


class CNN_regression_model_three_fc(nn.Module):
    def __init__(self, input_height=16, input_width=1024, output_size=460):
        super(CNN_regression_model_three_fc, self).__init__()

        # Set kernel sizes
        kernel_size_1 = (8, 1)
        kernel_size_2 = (1, 100)
        kernel_size_3 = (3, 3)
        pool_size_1 = (1, 2)  # Max pooling size
        pool_size_2 = (1, 10)
        pool_size_3 = (2, 2)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=kernel_size_1)
        self.pool1 = nn.MaxPool2d(pool_size_1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size_2)
        self.pool2 = nn.MaxPool2d(pool_size_2)
        self.bn2 = nn.BatchNorm2d(128)

        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size_3)
        # self.pool3 = nn.MaxPool2d(pool_size_3)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=kernel_size_3)

        # Compute output dimensions after convolutions and pooling
        conv1_out_height, conv1_out_width = self.compute_conv_output_size(input_height, input_width, kernel_size_1)
        # print(f"conv1 output size: ({conv1_out_height}, {conv1_out_width})")
        
        pool1_out_height, pool1_out_width = self.compute_pool_output_size(conv1_out_height, conv1_out_width, pool_size_1)
        # print(f"pool1 output size: ({pool1_out_height}, {pool1_out_width})")

        conv2_out_height, conv2_out_width = self.compute_conv_output_size(pool1_out_height, pool1_out_width, kernel_size_2)
        # print(f"conv2 output size: ({conv2_out_height}, {conv2_out_width})")
        
        pool2_out_height, pool2_out_width = self.compute_pool_output_size(conv2_out_height, conv2_out_width, pool_size_2)
        # print(f"pool2 output size: ({pool2_out_height}, {pool2_out_width})")

        # conv3_out_height, conv3_out_width = self.compute_conv_output_size(pool2_out_height, pool2_out_width, kernel_size_3)
        # print(f"conv3 output size: ({conv3_out_height}, {conv3_out_width})")

        # pool3_out_height, pool3_out_width = self.compute_pool_output_size(conv3_out_height, conv3_out_width, pool_size_3)

        # Dynamically calculate flattened size based on actual output dimensions after conv3
        self.flattened_size = 128 * pool2_out_height * pool2_out_width  # Update flattened size
        # print(f"Flattened size: {self.flattened_size}")

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 4096)  # Use calculated flattened_size
        self.bn_fc = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.bn_fc_1 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, output_size)

    def compute_conv_output_size(self, input_height, input_width, kernel_size, stride=1, padding=0):
        # Formula to calculate output size after convolution or pooling
        output_height = (input_height - kernel_size[0] + 2 * padding) // stride + 1
        output_width = (input_width - kernel_size[1] + 2 * padding) // stride + 1
        return output_height, output_width

    def compute_pool_output_size(self, input_height, input_width, kernel_size, padding=0):
        # Formula to calculate output size after convolution or pooling with stride=2
        output_height = (input_height - kernel_size[0] + 2 * padding) // kernel_size[0] + 1
        output_width = (input_width - kernel_size[1] + 2 * padding) // kernel_size[1] + 1
        
        return output_height, output_width

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"Shape after conv1: {x.shape}")  # Debugging shape
        
        x = self.pool1(x)
        # print(f"Shape after pool1: {x.shape}")  # Debugging shape

        x = F.relu(self.bn2(self.conv2(x)))
        # print(f"Shape after conv2: {x.shape}")  # Debugging shape
        
        x = self.pool2(x)
        # print(f"Shape after pool2: {x.shape}")  # Debugging shape

        # x = F.relu(self.conv3(x))
        # print(f"Shape after conv3: {x.shape}")  # Debugging shape

        # x = F.relu(self.bn3(self.conv3(x)))
        # print(f"Shape after conv2: {x.shape}")  # Debugging shape
        
        # x = self.pool3(x)
        # print(f"Shape after pool2: {x.shape}")  # Debugging shape


        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(f"Shape after flatten: {x.shape}")  # Debugging shape

        x = F.relu(self.bn_fc(self.fc1(x)))
        # print(f"Shape after fc1: {x.shape}")  # Debugging shape
        x = F.relu(self.bn_fc_1(self.fc2(x)))
        
        x = self.fc3(x)
        return x




############################################# ST-Net ##################################

class STNet(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(STNet, self).__init__()
        # Load DenseNet-121 backbone
        densenet = models.densenet121(pretrained=pretrained)
        # Remove the classifier (fc) layer
        self.features = densenet.features  # Output: (batch, 1024, 7, 7)
        
        # Global average pooling to get (batch, 1024)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer to predict gene expression
        self.fc = nn.Linear(1024, num_genes)
        
    def forward(self, x):
        # x: (batch, 3, 224, 224)
        features = self.features(x)  # (batch, 1024, 7, 7)
        pooled = self.global_pool(features)  # (batch, 1024, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch, 1024)
        out = self.fc(pooled)  # (batch, num_genes)
        return out


######################################### Custom Resnet ################################################

class Custom_ResNet(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(Custom_ResNet, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Up to conv5_x block

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (batch, 2048, 1, 1)
        
        # Two-layer MLP head
        self.fc1 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, num_genes)
        
    def forward(self, x):
        # x: (batch, 3, 224, 224)
        features = self.features(x)               # (batch, 2048, 7, 7)
        pooled = self.global_pool(features)       # (batch, 2048, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch, 2048)
        
        x = self.fc1(pooled)                      # (batch, 1024)
        x = self.relu(x)
        out = self.fc2(x)                         # (batch, num_genes)
        return out

############################### UNI #################################################################

class Custom_UNI(nn.Module):
    def __init__(self, pretrained_weights_path=None, num_genes=460):
        super(Custom_UNI, self).__init__()
        self.backbone = create_model(
            "vit_large_patch16_224",
            pretrained=False,  # we'll load our custom pretrained weights
            num_classes=0,     # remove classification head
        )
        if pretrained_weights_path is not None:
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            # Remove "module." if saved with DataParallel
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=False)

        self.regressor = nn.Linear(self.backbone.num_features, num_genes)

    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features)


############################################# EfficientNet ###########################################
class EfficientNet(nn.Module):
    def __init__(self, out_features=460, pretrained=True):
        super(EfficientNet, self).__init__()
        
        # Load EfficientNet backbone
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        print(f"No pretrained weights:::: {pretrained}")

        # Remove classifier and replace with custom regression head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # Remove original classifier
        
        # Regression head for gene expression
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(1024, out_features)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.regressor(features)
        return output





#########################################################################################

class EfficientNetB4GeneRegressor(nn.Module):
    def __init__(self, num_outputs=460):
        super(EfficientNetB4GeneRegressor, self).__init__()
        self.backbone = efficientnet_b4(pretrained=True)
        
        # Get number of input features from the classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace the classification head with a regression head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_outputs)
        )

    def forward(self, x):
        return self.backbone(x)


###########################################################################################


class Custom_VGG16(nn.Module):
    def __init__(self, output_dim=460, pretrained=True):
        super(Custom_VGG16, self).__init__()
        
        # Load VGG16 backbone
        vgg = models.vgg16(pretrained=pretrained)
        
        # Remove original classifier (FC layers)
        self.features = vgg.features  # convolutional layers
        
        # Define a custom regressor
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),  # VGG16 default flatten size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)  # Output: 460 genes
        )
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x



#############################################################  New stratgy for crunch  ##########################################


class GeneGNN(nn.Module):
    def __init__(self, gene_graph, in_dim=1, hidden_dim=640, out_dim=1280):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        self.graph = gene_graph  # torch_geometric.data.Data with edge_index

    def forward(self, x):  # x: [batch_size, num_genes]
        x = x.unsqueeze(-1)  # -> [batch_size, num_genes, 1]
        batch_size, num_genes, _ = x.size()
        
        x = x.view(-1, 1)  # [batch_size * num_genes, 1]
        edge_index = self.graph.edge_index

        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = x.view(batch_size, num_genes, -1)  # [batch_size, num_genes, out_dim]
        return x.mean(dim=1)  # [batch_size, out_dim]

class EfficientNetB0(nn.Module):
    def __init__(self, out_dim=1280, final_dim=460, pretrained=True):
        super(EfficientNetB0, self).__init__()
        
        # Load EfficientNet backbone
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        self.backbone.classifier = nn.Identity()  # Remove original classifier
        self.projector = nn.Linear(1280, out_dim)
        self.final_layer = nn.Linear(out_dim, final_dim)  # Final prediction
        
    def forward(self, x):
        feat = self.backbone(x)  # [B, 1280]
        emb = self.projector(feat)  # [B, 1280]
        out = self.final_layer(emb)
        # print("emb shape:", emb.shape)
        # print("out shape:", out.shape)
        
        return emb, out

# class ContrastiveGeneModel(nn.Module):
#     def __init__(self, gene_graph, temperature=0.07):
#         super().__init__()
#         self.gene_gnn = GeneGNN(gene_graph)
#         self.image_branch = EfficientNetB0()
#         self.temperature = temperature

#     def contrastive_loss(self, I_t, T_t):
#         # I_t, T_t: [B, 1280]
#         logits = I_t @ T_t.T / self.temperature  # [B, B]
#         labels = torch.arange(I_t.size(0)).to(I_t.device)  # Ground truth diag
#         loss_I = F.cross_entropy(logits, labels)
#         loss_T = F.cross_entropy(logits.T, labels)
#         return (loss_I + loss_T) / 2.0

#     def forward(self, gene_input=None, image_input=None, mode="train"):
#         if mode == "train":
#             gene_emb = self.gene_gnn(gene_input)  # [B, 1280]
#             image_emb, image_pred = self.image_branch(image_input)  # [B, 1280], [B, 460]
#             return gene_emb, image_emb, image_pred
#         elif mode == "test":
#             _, image_pred = self.image_branch(image_input)  # Only image-to-gene
#             return image_pred


class ContrastiveGeneModel(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.gene_proj = nn.Sequential(
            nn.Linear(460, 1280),
            nn.ReLU(),
            # nn.Linear(640, 1280)
        )
        self.image_branch = EfficientNetB0()
        self.temperature = temperature

    def contrastive_loss(self, I_t, T_t):
        # I_t, T_t: [B, 1280]
        logits = I_t @ T_t.T / self.temperature  # [B, B]
        labels = torch.arange(I_t.size(0)).to(I_t.device)  # Ground truth diag
        loss_I = F.cross_entropy(logits, labels)
        loss_T = F.cross_entropy(logits.T, labels)
        return (loss_I + loss_T) / 2.0

    def forward(self, gene_input=None, image_input=None, mode="train"):
        if mode == "train":
            gene_emb = self.gene_proj(gene_input)  # [B, 1280]
            print(image_input.shape)
            image_emb, image_pred = self.image_branch(image_input)  # [B, 1280], [B, 460]
            return gene_emb, image_emb, image_pred
        elif mode == "test":
            _, image_pred = self.image_branch(image_input)  # Only image-to-gene
            return image_pred


############################################################################################################