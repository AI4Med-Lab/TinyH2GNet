import sys
sys.path.append('/code_model_training/models/TCGN')
sys.path.append('/code_model_training/models/THItoGene')

from CMT_block import *
from gnn_block import Graph_Encoding_Block_big, Graph_Encoding_Block
from transformer_block import Channel_Attention
from functools import partial
from collections import OrderedDict

import timm
import torch
import os
import h5py
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import efficientnet_b0
from torchvision.models import densenet121
from torchvision.transforms.functional import crop

from torchvision.models.densenet import _DenseBlock, _Transition
from torchvision.models.efficientnet import EfficientNet_B0_Weights, efficientnet_b0

from transformer import ViT


from utils import *
from vis_model import THItoGene


class THItoGeneModel(nn.Module):
    def __init__(self, num_genes, learning_rate=1e-4, route_dim=64, caps=20, heads=[16, 8], n_layers=4):
        super().__init__()
        self.model = THItoGene(
            n_genes=num_genes,
            learning_rate=learning_rate,
            route_dim=route_dim,
            caps=caps,
            heads=heads,
            n_layers=n_layers,
        )

    def forward(self, patches, centers, adj):
        """
        patches : (B, N, 3, 112, 112)
        centers : (B, N, 2)    # pixel_x, pixel_y
        adj     : (B, N, N)
        """
        return self.model(patches, centers, adj)

########################### STNet ################################################

class STNet(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(STNet, self).__init__()
        densenet = models.densenet121(pretrained=pretrained)

        self.features = densenet.features  # Output: (batch, 1024, 7, 7)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(1024, num_genes)
        
    def forward(self, x):
        features = self.features(x)  # (batch, 1024, 7, 7)
        pooled = self.global_pool(features)  # (batch, 1024, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch, 1024)
        out = self.fc(pooled)  # (batch, num_genes)
        return out

########################### EfftNet ################################################

class EfficientNet(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(EfficientNet, self).__init__()
        
        # Load EfficientNet backbone
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Remove classifier and replace with custom regression head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # Remove original classifier
        
        # Regression head for gene expression
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(1024, num_genes)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.regressor(features)
        return output


###################################### DeepSpaCe #####################################################


class Custom_VGG16(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
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
            nn.Linear(4096, num_genes)  # Output: 460 genes
        )
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
    
###################################### HisToGene #####################################################


class HisToGene(nn.Module):
    def __init__(self, patch_size=16, n_layers=4, n_genes=1000, dim=1024, dropout=0.1, n_pos=64):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim

        patch_dim = 3 * patch_size * patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)

        self.vit = ViT(
            dim=dim,
            depth=n_layers,
            heads=16,
            mlp_dim=2 * dim,
            dropout=dropout,
            emb_dropout=dropout
        )

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, images):
        """
        images: [B, 3, 224, 224]
        Automatically splits into patches and generates centers
        """
        B, C, H, W = images.shape
        patch_size = self.patch_size

        # Split image into non-overlapping patches
        patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
        patches = patches.transpose(1, 2)  # [B, N_patches, 3*patch_size*patch_size]

        # Compute coordinates for each patch
        grid_size = int((patches.shape[1]) ** 0.5)
        coords = torch.stack(torch.meshgrid(
            torch.arange(grid_size, device=images.device),
            torch.arange(grid_size, device=images.device),
            indexing='ij'
        ), dim=-1).reshape(-1, 2).unsqueeze(0).repeat(B, 1, 1)  # [B, N_patches, 2]

        # Positional embeddings
        coords = torch.clamp(coords, 0, self.x_embed.num_embeddings - 1)
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(coords[:, :, 0])
        centers_y = self.y_embed(coords[:, :, 1])

        x = patches + centers_x + centers_y
        h = self.vit(x)
        x = self.gene_head(h)
        x = x.mean(dim=1)  # mean pooling across patches
        return x
    
    
###################################### #####################################################

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))



class TCGN(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=785, embed_dims=[46, 92, 184, 368], stem_channel=16,
                 fc_dim=1280,
                 num_heads=[1, 2, 4, 8], mlp_ratios=[3.6, 3.6, 3.6, 3.6], qkv_bias=True, qk_scale=None,
                 representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, hybrid_backbone=None, norm_layer=None,
                 depths=[2, 2, 10, 2], qk_ratio=1, sr_ratios=[8, 4, 2, 1], dp=0.1):
        super().__init__()
        self.fc_dim = fc_dim
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.stem_conv1 = nn.Conv2d(3, stem_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.patch_embed_a = PatchEmbed(
            img_size=img_size // 2, patch_size=2, in_chans=stem_channel, embed_dim=embed_dims[0])
        self.patch_embed_b = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed_c = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed_d = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.relative_pos_a = nn.Parameter(torch.randn(
            num_heads[0], self.patch_embed_a.num_patches,
            self.patch_embed_a.num_patches // sr_ratios[0] // sr_ratios[0]))
        self.relative_pos_b = nn.Parameter(torch.randn(
            num_heads[1], self.patch_embed_b.num_patches,
            self.patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1]))
        self.relative_pos_c = nn.Parameter(torch.randn(
            num_heads[2], self.patch_embed_c.num_patches,
            self.patch_embed_c.num_patches // sr_ratios[2] // sr_ratios[2]))
        self.relative_pos_d = nn.Parameter(torch.randn(
            num_heads[3], self.patch_embed_d.num_patches,
            self.patch_embed_d.num_patches // sr_ratios[3] // sr_ratios[3]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks_a = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        cur += depths[0]
        self.blocks_b = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        self.blocks_c = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        self.blocks_d = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, representation_size)),
                ('act', nn.Tanh())#('act', nn.GELU())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Graph learning
        self.gnn_block0 = Graph_Encoding_Block(img_size=56, patch_size=4, num_feature_in=embed_dims[0],
                                               embed_dim=embed_dims[0]*2, num_feature_graph_hidden=embed_dims[0]*2
                                               , num_feature_out=48, flatten=False, num_heads=2)
        self.channel_attention0 = Channel_Attention(num_nodes=196)

        self.gnn_block1 = Graph_Encoding_Block(img_size=28, patch_size=2, num_feature_in=embed_dims[1],
                                               embed_dim=embed_dims[1]*2, num_feature_graph_hidden=embed_dims[1]*2
                                               , num_feature_out=48, flatten=False, num_heads=2)
        self.channel_attention1 = Channel_Attention(num_nodes=196)

        # Classifier head
        self._fc = nn.Conv2d(embed_dims[-1], fc_dim, kernel_size=1)
        self._bn = nn.BatchNorm2d(fc_dim, eps=1e-5)
        self._swish = MemoryEfficientSwish()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._drop = nn.Dropout(dp)
        self.head = nn.Sequential(nn.Linear(fc_dim, fc_dim*4, bias=True),nn.GELU(),nn.Linear(fc_dim*4,num_classes, bias=True))
        # nn.Linear(fc_dim, num_classes, bias=True) if num_classes > 0 else nn.Identity()
        # nn.Sequential(nn.Linear(fc_dim, fc_dim*4, bias=True),nn.Linear(fc_dim*4,num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Attention):
                m.update_temperature()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.fc_dim, num_classes, bias=True) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)

        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)

        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)

        x, (H, W) = self.patch_embed_a(x)
        for i, blk in enumerate(self.blocks_a):
            x = blk(x, H, W, self.relative_pos_a)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        graph_feature0 = self.gnn_block0(x,self.relative_pos_a)

        x, (H, W) = self.patch_embed_b(x)
        for i, blk in enumerate(self.blocks_b):
            x = blk(x, H, W, self.relative_pos_b)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        graph_feature1 = self.gnn_block1(x, self.relative_pos_c)

        x, (H, W) = self.patch_embed_c(x)

        x = self.channel_attention0(x, graph_feature0)


        for i, blk in enumerate(self.blocks_c):
            x = blk(x, H, W, self.relative_pos_c)

        x=self.channel_attention1(x,graph_feature1)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x, (H, W) = self.patch_embed_d(x)

        for i, blk in enumerate(self.blocks_d):
            x = blk(x, H, W, self.relative_pos_d)

        #print("+", end="")

        B, N, C = x.shape
        x = self._fc(x.permute(0, 2, 1).reshape(B, C, H, W))
        x = self._bn(x)
        x = self._swish(x)
        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._drop(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

##########################################################################################    
    
from torchvision.models import mobilenet_v2

class MobileNetV2Regressor(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super().__init__()
        base = mobilenet_v2(pretrained=pretrained)
        
        self.backbone = base.features  # feature extractor
        in_features = base.classifier[1].in_features

        # Custom regression head
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_genes)
        )

    def forward(self, x):
        features = self.backbone(x)               # (B, C, H, W)
        pooled = features.mean([2, 3])            # Global average pooling
        out = self.regressor(pooled)              # (B, num_genes)
        return out