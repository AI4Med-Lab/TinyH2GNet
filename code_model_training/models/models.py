import sys
sys.path.append('/home/puneet/mk/code_model_training/models/TCGN')
sys.path.append('/home/puneet/mk/code_model_training/models/THItoGene')

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
    

class EfficientNetB4GeneRegressor(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(EfficientNetB4GeneRegressor, self).__init__()
        self.backbone = models.efficientnet_b4(pretrained=False)
        
        # Get number of input features from the classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace the classification head with a regression head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_genes)
        )

    def forward(self, x):
        return self.backbone(x)


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
 
 
 
 
   
    
######################  EfficientNet_GeneCaptionContrastive  ######################### 

class EfficientNet_GeneCaptionContrastive(nn.Module):
    """
    EfficientNet-B0 + Biological Bottleneck + Contrastive Regularization
    Works for any gene count (171 / 250 / 560 / 845)
    """

    def __init__(
        self,
        num_genes,
        pretrained=True,
        contrastive_weight=0.1,
        temperature=0.07
    ):
        super().__init__()

        self.num_genes = num_genes
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

        # =========================
        # Auto dimension scaling
        # =========================
        self.latent_dim = min(256, max(64, num_genes // 2))
        self.proj_dim = min(256, max(64, num_genes // 4))

        print(f"[INFO] Using latent_dim={self.latent_dim}, proj_dim={self.proj_dim} for {num_genes} genes")

        # =========================
        # EfficientNet-B0 Backbone
        # =========================
        backbone = models.efficientnet_b0(pretrained=pretrained)
        self.backbone = backbone.features            # (B, 1280, H', W')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.enc_dim = 1280

        # =========================
        # Biological Bottleneck
        # =========================
        self.bottleneck = nn.Sequential(
            nn.Linear(self.enc_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # =========================
        # Gene Decoder (Caption Head)
        # =========================
        self.gene_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, max(256, self.latent_dim)),
            nn.ReLU(),
            nn.Linear(max(256, self.latent_dim), num_genes)
        )

        # =========================
        # Projection Heads (Contrastive)
        # =========================
        self.img_proj = nn.Sequential(
            nn.Linear(self.enc_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim)
        )

        self.gene_proj = nn.Sequential(
            nn.Linear(num_genes, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim)
        )

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, images, gene_values=None):
        """
        images: (B,3,224,224)
        gene_values:  (B,G) or None

        returns:
            gene_pred: (B,G)
            contrastive_loss: scalar or None
        """

        B = images.size(0)

        # -------- Image Encoding --------
        feat = self.backbone(images)                 # (B,1280,H',W')
        feat = self.pool(feat).view(B, -1)           # (B,1280)

        # -------- Bottleneck --------
        z = self.bottleneck(feat)                    # (B,latent_dim)

        # -------- Gene Prediction --------
        gene_pred = self.gene_decoder(z)             # (B,num_genes)

        # -------- Contrastive --------
        contrastive_loss = None
        if gene_values is not None:
            img_emb = self.img_proj(feat)            # (B,proj_dim)
            gene_emb = self.gene_proj(gene_values)         # (B,proj_dim)

            contrastive_loss = self.info_nce(img_emb, gene_emb)

        return gene_pred, contrastive_loss

    # ============================================================
    # InfoNCE Loss
    # ============================================================
    def info_nce(self, img_emb, gene_emb):
        img_emb = F.normalize(img_emb, dim=1)
        gene_emb = F.normalize(gene_emb, dim=1)

        logits = torch.matmul(img_emb, gene_emb.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i2g = F.cross_entropy(logits, labels)
        loss_g2i = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss_i2g + loss_g2i)
    
######################  EfficientNet_GeneCaptionContrastive  #########################     


def compute_sobel_channels(image_tensor):
    # image_tensor: (B, 3, H, W) normalized
    sobel_kernel_x = torch.tensor([[[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]]], dtype=torch.float32)
    sobel_kernel_y = torch.tensor([[[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]]], dtype=torch.float32)

    sobel_kernel_x = sobel_kernel_x.expand(3, 1, 3, 3)
    sobel_kernel_y = sobel_kernel_y.expand(3, 1, 3, 3)

    if image_tensor.is_cuda:
        sobel_kernel_x = sobel_kernel_x.cuda()
        sobel_kernel_y = sobel_kernel_y.cuda()

    grad_x = F.conv2d(image_tensor, sobel_kernel_x, padding=1, groups=3)
    grad_y = F.conv2d(image_tensor, sobel_kernel_y, padding=1, groups=3)

    grad_x = grad_x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    grad_y = grad_y.mean(dim=1, keepdim=True)

    return grad_x, grad_y


class GEMResNet18(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super().__init__()
        base = resnet18(pretrained=pretrained)

        # Modify first conv layer
        old_conv = base.conv1
        new_conv = nn.Conv2d(5, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight  # Copy RGB weights
            new_conv.weight[:, 3:] = old_conv.weight[:, :2]  # Init gradients

        base.conv1 = new_conv
        self.backbone = base

        # Replace final FC with gene regressor
        self.backbone.fc = nn.Sequential(
            nn.Linear(base.fc.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_genes)
        )

    def forward(self, x_rgb):
        grad_x, grad_y = compute_sobel_channels(x_rgb)
        x = torch.cat([x_rgb, grad_x, grad_y], dim=1)  # (B, 5, H, W)
        return self.backbone(x)


class GEMEfficientNetB0(nn.Module):
    def __init__(self, num_genes=460, pretrained=True):
        super(GEMEfficientNetB0, self).__init__()
        base = efficientnet_b0(pretrained=pretrained)

        # Modify first conv layer for 5-channel input
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(5, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight  # Copy RGB weights
            new_conv.weight[:, 3:] = old_conv.weight[:, :2]  # Init Sobel channels

        base.features[0][0] = new_conv
        self.encoder = base

        # Custom regression head
        in_features = base.classifier[1].in_features
        self.regressor = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_genes)
        )

    def forward(self, x_rgb):
        grad_x, grad_y = compute_sobel_channels(x_rgb)
        x = torch.cat([x_rgb, grad_x, grad_y], dim=1)  # (B, 5, H, W)
        features = self.encoder.features(x)
        features = self.encoder.avgpool(features)
        features = torch.flatten(features, 1)
        output = self.regressor(features)
        return output

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


####################################################################################


def extract_center_crops(x, sizes=[64, 224]):
    """
    Extract center crops of sizes from a batch of 224x224 patches.

    Args:
        x: Tensor of shape (B, C, 224, 224)
        sizes: List of crop sizes

    Returns:
        Dict of {size: tensor of shape (B, C, size, size)}
    """
    B, C, H, W = x.shape
    assert H == 224 and W == 224, "Input must be 224x224"

    crops = {}
    for size in sizes:
        top = (H - size) // 2
        left = (W - size) // 2
        crops[size] = crop(x, top, left, size, size)
    return crops


class ScaleAttentionFusion(nn.Module):
    def __init__(self, in_dim=1024, num_scales=2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, features):
        # features: list of [B, C]
        stacked = torch.stack(features, dim=1)  # [B, S, C]
        attn_weights = self.attn(stacked)       # [B, S, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        fused = (stacked * attn_weights).sum(dim=1)  # [B, C]
        return fused


class HierarchicalDenseNet(nn.Module):
    def __init__(self, num_genes=460, pretrained=True, shared_backbone=True):
        super(HierarchicalDenseNet, self).__init__()
        self.shared_backbone = shared_backbone

        def get_encoder():
            model = densenet121(pretrained=pretrained)
            features = model.features  # Exclude classifier
            return features

        if shared_backbone:
            self.encoder = get_encoder()
        else:
            # self.encoder_small = get_encoder()
            self.encoder_mid = get_encoder()
            self.encoder_large = get_encoder()

        # DenseNet121 output: (B, 1024, 7, 7)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fused_dim = 1024  # concatenate 3 scale features
        
        # Attention-based fusion
        self.scale_fusion = ScaleAttentionFusion(in_dim=1024, num_scales=2)
        
        self.fusion_bn_dropout = nn.Sequential(
            nn.BatchNorm1d(self.fused_dim),
            nn.Dropout(p=0.3)
        )

        self.regressor = nn.Sequential(
            nn.Linear(self.fused_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_genes)
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        crops = extract_center_crops(x, sizes=[64, 224])
        
        # Resize to 224x224
        crops = {size: F.interpolate(c, size=(224, 224), mode='bilinear') for size, c in crops.items()}
        x_mid, x_large = crops[64], crops[224]

        if self.shared_backbone:
            # f_small = self.global_pool(self.encoder(x_small)).squeeze(-1).squeeze(-1)
            f_mid   = self.global_pool(self.encoder(x_mid)).squeeze(-1).squeeze(-1)
            f_large = self.global_pool(self.encoder(x_large)).squeeze(-1).squeeze(-1)
        else:
            # f_small = self.global_pool(self.encoder_small(x_small)).squeeze(-1).squeeze(-1)
            f_mid   = self.global_pool(self.encoder_mid(x_mid)).squeeze(-1).squeeze(-1)
            f_large = self.global_pool(self.encoder_large(x_large)).squeeze(-1).squeeze(-1)

        fused = self.scale_fusion([f_mid, f_large])  # [B, 1024]
        fused = self.fusion_bn_dropout(fused)
        out = self.regressor(fused)  # [B, num_genes]
        
        # fused = torch.cat([f_small, f_mid, f_large], dim=1)
        
        return out


#################################       TinySTNet        #########################################


class TinySTNet(nn.Module):
    def __init__(self, num_genes=460, growth_rate=16, num_init_features=32, pretrained=False):
        super(TinySTNet, self).__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense Blocks (fewer layers per block)
        num_features = num_init_features
        block_config = [3, 4, 6]  # Reduced layers from [6, 12, 24] in DenseNet121

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=4, growth_rate=growth_rate, drop_rate=0)
            self.features.add_module(f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_genes)
        )

    def forward(self, x):
        features = self.features(x)
        out = self.avg_pool(F.relu(features)).view(x.size(0), -1)
        out = self.regressor(out)
        return out

#################################       TinySTNet        #########################################

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * scale

class RefinedTinySTNet(nn.Module):
    def __init__(self, num_genes=460, growth_rate=24, num_init_features=48, pretrained=False):
        super(RefinedTinySTNet, self).__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense Blocks with modified configuration
        num_features = num_init_features
        block_config = [4, 6, 8]

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=4, growth_rate=growth_rate, drop_rate=0)
            self.features.add_module(f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = num_features // 2

        # Final layers
        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))
        self.se_block = SEBlock(num_features)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_genes)
        )

    def forward(self, x):
        features = self.features(x)
        features = self.se_block(features)
        out = self.avg_pool(F.relu(features)).view(x.size(0), -1)
        out = self.regressor(out)
        return out

##########################################  TinyEfficientNet    ###########################################

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class SqueezeExcite(nn.Module):
    def __init__(self, in_ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1)
        return x * scale

class TinyEfficientBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=1, stride=1, use_se=True):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_res = (in_ch == out_ch and stride == 1)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_ch, mid_ch, kernel_size=1, padding=0))

        layers.extend([
            ConvBNReLU(mid_ch, mid_ch, kernel_size=3, stride=stride),
            SqueezeExcite(mid_ch) if use_se else nn.Identity(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out += x
        return F.relu(out)

class TinyEfficientNet(nn.Module):
    def __init__(self, num_genes=460):
        super().__init__()

        base_channels = 16
        self.stem = ConvBNReLU(3, base_channels, stride=2)

        self.blocks = nn.Sequential(
            TinyEfficientBlock(base_channels, 24, expand_ratio=1, stride=1),
            TinyEfficientBlock(24, 32, expand_ratio=2, stride=2),
            TinyEfficientBlock(32, 48, expand_ratio=2, stride=2),
            TinyEfficientBlock(48, 64, expand_ratio=2, stride=1),
            TinyEfficientBlock(64, 96, expand_ratio=2, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(96, 256),  #96 to 256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_genes)  #256
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.head(x)
        return x

###################################################################################

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
#         super().__init__(
#             nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

# class ECABlock(nn.Module):
#     def __init__(self, channels, k_size=3):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)  # (B, C, 1, 1)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)

# class TinyEfficientBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, expand_ratio=1, stride=1, use_eca=True):
#         super().__init__()
#         # mid_ch = in_ch * expand_ratio
#         mid_ch = int(in_ch * expand_ratio)
#         self.use_res = (in_ch == out_ch and stride == 1)

#         layers = []
#         if expand_ratio != 1:
#             layers.append(ConvBNReLU(in_ch, mid_ch, kernel_size=1, padding=0))

#         layers.extend([
#             ConvBNReLU(mid_ch, mid_ch, kernel_size=3, stride=stride),
#             ECABlock(mid_ch) if use_eca else nn.Identity(),
#             nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_ch)
#         ])

#         self.block = nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.block(x)
#         if self.use_res:
#             out += x
#         return F.relu(out)

# class DropBlock2D(nn.Module):
#     def __init__(self, block_size=3, drop_prob=0.1):
#         super().__init__()
#         self.block_size = block_size
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         if not self.training or self.drop_prob == 0.:
#             return x
#         gamma = self.drop_prob * x.numel() / (x.shape[0] * x.shape[1] * (x.shape[2] - self.block_size + 1) * (x.shape[3] - self.block_size + 1))
#         mask = (torch.rand_like(x[:, :, self.block_size//2::, self.block_size//2::]) < gamma).float()
#         mask = F.pad(mask, [0, self.block_size//2, 0, self.block_size//2, 0, 0, 0, 0])
#         mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
#         return x * (1 - mask)

# # class RefinedTinyEfficientNet(nn.Module):
# #     def __init__(self, num_genes=460):
# #         super().__init__()

# #         base_channels = 12
# #         self.stem = ConvBNReLU(3, base_channels, stride=2)

# #         self.blocks = nn.Sequential(
# #             TinyEfficientBlock(base_channels, 16, expand_ratio=1, stride=1),
# #             TinyEfficientBlock(16, 24, expand_ratio=1.25, stride=2),
# #             TinyEfficientBlock(24, 32, expand_ratio=1.25, stride=2),
# #             TinyEfficientBlock(32, 48, expand_ratio=1.5, stride=1),
# #             TinyEfficientBlock(48, 64, expand_ratio=1.5, stride=1),
# #         )

# #         self.dropblock = DropBlock2D(block_size=3, drop_prob=0.1)
# #         self.pool = nn.AdaptiveAvgPool2d(1)
# #         self.head = nn.Sequential(
# #             nn.Linear(64, 256),
# #             nn.ReLU(),
# #             nn.Dropout(0.2),
# #             nn.Linear(256, num_genes)
# #         )

# #     def forward(self, x):
# #         x = self.stem(x)
# #         x = self.blocks(x)
# #         x = self.dropblock(x)
# #         x = self.pool(x).view(x.size(0), -1)
# #         x = self.head(x)
# #         return x

# class RefinedTinyEfficientNet(nn.Module):
#     def __init__(self, num_genes=460):
#         super().__init__()

#         base_channels = 16
#         self.stem = ConvBNReLU(3, base_channels, stride=2)

#         self.blocks = nn.Sequential(
#             TinyEfficientBlock(base_channels, 24, expand_ratio=1, stride=1, use_eca=False),
#             TinyEfficientBlock(24, 32, expand_ratio=2, stride=2, use_eca=False),
#             TinyEfficientBlock(32, 48, expand_ratio=2, stride=2, use_eca=True),
#             TinyEfficientBlock(48, 64, expand_ratio=2, stride=1, use_eca=True),
#             TinyEfficientBlock(64, 80, expand_ratio=2, stride=1, use_eca=True),
#         )

#         self.pool = nn.AdaptiveAvgPool2d(1)

#         self.head = nn.Sequential(
#             nn.Linear(80, 384),
#             nn.ReLU(),
#             nn.Dropout(0.25),
#             nn.Linear(384, num_genes)
#         )

#     def forward(self, x):
#         x = self.stem(x)
#         x = self.blocks(x)
#         x = self.pool(x).view(x.size(0), -1)
#         x = self.head(x)
#         return x

class ConvBNSiLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class TinyEfficientBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=1, stride=1, use_eca=False):
        super().__init__()
        mid_ch = int(in_ch * expand_ratio)
        self.use_res = (in_ch == out_ch and stride == 1)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNSiLU(in_ch, mid_ch, kernel_size=1, padding=0))

        layers.extend([
            ConvBNSiLU(mid_ch, mid_ch, kernel_size=3, stride=stride),
            ECABlock(mid_ch) if use_eca else nn.Identity(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out += x
        return F.silu(out)

class RefinedTinyEfficientNet(nn.Module):
    def __init__(self, num_genes=460):
        super().__init__()
        base_channels = 16

        self.stem = ConvBNSiLU(3, base_channels, stride=2)

        self.blocks = nn.Sequential(
            TinyEfficientBlock(base_channels, 24, expand_ratio=1, stride=1, use_eca=False),
            TinyEfficientBlock(24, 32, expand_ratio=2, stride=2, use_eca=False),
            TinyEfficientBlock(32, 48, expand_ratio=2, stride=2, use_eca=True),
            TinyEfficientBlock(48, 64, expand_ratio=2, stride=1, use_eca=True),
            TinyEfficientBlock(64, 80, expand_ratio=2, stride=1, use_eca=True)
        )

        self.conv_head = ConvBNSiLU(80, 96, kernel_size=1, stride=1, padding=0)
        self.dropout2d = nn.Dropout2d(p=0.3)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(96, 192),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(192, num_genes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.dropout2d(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.head(x)
        return x
