import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import models



class LoCIGENv2(nn.Module):
    """
    Location-Conditioned Image-to-Gene Network (v2)

    - CNN backbone (EfficientNet)
    - Spatial FiLM conditioning with (x, y)
    - Gene-aware regressor
    """

    def __init__(
        self,
        num_genes: int,
        coord_dim: int = 2,
        coord_embed_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()

        # ------------------------------
        # 1. EfficientNet backbone
        # ------------------------------
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            backbone = efficientnet_b0(weights=weights)
        else:
            backbone = efficientnet_b0(weights=None)

        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = 1280  # EfficientNet-B0 output

        # ------------------------------
        # 2. Coordinate encoder
        # ------------------------------
        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_dim, coord_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(coord_embed_dim, coord_embed_dim),
            nn.ReLU(inplace=True),
        )

        # ------------------------------
        # 3. FiLM modulation
        # ------------------------------
        self.film_gamma = nn.Linear(coord_embed_dim, feat_dim)
        self.film_beta = nn.Linear(coord_embed_dim, feat_dim)

        # ------------------------------
        # 4. Gene-aware regressor
        # ------------------------------
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_genes),
        )

        # ------------------------------
        # 5. Optional uncertainty head
        # ------------------------------
        self.log_var_head = nn.Linear(feat_dim, num_genes)

        self._init_weights()

    def _init_weights(self):
        for m in self.coord_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)

        nn.init.xavier_uniform_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, images: torch.Tensor, coords: torch.Tensor):
        """
        Args:
            images: (B, 3, H, W)
            coords: (B, 2)  -> (x, y)

        Returns:
            dict with:
                - pred: (B, num_genes)
                - log_var: (B, num_genes)
        """

        # CNN feature extraction
        feat = self.backbone(images)        # (B, 1280, H', W')
        feat = self.pool(feat).flatten(1)   # (B, 1280)

        # Coordinate embedding
        coord_emb = self.coord_encoder(coords)  # (B, coord_embed_dim)

        # FiLM modulation
        gamma = self.film_gamma(coord_emb)
        beta = self.film_beta(coord_emb)
        feat_mod = gamma * feat + beta

        # Gene prediction
        pred = self.regressor(feat_mod)

        # Optional uncertainty
        log_var = self.log_var_head(feat_mod)

        return {
            "pred": pred,
            "log_var": log_var,
            "features": feat_mod,
        }


class LoCIGEN_v2_2(nn.Module):
    """
    LoCIGEN-v2.2
    - Same CNN backbone (EfficientNet)
    - Same coord FiLM conditioning
    - NEW: gene-token cross-attention decoder
    """

    def __init__(
        self,
        num_genes: int,
        backbone_name: str = "efficientnet_b0",
        embed_dim: int = 1280,
        gene_token_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()

        # --------------------------------------------------
        # Backbone (unchanged)
        # --------------------------------------------------
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # --------------------------------------------------
        # Coordinate encoder + FiLM (unchanged)
        # --------------------------------------------------
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

        self.film_gamma = nn.Linear(128, embed_dim)
        self.film_beta = nn.Linear(128, embed_dim)

        # --------------------------------------------------
        # NEW: Gene tokens (learned)
        # --------------------------------------------------
        self.num_genes = num_genes
        self.gene_tokens = nn.Parameter(
            torch.randn(num_genes, gene_token_dim)
        )

        # Project image embedding to gene-token space
        self.img_proj = nn.Linear(embed_dim, gene_token_dim)

        # --------------------------------------------------
        # Cross-attention: genes attend to image
        # --------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=gene_token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_norm = nn.LayerNorm(gene_token_dim)

        # --------------------------------------------------
        # Gene-wise regressor
        # --------------------------------------------------
        self.gene_mlp = nn.Sequential(
            nn.Linear(gene_token_dim, gene_token_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gene_token_dim, 1),
        )

        # Optional uncertainty head (kept for v2.1 loss)
        self.log_var_head = nn.Linear(embed_dim, num_genes)

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, images, coords):
        """
        images: (B,3,224,224)
        coords: (B,2)
        """

        B = images.size(0)

        # ---- Image backbone ----
        feat = self.backbone(images)
        feat = self.pool(feat).flatten(1)  # (B, embed_dim)

        # ---- FiLM conditioning ----
        coord_feat = self.coord_encoder(coords)
        gamma = self.film_gamma(coord_feat)
        beta = self.film_beta(coord_feat)
        feat = gamma * feat + beta

        # ---- Prepare gene queries ----
        gene_tokens = self.gene_tokens.unsqueeze(0).expand(B, -1, -1)
        img_tokens = self.img_proj(feat).unsqueeze(1)  # (B,1,Dg)

        # ---- Cross-attention (genes attend to image) ----
        attn_out, _ = self.cross_attn(
            query=gene_tokens,
            key=img_tokens,
            value=img_tokens,
        )

        gene_repr = self.attn_norm(attn_out + gene_tokens)

        # ---- Gene-wise prediction ----
        y_pred = self.gene_mlp(gene_repr).squeeze(-1)

        # ---- Uncertainty (same as v2.1) ----
        log_var = self.log_var_head(feat)

        return {
            "pred": y_pred,
            "log_var": log_var,
        }

################################### Version 3 ###########################

# ==========================================================
# Gene Transformer Block
# ==========================================================
class GeneTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, G, D)
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


# ==========================================================
# LoCIGEN-v3
# ==========================================================
class LoCIGEN_v3(nn.Module):
    def __init__(
        self,
        num_genes,
        gene_dim=256,
        coord_dim=128,
        num_gene_layers=4,
        pretrained=True
    ):
        super().__init__()

        # --------------------------------------------------
        # 1. EfficientNet-B0 backbone (internal)
        # --------------------------------------------------
        backbone = efficientnet_b0(pretrained=pretrained)
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.pool = nn.AdaptiveAvgPool2d(1)
        img_feat_dim = 1280

        # --------------------------------------------------
        # 2. Coordinate encoder
        # --------------------------------------------------
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, coord_dim),
            nn.ReLU(inplace=True),
            nn.Linear(coord_dim, coord_dim),
            nn.ReLU(inplace=True)
        )

        # --------------------------------------------------
        # 3. Conditioning projection
        # --------------------------------------------------
        self.cond_proj = nn.Linear(
            img_feat_dim + coord_dim,
            gene_dim
        )

        # --------------------------------------------------
        # 4. Gene embeddings
        # --------------------------------------------------
        self.num_genes = num_genes
        self.gene_embed = nn.Embedding(num_genes, gene_dim)

        # --------------------------------------------------
        # 5. Gene–gene Transformer
        # --------------------------------------------------
        self.gene_blocks = nn.ModuleList([
            GeneTransformerBlock(gene_dim)
            for _ in range(num_gene_layers)
        ])

        # --------------------------------------------------
        # 6. Output heads
        # --------------------------------------------------
        self.expr_head = nn.Linear(gene_dim, 1)
        self.log_var_head = nn.Linear(gene_dim, 1)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------
    def forward(self, images, coords):
        """
        images: (B, 3, 224, 224)
        coords : (B, 2)
        """
        B = images.size(0)
        G = self.num_genes

        # ----- Image features -----
        feat = self.backbone(images)          # (B, 1280) for EfficientNet
        if feat.dim() == 4:                   # safety for other backbones
            feat = self.pool(feat).flatten(1)

        # ----- Coordinate features -----
        coord_feat = self.coord_encoder(coords)  # (B, coord_dim)

        # ----- Conditioning vector -----
        cond = torch.cat([feat, coord_feat], dim=1)  # (B, 1280+coord_dim)
        cond = self.cond_proj(cond)                  # (B, gene_dim)

        # ----- Gene tokens -----
        gene_ids = torch.arange(G, device=images.device)
        gene_tokens = self.gene_embed(gene_ids)     # (G, D)
        gene_tokens = gene_tokens.unsqueeze(0).expand(B, G, -1)

        # Condition genes on image + location
        gene_tokens = gene_tokens + cond.unsqueeze(1)

        # ----- Gene–gene attention -----
        for block in self.gene_blocks:
            gene_tokens = block(gene_tokens)

        # ----- Predictions -----
        y_pred = self.expr_head(gene_tokens).squeeze(-1)      # (B, G)
        log_var = self.log_var_head(gene_tokens).squeeze(-1)  # (B, G)

        return {
            "pred": y_pred,
            "log_var": log_var
        }
        
        


# ==========================================================
# Mil new version 
# ==========================================================

class GeneInstanceAttention(nn.Module):
    def __init__(self, dim=256, heads=4, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, gene_q, inst):
        # gene_q: (B, G, D)
        # inst:   (B, K, D)

        attn_out, _ = self.attn(gene_q, inst, inst)
        gene_q = gene_q + attn_out
        gene_q = gene_q + self.ffn(self.norm(gene_q))

        return gene_q
    
    
class LoCIGEN_MIL(nn.Module):
    """
    CNN-strong MIL LoCIGEN
    - EfficientNet backbone
    - Latent instance pooling
    - Coordinate-conditioned instances
    - Gene-conditioned attention
    """

    def __init__(
        self,
        num_genes: int,
        num_instances: int = 8,
        embed_dim: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()

        # ----------------------------------
        # 1. EfficientNet backbone
        # ----------------------------------
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.backbone = backbone.features      # (B, 1280, H, W)

        # ----------------------------------
        # 2. Instance projection
        # ----------------------------------
        self.inst_proj = nn.Conv2d(1280, embed_dim, kernel_size=1)

        self.num_instances = num_instances

        # ----------------------------------
        # 3. Coordinate encoder (CRITICAL)
        # ----------------------------------
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim)
        )

        # ----------------------------------
        # 4. Gene queries
        # ----------------------------------
        self.gene_embed = nn.Embedding(num_genes, embed_dim)

        # ----------------------------------
        # 5. Gene–instance attention
        # ----------------------------------
        self.gene_attn = GeneInstanceAttention(
            dim=embed_dim,
            heads=4
        )

        # ----------------------------------
        # 6. Output heads
        # ----------------------------------
        self.expr_head = nn.Linear(embed_dim, 1)
        self.log_var_head = nn.Linear(embed_dim, 1)

    def forward(self, images, coords):
        """
        images: (B, 3, 224, 224)
        coords: (B, 2)
        """

        B = images.size(0)
        G = self.gene_embed.num_embeddings

        # ----------------------------------
        # CNN feature extraction
        # ----------------------------------
        feat = self.backbone(images)                # (B, 1280, H, W)
        feat = self.inst_proj(feat)                 # (B, D, H, W)

        # ----------------------------------
        # Build latent instances (MIL)
        # ----------------------------------
        feat = feat.flatten(2).transpose(1, 2)      # (B, HW, D)

        inst = F.adaptive_avg_pool1d(
            feat.transpose(1, 2),
            self.num_instances
        ).transpose(1, 2)                           # (B, K, D)

        # ----------------------------------
        # Coordinate conditioning (IMPORTANT)
        # ----------------------------------
        coord_emb = self.coord_encoder(coords)      # (B, D)
        inst = inst + coord_emb.unsqueeze(1)        # (B, K, D)

        # ----------------------------------
        # Gene queries
        # ----------------------------------
        gene_q = self.gene_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # ----------------------------------
        # Gene–instance attention
        # ----------------------------------
        gene_feat = self.gene_attn(gene_q, inst)    # (B, G, D)

        # ----------------------------------
        # Predictions
        # ----------------------------------
        y_pred = self.expr_head(gene_feat).squeeze(-1)
        log_var = self.log_var_head(gene_feat).squeeze(-1)

        return {
            "pred": y_pred,
            "log_var": log_var
        }

#################### version 3 with neighborhod ###############

# --------------------------------------------------
# Gene Transformer Block (UNCHANGED)
# --------------------------------------------------
class GeneTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


# ==================================================
# 🔥 Neighborhood-Aware LoCIGEN-v3 (DROP-IN)
# ==================================================
class LoCIGEN_v3_Neighborhood(nn.Module):
    """
    Drop-in replacement for LoCIGEN_v3
    - Same forward signature
    - Same outputs
    - Adds neighborhood-aware smoothing INSIDE gene space
    """

    def __init__(
        self,
        num_genes,
        gene_dim=256,
        coord_dim=128,
        num_gene_layers=4,
        pretrained=True
    ):
        super().__init__()

        # --------------------------------------------------
        # 1. EfficientNet-B0 backbone (UNCHANGED)
        # --------------------------------------------------
        backbone = efficientnet_b0(pretrained=pretrained)
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.pool = nn.AdaptiveAvgPool2d(1)
        img_feat_dim = 1280

        # --------------------------------------------------
        # 2. Coordinate encoder
        # --------------------------------------------------
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, coord_dim),
            nn.ReLU(inplace=True),
            nn.Linear(coord_dim, coord_dim),
            nn.ReLU(inplace=True)
        )

        # --------------------------------------------------
        # 3. Conditioning projection
        # --------------------------------------------------
        self.cond_proj = nn.Linear(
            img_feat_dim + coord_dim,
            gene_dim
        )

        # --------------------------------------------------
        # 4. Gene embeddings
        # --------------------------------------------------
        self.num_genes = num_genes
        self.gene_embed = nn.Embedding(num_genes, gene_dim)

        # --------------------------------------------------
        # 🔥 NEW: Neighborhood smoother (gene-wise)
        # --------------------------------------------------
        self.neigh_smoother = nn.Sequential(
            nn.Linear(gene_dim, gene_dim),
            nn.GELU(),
            nn.Linear(gene_dim, gene_dim)
        )

        # --------------------------------------------------
        # 5. Gene–gene Transformer
        # --------------------------------------------------
        self.gene_blocks = nn.ModuleList([
            GeneTransformerBlock(gene_dim)
            for _ in range(num_gene_layers)
        ])

        # --------------------------------------------------
        # 6. Output heads
        # --------------------------------------------------
        self.expr_head = nn.Linear(gene_dim, 1)
        self.log_var_head = nn.Linear(gene_dim, 1)

    # --------------------------------------------------
    # Forward (UNCHANGED SIGNATURE)
    # --------------------------------------------------
    def forward(self, images, coords):
        """
        images: (B, 3, 224, 224)
        coords : (B, 2)
        """

        B = images.size(0)
        G = self.num_genes

        # ----- Image features -----
        feat = self.backbone(images)
        if feat.dim() == 4:
            feat = self.pool(feat).flatten(1)

        # ----- Coordinate features -----
        coord_feat = self.coord_encoder(coords)

        # ----- Conditioning vector -----
        cond = torch.cat([feat, coord_feat], dim=1)
        cond = self.cond_proj(cond)

        # ----- Gene tokens -----
        gene_ids = torch.arange(G, device=images.device)
        gene_tokens = self.gene_embed(gene_ids)
        gene_tokens = gene_tokens.unsqueeze(0).expand(B, G, -1)

        # Condition genes on image + location
        gene_tokens = gene_tokens + cond.unsqueeze(1)

        # 🔥 Neighborhood smoothing (RESIDUAL)
        smooth = self.neigh_smoother(gene_tokens)
        gene_tokens = gene_tokens + 0.1 * smooth   # small, stable weight

        # ----- Gene–gene attention -----
        for block in self.gene_blocks:
            gene_tokens = block(gene_tokens)

        # ----- Predictions -----
        y_pred = self.expr_head(gene_tokens).squeeze(-1)
        log_var = self.log_var_head(gene_tokens).squeeze(-1)

        return {
            "pred": y_pred,
            "log_var": log_var
        }
