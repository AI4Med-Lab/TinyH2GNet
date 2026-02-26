import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import models


class LoCIGEN(nn.Module):
    """
    Unified LoCIGEN for Visium + Xenium

    mode = "visium"  → strong spatial conditioning
    mode = "xenium"  → damped spatial conditioning
    """

    def __init__(
        self,
        num_genes: int,
        mode: str = "visium",
        coord_embed_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()

        assert mode in ["visium", "xenium"]
        self.mode = mode

        # ---------------- Backbone ----------------
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = 1280

        # ---------------- Coord encoder ----------------
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, coord_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(coord_embed_dim, coord_embed_dim),
            nn.ReLU(inplace=True),
        )

        # ---------------- FiLM ----------------
        self.film_gamma = nn.Linear(coord_embed_dim, feat_dim)
        self.film_beta  = nn.Linear(coord_embed_dim, feat_dim)

        # ---------------- Mode switch ----------------
        if mode == "visium":
            self.coord_scale = 1.0
            self.detach_coords = False
        else:  # xenium
            self.coord_scale = 0.1
            self.detach_coords = True

        # ---------------- Regressor ----------------
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_genes),
        )

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

    def forward(self, images, coords):
        feat = self.backbone(images)
        feat = self.pool(feat).flatten(1)

        if self.detach_coords:
            coords = coords.detach()

        coord_emb = self.coord_encoder(coords)

        # ✅ BOUNDED + DAMPED FiLM
        gamma = torch.tanh(self.film_gamma(coord_emb)) * self.coord_scale
        beta  = torch.tanh(self.film_beta(coord_emb))  * self.coord_scale

        feat_mod = feat + gamma * feat + beta

        pred = self.regressor(feat_mod)
        log_var = self.log_var_head(feat_mod)

        return {
            "pred": pred,
            "log_var": log_var,
            "features": feat_mod,
        }
