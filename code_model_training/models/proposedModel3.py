import torch
import torch.nn as nn
import torchvision.models as models
import math


# --- Sinusoidal Positional Embedding Utility -------------------------------
def sinusoidal_positional_encoding(n_positions, d_model, device):
    """
    Create sinusoidal positional embeddings like in 'Attention Is All You Need'.
    """
    position = torch.arange(n_positions, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(n_positions, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # shape: (1, n_positions, d_model)
    return pe


# --- Cross-Attention Block --------------------------------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, context):
        attn_out, _ = self.cross_attn(query, context, context)
        x = self.norm1(query + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


# --- Image Projector (EfficientNet-B0) with Sinusoidal Pos Embedding -------
class EfficientNetImageProjector(nn.Module):
    def __init__(self, d_model=512, pretrained=True):
        super().__init__()
        effnet = models.efficientnet_b0(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(effnet.features.children()))
        self.proj = nn.Linear(1280, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, images):
        feats = self.backbone(images)  # (B, 1280, H', W')
        B, C, H, W = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)  # (B, N=H'*W', C)
        tokens = self.proj(tokens)  # (B, N, d_model)

        # Sinusoidal positional embedding
        N = H * W
        pos_embed = sinusoidal_positional_encoding(N, self.d_model, tokens.device)
        tokens = tokens + pos_embed

        return self.norm(tokens)  # (B, N, d_model)


# --- Gene Projector with Sinusoidal Pos Embedding --------------------------
class GeneProjector(nn.Module):
    def __init__(self, num_genes, d_model):
        super().__init__()
        self.linear = nn.Linear(1, d_model)
        self.num_genes = num_genes
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gene_values):
        if gene_values.dim() == 2:
            gene_values = gene_values.unsqueeze(-1)
        g = self.linear(gene_values)  # (B, G, d_model)

        # Sinusoidal positional embedding for genes
        pos_embed = sinusoidal_positional_encoding(self.num_genes, self.d_model, g.device)
        g = g + pos_embed

        return self.norm(g)  # (B, G, d_model)


# --- Full Cross-Transformer ------------------------------------------------
class ImageGeneCrossTransformerWithSinePE(nn.Module):
    def __init__(self, num_genes, d_model=512, nhead=8, dim_feedforward=1024, dropout=0.1, pretrained=True):
        super().__init__()
        self.img_proj = EfficientNetImageProjector(d_model=d_model, pretrained=pretrained)
        self.gene_proj = GeneProjector(num_genes, d_model)

        # Learned gene query tokens
        self.gene_query_tokens = nn.Parameter(torch.randn(1, num_genes, d_model))
        nn.init.trunc_normal_(self.gene_query_tokens, std=0.02)

        self.cross_block = CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, images, gene_values=None):
        B = images.shape[0]
        img_tokens = self.img_proj(images)

        if gene_values is not None:
            gene_tokens = self.gene_proj(gene_values)
        else:
            gene_tokens = self.gene_query_tokens.expand(B, -1, -1)

        H_cross = self.cross_block(gene_tokens, img_tokens)
        y_hat = self.output_head(H_cross).squeeze(-1)
        return y_hat, H_cross

if __name__ == "__main__":
    main()




