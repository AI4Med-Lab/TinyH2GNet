import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# -----------------------------
# Rotary positional embedding
# -----------------------------
class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding (RoPE).
    Produces cos, sin tensors for given sequence length and head_dim.
    head_dim must be even.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        # register inv_freq as buffer so it moves with the module but is not a parameter
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        """
        Return cos, sin for sequence length.
        cos, sin shapes: (seq_len, head_dim//2)
        """
        t = torch.arange(seq_len, device=device).float()  # (seq_len,)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, head_dim//2)
        # sin and cos will be used on the interleaved (even/odd) pairs
        sin = torch.sin(freqs)  # (seq_len, head_dim//2)
        cos = torch.cos(freqs)  # (seq_len, head_dim//2)
        return cos, sin

    @staticmethod
    def apply_rotary(q, k, cos, sin):
        """
        Apply RoPE to q and k.
        q: (B, L_q, nhead, head_dim)
        k: (B, L_k, nhead, head_dim)
        cos/sin: (L, head_dim//2) where L equals respective sequence length
        We expect to call this separately for q and k with their own cos/sin.
        """
        # x[..., ::2] = even dims, x[..., 1::2] = odd dims
        def rotate(x, cos, sin):
            # x: (..., head_dim)
            x1 = x[..., ::2]  # (..., head_dim/2)
            x2 = x[..., 1::2] # (..., head_dim/2)
            # cos, sin must be broadcastable to x1/x2
            # cos: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
            cos_b = cos.unsqueeze(0).unsqueeze(2)
            sin_b = sin.unsqueeze(0).unsqueeze(2)
            x_rot_even = x1 * cos_b - x2 * sin_b
            x_rot_odd  = x1 * sin_b + x2 * cos_b
            # interleave back to head_dim
            x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
            return x_rot

        q_rot = rotate(q, cos, sin)
        k_rot = rotate(k, cos, sin)
        return q_rot, k_rot

# -----------------------------
# Cross-attention block using RoPE
# -----------------------------
class CrossAttentionBlock(nn.Module):
    """
    Cross-attention with RoPE applied to Q and K.
    Query: (B, G, d_model)  - genes
    Context: (B, N, d_model) - image tokens
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # linear projections for q,k,v and output
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        # Rotary embedding for head_dim
        self.rotary = RotaryEmbedding(self.head_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, context):
        B, G, D = query.shape
        N = context.shape[1]

        # Linear projections
        q = self.Wq(query).view(B, G, self.nhead, self.head_dim)
        k = self.Wk(context).view(B, N, self.nhead, self.head_dim)
        v = self.Wv(context).view(B, N, self.nhead, self.head_dim)

        # Generate RoPE angles for BOTH sequences (q_len=G, k_len=N)
        cos_q, sin_q = self.rotary(G, query.device)
        cos_k, sin_k = self.rotary(N, context.device)

        # Apply RoPE correctly
        q_rot, _ = RotaryEmbedding.apply_rotary(q, q, cos_q, sin_q)
        _, k_rot = RotaryEmbedding.apply_rotary(k, k, cos_k, sin_k)

        # Prepare for attention
        q_ = q_rot.permute(0, 2, 1, 3)  # (B, H, G, d)
        k_ = k_rot.permute(0, 2, 1, 3)  # (B, H, N, d)
        v_ = v.permute(0, 2, 1, 3)      # (B, H, N, d)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.softmax(torch.matmul(q_, k_.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v_)  # (B, H, G, d)

        # Combine heads
        out = out.permute(0, 2, 1, 3).reshape(B, G, D)
        out = self.out(out)

        # Residual + FFN
        x = self.norm1(query + out)
        x = self.norm2(x + self.ff(x))

        return x


# -----------------------------
# Image projectors (no pos_embed!)
# -----------------------------
class EfficientNetImageProjector(nn.Module):
    def __init__(self, d_model=512, pretrained=True):
        super().__init__()
        eff = models.efficientnet_b0(pretrained=pretrained)
        # use features only
        self.backbone = nn.Sequential(*list(eff.features.children()))
        # EfficientNet-B0 last channels = 1280
        self.proj = nn.Linear(1280, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, images):
        feats = self.backbone(images)  # (B, C, H', W')
        B, C, H, W = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = self.proj(tokens)                 # (B, N, d_model)
        tokens = self.norm(tokens)
        return tokens  # (B, N, d_model)

class DenseNetImageProjector(nn.Module):
    def __init__(self, d_model=512, pretrained=True):
        super().__init__()
        densenet = models.densenet121(pretrained=pretrained)
        self.backbone = densenet.features  # (B, 1024, H', W')
        self.proj = nn.Linear(1024, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, images):
        feats = self.backbone(images)  # (B, C, H', W')
        B, C, H, W = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)
        tokens = self.norm(tokens)
        return tokens

class VGG16ImageProjector(nn.Module):
    def __init__(self, d_model=512, pretrained=True):
        super().__init__()
        vgg = models.vgg16(pretrained=pretrained)
        self.backbone = vgg.features  # (B, 512, H', W')
        self.proj = nn.Linear(512, d_model) if d_model != 512 else nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, images):
        feats = self.backbone(images)  # (B, C, H', W')
        B, C, H, W = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = self.proj(tokens) if not isinstance(self.proj, nn.Identity) else tokens
        tokens = self.norm(tokens)
        return tokens

# -----------------------------
# Gene projector (no pos_embed)
# -----------------------------
class GeneProjector(nn.Module):
    """Project scalar gene values to d_model embeddings (no positional embedding)."""
    def __init__(self, num_genes, d_model):
        super().__init__()
        self.linear = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gene_values):
        # gene_values: (B, G) or (B, G, 1)
        if gene_values.dim() == 2:
            gene_values = gene_values.unsqueeze(-1)  # (B, G, 1)
        g = self.linear(gene_values)  # (B, G, d_model)
        return self.norm(g)

# -----------------------------
# ImageGeneCrossTransformer
# -----------------------------
class ImageGeneCrossTransformerRoPE(nn.Module):
    """
    Uses RoPE in cross-attention. No learnable positional embeddings.
    """
    def __init__(
        self,
        num_genes,
        d_model=512,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
        pretrained=True,
        img_proj="denseNet"
    ):
        super().__init__()

        # choose image projector
        if img_proj == "effNet":
            self.img_proj = EfficientNetImageProjector(d_model=d_model, pretrained=pretrained)
            print("Using EfficientNetImageProjector")
        elif img_proj == "vgg16":
            self.img_proj = VGG16ImageProjector(d_model=d_model, pretrained=pretrained)
            print("Using VGG16ImageProjector")
        else:
            self.img_proj = DenseNetImageProjector(d_model=d_model, pretrained=pretrained)
            print("Using DenseNetImageProjector")

        self.gene_proj = GeneProjector(num_genes, d_model)

        # Learned query tokens (no positional)
        self.gene_query_tokens = nn.Parameter(torch.randn(1, num_genes, d_model))
        nn.init.trunc_normal_(self.gene_query_tokens, std=0.02)

        # Cross-attention
        self.cross_block = CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)

        # Output head
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, images, gene_values=None):
        B = images.shape[0]

        img_tokens = self.img_proj(images)  # (B, N, d)
        if gene_values is not None:
            gene_tokens = self.gene_proj(gene_values)  # (B, G, d)
        else:
            gene_tokens = self.gene_query_tokens.expand(B, -1, -1)  # (B, G, d)

        # Cross attention (genes query image tokens). Rotary applied inside.
        H_cross = self.cross_block(gene_tokens, img_tokens)  # (B, G, d)

        # Predict per-gene expression
        y_hat = self.output_head(H_cross).squeeze(-1)  # (B, G)
        return y_hat, H_cross

# -----------------------------
# Quick sanity test (toy)
# -----------------------------
if __name__ == "__main__":
    # toy inputs
    B = 2
    C = 3
    H = 128
    W = 128
    num_genes = 10
    d_model = 64
    nhead = 8

    model = ImageGeneCrossTransformer(num_genes=num_genes, d_model=d_model, nhead=nhead, img_proj="vgg16", pretrained=False)
    images = torch.randn(B, 3, H, W)
    gene_values = torch.randn(B, num_genes)  # scalar gene values per gene

    y_hat, H_cross = model(images, gene_values)
    print("y_hat shape:", y_hat.shape)      # (B, G)
    print("H_cross shape:", H_cross.shape)  # (B, G, d_model)
