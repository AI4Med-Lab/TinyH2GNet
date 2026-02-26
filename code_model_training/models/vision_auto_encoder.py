import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# Custom Positional Embedding Function
def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result.clone().detach()

# Patch Embeddings Module
class PatchEmbeddings(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = rearrange(x, 'b c h w -> b (h w) c')  # Flatten patches
        return x

# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x):
        # Permute to (sequence_length, batch_size, embed_dim) for MultiheadAttention
        x = x.permute(1, 0, 2)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x.permute(1, 0, 2)  # Back to (batch_size, sequence_length, embed_dim)
        x = x + self.mlp(self.norm2(x))
        return x

# Vision Transformer Encoder
class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder_blocks = nn.ModuleList([ 
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Get positional embeddings
        sequence_length = (img_size // patch_size) ** 2 + 1  # CLS token + patches
        self.pos_embed = get_positional_embeddings(sequence_length, embed_dim)

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, num_patches, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # Add CLS token
        x = x + self.pos_embed[:num_patches + 1, :]  # Add positional embeddings
        for block in self.encoder_blocks:
            x = block(x)
        x = self.norm(x)
        return x

# Decoder to reconstruct the image
class VisionTransformerDecoder(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, out_channels):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x[:, 1:, :]  # Remove CLS token
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.img_size // self.patch_size)
        x = self.proj(x)
        return x

# Full Vision Transformer (Encoder-Decoder)
class VisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=3, embed_dim=768, depth=6, num_heads=8, mlp_dim=1024):
        super().__init__()
        self.encoder = VisionTransformerEncoder(img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim)
        self.decoder = VisionTransformerDecoder(img_size, patch_size, embed_dim, in_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

# Example Usage
if __name__ == "__main__":
    img_size = 128
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    model = VisionTransformer(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)

    dummy_input = torch.randn(1, in_channels, img_size, img_size)  # Example input
    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
