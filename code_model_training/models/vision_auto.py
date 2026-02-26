import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbeddings(nn.Module):
    def __init__(self, img_size=96, patch_size=16, hidden_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Conv2d(in_channels=3, out_channels=hidden_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, X):
        X = self.conv(X)
        X = X.flatten(2)
        X = X.transpose(1, 2)
        return X

class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout=0.1, is_decoder=False):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.is_decoder = is_decoder

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        if self.is_decoder:
            tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()
        assert n_embd % num_heads == 0, "n_embd must be divisible by num_heads"
        self.heads = nn.ModuleList([
            Head(n_embd, n_embd // num_heads, dropout, is_decoder)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = [h(x) for h in self.heads]
        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, num_heads, dropout, is_decoder)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        original_x = x
        x = self.ln1(x)
        attn_output = self.attn(x)
        x = original_x + attn_output
        x = self.ln2(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        return x

class VisionEncoder(nn.Module):
    def __init__(self, img_size=128, patch_size=16, num_hiddens=1024, num_heads=8, num_blks=6, emb_dropout=0.1, blk_dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbeddings(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = self.get_positional_embeddings(num_patches + 1, num_hiddens)
        self.dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([
            Block(num_hiddens, num_heads, blk_dropout, is_decoder=False) for _ in range(num_blks)
        ])
        self.layer_norm = nn.LayerNorm(num_hiddens)

    def get_positional_embeddings(self, sequence_length, d):
        result = np.zeros((sequence_length, d), dtype=np.float32)
        for i in range(sequence_length):
            for j in range(d):
                result[i, j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return torch.from_numpy(result)

    def forward(self, X):
        x = self.patch_embedding(X)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)  # Apply LayerNorm to the entire sequence
        return x  # Return the full sequence, not just the CLS token

class VisionDecoder(nn.Module):
    def __init__(self, img_size=128, patch_size=16, num_hiddens=1024, num_heads=8, num_blks=6, blk_dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(num_hiddens, num_heads, blk_dropout, is_decoder=True) for _ in range(num_blks)
        ])
        self.patch_proj = nn.Linear(num_hiddens, patch_size * patch_size * 3)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_dim = img_size // patch_size  # Calculate patches along one dimension
        self.num_patches = self.num_patches_per_dim ** 2

    def forward(self, x):
        batch_size = x.shape[0]
        for block in self.blocks:
            x = block(x)
        x = self.patch_proj(x)  # [batch_size, num_patches + 1, patch_size * patch_size * 3]
        
        # Remove the CLS token before reshaping
        x = x[:, 1:, :]  # Remove the CLS token (assumes CLS token at position 0)
        
        # Reshape to patches
        x = x.view(batch_size, self.num_patches_per_dim, self.num_patches_per_dim, self.patch_size, self.patch_size, 3)
        
        # Rearrange to image
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [batch_size, 3, H, W]
        x = x.view(batch_size, 3, self.img_size, self.img_size)
        return x

class VisionAutoEncoder(nn.Module):
    def __init__(self, img_size=128, patch_size=16, num_hiddens=1024, num_heads=8, num_blks=6, emb_dropout=0.1, blk_dropout=0.1):
        super().__init__()
        self.encoder = VisionEncoder(img_size, patch_size, num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout)
        self.decoder = VisionDecoder(img_size, patch_size, num_hiddens, num_heads, num_blks, blk_dropout)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

# Example Usage
if __name__ == "__main__":
    img_size = 128
    patch_size = 16
    num_hiddens = 1024
    num_heads = 2
    num_blks = 5

    model = VisionAutoEncoder(img_size, patch_size, num_hiddens, num_heads, num_blks)

    # x = torch.randn(4, 3, img_size, img_size)

    # reconstructed = model(x)

    # print("Input shape:", x.shape)
    # print("Reconstructed shape:", reconstructed.shape)
    print(model)
