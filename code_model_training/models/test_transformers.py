import torch
import torch.nn as nn
import numpy as np
import timm
import os
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, seq_length, input_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # Apply a linear transformation to project embeddings to hidden_dim
        x = self.projection(x)
        return x

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length, hidden_dim = x.size()
        qkv = self.qkv(x).reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, seq_length, hidden_dim)
        output = self.proj(attn_output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, seq_length, input_dim, hidden_dim, num_heads, num_blks, output_dim, dropout):
        super().__init__()
        self.patch_embedding = PatchEmbedding(seq_length, input_dim, hidden_dim)

        self.pos_embedding = nn.Parameter(torch.tensor(get_positional_embeddings(seq_length + 1, hidden_dim), dtype=torch.float32))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_blks)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embedding(x)

        # print("X:::::::::::::::::::::::::::", x.shape)

        x = x.view(x.size(0), x.size(2), -1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:x.size(1), :]
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])
        output = self.fc(x)
        return output

# Example Usage
# seq_length = 11
# input_dim = 1024
# hidden_dim = 512
# num_heads = 2
# num_blks = 4
# output_dim = 460
# dropout = 0.1

# vit = VisionTransformer(seq_length, input_dim, hidden_dim, num_heads, num_blks, output_dim, dropout)
# # x = torch.randn(32, seq_length, input_dim)  # Batch size = 32
# # output = vit(x)
# print(vit)




