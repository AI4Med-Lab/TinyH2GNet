import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Function to generate static positional embeddings
def get_positional_embeddings(sequence_length, d):
    result = torch.zeros(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


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

# Vision Transformer class
class VisionTransformer(nn.Module):
    def __init__(self, input_dim, seq_length, num_hiddens, num_heads, num_blks, output_dim, emb_dropout=0.1, blk_dropout=0.1):
        super().__init__()
        
        # Save input and sequence dimensions for dummy input generation
        self.input_dim = input_dim
        self.seq_length = seq_length
        
        # Linear projection to match input_dim to num_hiddens
        # print("input_dim",  input_dim)
        # print("num_hiddens", num_hiddens)

        self.input_projection = nn.Linear(input_dim, num_hiddens)
        
        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        
        # Static positional embeddings
        self.pos_embedding = nn.Parameter(
            get_positional_embeddings(seq_length + 1, num_hiddens), requires_grad=False
        )
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(num_hiddens, num_heads, blk_dropout, is_decoder=False)
            for _ in range(num_blks)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(num_hiddens)
        
        # Output projection
        self.proj = nn.Linear(num_hiddens, output_dim)

    def forward(self, x):
        # Generate dummy input

        
        # Project input to hidden dimension

        x = x.squeeze(1)

        x = self.input_projection(x)

        # x = x.view(x.size(0), x.size(2), -1)
        x = x.view(x.size(0), self.seq_length, -1)
      

        
        # Expand CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        
        # Concatenate CLS token with input sequence
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embedding.to(x.device)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Forward pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Extract CLS token representation
        x = self.layer_norm(x[:, 0])
        
        # Project to output dimension
        x = self.proj(x)
        return x