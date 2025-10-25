import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """Initialise RMSNorm scaling vector to stabilise transformer residual streams."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise activations channel-wise before they enter attention or feed-forward blocks."""
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * norm_x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 10_000):
        """Pre-compute sinusoidal table so tokens carry absolute position information."""
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10_000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Add positional phase offsets so attention can reason over token order."""
        seq_len = x.size(1)
        return x + self.pe[start_pos : start_pos + seq_len].unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int = None, dropout: float = 0.0):
        """Set up projections that let tokens attend to earlier context across multiple heads."""
        super().__init__()
        assert dim % num_heads == 0 or head_dim is not None, "dim must be divisible by num_heads or specify head_dim"
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.inner_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Project queries, keys, values and perform masked attention within the causal window."""
        batch, seqlen, _ = x.shape
        q = self.q_proj(x).view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seqlen, self.inner_dim)
        return self.o_proj(attn)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        """Construct the position-wise feed-forward network used between attention layers."""
        super().__init__()
        hidden = hidden_mult * dim
        self.fc1 = nn.Linear(dim, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GEGLU activation and projection to refine each token's representation."""
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        x = F.gelu(x1) * x2  # GEGLU activation
        x = self.fc2(x)
        return self.dropout(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        """Bundle attention and feed-forward sublayers with residual pathways."""
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, hidden_mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Run a single transformer decoder layer over the sequence with causal masking."""
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ff(self.norm2(x))
        return x
