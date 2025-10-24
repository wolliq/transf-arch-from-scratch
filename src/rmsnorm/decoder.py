from torch import nn

from attention.grouped_query_attention import GQA
from moe.MoeLayer import MoELayer


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, moe_experts, moe_topk, dropout=0.1):
        super().__init__()
        self.self_attn = GQA(hidden_dim, num_heads, num_kv_heads, dropout)
        self.mlp = MoELayer(hidden_dim, 4 * hidden_dim, moe_experts, moe_topk)
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, mask=None, rope=None):
        # self-attention sublayer
        out = self.norm1(x)
        out = self.self_attn(out, out, out, mask, rope)
        x = out + x
        # MLP sublayer
        out = self.norm2(x)
        out = self.mlp(out)
        return out + x