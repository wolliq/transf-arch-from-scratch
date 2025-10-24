from torch import nn

from pos_encoder.rope_encoder import RotaryPositionalEncoding
from rmsnorm.decoder import DecoderLayer

model_config = {
    "num_layers": 8,
    "num_heads": 8,
    "num_kv_heads": 4,
    "hidden_dim": 768,
    "moe_experts": 8,
    "moe_topk": 3,
    "max_seq_len": 512,
    "vocab_size": len(tokenizer.get_vocab()),
    "dropout": 0.1,
}


class TextGenerationModel(nn.Module):
    def __init__(self, num_layers, num_heads, num_kv_heads, hidden_dim,
                 moe_experts, moe_topk, max_seq_len, vocab_size, dropout=0.1):
        super().__init__()
        self.rope = RotaryPositionalEncoding(hidden_dim // num_heads, max_seq_len)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoders = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, num_kv_heads, moe_experts, moe_topk, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, ids, mask=None):
        x = self.embedding(ids)
        for decoder in self.decoders:
            x = decoder(x, mask, self.rope)
        x = self.norm(x)
        return self.out(x)

model = TextGenerationModel(**model_config)