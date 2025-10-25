import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DecoderBlock, RMSNorm, SinusoidalPositionalEncoding


class TinyDecoderLM(nn.Module):
    """Compact decoder-only Transformer for character-level language modelling."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        context_len: int = 256,
        dropout: float = 0.0,
    ):
        """Assemble token embeddings, positional encodings, and stacked decoder blocks."""
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_enc = SinusoidalPositionalEncoding(dim, max_len=context_len + 10_000)
        self.blocks = nn.ModuleList(
            [DecoderBlock(dim, num_heads, ff_mult=4, dropout=dropout) for _ in range(depth)]
        )
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        mask = torch.triu(torch.ones(context_len, context_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)
        self.context_len = context_len

    def forward(self, idx: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Compute next-token logits for each position in the provided context window."""
        tokens = self.token_emb(idx)
        tokens = self.pos_enc(tokens, start_pos=start_pos)

        attn_mask = self.causal_mask[: tokens.size(1), : tokens.size(1)]
        x = tokens
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.norm_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """Auto-regressively extend the prompt by sampling from the transformer's predictions."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_len :]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx
