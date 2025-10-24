#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_custom_transformer.py

A single-file script to train a small decoder-only Transformer language model from scratch with PyTorch.
- Character-level tokenizer (no external deps)
- Optional public-domain text download (Project Gutenberg)
- Sinusoidal positional encoding
- Multi-Head Self-Attention with causal mask
- RMSNorm + residual connections
- Feedforward (GEGLU)
- Training loop with next-token prediction
- Sampling/generation utility

Inspired by and consolidated for convenience from educational materials on transformer building blocks,
including MachineLearningMastery.com's 10-day mini-course on building transformers with PyTorch.

Usage:
    python train_custom_transformer.py --data_dir ./data --download_gutenberg \
        --context_len 256 --vocab "byte" --epochs 2 --batch_size 32

Author: ChatGPT (GPT-5 Thinking)
License: MIT
"""

import argparse
import math
import os
import random
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import requests  # only needed if --download_gutenberg is used
except Exception:
    requests = None


# -----------------------------
# Data utilities
# -----------------------------

GUTENBERG_URLS = {
    # Small set of public-domain English texts as a convenient starter
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    "sleepy_hollow": "https://www.gutenberg.org/ebooks/41.txt.utf-8",
    "common_sense": "https://www.gutenberg.org/ebooks/147.txt.utf-8",
}

def maybe_download_gutenberg(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if requests is None:
        print("[warn] requests not available; cannot auto-download.")
        return
    for name, url in GUTENBERG_URLS.items():
        path = os.path.join(out_dir, f"{name}.txt")
        if not os.path.exists(path):
            print(f"[info] downloading {name} ...")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)


def read_all_texts(data_dir: str) -> str:
    texts = []
    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".txt"):
            with open(os.path.join(data_dir, fn), "r", encoding="utf-8", errors="ignore") as f:
                t = f.read()
                # crude strip of Gutenberg headers/footers if present
                start = t.find("*** START OF THE PROJECT GUTENBERG EBOOK")
                if start != -1:
                    start = t.find("\n", start) + 1
                else:
                    start = 0
                end = t.find("*** END OF THE PROJECT GUTENBERG EBOOK")
                if end == -1:
                    end = len(t)
                t = t[start:end].strip()
                texts.append(t)
    if not texts:
        raise RuntimeError(f"No .txt files found in {data_dir}. Provide data or use --download_gutenberg.")
    return "\n\n".join(texts)


# -----------------------------
# Tokenizers
# -----------------------------

class CharTokenizer:
    def __init__(self, text: str, kind: str = "char"):
        if kind == "byte":
            # byte-level: fixed vocab of 256
            self.kind = "byte"
            self.vocab_size = 256
            self.stoi = None
            self.itos = None
        else:
            self.kind = "char"
            chars = sorted(list(set(text)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.vocab_size = len(self.stoi)

    def encode(self, s: str) -> List[int]:
        if self.kind == "byte":
            return list(s.encode("utf-8", errors="ignore"))
        else:
            return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        if self.kind == "byte":
            return bytes([int(i) % 256 for i in ids]).decode("utf-8", errors="ignore")
        else:
            return "".join(self.itos[i] for i in ids if i in self.itos)


# -----------------------------
# Dataset
# -----------------------------

class LMSequenceDataset(Dataset):
    def __init__(self, ids: List[int], context_len: int, step: int = None):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context_len = context_len
        self.step = step or context_len  # non-overlapping by default

        self.num_sequences = max(0, (len(self.ids) - context_len - 1) // self.step + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.step
        end = start + self.context_len
        x = self.ids[start:end]
        y = self.ids[start + 1 : end + 1]
        return x, y


# -----------------------------
# Model components
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * norm_x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 10_000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10_000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, dim)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # x: (B, T, C)
        T = x.size(1)
        return x + self.pe[start_pos:start_pos+T, :].unsqueeze(0)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int = None, dropout: float = 0.0):
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
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch scaled dot-product attention supports an additive mask (float) or boolean
        # For causal, we'll pass a boolean mask of shape (T, T)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = hidden_mult * dim
        self.fc1 = nn.Linear(dim, hidden * 2, bias=False)  # for GEGLU
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.fc1(x).chunk(2, dim=-1)  # (B,T,H), (B,T,H)
        x = F.gelu(x1) * x2  # GEGLU
        x = self.fc2(x)
        return self.dropout(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, hidden_mult=ff_mult, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ff(self.norm2(x))
        return x


class TinyDecoderLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, depth: int = 6, num_heads: int = 8, context_len: int = 256, dropout: float = 0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_enc = SinusoidalPositionalEncoding(dim, max_len=context_len + 10_000)
        self.blocks = nn.ModuleList([DecoderBlock(dim, num_heads, ff_mult=4, dropout=dropout) for _ in range(depth)])
        self.norm_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.context_len = context_len

        # Causal mask (T, T): True for positions that should be masked
        # We'll build a boolean mask once for max context length and slice in forward
        mask = torch.triu(torch.ones(context_len, context_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, idx: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # idx: (B, T)
        B, T = idx.shape
        x = self.token_emb(idx)  # (B,T,C)
        x = self.pos_enc(x, start_pos=start_pos)

        attn_mask = self.causal_mask[:T, :T]  # (T, T) bool
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B,T,V)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_len:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = v[:, -1].unsqueeze(-1)
                logits[logits < thresh] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# -----------------------------
# Training
# -----------------------------

def split_data(ids: List[int], train_ratio: float = 0.9) -> Tuple[List[int], List[int]]:
    n = len(ids)
    cut = int(n * train_ratio)
    return ids[:cut], ids[cut:]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[info] device: {device}")

    if args.download_gutenberg:
        maybe_download_gutenberg(args.data_dir)

    raw_text = read_all_texts(args.data_dir)

    tokenizer = CharTokenizer(raw_text, kind=args.vocab)
    print(f"[info] vocab_size={tokenizer.vocab_size} (kind={tokenizer.kind})")

    ids_all = tokenizer.encode(raw_text)
    train_ids, val_ids = split_data(ids_all, train_ratio=0.9)

    train_ds = LMSequenceDataset(train_ids, context_len=args.context_len, step=args.stride or args.context_len)
    val_ds = LMSequenceDataset(val_ids, context_len=args.context_len, step=args.context_len)

    def collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate)

    model = TinyDecoderLM(
        vocab_size=tokenizer.vocab_size,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        context_len=args.context_len,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, len(train_loader)*args.epochs))

    best_val = float("inf")

    def run_epoch(loader, train_mode: bool):
        model.train(train_mode)
        total_loss, total_tokens = 0.0, 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
        return total_loss / max(1, total_tokens)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(train_loader, train_mode=True)
        val_loss = run_epoch(val_loader, train_mode=False)
        dt = time.time() - t0
        print(f"[epoch {epoch:03d}] train_bpc={train_loss/math.log(2):.4f} val_bpc={val_loss/math.log(2):.4f} ({dt:.1f}s)")

        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, "model.pt")
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "vocab_size": tokenizer.vocab_size,
                    "dim": args.dim,
                    "depth": args.depth,
                    "heads": args.heads,
                    "context_len": args.context_len,
                },
                "tokenizer": {"kind": tokenizer.kind, "stoi": getattr(tokenizer, "stoi", None), "itos": getattr(tokenizer, "itos", None)},
            }, ckpt_path)
            print(f"[info] saved checkpoint to {ckpt_path}")

    # quick sample
    if args.sample_len > 0:
        print("[info] sampling...")
        model.eval()
        start = "The"
        idx = torch.tensor([tokenizer.encode(start)], dtype=torch.long, device=device)
        out_ids = model.generate(idx, max_new_tokens=args.sample_len, temperature=args.temperature, top_k=args.top_k)
        print(tokenizer.decode(out_ids[0].tolist()))


def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny decoder-only Transformer from scratch.")
    p.add_argument("--data_dir", type=str, default="./data", help="directory containing .txt files")
    p.add_argument("--out_dir", type=str, default="./checkpoints", help="where to save checkpoints")
    p.add_argument("--download_gutenberg", action="store_true", help="download a few public-domain texts to --data_dir")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")

    # model
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--context_len", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    # training
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--stride", type=int, default=None, help="sequence stride; default None = non-overlapping")

    # sampling
    p.add_argument("--sample_len", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)

    # tokenizer
    p.add_argument("--vocab", type=str, choices=["char", "byte"], default="byte", help="byte-level has fixed vocab=256")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
