import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import data
from .models import CharTokenizer, TinyDecoderLM


@dataclass
class TrainingArtifacts:
    tokenizer: CharTokenizer
    model: TinyDecoderLM


class Trainer:
    """Orchestrates data loading, model training, checkpointing, and sampling."""

    def __init__(self, args):
        """Record CLI configuration and choose device for transformer training."""
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    def run(self) -> None:
        """End-to-end driver that prepares data, builds the model, and kicks off optimisation."""
        print(f"[info] device: {self.device}")
        if self.args.download_gutenberg:
            data.maybe_download_gutenberg(self.args.data_dir)

        raw_text = data.read_all_texts(self.args.data_dir)
        tokenizer = CharTokenizer(raw_text, kind=self.args.vocab)
        print(f"[info] vocab_size={tokenizer.vocab_size} (kind={tokenizer.kind})")

        ids_all = tokenizer.encode(raw_text)
        train_ids, val_ids = data.split_data(ids_all, train_ratio=0.9)
        train_loader, val_loader = data.build_dataloaders(
            train_ids=train_ids,
            val_ids=val_ids,
            context_len=self.args.context_len,
            stride=self.args.stride,
            batch_size=self.args.batch_size,
        )

        model = TinyDecoderLM(
            vocab_size=tokenizer.vocab_size,
            dim=self.args.dim,
            depth=self.args.depth,
            num_heads=self.args.heads,
            context_len=self.args.context_len,
            dropout=self.args.dropout,
        ).to(self.device)

        artifacts = TrainingArtifacts(tokenizer=tokenizer, model=model)
        self._train(artifacts, train_loader, val_loader)
        if self.args.sample_len > 0:
            self._sample(artifacts)

    def _train(self, artifacts: TrainingArtifacts, train_loader, val_loader) -> None:
        """Iterate over epochs to optimise the decoder-only transformer and checkpoint the best weights."""
        model = artifacts.model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, len(train_loader) * self.args.epochs)
        )

        best_val = float("inf")
        for epoch in range(1, self.args.epochs + 1):
            start_time = time.time()
            train_loss = self._run_epoch(model, train_loader, optimizer, scheduler, train_mode=True)
            val_loss = self._run_epoch(model, val_loader, optimizer, None, train_mode=False)
            duration = time.time() - start_time
            print(
                f"[epoch {epoch:03d}] train_bpc={train_loss / math.log(2):.4f} "
                f"val_bpc={val_loss / math.log(2):.4f} ({duration:.1f}s)"
            )

            if val_loss < best_val:
                best_val = val_loss
                self._save_checkpoint(artifacts)

    def _run_epoch(self, model, loader, optimizer, scheduler, train_mode: bool) -> float:
        """Process one pass through a dataloader, updating parameters when train_mode is True."""
        model.train(train_mode)
        total_loss, total_tokens = 0.0, 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
        return total_loss / max(1, total_tokens)

    def _save_checkpoint(self, artifacts: TrainingArtifacts) -> None:
        """Persist model weights and tokenizer metadata for later reuse."""
        os.makedirs(self.args.out_dir, exist_ok=True)
        ckpt_path = os.path.join(self.args.out_dir, "model.pt")
        tokenizer = artifacts.tokenizer
        torch.save(
            {
                "model_state": artifacts.model.state_dict(),
                "config": {
                    "vocab_size": tokenizer.vocab_size,
                    "dim": self.args.dim,
                    "depth": self.args.depth,
                    "heads": self.args.heads,
                    "context_len": self.args.context_len,
                },
                "tokenizer": {
                    "kind": tokenizer.kind,
                    "stoi": getattr(tokenizer, "stoi", None),
                    "itos": getattr(tokenizer, "itos", None),
                },
            },
            ckpt_path,
        )
        print(f"[info] saved checkpoint to {ckpt_path}")

    def _sample(self, artifacts: TrainingArtifacts) -> None:
        """Generate a short continuation so we can qualitatively inspect the trained transformer."""
        print("[info] sampling...")
        artifacts.model.eval()
        start = "The"
        idx = torch.tensor([artifacts.tokenizer.encode(start)], dtype=torch.long, device=self.device)
        out_ids = artifacts.model.generate(
            idx,
            max_new_tokens=self.args.sample_len,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
        )
        print(artifacts.tokenizer.decode(out_ids[0].tolist()))
