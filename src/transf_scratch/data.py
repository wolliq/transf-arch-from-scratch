import os
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None

from loguru import logger

GUTENBERG_URLS: Dict[str, str] = {
    "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
    "sleepy_hollow": "https://www.gutenberg.org/ebooks/41.txt.utf-8",
    "common_sense": "https://www.gutenberg.org/ebooks/147.txt.utf-8",
}


def maybe_download_gutenberg(out_dir: str) -> None:
    """Download a small bundle of public-domain texts if they are missing."""
    os.makedirs(out_dir, exist_ok=True)
    if requests is None:
        logger.warning("'requests' not available; skipping Gutenberg download.")
        return
    for name, url in GUTENBERG_URLS.items():
        path = os.path.join(out_dir, f"{name}.txt")
        if os.path.exists(path):
            continue
        logger.info("Downloading Gutenberg text: {}", name)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(path, "wb") as handle:
            handle.write(resp.content)


def read_all_texts(data_dir: str) -> str:
    """Concatenate every .txt file in data_dir after stripping Gutenberg boilerplate."""
    texts: List[str] = []
    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(".txt"):
            continue
        logger.debug("Reading corpus file {}", filename)
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
        start = content.find("*** START OF THE PROJECT GUTENBERG EBOOK")
        if start != -1:
            start = content.find("\n", start) + 1
        else:
            start = 0
        end = content.find("*** END OF THE PROJECT GUTENBERG EBOOK")
        if end == -1:
            end = len(content)
        texts.append(content[start:end].strip())
    if not texts:
        raise RuntimeError(f"No .txt files found in {data_dir}. Provide data or enable --download_gutenberg.")
    logger.info("Loaded {} text files from {}", len(texts), data_dir)
    return "\n\n".join(texts)


class LMSequenceDataset(Dataset):
    """Language modelling dataset producing (context, target) slices."""

    def __init__(self, ids: List[int], context_len: int, step: int = None):
        """Pre-slice the token stream into fixed-length windows for autoregressive training."""
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context_len = context_len
        self.step = step or context_len
        self.num_sequences = max(0, (len(self.ids) - context_len - 1) // self.step + 1)

    def __len__(self) -> int:
        """Return number of available training examples given stride and context length."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one context-target pair where the target is the next-token shift of the context."""
        start = idx * self.step
        end = start + self.context_len
        x = self.ids[start:end]
        y = self.ids[start + 1 : end + 1]
        return x, y


def split_data(ids: List[int], train_ratio: float = 0.9) -> Tuple[List[int], List[int]]:
    """Partition token ids into training and validation splits for model evaluation."""
    cut = int(len(ids) * train_ratio)
    return ids[:cut], ids[cut:]


def _stack_batch(batch: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function that stacks variable into batch tensors for the transformer."""
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def build_dataloaders(
    train_ids: List[int],
    val_ids: List[int],
    context_len: int,
    stride: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders feeding the transformer with fixed windows."""
    train_dataset = LMSequenceDataset(train_ids, context_len=context_len, step=stride or context_len)
    val_dataset = LMSequenceDataset(val_ids, context_len=context_len, step=context_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_stack_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_stack_batch,
    )
    return train_loader, val_loader
