import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

from .models import CharTokenizer, TinyDecoderLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a trained TinyDecoderLM checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt", help="Path to model checkpoint.")
    parser.add_argument("--prompt", type=str, default="The ", help="Initial prompt to condition generation.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of new tokens to sample.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for sampling.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k truncation (0 disables).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    return parser


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    config = ckpt["config"]
    tokenizer_meta = ckpt["tokenizer"]
    if tokenizer_meta["kind"] == "byte":
        # Byte tokenizer does not require vocab reconstruction
        tokenizer = CharTokenizer("", kind="byte")
    else:
        tokenizer = CharTokenizer("", kind="char")
        tokenizer.stoi = tokenizer_meta["stoi"]
        tokenizer.itos = tokenizer_meta["itos"]
        tokenizer.vocab_size = len(tokenizer.stoi)
    model = TinyDecoderLM(
        vocab_size=config["vocab_size"],
        dim=config["dim"],
        depth=config["depth"],
        num_heads=config["heads"],
        context_len=config["context_len"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return tokenizer, model


def main(argv=None):
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    args = build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info("Loading checkpoint {} on {}", args.checkpoint, device)

    path = Path(args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    tokenizer, model = load_checkpoint(path, device=device)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("Prompt must produce at least one token with the checkpoint tokenizer.")

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logger.info("Generating with prompt {!r}, max_new_tokens={}, temperature={}, top_k={}",
                args.prompt, args.max_new_tokens, args.temperature, args.top_k)
    out_ids = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(out_ids[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
