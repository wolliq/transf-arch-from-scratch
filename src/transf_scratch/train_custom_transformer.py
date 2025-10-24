import argparse

from .trainer import Trainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a tiny decoder-only Transformer language model from scratch."
    )

    # General setup
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing .txt files.")
    parser.add_argument("--out_dir", type=str, default="./checkpoints", help="Where to save checkpoints.")
    parser.add_argument(
        "--download_gutenberg",
        action="store_true",
        help="Download a small bundle of public-domain texts into --data_dir.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")

    # Model
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--depth", type=int, default=6, help="Number of decoder blocks.")
    parser.add_argument("--heads", type=int, default=8, help="Attention heads per block.")
    parser.add_argument("--context_len", type=int, default=256, help="Maximum context window.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability.")

    # Training
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Sequences per optimisation step.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument(
        "--stride", type=int, default=None, help="Stride between sequence windows; default is non-overlapping."
    )

    # Sampling
    parser.add_argument("--sample_len", type=int, default=200, help="Number of tokens to sample after training.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature during sampling.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling cutoff; 0 disables truncation.")

    # Tokeniser
    parser.add_argument(
        "--vocab",
        type=str,
        choices=["char", "byte"],
        default="byte",
        help="Character-level or byte-level tokeniser.",
    )

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    trainer = Trainer(args)
    trainer.run()
