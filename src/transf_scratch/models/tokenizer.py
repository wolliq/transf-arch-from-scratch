from typing import Dict, List, Optional


class CharTokenizer:
    """Character or byte-level tokenizer implemented without external dependencies."""

    def __init__(self, text: str, kind: str = "char"):
        """Analyse corpus to build lookup tables or configure byte-level ids."""
        if kind == "byte":
            self.kind = "byte"
            self.vocab_size = 256
            self.stoi: Optional[Dict[str, int]] = None
            self.itos: Optional[Dict[int, str]] = None
        else:
            self.kind = "char"
            chars = sorted({ch for ch in text})
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.vocab_size = len(self.stoi)

    def encode(self, value: str) -> List[int]:
        """Convert raw text into integer token ids ready for the transformer embedding."""
        if self.kind == "byte":
            return list(value.encode("utf-8", errors="ignore"))
        if self.stoi is None:
            raise ValueError("Character tokenizer requires stoi mapping to encode text.")
        return [self.stoi[ch] for ch in value if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        """Map model outputs back into readable text for inspection."""
        if self.kind == "byte":
            return bytes([int(i) % 256 for i in ids]).decode("utf-8", errors="ignore")
        if self.itos is None:
            raise ValueError("Character tokenizer requires itos mapping to decode ids.")
        return "".join(self.itos[i] for i in ids if i in self.itos)
