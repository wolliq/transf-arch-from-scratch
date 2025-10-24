from __future__ import annotations

from pathlib import Path

from downloader.processor import Downloader
from models.processor import Processor

from loguru import logger


class Loader(Processor):

    @classmethod
    def preprocess_gutenberg(cls, filepath: Path) -> str:
        text = filepath.read_text(encoding="utf-8")

        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        s = text.find(start_marker)
        e = text.find(end_marker)

        if s != -1:
            s = text.find("\n", s) + 1
        else:
            s = 0
        if e == -1:
            e = len(text)

        body = text[s:e].strip()
        return "\n".join(line.strip() for line in body.splitlines() if line.strip())

    def compute(self) -> list[str]:
        # 1) Ensure files exist in x-project/downloaded_data
        ddir = Downloader.compute()

        # 2) Read & process
        all_text: list[str] = []
        for filename, _ in Downloader.get_datasources():
            path = ddir / f"{filename}.txt"
            if not path.exists():
                raise FileNotFoundError(f"Expected file missing: {path}")
            logger.debug(f"Loaded {filename}")
            all_text.append(self.preprocess_gutenberg(path))
        return all_text


if __name__ == "__main__":
    # Run from project root:
    #   PYTHONPATH=src python -m loader.proc
    p = Loader(name="loader_processor")
    texts = p.compute()
