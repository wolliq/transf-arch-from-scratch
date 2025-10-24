from __future__ import annotations

from pathlib import Path
import re
import requests
from typing import ClassVar, Dict, Iterable, Tuple

from models.processor import Processor


def project_root() -> Path:
    # .../src/downloader/proc.py → .../src → project root
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "downloaded_data"


class Downloader(Processor):
    _datasources: ClassVar[Dict[str, str]] = {
        "memoirs_of_grant": "https://www.gutenberg.org/ebooks/4367.txt.utf-8",
        "frankenstein": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
        "sleepy_hollow": "https://www.gutenberg.org/ebooks/41.txt.utf-8",
        "origin_of_species": "https://www.gutenberg.org/ebooks/2009.txt.utf-8",
        "makers_of_many_things": "https://www.gutenberg.org/ebooks/28569.txt.utf-8",
        "common_sense": "https://www.gutenberg.org/ebooks/147.txt.utf-8",
        "economic_peace": "https://www.gutenberg.org/ebooks/15776.txt.utf-8",
        "the_great_war_3": "https://www.gutenberg.org/ebooks/29265.txt.utf-8",
        "elements_of_style": "https://www.gutenberg.org/ebooks/37134.txt.utf-8",
        "problem_of_philosophy": "https://www.gutenberg.org/ebooks/5827.txt.utf-8",
        "nights_in_london": "https://www.gutenberg.org/ebooks/23605.txt.utf-8",
    }

    @classmethod
    def get_datasources(cls) -> Iterable[Tuple[str, str]]:
        return cls._datasources.items()

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^\w.-]+", "_", name)

    @classmethod
    def compute(cls) -> Path:
        ddir = data_dir()
        ddir.mkdir(parents=True, exist_ok=True)
        for filename, url in cls.get_datasources():
            safe = cls._safe_name(filename)
            path = ddir / f"{safe}.txt"
            if path.exists():
                continue
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(64 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(path)
        return ddir


if __name__ == "__main__":
    p = Downloader(name="downloader_processor")
    p.compute()
