#!/usr/bin/env python3
"""Compatibility wrapper to run the refactored transformer trainer package."""

from pathlib import Path
import sys

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from transf_scratch import main

    main()
