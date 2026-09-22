"""Pytest session setup: redirect HF caches to a writable location.

Some sandboxes mount ~/.cache read-only; the LeRobot v3 dataset reader caches
parquet under $HF_DATASETS_CACHE at library-import time, so the env vars must
be set before any test module imports ``datasets``/``lerobot.datasets``.
"""
import os
import tempfile
from pathlib import Path

_cache = Path(tempfile.gettempdir()) / "hf_datasets_cache_rlt_pytest"
os.environ.setdefault("HF_HOME", str(_cache))
os.environ.setdefault("HF_DATASETS_CACHE", str(_cache / "datasets"))
_cache.mkdir(parents=True, exist_ok=True)
