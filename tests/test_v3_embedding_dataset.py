"""V3 LeRobotDataset custom array features (prefix embedding) round trip.

Validates the v3 embedding-dataset data layer: a v3 dataset can carry per-frame custom
array features such as ``prefix_embedding (L, D)`` / ``prefix_mask (L,)``
alongside the usual state/action/task, and read them back exactly.

Findings pinned here:
- feature specs must be passed to ``LeRobotDataset.create`` with TUPLE shapes
  (write-time validation compares against np.ndarray.shape tuples);
- ``float16`` array features are NOT supported (quantile stats + parquet), so
  embeddings must be stored as float32 (or quantized);
- frames require a ``task`` field; episodes are closed with ``save_episode``
  and the dataset with ``finalize``.

Requires the workspace lerobot package with its dataset dependencies
(for example the environment created for this project); skipped otherwise.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# The v3 parquet reader caches under ~/.cache by default, which is read-only in
# some sandboxes; redirect to a writable temp dir before any dataset access.
_CACHE = Path(tempfile.gettempdir()) / "hf_datasets_cache_zsw_test"
os.environ["HF_HOME"] = str(_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(_CACHE / "datasets")
_CACHE.mkdir(parents=True, exist_ok=True)

LEROBOT_SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
LeRobotDataset = lerobot_dataset.LeRobotDataset

TASK = "Insert the straw into the cup"


def _make_dataset(root: Path, dtype: str = "float32"):
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,)},
        "action": {"dtype": "float32", "shape": (6,)},
        "prefix_embedding": {"dtype": dtype, "shape": (4, 32)},
        "prefix_mask": {"dtype": "bool", "shape": (4,)},
    }
    ds = LeRobotDataset.create("local/embedding_test", fps=30, features=features,
                               root=root, robot_type="so_follower", use_videos=False)
    rng = np.random.default_rng(0)
    written = []
    for ep in range(2):
        for f in range(3):
            emb = rng.standard_normal((4, 32)).astype(np.float32 if dtype == "float32" else np.float16)
            frame = {
                "observation.state": rng.standard_normal(6).astype(np.float32),
                "action": rng.standard_normal(6).astype(np.float32),
                "prefix_embedding": emb,
                "prefix_mask": np.ones(4, dtype=np.bool_),
                "task": TASK,
            }
            ds.add_frame(frame)
            written.append(emb[0].copy())
        ds.save_episode()
    ds.finalize()
    return written


def test_v3_dataset_round_trips_custom_embedding_features(tmp_path):
    root = tmp_path / "ds1"
    written = _make_dataset(root)
    ds = LeRobotDataset("local/embedding_test", root=root)
    assert len(ds) == 6
    item = ds[0]
    assert tuple(item["prefix_embedding"].shape) == (4, 32)
    assert item["prefix_embedding"].dtype == torch.float32
    assert item["prefix_mask"].dtype == torch.bool
    np.testing.assert_allclose(item["prefix_embedding"][0].numpy(), written[0], rtol=1e-6)
    assert item["task"] == TASK
    # stats.json is written for the numerical features (required by tooling)
    assert (root / "meta" / "stats.json").is_file()


def test_v3_dataset_accepts_float16_embedding_features(tmp_path):
    """float16 array features round-trip end-to-end on current lerobot.

    The v3 design doc pinned the old behaviour where float16 raised
    (quantile stats over float16); upstream now accepts it, so pin the
    acceptance instead. Our convert pipeline still writes float32.
    """
    root = tmp_path / "fp16"
    written = _make_dataset(root, dtype="float16")
    ds = LeRobotDataset("local/embedding_test", root=root)
    item = ds[0]
    # Upstream upcasts to float32 on read (quantile-stats path); pin values only.
    assert item["prefix_embedding"].dtype == torch.float32
    np.testing.assert_allclose(item["prefix_embedding"][0].numpy(), written[0], rtol=1e-3)


def test_v3_dataset_coerces_list_shapes_at_create(tmp_path):
    """List shapes at create() are coerced to tuples (upstream PR #4232 behaviour).

    The v3 design doc pinned the old behaviour where a list shape raised
    ValueError at write time; current lerobot normalises list shapes to
    tuples instead, so pin the coercion (and round-trip) explicitly.
    """
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,)},
        "action": {"dtype": "float32", "shape": (6,)},
        "prefix_embedding": {"dtype": "float32", "shape": [4, 32]},  # list on purpose
        "prefix_mask": {"dtype": "bool", "shape": (4,)},
    }
    root = tmp_path / "listshape"
    ds = LeRobotDataset.create("local/embedding_test2", fps=30, features=features,
                               root=root, robot_type="so_follower", use_videos=False)
    assert tuple(ds.meta.info["features"]["prefix_embedding"]["shape"]) == (4, 32)
    frame = {
        "observation.state": np.zeros(6, dtype=np.float32),
        "action": np.zeros(6, dtype=np.float32),
        "prefix_embedding": np.zeros((4, 32), dtype=np.float32),
        "prefix_mask": np.ones(4, dtype=np.bool_),
        "task": TASK,
    }
    ds.add_frame(frame)
    ds.save_episode()
    ds.finalize()
    reloaded = LeRobotDataset("local/embedding_test2", root=root)
    item = reloaded[0]
    assert tuple(item["prefix_embedding"].shape) == (4, 32)
    assert item["prefix_embedding"].dtype == torch.float32
