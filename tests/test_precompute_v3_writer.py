"""V3EmbeddingWriter (precompute_pi05_embeddings.py) round trip test.

Pins the v3 dataset writer used by ``precompute --v3-dataset-root``: frame
schema (observation.state/action + vlm_embeddings fp32 + prefix_mask bool),
per-episode grouping, and value round trip. The writer output must be directly
trainable by ``lerobot-train --policy.type=pi05_rlt --policy.mode=rlt_training``.

Requires the workspace lerobot package with dataset dependencies
(for example the environment created for this project); skipped otherwise.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parents[1]

sys.path.insert(0, str(_ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "precompute_pi05_embeddings", _ROOT / "scripts" / "precompute_pi05_embeddings.py"
)
precompute = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precompute)

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
LeRobotDataset = lerobot_dataset.LeRobotDataset

TASK = "Insert the straw into the cup"


def test_v3_writer_round_trip_and_episode_grouping(tmp_path):
    root = tmp_path / "v3out"
    writer = precompute.V3EmbeddingWriter(root, seq_len=4, vlm_dim=32, fps=30)
    rng = np.random.default_rng(0)
    expected = []
    # episodes 0,0,1,1 -> two episodes of two frames each
    for ep, frame in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        emb = rng.standard_normal((4, 32)).astype(np.float32)
        writer.add_row(
            episode_index=ep,
            state=np.arange(6, dtype=np.float32) + frame,
            action=np.arange(6, dtype=np.float32) - frame,
            task=TASK,
            embeddings=emb,
            mask=np.ones(4, dtype=np.bool_),
        )
        expected.append((ep, emb[0].copy(), float(frame)))
    assert writer.n_rows == 4
    writer.close()

    # meta: two episodes recorded
    import json

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2 and info["total_frames"] == 4

    ds = LeRobotDataset("local/v3out", root=root)
    assert len(ds) == 4
    for i, (ep, first_emb, state0) in enumerate(expected):
        item = ds[i]
        assert int(np.asarray(item["episode_index"]).reshape(-1)[0]) == ep
        assert tuple(item["vlm_embeddings"].shape) == (4, 32)
        np.testing.assert_allclose(item["vlm_embeddings"][0].numpy(), first_emb, rtol=1e-6)
        assert item["prefix_mask"].dtype.__str__() == "torch.bool"
        assert item["task"] == TASK
        # raw state carried verbatim (state[0] = 0 + frame)
        state = item["observation.state"].numpy()
        assert state.shape == (6,)
        assert state[0] == state0
