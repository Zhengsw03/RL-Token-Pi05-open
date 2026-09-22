"""rlt_core.write_warmup_interval_records: journal schema single-source test.

The stride-2 warmup journal writer is shared by extract_annotated_warmup.py
and extract_warmup_from_cache.py (previously two byte-identical copies).
This test pins the 18-field record schema that must stay identical to the
Stage-2 online transitions (Transition.to_numpy), including dtypes, the
WARMUP collection-phase id and the BASE source.
"""
import io
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from rlt_core import COLLECTION_PHASE_WARMUP, TransitionSource, write_warmup_interval_records

RECORD_FIELDS = [
    "z_rl", "proprio", "ref_chunk", "action_chunk", "rewards", "done",
    "next_z_rl", "next_proprio", "next_ref_chunk", "source", "source_chunk",
    "valid_mask", "executed_steps", "collection_phase_id", "success",
    "intervention_flag", "episode_id", "step_id",
]


def _make_fixture():
    """Episode 0, annotated frames 10..24 (15 frames), window 10 / stride 2."""
    frame_data = {}
    for frame in range(10, 25):
        frame_data[(0, frame)] = (
            np.full(4, float(frame), dtype=np.float16),          # z_rl
            np.arange(6, dtype=np.float32) + frame,              # proprio
            np.full(6, 0.5 * frame, dtype=np.float32),           # action
        )
    return frame_data


def test_writes_stride2_windows_with_full_schema():
    frame_data = _make_fixture()
    out = io.BytesIO()
    n = write_warmup_interval_records(
        out, ep=0, frames_in_interval=list(range(10, 25)),
        frame_data=frame_data, horizon=10, stride=2, action_dim=6,
    )
    # 15 frames -> windows at pos 0,2,...,14 (8 windows)
    assert n == 8

    out.seek(0)
    records = []
    while True:
        try:
            records.append(pickle.load(out))
        except EOFError:
            break
    assert len(records) == 8

    for rec in records:
        assert list(rec) == RECORD_FIELDS, list(rec)
        assert rec["z_rl"].dtype == np.float16 and rec["z_rl"].shape == (4,)
        assert rec["proprio"].dtype == np.float32 and rec["proprio"].shape == (6,)
        for key in ("ref_chunk", "action_chunk", "next_ref_chunk"):
            assert rec[key].dtype == np.float16 and rec[key].shape == (10, 6)
        assert rec["rewards"].dtype == np.float32 and rec["rewards"].shape == (10,)
        assert rec["done"] == np.asarray(False, dtype=np.bool_)
        assert rec["source"] == np.asarray(TransitionSource.BASE, dtype=np.uint8)
        assert rec["source_chunk"].dtype == np.uint8 and rec["source_chunk"].shape == (10,)
        assert rec["collection_phase_id"] == np.asarray(COLLECTION_PHASE_WARMUP, dtype=np.uint8)
        assert rec["success"] == np.asarray(0, dtype=np.int8)
        assert rec["intervention_flag"] == np.asarray(False, dtype=np.bool_)
        assert rec["episode_id"] == np.asarray(0, dtype=np.int32)


def test_window_content_semantics():
    frame_data = _make_fixture()
    out = io.BytesIO()
    write_warmup_interval_records(
        out, ep=0, frames_in_interval=list(range(10, 25)),
        frame_data=frame_data, horizon=10, stride=2, action_dim=6,
    )
    out.seek(0)
    records = [pickle.load(out) for _ in range(2)]  # pos 0 (frames 10-19) & pos 2 (12-21)
    r0, r1 = records
    assert r0["executed_steps"] == np.asarray(10, dtype=np.int16)
    assert r0["valid_mask"].all()
    assert r0["step_id"] == np.asarray(10, dtype=np.int32)
    # first window: next_ref_chunk = frames 20..24 (actions 0.5*frame) then zero padding
    expected_next = np.full((5, 6), 0.5 * np.arange(20, 25, dtype=np.float32)[:, None],
                            dtype=np.float16)
    assert np.allclose(r0["next_ref_chunk"][:5], expected_next)
    assert (r0["next_ref_chunk"][5:] == 0).all()
    # z_rl of window start == z of frame 10; next_z_rl == z of the frame right
    # after the window (frame 20 for window 0), clamped to the interval tail
    assert r0["z_rl"][0] == np.float16(10.0)
    assert r0["next_z_rl"][0] == np.float16(20.0)
    # second window starts at frame 12; its next window (22..24) has 3 real frames
    assert r1["step_id"] == np.asarray(12, dtype=np.int32)
    assert (r1["next_ref_chunk"][:3] != 0).all() and (r1["next_ref_chunk"][3:] == 0).all()


def test_partial_tail_window_and_action_copy():
    frame_data = _make_fixture()
    out = io.BytesIO()
    write_warmup_interval_records(
        out, ep=0, frames_in_interval=list(range(10, 25)),
        frame_data=frame_data, horizon=10, stride=2, action_dim=6,
    )
    out.seek(0)
    records = []
    while True:
        try:
            records.append(pickle.load(out))
        except EOFError:
            break
    last = records[-1]  # pos 14 -> single frame 24
    assert last["executed_steps"] == np.asarray(1, dtype=np.int16)
    assert last["valid_mask"].sum() == 1
    # executed action is copied verbatim from frame 24
    assert np.allclose(last["action_chunk"][0], np.full(6, 12.0, dtype=np.float32))
