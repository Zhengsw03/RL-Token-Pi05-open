"""Shared rlt_core parsers/quantile helpers + eval parse_args smoke.

These pure functions gate the camera/limit contract shared by the Stage-2
trainer and the evaluator, so their behavior is pinned here (previously
untested duplicated copies lived in both scripts).
"""
import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from rlt_core import (  # noqa: E402
    _validate_quantiles,
    parse_camera_map,
    parse_max_relative_target,
    quantile_normalize,
)


# ── parse_camera_map ────────────────────────────────────────────────────────

def test_camera_map_exact_contract_passes():
    mapping = parse_camera_map(
        '{"top": "observation.images.top", "wrist": "observation.images.wrist"}',
        required_features=["observation.images.top", "observation.images.wrist"],
        configured_cameras={"top", "wrist"},
    )
    assert mapping == {"top": "observation.images.top", "wrist": "observation.images.wrist"}


def test_camera_map_contract_violations_raise():
    feats = ["observation.images.top", "observation.images.wrist"]
    with pytest.raises(ValueError, match="required"):
        parse_camera_map("", feats, {"top"})
    with pytest.raises(ValueError, match="valid JSON"):
        parse_camera_map("{oops", feats, {"top"})
    with pytest.raises(ValueError, match="exactly match configured cameras"):
        parse_camera_map('{"top": "observation.images.top"}', feats, {"top", "wrist"})
    with pytest.raises(ValueError, match="distinct"):
        parse_camera_map(
            '{"top": "observation.images.top", "wrist": "observation.images.top"}',
            feats, {"top", "wrist"},
        )
    with pytest.raises(ValueError, match="unknown checkpoint features"):
        parse_camera_map(
            '{"top": "observation.images.nope", "wrist": "observation.images.wrist"}',
            feats, {"top", "wrist"},
        )
    with pytest.raises(ValueError, match="Missing="):
        parse_camera_map('{"top": "observation.images.top"}', feats, {"top"})


def test_camera_map_pad_missing_is_test_only():
    feats = ["observation.images.top", "observation.images.wrist"]
    with mock.patch("rlt_core.logger.warning") as warn:
        mapping = parse_camera_map(
            '{"top": "observation.images.top"}', feats, {"top"}, pad_missing=True,
        )
    warn.assert_called_once()
    assert mapping == {"top": "observation.images.top"}


# ── parse_max_relative_target ───────────────────────────────────────────────

def test_max_relative_target_scalar_and_dict():
    assert parse_max_relative_target(None) is None
    assert parse_max_relative_target("5.0") == 5.0
    assert parse_max_relative_target("5") == 5.0
    assert parse_max_relative_target('{"shoulder_pan": 7.0, "gripper": 6.5}') == {
        "shoulder_pan": 7.0, "gripper": 6.5,
    }


def test_max_relative_target_rejects_bad_values():
    for bad in ("0", "-3", "abc", "true", "{}", '{"a": 0}', '{"a": -1}', '{"a": "x"}'):
        with pytest.raises(ValueError):
            parse_max_relative_target(bad)


# ── _validate_quantiles ─────────────────────────────────────────────────────

def _stats(q01=(-10, -9, -8, -7, -6, -5), q99=(10, 9, 8, 7, 6, 5)):
    return {
        "observation.state": {"q01": list(q01), "q99": list(q99)},
        "action": {"q01": list(q01), "q99": list(q99)},
    }


def test_quantiles_accept_valid_6d_stats():
    out = _validate_quantiles(_stats())
    assert set(out) == {"observation.state", "action"}
    assert out["action"]["q01"] == [-10, -9, -8, -7, -6, -5]
    assert out["action"]["q99"] == [10, 9, 8, 7, 6, 5]


def test_quantiles_reject_bad_stats():
    with pytest.raises(ValueError, match="missing"):
        _validate_quantiles({"action": {"q01": [0] * 6, "q99": [1] * 6}})
    with pytest.raises(ValueError, match="exactly 6D"):
        _validate_quantiles(_stats(q01=(1, 2, 3)))
    with pytest.raises(ValueError, match="finite"):
        _validate_quantiles(_stats(q01=(float("nan"),) + (0,) * 5))
    with pytest.raises(ValueError, match="q99 > q01"):
        _validate_quantiles(_stats(q99=(-10, -9, -8, -7, -6, -5)))


# ── quantile_normalize ──────────────────────────────────────────────────────

def test_quantile_normalize_maps_to_minus_one_one():
    q01 = torch.tensor([0.0, 0.0])
    q99 = torch.tensor([10.0, 100.0])
    out = quantile_normalize(torch.tensor([[0.0, 100.0], [5.0, 50.0], [10.0, 0.0]]), q01, q99)
    torch.testing.assert_close(out, torch.tensor([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]]))


def test_quantile_normalize_guards_zero_range():
    q01 = torch.tensor([5.0])
    q99 = torch.tensor([5.0])  # degenerate: must not divide by zero
    out = quantile_normalize(torch.tensor([[5.0], [7.0]]), q01, q99)
    assert torch.isfinite(out).all()


# ── eval parse_args smoke ───────────────────────────────────────────────────

def _load_eval_module():
    path = Path(__file__).parents[1] / "scripts" / "eval_rlt_pi05_so101.py"
    spec = importlib.util.spec_from_file_location("eval_rlt_pi05_so101", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_parse_args_minimal_and_config_file():
    module = _load_eval_module()
    argv = ["eval_rlt_pi05_so101.py",
            "--pi05_path", "p", "--rlt_checkpoint", "c", "--actor_checkpoint", "a",
            "--task", "t", "--follower_port", "/dev/ttyACM0",
            "--cameras", '{"top": {"type": "opencv", "index_or_path": 3}}',
            "--camera_map", '{"top": "observation.images.top"}',
            "--dry_run"]
    with mock.patch.object(module.sys, "argv", argv):
        args = module.parse_args()
    assert args.dry_run and args.episodes == 10  # parser defaults
    assert args.camera_map == '{"top": "observation.images.top"}'

    # Real config file path must keep parsing (pure JSON eval config).
    config_path = Path(__file__).parents[1] / "configs" / "eval_straw_pi05.json"
    if config_path.is_file():
        argv2 = ["eval_rlt_pi05_so101.py", "--config", str(config_path), "--dry_run"]
        with mock.patch.object(module.sys, "argv", argv2):
            args2 = module.parse_args()
        assert args2.task  # loaded from the config file
        assert args2.dry_run
