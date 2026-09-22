#!/usr/bin/env python3
"""CPU probe of the record-style inference chain on an exported RLT policy.

Replicates exactly what ``lerobot-record`` does per control step
(``lerobot.common.control_utils.predict_action``):
raw observation -> prepare_observation_for_inference -> preprocessor ->
``PI05RLTPolicy.select_action`` (lazily attaches the frozen π0.5 from
``config.pi05_pretrained_path``, runs sample_actions -> image-only prefix ->
z_rl -> actor) -> postprocessor.

Synthetic zero images stand in for the cameras: the point is the chain, not
the action quality. Slow on CPU (π0.5 load + diffusion, ~2-4 min).

Usage:
    python scripts/probe_record_inference.py
    python scripts/probe_record_inference.py --pretrained-dir <dir>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys

LEROBOT_SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy
from lerobot.common.control_utils import predict_action

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "rlt_export_check" / "pretrained_model",
                    help="Exported pretrained_model directory (see scripts/export_rlt_pretrained.py)")
    ap.add_argument("--task", default="Insert the straw into the cup")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not args.pretrained_dir.is_dir():
        raise SystemExit(
            f"Pretrained dir not found: {args.pretrained_dir}\n"
            "Export one first: python scripts/export_rlt_pretrained.py "
            "--stage1-checkpoint <stage1.pt> --stage2-checkpoint <stage2.pt> "
            "--pi05-path <pi05/pretrained_model> --output-dir outputs/rlt_export_check"
        )

    pt = str(args.pretrained_dir)
    cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=pt)
    policy = PI05RLTPolicy.from_pretrained(pt).to(args.device)
    overrides = {}
    if args.device != "cuda":
        overrides = {
            "preprocessor_overrides": {"device_processor": {"device": args.device}},
            "postprocessor_overrides": {"device_processor": {"device": args.device}},
        }
    pre, post = make_pre_post_processors(cfg, pretrained_path=pt, **overrides)
    print(f"[probe] policy + processors loaded from {pt}")

    raw = {
        "observation.images.top": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation.state": np.zeros(6, dtype=np.float32),
    }
    t0 = time.time()
    action = predict_action(raw, policy, torch.device(args.device), pre, post,
                            use_amp=False, task=args.task)
    print(f"[probe] predict_action (record-style) done in {time.time() - t0:.1f}s "
          f"-> shape {tuple(action.shape)} dtype {action.dtype}")
    assert tuple(action.shape) == (1, 6), action.shape  # record feeds it to make_robot_action
    assert torch.isfinite(action).all()
    print("[OK] full record inference chain (prepare->pre->select_action->post) works")


if __name__ == "__main__":
    main()
