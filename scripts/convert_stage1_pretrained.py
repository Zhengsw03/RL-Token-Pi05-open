#!/usr/bin/env python3
"""Convert a lerobot-trained Stage-1 pi05_rlt checkpoint into the legacy .pt.

Stage 1 can now be trained with the standard toolchain
(``lerobot-train --policy.type=pi05_rlt --policy.mode=rlt_training``), whose
checkpoints live under ``outputs/<run>/checkpoints/<step>/pretrained_model/``.
The Stage-2 online-RL trainer (``train_rlt_stage2_pi05.py``) still consumes the
legacy Stage-1 ``.pt`` bundle (``encoder_state_dict``/``decoder_state_dict`` +
``rlt_architecture``/``image_only`` metadata). This script bridges the two:

    python scripts/convert_stage1_pretrained.py \
        --pretrained-dir outputs/ltrain_synth/checkpoints/000006/pretrained_model \
        --output outputs/rlt_stage1/best_checkpoint.pt

Round trip: the produced .pt loads strict=True into the same package modules
(``RLTokenEncoder``/``RLTokenDecoder``) that Stage 2 instantiates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys

LEROBOT_SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))


def convert_stage1_pretrained(pretrained_dir: Path, out_pt: Path, *, device: str = "cpu") -> Path:
    """Load a lerobot pi05_rlt pretrained_model dir and write the legacy .pt."""
    from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy

    if not pretrained_dir.is_dir():
        raise SystemExit(f"Pretrained dir not found: {pretrained_dir}")
    policy = PI05RLTPolicy.from_pretrained(str(pretrained_dir)).to(device)
    cfg = policy.config
    if cfg.mode != "rlt_training":
        raise SystemExit(
            f"Expected a Stage-1 (mode=rlt_training) checkpoint, got mode={cfg.mode!r}"
        )

    payload = {
        "rlt_architecture": "paper_v1",
        "image_only": True,
        "step": None,
        "encoder_state_dict": {k: v.detach().cpu() for k, v in policy.rlt_encoder.state_dict().items()},
        "decoder_state_dict": {k: v.detach().cpu() for k, v in policy.rlt_decoder.state_dict().items()},
        "config": {
            # Fields the Stage-2 loader validates against its own PI05RLTConfig.
            "vlm_hidden_dim": cfg.vlm_hidden_dim,
            "rlt_hidden_dim": cfg.rlt_hidden_dim,
            "rlt_encoder_layers": cfg.rlt_encoder_layers,
            "rlt_decoder_layers": cfg.rlt_decoder_layers,
            "rlt_num_heads": cfg.rlt_num_heads,
            "rlt_dropout": cfg.rlt_dropout,
        },
        "provenance": {
            "legacy": False,
            "source": "lerobot-train",
            "pretrained_dir": str(pretrained_dir.resolve()),
        },
    }
    out_pt = Path(out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)
    return out_pt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = convert_stage1_pretrained(args.pretrained_dir, args.output, device=args.device)
    print(f"[OK] converted Stage-1 checkpoint -> {out}")


if __name__ == "__main__":
    main()
