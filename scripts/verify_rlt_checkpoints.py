#!/usr/bin/env python3
"""Verify that real trained RL-Token-Pi05 checkpoints load into the lerobot
``pi05_rlt`` policy package (single source of truth) with strict=True.

This is the compatibility gate for the "lerobot-native pipeline" refactor:
Stage-1 encoder/decoder checkpoints and Stage-2 (schema v3,
``paper_full_output_v1``) actor/critic checkpoints must be loadable by
``lerobot.policies.pi05_rlt`` modules without any key/shape conversion.

Usage:
    python scripts/verify_rlt_checkpoints.py \
        [--stage1-checkpoint <path>] [--stage2-checkpoint <path>]

Defaults point at the workspace artifacts produced by the v1.x pipeline.
Exits 0 only when every component loads strict=True and a forward pass
produces the expected shapes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Resolve the workspace lerobot package (editable install) even when running
# from a bare checkout.
import sys

LEROBOT_SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import (
    RLTokenDecoder,
    RLTokenEncoder,
    RLTChunkActor,
    RLTTwinCritic,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg_from_stage1(ckpt: dict, device: str) -> PI05RLTConfig:
    """Rebuild the config from the checkpoint's own saved fields when present."""
    saved = ckpt.get("config", {})
    kwargs = {
        "mode": "rlt_training",
        "device": device,
    }
    for field in (
        "vlm_hidden_dim", "rlt_hidden_dim", "rlt_encoder_layers",
        "rlt_decoder_layers", "rlt_num_heads", "rlt_dropout",
    ):
        if field in saved:
            kwargs[field] = saved[field]
    return PI05RLTConfig(**kwargs)


def verify_stage1(path: Path, device: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("rlt_architecture") != "paper_v1":
        raise SystemExit(f"Not a paper_v1 Stage-1 checkpoint: {path}")
    cfg = _cfg_from_stage1(ckpt, device)
    enc = RLTokenEncoder(cfg).to(device).eval()
    enc.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    dec = RLTokenDecoder(cfg).to(device).eval()
    dec.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    # Forward smoke on the trained dims.
    d = cfg.rlt_hidden_dim
    z = torch.randn(1, d, device=device)
    targets = torch.randn(1, 8, d, device=device)
    with torch.no_grad():
        out = dec(z, targets)
        z_rl = enc(targets)
    assert tuple(out.shape) == (1, 8, d), out.shape
    assert tuple(z_rl.shape) == (1, d), z_rl.shape
    print(f"[OK] Stage-1 encoder+decoder  strict load + forward  ({path.name})")


def verify_stage2(path: Path, device: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("schema_version") != 3 or ckpt.get("actor_contract") != "paper_full_output_v1":
        raise SystemExit(f"Not a schema-v3 paper_full_output_v1 Stage-2 checkpoint: {path}")
    # Stage-2 ckpts do not store the full config; infer from state-dict shapes.
    asd = ckpt["actor_state_dict"]
    z_state_ref_in = asd["net.0.weight"].shape[1]
    cfg = PI05RLTConfig(
        mode="online_rl",
        state_dim=6,
        action_dim=6,
        action_stride=2,
        n_action_steps_rl=ckpt["rl_chunk_length"],
        rlt_hidden_dim=z_state_ref_in - 6 - ckpt["rl_chunk_length"] * 6,
        actor_hidden_dim=asd["net.0.weight"].shape[0],
        actor_num_layers=len({k.split(".")[1] for k in asd if k.startswith("net.")}),
        critic_hidden_dim=ckpt["critic_state_dict"]["q1.0.weight"].shape[0],
        critic_num_layers=len({k.split(".")[1] for k in ckpt["critic_state_dict"] if k.startswith("q1.")}),
        device=device,
    )
    actor = RLTChunkActor(cfg).to(device).eval()
    actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
    critic = RLTTwinCritic(cfg).to(device).eval()
    critic.load_state_dict(ckpt["critic_state_dict"], strict=True)
    B = 2
    z = torch.randn(B, cfg.rlt_hidden_dim, device=device)
    state = torch.randn(B, cfg.state_dim, device=device)
    ref = torch.randn(B, cfg.n_action_steps_rl, cfg.action_dim, device=device)
    with torch.no_grad():
        mean, std = actor(z, state, ref)
        q1, q2 = critic(z, state, ref)
    assert tuple(mean.shape) == (B, cfg.n_action_steps_rl, cfg.action_dim)
    assert torch.allclose(std, torch.full_like(std, cfg.policy_fixed_std))
    assert tuple(q1.shape) == (B, 1) and tuple(q2.shape) == (B, 1)
    print(f"[OK] Stage-2 actor+critic     strict load + forward  ({path.name})")


def verify_pretrained(path: Path, device: str, source_stage2: Path | None = None) -> None:
    """Load a lerobot-native pretrained_model dir exported by export_rlt_pretrained.py.

    Checks the round trip: from_pretrained succeeds, no frozen-VLA keys are
    present, RLT components forward on the trained dims, and (when a source
    stage-2 checkpoint is given) every actor/critic weight matches the
    original artifact exactly.
    """
    from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy

    policy = PI05RLTPolicy.from_pretrained(str(path)).to(device)
    assert not any("frozen_vla" in k for k in policy.state_dict()), "exported checkpoint contains VLA keys!"
    cfg = policy.config
    print(f"[OK] from_pretrained {path.name}  mode={cfg.mode} z_dim={cfg.rlt_hidden_dim} "
          f"chunk={cfg.n_action_steps_rl} pi05_pretrained_path={cfg.pi05_pretrained_path!r}")
    B = 2
    z = torch.randn(B, cfg.rlt_hidden_dim, device=device)
    state = torch.randn(B, cfg.state_dim, device=device)
    ref = torch.randn(B, cfg.n_action_steps_rl, cfg.action_dim, device=device)
    with torch.no_grad():
        mean, std = policy.actor(z, state, ref)
        q1, _ = policy.critic(z, state, ref)
    assert tuple(mean.shape) == (B, cfg.n_action_steps_rl, cfg.action_dim)
    assert torch.allclose(std, torch.full_like(std, cfg.policy_fixed_std))
    assert tuple(q1.shape) == (B, 1)
    print("[OK] actor/critic forward on exported policy")

    if source_stage2 is not None:
        ck2 = torch.load(source_stage2, map_location="cpu", weights_only=False)
        attr_of = {"actor_state_dict": "actor", "actor_target_state_dict": "actor_target",
                   "critic_state_dict": "critic", "critic_target_state_dict": "critic_target"}
        for name, src in ck2.items():
            if name not in attr_of:
                continue
            target = getattr(policy, attr_of[name]).state_dict()
            assert set(target) == set(src), f"key mismatch in {name}"
            for key in src:
                torch.testing.assert_close(target[key], src[key], msg=f"{name}.{key}")
        print("[OK] exported weights identical to source stage-2 checkpoint")

    # Processor artifacts: load exactly like lerobot toolchains do
    # (record/eval load policy_preprocessor.json / policy_postprocessor.json).
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    pcfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=str(path))
    overrides = {}
    if device != "cuda":
        overrides = {
            "preprocessor_overrides": {"device_processor": {"device": device}},
            "postprocessor_overrides": {"device_processor": {"device": device}},
        }
    pre, post = make_pre_post_processors(pcfg, pretrained_path=str(path), **overrides)
    print(f"[OK] pre/post processors loaded via make_pre_post_processors "
          f"({len(pre.steps)} pre-steps, {len(post.steps)} post-steps)")


def verify_record_style_load(path: Path, dataset_root: Path, device: str) -> None:
    """Load the exported dir exactly like ``lerobot-record --policy.path`` does.

    ``make_policy(cfg, ds_meta)`` backfills the policy features from the dataset
    metadata and instantiates the policy from the exported pretrained dir;
    ``make_pre_post_processors(pretrained_path=...)`` loads the exported
    processor files. The frozen π0.5 backbone stays lazy (attached on first
    inference via ``config.pi05_pretrained_path`` on the robot machine).
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    meta = LeRobotDatasetMetadata(repo_id="local/export_check", root=dataset_root)
    cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=str(path))
    cfg.pretrained_path = str(path)
    policy = make_policy(cfg=cfg, ds_meta=meta)
    assert type(policy).__name__ == "PI05RLTPolicy", type(policy)
    assert not policy._vla_loaded  # backbone stays lazy until robot inference
    overrides = {}
    if device != "cuda":
        overrides = {
            "preprocessor_overrides": {"device_processor": {"device": device}},
            "postprocessor_overrides": {"device_processor": {"device": device}},
        }
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(path), **overrides)
    print(f"[OK] record-style load: make_policy -> {type(policy).__name__}, "
          f"processors {len(pre.steps)}/{len(post.steps)} steps, "
          f"pi05_pretrained_path={cfg.pi05_pretrained_path!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-checkpoint", type=Path, default=None,
                    help="Stage-1 .pt to verify (required unless --pretrained-dir is given)")
    ap.add_argument("--stage2-checkpoint", type=Path, default=None,
                    help="Stage-2 .pt to verify (required unless --pretrained-dir is given)")
    ap.add_argument("--pretrained-dir", type=Path, default=None,
                    help="Lerobot-native pretrained_model dir to verify (instead of legacy .pt files)")
    ap.add_argument("--dataset-root", type=Path, default=None,
                    help="Optional LeRobotDataset root for the record-style load check "
                         "(make_policy + make_pre_post_processors, used with --pretrained-dir)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.pretrained_dir is not None:
        if not args.pretrained_dir.is_dir():
            raise SystemExit(f"Pretrained dir not found: {args.pretrained_dir}")
        verify_pretrained(args.pretrained_dir, args.device, args.stage2_checkpoint)
        if args.dataset_root is not None:
            if not args.dataset_root.is_dir():
                raise SystemExit(f"Dataset root not found: {args.dataset_root}")
            verify_record_style_load(args.pretrained_dir, args.dataset_root, args.device)
        return

    if args.stage1_checkpoint is None or args.stage2_checkpoint is None:
        raise SystemExit(
            "Pass --pretrained-dir <exported pretrained_model dir>, or both "
            "--stage1-checkpoint and --stage2-checkpoint .pt files to verify."
        )
    for p in (args.stage1_checkpoint, args.stage2_checkpoint):
        if not p.is_file():
            raise SystemExit(f"Checkpoint not found: {p}")
    verify_stage1(args.stage1_checkpoint, args.device)
    verify_stage2(args.stage2_checkpoint, args.device)
    print("All real checkpoints verified against lerobot.policies.pi05_rlt.")


if __name__ == "__main__":
    main()
