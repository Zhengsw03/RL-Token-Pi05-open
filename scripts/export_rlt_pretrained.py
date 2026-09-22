#!/usr/bin/env python3
"""Export RL-Token-Pi05 stage checkpoints into a lerobot-native pretrained dir.

Bridges the legacy v1.x artifacts (stage1 ``best_checkpoint.pt`` +
stage2 ``final_checkpoint.pt``, schema v3) into the standard lerobot layout
``<output>/pretrained_model/{config.json, model.safetensors}`` understood by
``PI05RLTPolicy.from_pretrained`` / ``lerobot`` checkpoint tooling (hub push,
record-style loading, future eval).

The saved checkpoint contains ONLY the compact RLT components
(encoder/decoder/actor/critic/target); the frozen π0.5 backbone is not dumped
— it is referenced through ``config.pi05_pretrained_path`` and attached lazily
at inference time (``attach_frozen_vla_from_config``).

Usage:
    python scripts/export_rlt_pretrained.py \
        --stage1-checkpoint outputs/rlt_stage1/best_checkpoint.pt \
        --stage2-checkpoint outputs/rlt_stage2/final_checkpoint.pt \
        --pi05-path /path/to/pi05/pretrained_model \
        --output-dir outputs/rlt_export_check

Verification (round trip, no GPU needed):
    python scripts/verify_rlt_checkpoints.py --pretrained-dir <output>/pretrained_model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

LEROBOT_SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))


_CONFIG_FIELDS = {
    "mode", "device", "vlm_hidden_dim", "rlt_hidden_dim", "rlt_encoder_layers",
    "rlt_decoder_layers", "rlt_num_heads", "rlt_dropout", "state_dim", "action_dim",
    "action_stride", "n_action_steps_rl", "actor_critic_style", "actor_hidden_dim",
    "actor_num_layers", "critic_hidden_dim", "critic_num_layers", "policy_fixed_std",
    "actor_residual_scale", "discount", "bc_weight", "target_tau", "action_feature_names",
}


def _linear_width_and_depth(state_dict: dict, prefix: str) -> tuple[int | None, int | None]:
    weights = []
    for key, value in state_dict.items():
        if key.startswith(prefix) and key.endswith(".weight") and hasattr(value, "shape"):
            try:
                layer_index = int(key.split(".")[1])
            except (IndexError, ValueError):
                continue
            weights.append((layer_index, value))
    if not weights:
        return None, None
    weights.sort(key=lambda item: item[0])
    return int(weights[0][1].shape[0]), len(weights)


def build_export_config_kwargs(stage1_ckpt: dict, stage2_ckpt: dict, device: str = "cpu") -> dict:
    """Reconstruct the exact RLT architecture needed by a native policy load.

    Stage-1 checkpoints carry encoder/decoder fields while Stage-2 carries the
    authoritative actor contract and chunk length.  Combining both avoids the
    old exporter silently falling back to package defaults.
    """
    saved = stage1_ckpt.get("config", {})
    if not isinstance(saved, dict):
        saved = {}
    provenance = stage2_ckpt.get("runtime_provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    from rlt_core import SO101_JOINT_NAMES

    kwargs = {"mode": "online_rl", "device": device}
    for field in _CONFIG_FIELDS - {"mode", "device"}:
        value = saved.get(field)
        if value is not None:
            kwargs[field] = value

    state_dim = int(kwargs.get("state_dim", 6))
    action_dim = int(kwargs.get("action_dim", 6))
    chunk_length = int(stage2_ckpt.get("rl_chunk_length", kwargs.get("n_action_steps_rl", 10)))
    stride = int(provenance.get("action_stride", kwargs.get("action_stride", 2)))
    actor_hidden, actor_layers = _linear_width_and_depth(stage2_ckpt.get("actor_state_dict", {}), "net.")
    critic_hidden, critic_layers = _linear_width_and_depth(stage2_ckpt.get("critic_state_dict", {}), "q1.")
    actor_first = stage2_ckpt.get("actor_state_dict", {}).get("net.0.weight")
    if actor_first is not None and hasattr(actor_first, "shape"):
        inferred_z = int(actor_first.shape[1]) - state_dim - chunk_length * action_dim
        if inferred_z > 0:
            kwargs["rlt_hidden_dim"] = inferred_z
    kwargs.update({
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_action_steps_rl": chunk_length,
        "action_stride": stride,
        "actor_critic_style": "paper_mlp",
        "actor_residual_scale": 0.0,
        "policy_fixed_std": float(provenance.get("policy_fixed_std", kwargs.get("policy_fixed_std", 0.05))),
        # Native LeRobot rollout reads this field to align state and action keys.
        "action_feature_names": [f"{name}.pos" for name in SO101_JOINT_NAMES],
    })
    if actor_hidden is not None:
        kwargs["actor_hidden_dim"] = actor_hidden
    if actor_layers is not None:
        kwargs["actor_num_layers"] = actor_layers
    if critic_hidden is not None:
        kwargs["critic_hidden_dim"] = critic_hidden
    if critic_layers is not None:
        kwargs["critic_num_layers"] = critic_layers
    return kwargs


def build_deployment_metadata(stage1_ckpt: dict, stage2_ckpt: dict, pi05_path: str) -> dict:
    """Return sidecar metadata consumed by deployment adapters and validators."""
    from rlt_core import SO101_JOINT_NAMES

    provenance = stage2_ckpt.get("runtime_provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    return {
        "schema_version": 1,
        "policy_family": "pi05_rlt",
        "actor_contract": stage2_ckpt.get("actor_contract"),
        "rl_chunk_length": int(stage2_ckpt.get("rl_chunk_length", 10)),
        "action_stride": int(provenance.get("action_stride", 2)),
        "state_names": [f"{name}.pos" for name in SO101_JOINT_NAMES],
        "action_names": [f"{name}.pos" for name in SO101_JOINT_NAMES],
        "pi05_pretrained_path": str(pi05_path),
        "stage1_step": stage1_ckpt.get("step"),
        "stage2_episode": stage2_ckpt.get("episode"),
        "source_runtime_provenance": provenance,
    }


def validate_export_shapes(config_kwargs: dict, stage2_ckpt: dict) -> None:
    """Fail before writing a policy whose actor/critic shapes cannot load."""
    actor = stage2_ckpt.get("actor_state_dict", {})
    critic = stage2_ckpt.get("critic_state_dict", {})
    actor_first = actor.get("net.0.weight")
    critic_first = critic.get("q1.0.weight")
    if actor_first is None or critic_first is None:
        raise ValueError("Stage-2 checkpoint is missing paper MLP first-layer weights")
    expected_in = (
        int(config_kwargs["rlt_hidden_dim"])
        + int(config_kwargs["state_dim"])
        + int(config_kwargs["n_action_steps_rl"]) * int(config_kwargs["action_dim"])
    )
    if tuple(actor_first.shape)[1] != expected_in or tuple(critic_first.shape)[1] != expected_in:
        raise ValueError(
            "Stage-2 actor/critic input width does not match exported config: "
            f"actor={tuple(actor_first.shape)}, critic={tuple(critic_first.shape)}, expected_in={expected_in}"
        )
    if tuple(actor_first.shape)[0] != int(config_kwargs["actor_hidden_dim"]):
        raise ValueError("Stage-2 actor hidden width does not match exported config")
    if tuple(critic_first.shape)[0] != int(config_kwargs["critic_hidden_dim"]):
        raise ValueError("Stage-2 critic hidden width does not match exported config")


def validate_saved_action_order(config_path: Path, expected_names: list[str]) -> None:
    """Ensure the custom PI05 config serialized LeRobot's native order field."""
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    if saved.get("action_feature_names") != expected_names:
        raise RuntimeError(
            "Exported config.json did not preserve action_feature_names. Update "
            "PI05RLTConfig to declare `action_feature_names: list[str] | None = None` "
            "before using this policy with lerobot-rollout."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-checkpoint", type=Path, required=True,
                    help="Stage-1 .pt (encoder/decoder), paper_v1 image-only")
    ap.add_argument("--stage2-checkpoint", type=Path, required=True,
                    help="Stage-2 .pt (schema v3, paper_full_output_v1)")
    ap.add_argument("--pi05-path", type=str, required=True,
                    help="Frozen π0.5 pretrained_model directory (recorded in config, not saved)")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write <output-dir>/pretrained_model/")
    ap.add_argument("--device", default=None,
                    help="Device recorded in the exported config (default: cuda if available). "
                         "Must match how the policy will be deployed; a mismatch makes the "
                         "frozen π0.5 backbone and image tensors land on different devices.")
    args = ap.parse_args()

    for p in (args.stage1_checkpoint, args.stage2_checkpoint):
        if not p.is_file():
            raise SystemExit(f"Checkpoint not found: {p}")

    ck1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    if ck1.get("rlt_architecture") != "paper_v1" or not ck1.get("image_only", False):
        raise SystemExit(f"Stage-1 checkpoint must be paper_v1 image-only: {args.stage1_checkpoint}")
    ck2 = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=False)
    if ck2.get("schema_version") != 3 or ck2.get("actor_contract") != "paper_full_output_v1":
        raise SystemExit(
            f"Stage-2 checkpoint must be schema v3 paper_full_output_v1: {args.stage2_checkpoint}"
        )

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    config_kwargs = build_export_config_kwargs(ck1, ck2, args.device)
    validate_export_shapes(config_kwargs, ck2)
    from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig

    cfg = PI05RLTConfig(**config_kwargs)
    # Inherit the observation/action feature contract from the frozen π0.5
    # checkpoint. The stage-2 config kwargs carry only RLT fields, so without
    # this the exported config.json serializes empty input_features and the
    # deployed policy finds no cameras (vision-blind inference).
    from lerobot.configs.policies import PreTrainedConfig as _PTConfig
    from lerobot.configs.types import FeatureType

    src_pi05_cfg = _PTConfig.from_pretrained(str(Path(args.pi05_path).expanduser()))
    if not cfg.input_features:
        cfg.input_features = dict(src_pi05_cfg.input_features)
    if not cfg.output_features:
        cfg.output_features = dict(src_pi05_cfg.output_features)
    if not any(f.type == FeatureType.VISUAL for f in cfg.input_features.values()):
        raise ValueError(
            "Source π0.5 config carries no camera features; refusing to export "
            "a vision-blind RLT policy."
        )
    cfg.pi05_pretrained_path = args.pi05_path
    from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy

    policy = PI05RLTPolicy(cfg)

    # Load the real trained weights into the package modules (strict).
    policy.rlt_encoder.load_state_dict(ck1["encoder_state_dict"], strict=True)
    policy.rlt_decoder.load_state_dict(ck1["decoder_state_dict"], strict=True)
    policy.actor.load_state_dict(ck2["actor_state_dict"], strict=True)
    policy.actor_target.load_state_dict(ck2["actor_target_state_dict"], strict=True)
    policy.critic.load_state_dict(ck2["critic_state_dict"], strict=True)
    policy.critic_target.load_state_dict(ck2["critic_target_state_dict"], strict=True)

    out = args.output_dir / "pretrained_model"
    out.mkdir(parents=True, exist_ok=False)  # fail if it already exists
    policy.save_pretrained(out)
    validate_saved_action_order(out / "config.json", config_kwargs["action_feature_names"])
    (out / "rlt_deployment.json").write_text(
        json.dumps(build_deployment_metadata(ck1, ck2, args.pi05_path), indent=2) + "\n",
        encoding="utf-8",
    )
    copied = copy_processor_artifacts(Path(args.pi05_path).expanduser(), out)
    files = sorted(p.name for p in out.iterdir())
    print(f"[OK] exported {len(files)} files to {out}: {files}")
    print(f"[OK] processor artifacts copied from π0.5 dir: {copied}")
    print(f"[OK] config.mode={policy.config.mode} pi05_pretrained_path={policy.config.pi05_pretrained_path}")
    print("Verify with: python scripts/verify_rlt_checkpoints.py --pretrained-dir", out)


def copy_processor_artifacts(pi05_pretrained_model_dir: Path, out_pretrained_dir: Path) -> list[str]:
    """Copy the π0.5 processor artifacts into the exported RLT pretrained dir.

    RLT shares π0.5's exact input preprocessing and (un)normalization at robot
    runtime (the Stage-2 trainer/evaluator use the same processors), so the
    ``policy_preprocessor.json`` / ``policy_postprocessor.json`` (+ their
    ``*_processor.safetensors`` state files) are copied verbatim, together with
    the ``tokenizer/`` directory referenced by the preprocessor's
    ``tokenizer_name`` artifact. This makes the exported directory loadable by
    ``lerobot`` toolchains that resolve processor artifacts relative to
    ``config.json`` / ``model.safetensors`` (e.g. record-style loading via
    ``make_pre_post_processors(pretrained_path=...)``).
    """
    import shutil

    copied = []
    for src in sorted(pi05_pretrained_model_dir.iterdir()):
        if not src.is_file():
            continue
        if "processor" not in src.name or src.suffix not in (".json", ".safetensors"):
            continue
        shutil.copy2(src, out_pretrained_dir / src.name)
        copied.append(src.name)
    if not copied:
        raise SystemExit(
            f"No processor artifacts found under {pi05_pretrained_model_dir} "
            "(expected policy_preprocessor.json / policy_postprocessor.json + step files)."
        )

    # The preprocessor declares a ``tokenizer`` artifact resolved next to
    # policy_preprocessor.json, so the tokenizer directory must travel with it;
    # without it record/rollout loading fails with "Missing processor artifact".
    tokenizer_src = pi05_pretrained_model_dir / "tokenizer"
    if not tokenizer_src.is_dir():
        raise SystemExit(
            f"Tokenizer directory not found under {pi05_pretrained_model_dir}; the exported "
            "pretrained_model dir would not be loadable by record/rollout."
        )
    tokenizer_dst = out_pretrained_dir / "tokenizer"
    if tokenizer_dst.exists():
        shutil.rmtree(tokenizer_dst)
    shutil.copytree(tokenizer_src, tokenizer_dst)
    copied.append("tokenizer/")
    return copied


if __name__ == "__main__":
    main()
