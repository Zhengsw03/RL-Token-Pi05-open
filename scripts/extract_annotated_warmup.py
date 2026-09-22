#!/usr/bin/env python3
"""Extract annotated critical-phase intervals into a Stage-2 replay journal (warmup pool).

Reads annotations.json (from annotate_dataset.py), walks every annotated
interval of the dataset, computes z_rl for each frame with the frozen π0.5 +
Stage-1 RLT encoder (same code path as precompute_critical_zrl.py), and writes
Stage-2 replay journal records (pickle stream) tagged as collection_phase=warmup.

Stage 2 then loads the journal with --restore_replay, so the learner's warmup
batches (warmup_demo_ratio=0.3) are drawn from the operator's annotated key
phases instead of only online-collected episodes.

Replay window semantics match the online loop: stride=2, horizon=10
(n_action_steps_rl), source=BASE, rewards=0, done=False.

Usage:
    python scripts/extract_annotated_warmup.py \
        --dataset_path /path/to/my_dataset \
        --annotations annotations.json \
        --pi05_path /path/to/pi05/pretrained_model \
        --tokenizer_path /path/to/paligemma_tokenizer \
        --rlt_checkpoint outputs/rlt_stage1/best_checkpoint.pt \
        --output warmup_replay_journal.pkl \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_critical_classifier import load_annotations  # noqa: E402

from rlt_core import write_warmup_interval_records  # noqa: E402 (shared, see rlt_core.py)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--pi05_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--rlt_checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True,
                        help="Output replay journal .pkl (feed to Stage 2 --restore_replay).")
    parser.add_argument("--horizon", type=int, default=10,
                        help="n_action_steps_rl (must match Stage 2, default 10).")
    parser.add_argument("--stride", type=int, default=2,
                        help="Replay stride (must match Stage 2, default 2).")
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--state_dim", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Optional cap on how many annotated episodes to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    annotations = load_annotations(args.annotations)
    if not annotations:
        raise ValueError("Annotations file contains no intervals")
    total_intervals = sum(len(v) for v in annotations.values())
    logger.info("Loaded %d intervals across %d episodes", total_intervals, len(annotations))
    if args.max_episodes is not None:
        annotations = {ep: iv for ep, iv in sorted(annotations.items())
                       if ep < args.max_episodes}
        logger.info("Capped to first %d annotated episodes (%d intervals)",
                    args.max_episodes, sum(len(v) for v in annotations.values()))

    # ── Load frozen π0.5 + preprocessor (same path as Stage 2 / precompute) ──
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05_rlt.vla_compat import (
        extract_embeddings as extract_pi05_embeddings,
    )
    from lerobot.policies.pi05_rlt.vla_compat import resize_with_pad as resize_with_pad_torch
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    pi05_policy = PI05Policy.from_pretrained(pretrained_name_or_path=str(args.pi05_path))
    pi05_config = pi05_policy.config
    pi05_config.validate_features()
    pi05_policy.to(device)
    pi05_policy.eval()
    for parameter in pi05_policy.parameters():
        parameter.requires_grad = False
    pi05_model = pi05_policy.model
    image_features = [k for k in pi05_config.input_features if k.startswith("observation.images.")]
    logger.info("π0.5 loaded and frozen; image features: %s", image_features)

    dataset_path = Path(args.dataset_path)
    dataset = LeRobotDataset(dataset_path.name, root=dataset_path, video_backend="pyav")

    def _to_dict_features(features):
        from lerobot.configs.types import PolicyFeature
        return {
            k: {"type": v.type.name, "shape": list(v.shape)}
            if isinstance(v, PolicyFeature) else v
            for k, v in features.items()
        }

    pi05_config.input_features = _to_dict_features(pi05_config.input_features)
    pi05_config.output_features = _to_dict_features(pi05_config.output_features)

    _tk = Path(args.tokenizer_path).expanduser()
    if _tk.exists():
        pi05_config.text_tokenizer_name = str(_tk)  # the current factory reads the tokenizer from the config

    preprocessor, _ = make_pi05_pre_post_processors(
        config=pi05_config,
        dataset_stats=dataset.meta.stats,
    )
    if hasattr(preprocessor, "steps"):
        preprocessor.steps = [
            step for step in preprocessor.steps
            if step.__class__.__name__ not in ("DeviceProcessorStep",)
        ]

    # ── Load frozen RLT encoder (Stage 1 checkpoint) ─────────────────────
    from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
    from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTokenEncoder

    rlt_config = PI05RLTConfig(mode="online_rl")
    encoder = RLTokenEncoder(rlt_config).to(device)
    ckpt = torch.load(args.rlt_checkpoint, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    logger.info("RLT encoder loaded and frozen (z dim %d).", rlt_config.rlt_hidden_dim)

    # ── Collect every frame that falls inside an annotated interval ──────
    episode_ids = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64)
    rows: list[tuple[int, int, int]] = []  # (global_idx, episode, frame)
    for ep, intervals in sorted(annotations.items()):
        for start_f, end_f in intervals:
            sel = np.flatnonzero((episode_ids == ep) & (frame_ids >= start_f) & (frame_ids <= end_f))
            for g in sel:
                rows.append((int(g), ep, int(frame_ids[g])))
    rows.sort(key=lambda r: (r[1], r[2]))
    logger.info("Total annotated frames: %d", len(rows))
    if not rows:
        raise ValueError("No frames matched the annotations (episode/frame indices out of range?)")

    # ── Compute z_rl / proprio / action for every annotated frame ────────
    image_resolution = pi05_config.image_resolution
    max_action_dim = pi05_config.max_action_dim
    chunk_size = pi05_config.chunk_size

    def process_frame(frame: dict) -> dict:
        item = preprocessor(frame)
        return {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == 1 else v
            for k, v in item.items()
        }

    # frame_key: (episode, frame) -> (z_rl fp16 (z_dim,), proprio fp32 (state_dim,), action fp32 (action_dim,))
    frame_data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        processed = [process_frame(dataset[global_idx]) for global_idx, _, _ in batch_rows]
        batch = {
            k: torch.utils.data.default_collate([item[k] for item in processed])
            if isinstance(processed[0][k], torch.Tensor) else [item[k] for item in processed]
            for k in processed[0].keys()
        }
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        bsize = batch["action"].shape[0]

        images, img_masks = [], []
        for feat_name in image_features:
            if feat_name in batch:
                img = batch[feat_name].float()
                img = img.permute(0, 2, 3, 1)
                img = resize_with_pad_torch(img, *image_resolution)
                img = img * 2.0 - 1.0
                img = img.permute(0, 3, 1, 2)
                images.append(img.to(device))
                img_masks.append(torch.ones(bsize, dtype=torch.bool, device=device))
            else:
                images.append(torch.full((bsize, 3, *image_resolution), -1.0, device=device))
                img_masks.append(torch.zeros(bsize, dtype=torch.bool, device=device))

        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = torch.zeros(bsize, chunk_size, max_action_dim, dtype=torch.float32, device=device)

        with torch.no_grad():
            prefix_img, _, prefix_mask = extract_pi05_embeddings(pi05_model, images, img_masks, tokens, masks, actions,
            chunk_size=chunk_size, max_action_dim=max_action_dim, image_only=True)
            z_rl = encoder(prefix_img.float(), mask=prefix_mask).float()  # (B, z_dim)

        if not torch.isfinite(z_rl).all().item():
            raise ValueError(f"Batch {start // args.batch_size}: z_rl contains NaN/Inf")

        for offset, (global_idx, ep, frame) in enumerate(batch_rows):
            item = processed[offset]
            proprio = item["observation.state"].detach().cpu().numpy()[:args.state_dim].astype(np.float32)
            act = item["action"].detach().cpu().numpy()[:args.action_dim].astype(np.float32)
            frame_data[(ep, frame)] = (
                z_rl[offset].detach().cpu().numpy().astype(np.float16),
                proprio,
                act,
            )
        logger.info("z_rl batch %d/%d done (%.1fs elapsed)",
                    start // args.batch_size + 1,
                    (len(rows) + args.batch_size - 1) // args.batch_size,
                    time.time() - t0)

    # ── Build stride-2 replay windows inside each annotated interval ─────
    horizon, stride, action_dim = args.horizon, args.stride, args.action_dim
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    with open(out_path, "wb") as f:
        for ep, intervals in sorted(annotations.items()):
            for interval_idx, (start_f, end_f) in enumerate(intervals):
                frames_in_interval = [frame for (_, epf, frame) in rows
                                      if epf == ep and start_f <= frame <= end_f]
                if len(frames_in_interval) < 1:
                    continue
                n_records += write_warmup_interval_records(
                    f, ep, frames_in_interval, frame_data,
                    horizon=horizon, stride=stride, action_dim=action_dim,
                )

    logger.info("Saved %d warmup replay transitions to %s", n_records, out_path)
    logger.info("Stage 2 usage: --restore_replay %s (keep --warmup_bc_weight high)", out_path)


if __name__ == "__main__":
    main()
