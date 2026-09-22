#!/usr/bin/env python3
"""Pre-compute z_rl (RL token) for critical-phase classifier training.

Samples frames from a teleop dataset according to manual critical-phase
annotations (positive = inside an annotated interval, negative = outside, from
annotated episodes only), then runs the frozen π0.5 (image-only prefix) + the
frozen Stage 1 RLT encoder to produce one z_rl per sampled frame.

Output NPZ (consumed by ``train_critical_classifier.py --precomputed_zrl``):
    z_rl      (N, rlt_hidden_dim) float16
    proprio   (N, 6) float32
    labels    (N,) int64 (1 = critical phase)
    episode   (N,) int64
    frame     (N,) int64 (frame index within episode)

Usage:
    python scripts/precompute_critical_zrl.py \
        --pi05_path ./pi05_base \
        --tokenizer_path ./paligemma_tokenizer \
        --rlt_checkpoint ../outputs/pi05_rlt_stage1/best_checkpoint.pt \
        --dataset_path /path/to/my_dataset_multitask \
        --annotations annotations.json \
        --output outputs/critical_zrl.npz \
        --batch_size 8 --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_critical_classifier import load_annotations, sample_dataset_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi05_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--rlt_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pos_per_interval", type=int, default=200)
    parser.add_argument("--neg_per_episode", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    annotations = load_annotations(args.annotations)
    if not annotations:
        raise ValueError("Annotations file contains no intervals")

    samples = sample_dataset_frames(
        args.dataset_path,
        annotations,
        pos_per_interval=args.pos_per_interval,
        neg_per_episode=args.neg_per_episode,
        seed=args.seed,
    )
    logger.info("Sampled %d frames for z_rl precomputation", len(samples))

    # ── Load frozen π0.5 + preprocessor (same path as Stage 2) ───────────
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

    # ── Compute z_rl per sampled frame ───────────────────────────────────
    image_resolution = pi05_config.image_resolution
    max_action_dim = pi05_config.max_action_dim
    chunk_size = pi05_config.chunk_size

    def process_frame(frame: dict) -> dict:
        item = preprocessor(frame)
        return {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == 1 else v
            for k, v in item.items()
        }

    episode_ids = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64)

    z_rl_out = np.zeros((len(samples), rlt_config.rlt_hidden_dim), dtype=np.float16)
    proprio_out = np.zeros((len(samples), 6), dtype=np.float32)
    labels_out = np.zeros(len(samples), dtype=np.int64)
    episode_out = np.zeros(len(samples), dtype=np.int64)
    frame_out = np.zeros(len(samples), dtype=np.int64)

    t0 = time.time()
    for start in range(0, len(samples), args.batch_size):
        batch_meta = samples[start:start + args.batch_size]
        processed = [process_frame(dataset[global_idx]) for global_idx, _ in batch_meta]
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

        end = start + bsize
        z_rl_out[start:end] = z_rl.detach().cpu().numpy().astype(np.float16)
        for offset, (global_idx, label) in enumerate(batch_meta):
            proprio_out[start + offset] = dataset[global_idx]["observation.state"].numpy().astype(np.float32)
            labels_out[start + offset] = label
            episode_out[start + offset] = episode_ids[global_idx]
            frame_out[start + offset] = frame_ids[global_idx]
        logger.info("z_rl batch %d/%d done (%.1fs elapsed)",
                    start // args.batch_size + 1,
                    (len(samples) + args.batch_size - 1) // args.batch_size,
                    time.time() - t0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        z_rl=z_rl_out,
        proprio=proprio_out,
        labels=labels_out,
        episode=episode_out,
        frame=frame_out,
        annotations=json.dumps(annotations),
        pos_per_interval=args.pos_per_interval,
        neg_per_episode=args.neg_per_episode,
    )
    logger.info("Saved %d z_rl samples to %s (%.1f%% positive)",
                len(samples), output, 100.0 * labels_out.mean())


if __name__ == "__main__":
    main()
