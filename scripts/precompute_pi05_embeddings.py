#!/usr/bin/env python3
"""Pre-compute π0.5 image-only VLM embeddings for RLT Stage 1 training.

Runs the frozen π0.5 model once over the entire dataset and saves
(image-only prefix_out, prefix_mask) pairs to mmap files on disk.
Stage 1 then trains the RLT encoder-decoder on these cached embeddings.

Aligns with openpi-RLT: image_only=True, mask preserved for masked
reconstruction loss.

Usage:
    python scripts/precompute_pi05_embeddings.py \
        --pi05_path ./pi05_base \
        --tokenizer_path /path/to/paligemma_tokenizer \
        --dataset_repo_id ./my_dataset2 \
        --output_dir ../outputs/pi05_embeddings \
        --batch_size 16 \
        --device cuda
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute π0.5 VLM embeddings")
    parser.add_argument("--pi05_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--dataset_repo_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--camera_map", type=str, default="",
                        help='JSON: dataset camera → checkpoint feature name')
    parser.add_argument("--shard_index", type=int, default=0,
                        help="Shard index (0-based). Together with --shard_count it splits the "
                             "episodes into N parts, each written to its own --output_dir, for "
                             "parallel multi-GPU precomputation.")
    parser.add_argument("--shard_count", type=int, default=1,
                        help="Total number of shards. 1 disables sharding (full dataset).")
    parser.add_argument("--shard_episodes", type=str, default="",
                        help="Optional: explicit episodes for this shard, e.g. '0-20' or "
                             "'0,3,5'. Empty means an even split by "
                             "--shard_index/--shard_count.")
    parser.add_argument("--v3-dataset-root", type=str, default=None,
                        help="Optional: ALSO stream embeddings into a v3 LeRobotDataset at this "
                             "root (features vlm_embeddings/prefix_mask), so Stage 1 can run via "
                             "the standard lerobot-train toolchain. fp32 (v3 arrays do not "
                             "support fp16).")
    return parser.parse_args()


def _atomic_write_json(path, payload):
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class V3EmbeddingWriter:
    """Stream image-only prefix embeddings into a v3 LeRobotDataset.

    Alternative output of :mod:`precompute_pi05_embeddings` (used with
    ``--v3-dataset-root``): instead of (or in addition to) the fp16 mmap cache,
    write per-frame ``vlm_embeddings (S, D) float32`` / ``prefix_mask (S,) bool``
    features into a standard v3 dataset, so Stage 1 can be trained with the
    standard ``lerobot-train`` toolchain.

    Frame order must be the dataset's global frame order (the precompute
    DataLoader runs with ``shuffle=False``); the caller fetches the raw
    state/action/task per frame from the source dataset.
    """

    def __init__(self, root, *, seq_len, vlm_dim, fps=30, robot_type="so_follower"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(root)
        features = {
            "observation.state": {"dtype": "float32", "shape": (6,)},
            "action": {"dtype": "float32", "shape": (6,)},
            "vlm_embeddings": {"dtype": "float32", "shape": (seq_len, vlm_dim)},
            "prefix_mask": {"dtype": "bool", "shape": (seq_len,)},
        }
        self._ds = LeRobotDataset.create(
            f"local/{root.name}", fps=fps, features=features, root=root,
            robot_type=robot_type, use_videos=False,
        )
        self._current_episode = None
        self._n_rows = 0

    def add_row(self, *, episode_index, state, action, task, embeddings, mask):
        """Add one frame. ``embeddings`` (S, D) float32; ``mask`` (S,) bool."""
        if episode_index != self._current_episode:
            if self._current_episode is not None:
                self._ds.save_episode()
            self._current_episode = episode_index
        state = np.asarray(state, dtype=np.float32).reshape(-1)[:6]
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:6]
        if state.shape != (6,) or action.shape != (6,):
            raise ValueError(
                f"Expected 6D state/action per frame, got state={state.shape} action={action.shape}"
            )
        emb = np.asarray(embeddings, dtype=np.float32)
        msk = np.asarray(mask, dtype=np.bool_)
        if emb.ndim != 2 or msk.shape != emb.shape[:1]:
            raise ValueError(
                f"Expected embeddings (S, D) and mask (S,), got {emb.shape} / {msk.shape}"
            )
        self._ds.add_frame({
            "observation.state": state,
            "action": action,
            "vlm_embeddings": emb,
            "prefix_mask": msk,
            "task": str(task),
        })
        self._n_rows += 1

    def close(self):
        if self._current_episode is not None:
            self._ds.save_episode()
        self._ds.finalize()

    @property
    def n_rows(self) -> int:
        return self._n_rows


def _validate_args(args):
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"--num_workers must be non-negative, got {args.num_workers}")

    pi05_path = Path(args.pi05_path).expanduser()
    if not pi05_path.is_dir():
        raise FileNotFoundError(f"π0.5 checkpoint directory does not exist: {pi05_path}")
    config_path = pi05_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"π0.5 config does not exist: {config_path}")
    weights_path = pi05_path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"π0.5 weights do not exist: {weights_path}. "
            "The current PI05 loader requires one root-level model.safetensors file."
        )

    tokenizer_path = Path(args.tokenizer_path).expanduser()
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")

    if args.camera_map:
        try:
            camera_map = json.loads(args.camera_map)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--camera_map must be valid JSON: {exc}") from exc
        if not isinstance(camera_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in camera_map.items()
        ):
            raise ValueError("--camera_map must be a JSON object mapping strings to strings")

    return pi05_path, config_path


def main():
    args = parse_args()
    pi05_path, _config_path = _validate_args(args)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load π0.5 ─────────────────────────────────────────────────────
    logger.info(f"Loading π0.5 from {pi05_path}...")
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05_rlt.vla_compat import (
        extract_embeddings as extract_pi05_embeddings,
    )
    from lerobot.policies.pi05_rlt.vla_compat import resize_with_pad as resize_with_pad_torch
    from lerobot.configs.types import PolicyFeature

    # Load through the policy factory so feature entries are materialized as
    # PolicyFeature objects, matching Stage 2 embedding extraction.
    pi05_policy = PI05Policy.from_pretrained(pretrained_name_or_path=str(pi05_path))
    pi05_config = pi05_policy.config
    pi05_config.validate_features()

    checkpoint_image_features = [
        k for k in pi05_config.input_features
        if k.startswith("observation.images.")
    ]
    if not checkpoint_image_features:
        raise ValueError("π0.5 checkpoint config has no observation.images.* input features")
    logger.info("Checkpoint image features: %s", checkpoint_image_features)

    if args.camera_map:
        camera_feature_map = json.loads(args.camera_map)
    else:
        camera_feature_map = {}
        dataset_cam_names = ["observation.images.top", "observation.images.wrist"]
        for ds_name, feat_name in zip(dataset_cam_names, checkpoint_image_features):
            camera_feature_map[ds_name] = feat_name
        logger.info("Auto-mapped cameras: %s", camera_feature_map)

    feat_to_dataset_cam = {v: k for k, v in camera_feature_map.items()}

    pi05_policy.to(device)
    pi05_policy.eval()
    for p in pi05_policy.parameters():
        p.requires_grad = False
    pi05_model = pi05_policy.model
    logger.info("π0.5 loaded and frozen.")

    image_resolution = pi05_config.image_resolution
    max_action_dim = pi05_config.max_action_dim
    chunk_size = pi05_config.chunk_size

    # ── Load dataset + preprocessor ───────────────────────────────────
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    dataset_path = Path(args.dataset_repo_id)
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path

    # ── Sharding: split episodes into shard_count parts; this process only
    # ── computes its own part.
    shard_episodes = None
    if args.shard_count < 1:
        raise ValueError("--shard_count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError(f"--shard_index must be within [0, {args.shard_count})")
    if args.shard_count > 1 or args.shard_episodes:
        _probe_root = dataset_path if dataset_path.exists() else None
        _probe = (LeRobotDataset(dataset_path.name, root=_probe_root, video_backend="pyav")
                  if _probe_root else LeRobotDataset(args.dataset_repo_id, video_backend="pyav"))
        all_eps = sorted({int(e) for e in _probe.hf_dataset["episode_index"]})
        if args.shard_episodes:
            sel: list[int] = []
            for part in args.shard_episodes.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    a, b = part.split("-", 1)
                    sel.extend(range(int(a), int(b) + 1))
                else:
                    sel.append(int(part))
            shard_episodes = sorted(set(sel))
            missing = [e for e in shard_episodes if e not in all_eps]
            if missing:
                raise ValueError(f"--shard_episodes contains unknown episodes: {missing[:10]}")
        else:
            shard_episodes = [e for i, e in enumerate(all_eps) if i % args.shard_count == args.shard_index]
        if not shard_episodes:
            raise ValueError(f"shard {args.shard_index}/{args.shard_count} was assigned no episodes")
        logger.info("Shard %d/%d: %d episode(s) (%s ... %s)",
                    args.shard_index, args.shard_count, len(shard_episodes),
                    shard_episodes[:5], shard_episodes[-3:])
        del _probe

    if dataset_path.exists():
        dataset = LeRobotDataset(dataset_path.name, root=dataset_path, video_backend="pyav",
                                 episodes=shard_episodes)
    else:
        dataset = LeRobotDataset(args.dataset_repo_id, video_backend="pyav",
                                 episodes=shard_episodes)

    def _to_dict_features(features):
        return {
            k: {"type": v.type.name, "shape": list(v.shape)}
            if isinstance(v, PolicyFeature) else v
            for k, v in features.items()
        }

    pi05_config.input_features = _to_dict_features(pi05_config.input_features)
    pi05_config.output_features = _to_dict_features(pi05_config.output_features)

    # The current make_pi05_pre_post_processors only accepts
    # (config, dataset_stats), and the tokenizer location comes from
    # config.text_tokenizer_name, so it must be written here as a local path:
    # the checkpoint stores the remote Hub name 'google/paligemma-3b-pt-224',
    # which fails to download on offline machines. --tokenizer_path overrides it.
    _tk_local = Path(args.tokenizer_path).expanduser()
    if _tk_local.exists():
        pi05_config.text_tokenizer_name = str(_tk_local)

    preprocessor, _ = make_pi05_pre_post_processors(
        config=pi05_config,
        dataset_stats=dataset.meta.stats,
    )

    if hasattr(preprocessor, 'steps'):
        preprocessor.steps = [
            step for step in preprocessor.steps
            if step.__class__.__name__ not in ('DeviceProcessorStep',)
        ]

    def collate_and_process(samples):
        processed = []
        for sample in samples:
            item = preprocessor(sample)
            item = {
                k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == 1 else v
                for k, v in item.items()
            }
            processed.append(item)
        keys = processed[0].keys()
        return {
            k: torch.utils.data.default_collate([item[k] for item in processed])
            if isinstance(processed[0][k], torch.Tensor) else [item[k] for item in processed]
            for k in keys
        }

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # preserve order for indexing
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_and_process,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    logger.info(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # ── Pre-compute: stream image-only prefix + mask to mmap ────────────
    logger.info("Pre-computing image-only embeddings (image_only=True, per RLT paper)...")
    t0 = time.time()
    N = len(dataset)
    if N <= 0:
        raise ValueError("Dataset is empty; there are no embeddings to pre-compute")

    prefix_path = output_dir / "prefix_out.mmap"
    mask_path = output_dir / "prefix_mask.mmap"
    meta_path = output_dir / "meta.json"
    provenance = {
        "pi05_path": str(pi05_path.resolve()),
        "dataset_repo_id": args.dataset_repo_id,
        "checkpoint_image_features": checkpoint_image_features,
        "camera_feature_map": camera_feature_map,
    }
    incomplete_meta = {
        "schema_version": 2,
        "complete": False,
        "n_samples": N,
        "has_mask": True,
        "dtype": "float16",
        "image_only": True,
        "embedding_stage": "post_transformer",
        "language_positions_dropped": True,
        **provenance,
    }
    # Invalidate any prior completed cache before touching its mmap payloads.
    _atomic_write_json(meta_path, incomplete_meta)

    prefix_mmap = None
    mask_mmap = None
    expected_shape = None
    written = 0
    v3_writer = None

    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Build images
        bsize = batch["action"].shape[0]
        images = []
        img_masks = []
        for feat_name in checkpoint_image_features:
            ds_cam = feat_to_dataset_cam.get(feat_name)
            if ds_cam and ds_cam in batch:
                img = batch[ds_cam].float()
                img = img.permute(0, 2, 3, 1)
                img = resize_with_pad_torch(img, *image_resolution)
                img = img * 2.0 - 1.0
                img = img.permute(0, 3, 1, 2)
                images.append(img.to(device))
                img_masks.append(torch.ones(bsize, dtype=torch.bool, device=device))
            else:
                img = torch.full((bsize, 3, *image_resolution), -1.0, device=device)
                images.append(img)
                img_masks.append(torch.zeros(bsize, dtype=torch.bool, device=device))

        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # Image-prefix outputs cannot attend to the action suffix in π0.5's
        # block attention pattern. Use deterministic dummy suffix inputs rather
        # than depending on the dataset's action tensor layout.
        actions = torch.zeros(
            bsize, chunk_size, max_action_dim, dtype=torch.float32, device=device
        )

        with torch.no_grad():
            prefix_img, _, prefix_mask = extract_pi05_embeddings(
                pi05_model, images, img_masks, tokens, masks, actions,
                chunk_size=chunk_size, max_action_dim=max_action_dim,
                image_only=True,  # drop language + state tokens
            )

        if prefix_img.ndim != 3:
            raise ValueError(
                f"Batch {batch_idx}: expected prefix shape [B, S, D], got {tuple(prefix_img.shape)}"
            )
        if prefix_mask.ndim != 2:
            raise ValueError(
                f"Batch {batch_idx}: expected mask shape [B, S], got {tuple(prefix_mask.shape)}"
            )
        if prefix_img.shape[0] != bsize or prefix_mask.shape[0] != bsize:
            raise ValueError(
                f"Batch {batch_idx}: output batch size mismatch: input={bsize}, "
                f"prefix={prefix_img.shape[0]}, mask={prefix_mask.shape[0]}"
            )
        if prefix_img.shape[:2] != prefix_mask.shape:
            raise ValueError(
                f"Batch {batch_idx}: prefix/mask shape mismatch: "
                f"prefix={tuple(prefix_img.shape)}, mask={tuple(prefix_mask.shape)}"
            )
        batch_shape = (prefix_img.shape[1], prefix_img.shape[2])
        if expected_shape is None:
            expected_shape = batch_shape
            seq_len, vlm_dim = expected_shape
            prefix_mmap = np.memmap(
                str(prefix_path), dtype="float16", mode="w+", shape=(N, seq_len, vlm_dim)
            )
            mask_mmap = np.memmap(
                str(mask_path), dtype="bool", mode="w+", shape=(N, seq_len)
            )
        elif batch_shape != expected_shape:
            raise ValueError(
                f"Batch {batch_idx}: unstable embedding shape: expected {expected_shape}, "
                f"got {batch_shape}"
            )
        if not torch.isfinite(prefix_img).all().item():
            raise ValueError(f"Batch {batch_idx}: prefix embeddings contain NaN or Inf")

        prefix_cpu = prefix_img.detach().to(device="cpu", dtype=torch.float16)
        mask_cpu = prefix_mask.detach().to(device="cpu", dtype=torch.bool)
        if not torch.isfinite(prefix_cpu).all().item():
            raise ValueError(
                f"Batch {batch_idx}: prefix embeddings overflowed to NaN or Inf in float16"
            )
        if not mask_cpu.any(dim=1).all().item():
            invalid = (~mask_cpu.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"Batch {batch_idx}: samples without any valid prefix token: {invalid}"
            )

        # ── Optional v3 dataset streaming (lerobot-train ready) ───────────
        if args.v3_dataset_root:
            if v3_writer is None:
                seq_len0, vlm_dim0 = expected_shape
                v3_writer = V3EmbeddingWriter(
                    Path(args.v3_dataset_root), seq_len=seq_len0, vlm_dim=vlm_dim0,
                    fps=int(getattr(dataset.meta, "fps", 30)),
                )
                logger.info("Streaming v3 dataset to %s (fp32 embeddings)", args.v3_dataset_root)
            embeddings_fp32 = prefix_cpu.float().numpy()  # (B, S, D)
            masks_np = mask_cpu.numpy()
            for i in range(bsize):
                # DataLoader order == dataset global frame order (shuffle=False):
                # refetch the raw frame to carry its original state/action/task.
                raw = dataset[written + i]
                v3_writer.add_row(
                    episode_index=int(np.asarray(raw["episode_index"]).reshape(-1)[0]),
                    state=np.asarray(raw["observation.state"]).reshape(-1),
                    action=np.asarray(raw["action"]).reshape(-1),
                    task=str(raw["task"]),
                    embeddings=embeddings_fp32[i],
                    mask=masks_np[i],
                )

        end = written + bsize
        if end > N:
            raise ValueError(f"Batch {batch_idx}: write would exceed dataset size {N}: {end}")
        prefix_mmap[written:end] = prefix_cpu.numpy()
        mask_mmap[written:end] = mask_cpu.numpy()
        prefix_mmap.flush()
        mask_mmap.flush()
        written = end

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            elapsed = time.time() - t0
            logger.info(f"  {written}/{N} samples | {elapsed:.0f}s")

    if prefix_mmap is None or mask_mmap is None or expected_shape is None:
        raise RuntimeError("DataLoader produced no batches")
    if written != N:
        raise ValueError(f"Embedding write count mismatch: wrote {written}, expected {N}")

    prefix_mmap.flush()
    mask_mmap.flush()
    del prefix_mmap, mask_mmap

    if v3_writer is not None:
        if v3_writer.n_rows != N:
            raise ValueError(f"v3 writer row count mismatch: {v3_writer.n_rows} != {N}")
        v3_writer.close()
        logger.info(f"Saved v3 dataset: {args.v3_dataset_root} ({v3_writer.n_rows} frames)")

    seq_len, vlm_dim = expected_shape
    complete_meta = {
        **incomplete_meta,
        "complete": True,
        "seq_len": seq_len,
        "vlm_hidden_dim": vlm_dim,
        "tokens_per_camera": seq_len // len(checkpoint_image_features),
    }
    _atomic_write_json(meta_path, complete_meta)

    total_time = time.time() - t0
    prefix_gb = (N * seq_len * vlm_dim * 2) / 1e9
    mask_gb = (N * seq_len) / 1e9
    logger.info(f"Done! {written} samples in {total_time:.0f}s")
    logger.info(f"Saved: {prefix_path} ({prefix_gb:.1f} GB) + {mask_path} ({mask_gb:.1f} GB)")
    logger.info(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
