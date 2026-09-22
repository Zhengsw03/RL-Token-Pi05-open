#!/usr/bin/env python3
"""Stage 1: Offline RLT Encoder-Decoder Training for π0.5.

Trains the RLT encoder and decoder on demonstration data to learn a compact
z_rl representation from frozen π0.5 image-only prefix embeddings.

Strictly follows the paper's Stage 1 objective:
  - image_only=True for fixed-instruction tasks
  - append one learned <rl> token and take the final encoder position (Eq. 1)
  - teacher-forced causal reconstruction from [z_rl, stopgrad(z_1:i-1)] (Eq. 2)
  - masked reconstruction loss for invalid camera/padding tokens

Recommended workflow (fast — ~100x faster):
  1. Pre-compute embeddings once:
     python scripts/precompute_pi05_embeddings.py \
         --pi05_path ./pi05_base --tokenizer_path /path/to/tok \
         --dataset_repo_id ./my_dataset2 --output_dir ../outputs/pi05_embeddings

  2. Train RLT on cached embeddings:
     python scripts/train_rlt_stage1_pi05.py \
         --precomputed_path ../outputs/pi05_embeddings \
         --output_dir ../outputs/pi05_rlt_stage1 \
         --steps 30000 --batch_size 256 --lr 1e-4

Legacy (slow) — runs π0.5 forward every step:
  python scripts/train_rlt_stage1_pi05.py \
      --pi05_path ./pi05_base --tokenizer_path /path/to/tok \
      --dataset_repo_id ./my_dataset2 --output_dir ../outputs/pi05_rlt_stage1 \
      --steps 30000 --batch_size 8 --lr 1e-4 --device cuda --gpus 0
"""

import argparse
import json
import logging
import math
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="π0.5 RLT Stage 1: Encoder-Decoder Training")
    parser.add_argument("--pi05_path", type=str, default="",
                        help="Path to pretrained π0.5 checkpoint (required for legacy mode)")
    parser.add_argument("--tokenizer_path", type=str, default="",
                        help="Local path to PaliGemma tokenizer (default: download from HF)")
    parser.add_argument("--camera_map", type=str, default="",
                        help='JSON map: physical camera names to checkpoint feature names')
    parser.add_argument("--precomputed_path", type=str, default="",
                        help="Path to precomputed embeddings directory (fast mode, skip π0.5)")
    parser.add_argument("--dataset_repo_id", type=str, default="",
                        help="HuggingFace dataset repo ID or local path for demo data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/pi05_rlt_stage1")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device: cpu, cuda, cuda:N, or mps")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Single CUDA GPU ID (multi-GPU is intentionally unsupported)")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (0=main process only)")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use device-appropriate AMP (BF16 when supported, otherwise FP16)")
    parser.add_argument("--resume_from", type=str, default="",
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--allow_schedule_change", action="store_true",
                        help="Allow resuming when --steps/--lr/--warmup_steps change the LR "
                             "schedule (e.g. extending a finished 10000-step cosine run to 30000). "
                             "The schedule is rebuilt over the new --steps and the checkpoint's "
                             "scheduler state is NOT loaded, so the LR does a short warm restart "
                             "instead of continuing the old decay.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="Run one finite forward/backward check, then exit")
    return parser.parse_args()


class MMapPrefixDataset(Dataset):
    """Lazily reads prefix embeddings and masks from one or more mmap-backed cache roots.

    Multi-shard support: precomputation can be split by episode across several
    GPUs, each writing its own cache directory; training then concatenates the
    directories into one logical dataset.
    The concatenation order follows the order of ``roots``, and frame order is
    preserved inside each shard (shuffling only happens in the training loop).
    """

    def __init__(self, roots, counts, seq_len: int, hidden_dim: int):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        if isinstance(counts, int):
            counts = [counts]
        if len(roots) != len(counts):
            raise ValueError(f"roots/counts length mismatch: {len(roots)} vs {len(counts)}")
        self.roots = [Path(r) for r in roots]
        self.counts = [int(c) for c in counts]
        self.n_samples = int(sum(self.counts))
        if self.n_samples <= 0:
            raise ValueError("multi-shard cache contains zero samples")
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self._offsets: list[int] = []
        acc = 0
        for c in self.counts:
            self._offsets.append(acc)
            acc += c
        self._prefix = None
        self._mask = None

    def _open(self):
        # Open independently in every DataLoader process; never pickle live mmap handles.
        if self._prefix is None:
            self._prefix, self._mask = [], []
            for root, n in zip(self.roots, self.counts):
                self._prefix.append(np.memmap(
                    str(Path(root) / "prefix_out.mmap"), dtype=np.float16, mode="r",
                    shape=(n, self.seq_len, self.hidden_dim),
                ))
                self._mask.append(np.memmap(
                    str(Path(root) / "prefix_mask.mmap"), dtype=np.bool_, mode="r",
                    shape=(n, self.seq_len),
                ))

    def __len__(self):
        return self.n_samples

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self.n_samples
        if not 0 <= index < self.n_samples:
            raise IndexError(index)
        for shard in range(len(self.roots) - 1, -1, -1):
            if index >= self._offsets[shard]:
                return shard, index - self._offsets[shard]
        raise IndexError(index)

    def __getitem__(self, index):
        self._open()
        shard, local = self._locate(index)
        # Copies are batch-sized/lazy and make the read-only mmap safe for torch tensors.
        prefix = torch.from_numpy(np.array(self._prefix[shard][local], copy=True))
        mask = torch.from_numpy(np.array(self._mask[shard][local], copy=True))
        if not mask.any():
            raise ValueError(f"Cache sample {index} has no valid image tokens.")
        if not torch.isfinite(prefix).all():
            raise ValueError(f"Cache sample {index} contains NaN or Inf prefix embeddings.")
        return prefix, mask

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_prefix"] = None
        state["_mask"] = None
        return state


def atomic_torch_save(obj, path: Path):
    """Write a torch checkpoint atomically in the destination directory."""
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def validate_args(args):
    positive_ints = ("steps", "batch_size", "save_every", "log_every")
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.warmup_steps < 0 or args.warmup_steps > args.steps:
        raise ValueError("--warmup_steps must be between 0 and --steps")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    if bool(args.precomputed_path) and bool(args.dataset_repo_id or args.pi05_path):
        logger.warning("--precomputed_path selected; --pi05_path/--dataset_repo_id are ignored")
    if not args.precomputed_path and (not args.pi05_path or not args.dataset_repo_id):
        raise ValueError("Without --precomputed_path, both --pi05_path and --dataset_repo_id are required")

    try:
        requested = torch.device(args.device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid --device {args.device!r}") from exc
    gpu_parts = [part.strip() for part in args.gpus.split(",") if part.strip()]
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if len(gpu_parts) != 1:
            raise ValueError("This script supports exactly one CUDA GPU; pass e.g. --gpus 0")
        try:
            gpu_id = int(gpu_parts[0])
        except ValueError as exc:
            raise ValueError("--gpus must contain one integer GPU ID") from exc
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA GPU {gpu_id} is unavailable; detected {torch.cuda.device_count()} device(s)"
            )
        if requested.index is not None and requested.index != gpu_id:
            raise ValueError("--device cuda:N and --gpus must select the same GPU")
        requested = torch.device("cuda", gpu_id)
    elif gpu_parts != ["0"]:
        raise ValueError("--gpus only applies to CUDA; leave it at the default for other devices")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    if requested.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("--device must be cpu, cuda, cuda:N, or mps")
    return requested


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """Restore RNG state, tolerating device-mapped byte tensors.

    ``torch.load(..., map_location=device)`` moves the saved RNG byte tensors to
    the GPU, but ``torch.set_rng_state``/``torch.cuda.set_rng_state*`` only accept
    CPU byte tensors -> resuming used to die with
    "TypeError: RNG state must be a torch.ByteTensor". Move them back to CPU, and
    never let RNG restoration abort a run: resume is already non-exact.
    """
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
    except Exception as exc:  # noqa: BLE001 - RNG exactness is best-effort
        logger.warning("python/numpy RNG state could not be restored: %s", exc)
    torch_state = state.get("torch")
    if torch_state is not None:
        if torch.is_tensor(torch_state):
            torch_state = torch_state.detach().to("cpu")
        try:
            torch.set_rng_state(torch_state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("torch RNG state could not be restored: %s", exc)
    cuda_states = state.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([
                s.detach().to("cpu") if torch.is_tensor(s) else s for s in cuda_states
            ])
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA RNG state could not be restored: %s", exc)


def main():
    args = parse_args()
    device = validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # ── Create RLT encoder + decoder ────────────────────────────────
    from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
    from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTokenDecoder, RLTokenEncoder

    rlt_config = PI05RLTConfig(
        mode="rlt_training",
        rlt_lr=args.lr,
        rlt_warmup_steps=args.warmup_steps,
        rlt_total_steps=args.steps,
    )
    encoder = RLTokenEncoder(rlt_config).to(device)
    decoder = RLTokenDecoder(rlt_config).to(device)

    # Multi-GPU wrapping is deliberately avoided: encoder/decoder are coupled modules.
    # A strict single-device boundary prevents unsafe split DataParallel behavior.

    # Count parameters
    enc_base = encoder.module if hasattr(encoder, 'module') else encoder
    dec_base = decoder.module if hasattr(decoder, 'module') else decoder
    enc_params = sum(p.numel() for p in enc_base.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in dec_base.parameters() if p.requires_grad)
    logger.info(f"RLT Encoder: {enc_params:,} params, Decoder: {dec_params:,} params")

    # ── Optimizer + scheduler ───────────────────────────────────────
    params = list(enc_base.parameters()) + list(dec_base.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    def lr_schedule(update_index):
        # LambdaLR evaluates index 0 at construction. Keep the first optimizer
        # update non-zero, then apply warmup followed by cosine decay.
        update_number = update_index + 1
        if args.warmup_steps > 0 and update_number <= args.warmup_steps:
            return update_number / args.warmup_steps
        cosine_updates = max(1, args.steps - args.warmup_steps)
        # Keep every requested optimizer update effective. The scheduler is
        # advanced after an update, so it reaches zero only for a hypothetical
        # update after the final requested one.
        progress = min(
            1.0,
            max(0.0, (update_number - args.warmup_steps - 1) / cosine_updates),
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Save / resume helpers ──────────────────────────────────────
    effective_args = vars(args).copy()
    effective_args["device"] = str(device)

    def save_checkpoint(output_dir, enc_base, dec_base, optimizer, rlt_config, step, loss_val,
                        scheduler=None, scaler=None, loss_history=None, best_loss=None):
        ckpt = {
            "step": step,
            "encoder_state_dict": enc_base.state_dict(),
            "decoder_state_dict": dec_base.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": rlt_config.__dict__,
            "rlt_architecture": rlt_config.rlt_architecture,
            "image_only": True,
            "loss": loss_val,
            # Preserve all historical Stage 2-consumed keys above; additions are optional.
            "effective_args": effective_args,
            "schedule_config": {
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "total_steps": args.steps,
            },
            "rng_state": capture_rng_state(),
            "resume_exact": False,
            "resume_note": "DataLoader sampler position is not checkpointed; resume is not exact.",
        }
        if scheduler is not None:
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        if best_loss is not None:
            ckpt["best_loss"] = best_loss
        if loss_history is not None:
            ckpt["loss_history"] = loss_history

        improved = best_loss is not None and loss_val < best_loss
        if improved:
            best_loss = loss_val
        if best_loss is not None:
            ckpt["best_loss"] = best_loss

        ckpt_path = output_dir / f"checkpoint_step{step}.pt"
        atomic_torch_save(ckpt, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

        if improved:
            atomic_torch_save(ckpt, output_dir / "best_checkpoint.pt")
            logger.info(f"New best loss: {best_loss:.4f}")
        return best_loss

    # ── Load data: fast path (precomputed) or slow path (online) ────
    if args.precomputed_path:
        # ── FAST MODE: load precomputed embeddings ──────────────────
        # Accepts comma-separated shard directories (parallel multi-GPU precompute
        # outputs); training concatenates them into one logical dataset.
        shard_paths = [Path(p.strip()) for p in str(args.precomputed_path).split(",") if p.strip()]
        if not shard_paths:
            raise ValueError("--precomputed_path is empty")
        logger.info(f"Loading precomputed embeddings from {len(shard_paths)} shard(s): "
                    f"{[str(p) for p in shard_paths]}...")
        precomputed_path = shard_paths[0]   # the first shard is the metadata reference; all shards must agree

        meta_path = precomputed_path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {precomputed_path}. "
                                    "Run precompute_pi05_embeddings.py first.")

        with open(meta_path) as f:
            meta = json.load(f)
        raw_cache_version = meta.get(
            "schema_version", meta.get("format_version", meta.get("version"))
        )
        if raw_cache_version is None:
            # Original v1 caches always carried actions; v2 deliberately omits them.
            raw_cache_version = 1 if (precomputed_path / "actions.mmap").exists() else 2
        cache_version = int(raw_cache_version)
        if cache_version == 2:
            if meta.get("complete") is not True:
                raise ValueError("v2 embedding cache is incomplete; rerun precomputation")
            if meta.get("image_only") is not True:
                raise ValueError("v2 cache must contain image-only Stage 1 embeddings")
            if meta.get("embedding_stage") != "post_transformer":
                raise ValueError("v2 cache must contain post-transformer prefix embeddings")
            if meta.get("language_positions_dropped") is not True:
                raise ValueError("v2 cache must drop language token positions")
        if cache_version not in (1, 2):
            raise ValueError(f"Unsupported embedding cache version {cache_version}; expected v1 or v2")
        N = int(meta["n_samples"])
        seq_len = int(meta["seq_len"])
        vlm_dim = int(meta["vlm_hidden_dim"])
        if N <= 0 or seq_len <= 0 or vlm_dim <= 0:
            raise ValueError("Cache dimensions must all be positive")
        has_mask = meta.get("has_mask", False)
        if not has_mask:
            raise ValueError(
                "Paper-aligned Stage 1 requires prefix_mask.mmap; regenerate the cache with "
                "precompute_pi05_embeddings.py."
            )
        if meta.get("dtype", "float16") != "float16":
            raise ValueError("Only float16 prefix_out.mmap caches are supported")
        if vlm_dim != rlt_config.vlm_hidden_dim:
            raise ValueError(
                f"Cache VLM dim {vlm_dim} does not match RLT config {rlt_config.vlm_hidden_dim}."
            )
        if seq_len > dec_base.pos_embed.shape[1]:
            raise ValueError(
                f"Cache sequence length {seq_len} exceeds decoder limit {dec_base.pos_embed.shape[1]}."
            )
        # Per-shard validation: each shard has its own n_samples, while
        # seq_len/vlm_dim/dtype must match exactly across shards.
        shard_counts: list[int] = []
        for sp in shard_paths:
            sp_meta_path = sp / "meta.json"
            if not sp_meta_path.exists():
                raise FileNotFoundError(f"shard is missing meta.json: {sp}")
            with open(sp_meta_path) as f:
                sp_meta = json.load(f)
            sp_n = int(sp_meta["n_samples"])
            for key, expect in (("seq_len", seq_len), ("vlm_hidden_dim", vlm_dim)):
                if int(sp_meta.get(key, -1)) != expect:
                    raise ValueError(
                        f"shard {sp} has {key}={sp_meta.get(key)}, which differs from the first shard ({expect})"
                    )
            if sp_meta.get("dtype", "float16") != "float16":
                raise ValueError(f"shard {sp} dtype is not float16")
            if sp_meta.get("has_mask", False) is not True:
                raise ValueError(f"shard {sp} is missing prefix_mask.mmap (has_mask=False)")
            if sp_meta.get("complete") is not True:
                raise ValueError(f"shard {sp} is incomplete (complete != true)")
            # Cache file size validation
            for filename, expected_size in (
                ("prefix_out.mmap", sp_n * seq_len * vlm_dim * np.dtype("float16").itemsize),
                ("prefix_mask.mmap", sp_n * seq_len * np.dtype("bool").itemsize),
            ):
                p = sp / filename
                if not p.exists() or p.stat().st_size != expected_size:
                    actual = p.stat().st_size if p.exists() else "missing"
                    raise ValueError(
                        f"Invalid cache file {p}: expected {expected_size} bytes, got {actual}."
                    )
            shard_counts.append(sp_n)
            logger.info("  shard %s: %d samples", sp.name, sp_n)
        N = int(sum(shard_counts))
        logger.info(
            "Using lazy mmap cache v%d: %d shard(s) / %d samples, seq_len=%d, vlm_hidden_dim=%d",
            cache_version, len(shard_paths), N, seq_len, vlm_dim,
        )

        dataset = MMapPrefixDataset(shard_paths, shard_counts, seq_len, vlm_dim)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            pin_memory=device.type == "cuda", num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
        data_iter = iter(dataloader)

        def get_batch():
            nonlocal data_iter
            try:
                prefix_out, mask = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                prefix_out, mask = next(data_iter)
            return (
                prefix_out.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"),
                mask.to(device=device, dtype=torch.bool, non_blocking=device.type == "cuda"),
            )

        logger.info("Using FAST mode (lazy image-only precomputed prefix/mask, no π0.5 needed)")
    else:
        # ── SLOW MODE: run π0.5 forward every step ──────────────────
        if not args.pi05_path or not args.dataset_repo_id:
            raise ValueError("Without --precomputed_path, both --pi05_path and --dataset_repo_id are required")

        logger.info(f"Loading π0.5 from {args.pi05_path}...")
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.pi05_rlt.vla_compat import (
            extract_embeddings as extract_pi05_embeddings,
        )
        from lerobot.policies.pi05_rlt.vla_compat import resize_with_pad as resize_with_pad_torch
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.configs.types import PolicyFeature

        config_path = Path(args.pi05_path) / "config.json"
        weights_path = Path(args.pi05_path) / "model.safetensors"
        if not config_path.is_file():
            raise FileNotFoundError(f"π0.5 config not found: {config_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"π0.5 weights not found: {weights_path}")
        with open(config_path) as f:
            config_dict = json.load(f)
        config_dict.pop("type", None)
        pi05_config = PI05Config(**config_dict)
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

        pi05_policy = PI05Policy.from_pretrained(
            pretrained_name_or_path=args.pi05_path,
            config=pi05_config,
        )
        pi05_policy.to(device)
        pi05_policy.eval()
        for p in pi05_policy.parameters():
            p.requires_grad = False
        pi05_model = pi05_policy.model
        logger.info("π0.5 loaded and frozen.")

        max_action_dim = pi05_config.max_action_dim
        chunk_size = pi05_config.chunk_size
        image_resolution = pi05_config.image_resolution

        logger.info(f"Loading dataset {args.dataset_repo_id}...")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        dataset_path = Path(args.dataset_repo_id)
        if not dataset_path.is_absolute():
            dataset_path = Path.cwd() / dataset_path
        if dataset_path.exists():
            dataset = LeRobotDataset(dataset_path.name, root=dataset_path, video_backend="pyav")
        else:
            dataset = LeRobotDataset(args.dataset_repo_id, video_backend="pyav")

        def _to_dict_features(features):
            return {
                k: {"type": v.type.name, "shape": list(v.shape)}
                if isinstance(v, PolicyFeature) else v
                for k, v in features.items()
            }

        pi05_config.input_features = _to_dict_features(pi05_config.input_features)
        pi05_config.output_features = _to_dict_features(pi05_config.output_features)

        # As above: the tokenizer location must be written into the config because
        # the current factory only consumes config/dataset_stats.
        if args.tokenizer_path:
            _tk = Path(args.tokenizer_path).expanduser()
            if _tk.exists():
                pi05_config.text_tokenizer_name = str(_tk)

        preprocessor, _ = make_pi05_pre_post_processors(
            config=pi05_config,
            dataset_stats=dataset.meta.stats,
        )

        if hasattr(preprocessor, 'steps'):
            preprocessor.steps = [
                step for step in preprocessor.steps
                if step.__class__.__name__ not in ('DeviceProcessorStep',)
            ]

        def collate_and_process(samples: list[dict[str, Any]]) -> dict[str, Any]:
            processed = []
            for sample in samples:
                item = preprocessor(sample)
                item = {
                    key: value.squeeze(0)
                    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 1
                    else value
                    for key, value in item.items()
                }
                processed.append(item)
            keys = processed[0].keys()
            return {
                key: torch.utils.data.default_collate([item[key] for item in processed])
                if isinstance(processed[0][key], torch.Tensor)
                else [item[key] for item in processed]
                for key in keys
            }

        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_and_process,
            persistent_workers=True if args.num_workers > 0 else False,
        )
        data_iter = iter(dataloader)
        logger.info(f"Dataset loaded: {len(dataset)} samples")

        def get_batch():
            """Run π0.5 with image_only=True and return prefix embeddings + mask."""
            nonlocal data_iter
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            images = []
            img_masks = []
            bsize = batch["action"].shape[0]
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

            actions = torch.zeros(
                bsize, chunk_size, max_action_dim, dtype=torch.float32, device=device
            )

            with torch.no_grad():
                prefix_img, _, prefix_mask = extract_pi05_embeddings(
                    pi05_model, images, img_masks, tokens, masks, actions,
                    chunk_size=chunk_size, max_action_dim=max_action_dim,
                    image_only=True,
                )

            return prefix_img.float(), prefix_mask.bool()

        logger.info("Using SLOW mode (image_only=True, running π0.5 forward every step)")

    # ── Mixed precision (optional) ─────────────────────────────────
    use_amp = args.use_amp
    amp_dtype = None
    use_scaler = False
    if use_amp:
        if device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device.type == "cpu":
            amp_dtype = torch.bfloat16
        elif device.type == "mps":
            raise ValueError("--use_amp is not supported on MPS by this training script")
        use_scaler = amp_dtype == torch.float16
        logger.info("Using AMP on %s (%s, GradScaler=%s)", device.type, amp_dtype, use_scaler)
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    # ── Resume from checkpoint (optional) ────────────────────────────
    start_step = 1
    loss_history = []
    best_loss = float("inf")
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        if ckpt.get("rlt_architecture") != rlt_config.rlt_architecture or not ckpt.get("image_only", False):
            raise ValueError(
                "Checkpoint is not a paper_v1 image-only Stage 1 checkpoint; legacy checkpoints "
                "must be retrained."
            )
        enc_base.load_state_dict(ckpt["encoder_state_dict"])
        dec_base.load_state_dict(ckpt["decoder_state_dict"])
        saved_schedule = ckpt.get("schedule_config")
        if saved_schedule is None:
            legacy_args = ckpt.get("effective_args", {})
            legacy_keys = ("lr", "warmup_steps", "steps")
            if all(key in legacy_args for key in legacy_keys):
                saved_schedule = {
                    "lr": legacy_args["lr"],
                    "warmup_steps": legacy_args["warmup_steps"],
                    "total_steps": legacy_args["steps"],
                }
        requested_schedule = {
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "total_steps": args.steps,
        }
        if saved_schedule is None:
            logger.warning(
                "Legacy checkpoint has no schedule metadata; assuming the requested LR schedule "
                "matches the original run."
            )
        elif saved_schedule != requested_schedule:
            if not args.allow_schedule_change:
                raise ValueError(
                    "Cannot resume with a different LR schedule. "
                    f"Checkpoint={saved_schedule}, requested={requested_schedule}. "
                    "Pass --allow_schedule_change to extend the run with a rebuilt schedule."
                )
            logger.warning(
                "LR schedule CHANGED with --allow_schedule_change: checkpoint=%s -> requested=%s. "
                "Rebuilding the schedule over %s steps and skipping the checkpoint scheduler state "
                "(warm restart).",
                saved_schedule, requested_schedule, args.steps,
            )
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        schedule_matches = saved_schedule == requested_schedule
        if "scheduler_state_dict" in ckpt and schedule_matches:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        elif start_step > 1 and schedule_matches:
            raise ValueError("Resume checkpoint is missing scheduler_state_dict")
        elif start_step > 1:
            logger.warning(
                "Scheduler state not restored: warmup %s steps then cosine decay to 0 at step %s "
                "(LR at step %s will be ~%.2e at peak).",
                args.warmup_steps, args.steps, start_step, args.lr,
            )
        if "scaler_state_dict" in ckpt and use_scaler:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "loss_history" in ckpt:
            loss_history = ckpt["loss_history"]
        if "best_loss" in ckpt:
            best_loss = ckpt["best_loss"]
        if "rng_state" in ckpt:
            restore_rng_state(ckpt["rng_state"])
            logger.warning(
                "Restored RNG state, but resume remains non-exact because DataLoader sampler "
                "position is not checkpointed."
            )
        else:
            logger.warning(
                "Legacy checkpoint has no RNG state; resume is explicitly non-exact and uses "
                "the current seeded RNG state."
            )
        if ckpt.get("effective_args") and ckpt["effective_args"] != effective_args:
            logger.warning("Effective CLI arguments differ from the checkpoint; resume is non-exact")
        logger.info(f"Resumed at step {start_step}, best_loss={best_loss:.4f}")

    # ── Training loop ───────────────────────────────────────────────
    logger.info(f"Starting training for {args.steps} steps (from step {start_step})...")
    start_time = time.time()

    encoder.train()
    decoder.train()
    enc_base.train()
    dec_base.train()

    updates_this_process = 0
    for step in range(start_step, args.steps + 1):
        prefix_out, prefix_mask = get_batch()

        # Clear stale gradients before forward so failed/dry runs cannot carry them onward.
        optimizer.zero_grad(set_to_none=True)

        # RLT forward: encode → z_rl → decode → masked reconstruction loss
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            # Detach targets (stop-gradient per RLT paper)
            target = prefix_out.detach()

            # Encode Eq. (1): [z_1:M, e_rl] → z_rl
            z_rl = encoder(target, mask=prefix_mask)

            # Decode Eq. (2): [z_rl, stopgrad(z_1:M-1)] with a causal mask
            vlm_recon = decoder(z_rl, target, mask=prefix_mask)

            # Masked MSE: only compute loss over valid (non-padding) tokens
            sq_error = (vlm_recon - target).pow(2)  # (B, L, D)
            mask_expanded = prefix_mask.float().unsqueeze(-1)  # (B, L, 1)
            loss = (sq_error * mask_expanded).sum() / (mask_expanded.sum() * target.shape[-1]).clamp(min=1.0)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite Stage 1 update at step {step}: loss={loss.item()}, grad_norm={grad_norm}"
            )
        if args.dry_run:
            logger.info(
                "Dry run passed: loss=%.6f, grad_norm=%.6f, valid_tokens=%d",
                loss.item(), float(grad_norm), int(prefix_mask.sum().item()),
            )
            return
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        updates_this_process += 1

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            updates_per_sec = updates_this_process / max(elapsed, 1e-9)
            logger.info(
                f"Step {step}/{args.steps} | Loss: {loss_val:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | {updates_per_sec:.1f} updates/s"
            )

        if step % args.save_every == 0 or step == args.steps:
            best_loss = save_checkpoint(
                output_dir, enc_base, dec_base, optimizer, rlt_config,
                step, loss_val,
                scheduler=scheduler, scaler=scaler if use_scaler else None,
                loss_history=loss_history, best_loss=best_loss,
            )

    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)

    total_time = time.time() - start_time
    logger.info(f"Training complete! {args.steps} steps in {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
