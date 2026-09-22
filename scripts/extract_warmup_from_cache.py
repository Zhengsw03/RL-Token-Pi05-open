#!/usr/bin/env python3
"""Extract annotated critical-phase intervals into a Stage-2 warmup replay journal
from the Stage-1 precomputed embedding cache (no π0.5 needed, RLT encoder only).

Reads prefix_out.mmap / prefix_mask.mmap (from precompute_pi05_embeddings.py,
schema v2, image-only post-transformer) and the dataset parquet rows, computes
z_rl with the frozen Stage-1 RLT encoder, and writes Stage-2 replay journal
records tagged collection_phase=warmup (same schema as extract_annotated_warmup.py).

Much faster than the π0.5 path: encoder is only ~400M params, so it fits in a
few GB of VRAM and runs in minutes.

Usage:
    python scripts/extract_warmup_from_cache.py \
        --cache_path outputs/pi05_embeddings \
        --rlt_checkpoint outputs/rlt_stage1/best_checkpoint.pt \
        --annotations annotations.json \
        --dataset_path /path/to/my_dataset \
        --output warmup_replay_journal.pkl \
        --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rlt_core import write_warmup_interval_records  # noqa: E402 (shared, see rlt_core.py)


def load_annotations(path: str | Path) -> dict[int, list[tuple[int, int]]]:
    """Read annotate_dataset.py JSON output: {episode: [(start, end), ...]}."""
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    annotations: dict[int, list[tuple[int, int]]] = {}
    for item in data.get("annotations", []):
        annotations.setdefault(int(item["episode"]), []).append(
            (int(item["start"]), int(item["end"]))
        )
    return annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_path", type=str, required=True,
                        help="Stage-1 precomputed embeddings dir (meta.json + prefix_*.mmap).")
    parser.add_argument("--rlt_checkpoint", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Dataset dir (only the data parquet is read for state/action).")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--state_dim", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    annotations = load_annotations(args.annotations)
    if not annotations:
        raise ValueError("Annotations file contains no intervals")
    logger.info("Loaded %d intervals across %d episodes",
                sum(len(v) for v in annotations.values()), len(annotations))

    # ── Cache metadata ───────────────────────────────────────────────────
    cache_path = Path(args.cache_path)
    with open(cache_path / "meta.json") as f:
        meta = json.load(f)
    assert meta.get("schema_version") == 2 and meta.get("image_only"), meta
    N, seq_len, dim = meta["n_samples"], meta["seq_len"], meta["vlm_hidden_dim"]
    logger.info("Cache: %d samples, seq_len=%d, dim=%d (%s)", N, seq_len, dim,
                meta.get("pi05_path", "?"))

    # ── Dataset rows: (episode, frame) -> parquet row ────────────────────
    data_files = sorted(Path(args.dataset_path).glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet under {args.dataset_path}/data")
    df = pd.read_parquet(data_files)
    df = df[["index", "episode_index", "frame_index", "action", "observation.state"]]
    df = df.sort_values("index").reset_index(drop=True)
    logger.info("Dataset parquet: %d rows", len(df))

    # ── RLT encoder (frozen, Stage 1) ────────────────────────────────────
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

    # ── mmap cache ───────────────────────────────────────────────────────
    prefix_out = np.memmap(cache_path / "prefix_out.mmap", dtype=np.float16, mode="r",
                           shape=(N, seq_len, dim))
    prefix_mask = np.memmap(cache_path / "prefix_mask.mmap", dtype=np.bool_, mode="r",
                            shape=(N, seq_len))

    # ── Collect annotated rows ───────────────────────────────────────────
    df_indexed = df.set_index("index")
    index_by_ep_frame = {
        (int(e), int(f)): int(i)
        for i, e, f in zip(df["index"], df["episode_index"], df["frame_index"])
    }
    rows: list[tuple[int, int, int]] = []  # (episode, frame, cache_index)
    for ep, intervals in sorted(annotations.items()):
        for start_f, end_f in intervals:
            for frame in range(int(start_f), int(end_f) + 1):
                key = (ep, frame)
                if key in index_by_ep_frame:
                    rows.append((ep, frame, index_by_ep_frame[key]))
    rows.sort()
    logger.info("Annotated frames: %d", len(rows))
    if not rows:
        raise ValueError("No annotated frames matched the dataset rows")

    # ── z_rl for every annotated frame (batched, encoder only) ───────────
    frame_data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    t0 = time.time()
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        idx = np.asarray([r[2] for r in batch_rows], dtype=np.int64)
        emb = torch.from_numpy(np.asarray(prefix_out[idx], dtype=np.float16)).to(device).float()
        mask = torch.from_numpy(np.asarray(prefix_mask[idx])).to(device)
        with torch.no_grad():
            z_rl = encoder(emb, mask=mask).float()  # (B, z_dim)
        if not torch.isfinite(z_rl).all().item():
            raise ValueError(f"Batch {start // args.batch_size}: z_rl contains NaN/Inf")
        z_np = z_rl.detach().cpu().numpy().astype(np.float16)
        for offset, (ep, frame, cache_idx) in enumerate(batch_rows):
            r = df_indexed.loc[cache_idx]
            proprio = np.asarray(r["observation.state"], dtype=np.float32)[:args.state_dim]
            action = np.asarray(r["action"], dtype=np.float32)[:args.action_dim]
            frame_data[(ep, frame)] = (z_np[offset], proprio, action)
        logger.info("z_rl batch %d/%d (%.1fs elapsed)",
                    start // args.batch_size + 1,
                    (len(rows) + args.batch_size - 1) // args.batch_size,
                    time.time() - t0)

    # ── Build stride-2 windows and write journal ─────────────────────────
    horizon, stride, action_dim = args.horizon, args.stride, args.action_dim
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    with open(out_path, "wb") as f:
        for ep, intervals in sorted(annotations.items()):
            for start_f, end_f in intervals:
                frames_in_interval = [frame for (epf, frame, _) in rows
                                      if epf == ep and start_f <= frame <= end_f]
                if len(frames_in_interval) < 1:
                    continue
                n_records += write_warmup_interval_records(
                    f, ep, frames_in_interval, frame_data,
                    horizon=horizon, stride=stride, action_dim=action_dim,
                )

    logger.info("Saved %d warmup replay transitions to %s", n_records, out_path)
    logger.info("Stage 2 usage: --restore_replay %s", out_path)


if __name__ == "__main__":
    main()
