#!/usr/bin/env python3
"""Offline evaluation of the critical-phase classifier (no real robot needed).

Replays the precomputed z_rl samples in temporal order through the exact
runtime pipeline (CriticalPhaseDetector: per-chunk probability, moving average,
hysteresis thresholds) and compares the auto-detected critical intervals
against the manual annotations:

  - per-sample classification report (precision / recall / F1 / accuracy),
    split by train vs held-out validation episodes;
  - per-episode interval comparison: detected vs annotated start/end frames,
    overlap IoU, and detection lag in frames.

Usage:
    python scripts/evaluate_critical_classifier.py \
        --precomputed_zrl outputs/critical_zrl_straw.npz \
        --classifier outputs/critical_classifier_straw.pt \
        --val_episodes 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.policies.pi05_rlt.critical import CriticalPhaseDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precomputed_zrl", type=str, default=None,
                        help="NPZ from precompute_critical_zrl.py (dataset evaluation).")
    parser.add_argument("--record_path", type=str, default=None,
                        help="Live 'c'-boundary log from train_rlt_stage2_pi05.py "
                             "(evaluate on REAL robot data; mutually exclusive-ish with precomputed_zrl).")
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--val_episodes", type=int, default=3,
                        help="Number of annotated episodes treated as held-out (same as training).")
    parser.add_argument("--threshold_on", type=float, default=0.6)
    parser.add_argument("--threshold_off", type=float, default=0.4)
    parser.add_argument("--smooth_steps", type=int, default=10)
    parser.add_argument("--min_on_chunks", type=int, default=0,
                        help="Debounce: hold ON for at least N updates before an OFF is accepted (0 = off).")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def intervals_from_flags(frames: np.ndarray, flags: np.ndarray) -> list[tuple[int, int]]:
    """Merge a per-sample flag sequence into (start, end) frame intervals."""
    intervals = []
    start = None
    for frame, flag in zip(frames, flags):
        if flag and start is None:
            start = frame
        elif not flag and start is not None:
            intervals.append((start, frame - 1))
            start = None
    if start is not None:
        intervals.append((start, frames[-1]))
    return intervals


def interval_union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals = sorted(intervals)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap_ratio(detected: list[tuple[int, int]], annotated: list[tuple[int, int]]) -> float:
    """Frame-level IoU between detected and annotated intervals (union of all intervals)."""
    det = interval_union(detected)
    ann = interval_union(annotated)
    if not det or not ann:
        return 0.0
    det_frames = set()
    for start, end in det:
        det_frames.update(range(start, end + 1))
    ann_frames = set()
    for start, end in ann:
        ann_frames.update(range(start, end + 1))
    intersection = len(det_frames & ann_frames)
    union = len(det_frames | ann_frames)
    return intersection / union if union else 0.0


def load_annotation_intervals(raw) -> dict[int, list[tuple[int, int]]]:
    """Accept both formats: {episode: [[start, end], ...]} (precompute NPZ)
    and {"annotations": [{"episode":..., "start":..., "end":...}, ...]}."""
    intervals: dict[int, list[tuple[int, int]]] = {}
    if isinstance(raw, dict):
        for episode, items in raw.items():
            intervals[int(episode)] = [tuple(map(int, item)) for item in items]
    else:
        for item in raw.get("annotations", []):
            intervals.setdefault(int(item["episode"]), []).append(
                (int(item["start"]), int(item["end"]))
            )
    return intervals


def main() -> None:
    args = parse_args()
    if not args.precomputed_zrl and not args.record_path:
        raise ValueError("Provide --precomputed_zrl or --record_path")
    if args.precomputed_zrl and args.record_path:
        raise ValueError("Provide only one of --precomputed_zrl or --record_path")
    if args.record_path:
        from train_critical_classifier import load_record_log
        features, _ = load_record_log(args.record_path)
        z_rl = features["z_rl"]
        proprio = features["proprio"]
        labels = features["labels"]
        episodes = features["episode"]
        frames = features["frame"]
        # Manual 'c' boundaries ARE the ground truth on real robot data:
        # rebuild per-episode intervals from the expanded labels.
        annotations = {}
        for episode in np.unique(episodes):
            mask = episodes == episode
            order = np.argsort(frames[mask])
            ep_frames = frames[mask][order]
            ep_labels = labels[mask][order]
            intervals = []
            start = None
            for i, lab in enumerate(ep_labels):
                if lab and start is None:
                    start = i
                elif not lab and start is not None:
                    intervals.append((int(ep_frames[start]), int(ep_frames[i - 1])))
                    start = None
            if start is not None:
                intervals.append((int(ep_frames[start]), int(ep_frames[-1])))
            if intervals:
                annotations[int(episode)] = intervals
        val_episodes: set[int] = set()
    else:
        data = np.load(args.precomputed_zrl, allow_pickle=False)
        z_rl = data["z_rl"].astype(np.float32)
        proprio = data["proprio"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        episodes = data["episode"].astype(np.int64)
        frames = data["frame"].astype(np.int64)
        annotations = load_annotation_intervals(json.loads(str(data["annotations"])))
        annotated_episodes = sorted(annotations)
        val_episodes = set(annotated_episodes[:args.val_episodes])

    detector = CriticalPhaseDetector(
        args.classifier,
        torch_device(args.device),
        threshold_on=args.threshold_on,
        threshold_off=args.threshold_off,
        smooth_steps=args.smooth_steps,
        min_on_chunks=args.min_on_chunks,
    )

    tp = fp = fn = 0
    rows = []
    for episode in sorted(np.unique(episodes)):
        mask = episodes == episode
        order = np.argsort(frames[mask])
        ep_z = z_rl[mask][order]
        ep_p = proprio[mask][order]
        ep_labels = labels[mask][order]
        ep_frames = frames[mask][order]

        # Simulate the online loop: feed samples in temporal order through the
        # detector (smoothing + hysteresis), as the training script would.
        detector.reset()
        detected_state = False
        preds = []
        for z, p, label in zip(ep_z, ep_p, ep_labels):
            candidate = detector.update(z, p)
            if candidate is not None:
                detected_state = candidate
            preds.append(detected_state)
        preds = np.asarray(preds, dtype=bool)

        tp += int((preds & (ep_labels == 1)).sum())
        fp += int((preds & (ep_labels == 0)).sum())
        fn += int((~preds & (ep_labels == 1)).sum())

        annotated = interval_union(annotations.get(episode, []))
        detected = intervals_from_flags(ep_frames, preds)
        iou = overlap_ratio(detected, annotated)
        split = "val " if episode in val_episodes else "train"
        rows.append((episode, split, annotated, detected, iou))

    # Per-sample report
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + (len(labels) - tp - fp - fn)) / len(labels)
    print("\n--- Per-sample classification report (simulated online detection) ---")
    print(f"  accuracy : {accuracy:.3f}")
    print(f"  precision: {precision:.3f}   recall: {recall:.3f}   F1: {f1:.3f}")

    print("\n--- Per-episode interval comparison (annotation vs auto detection) ---")
    print(f"{'ep':>3} {'split':>5} {'annotated':>28} {'detected':>28} {'IoU':>5} {'on lag':>6} {'off lag':>7}")
    ious_train, ious_val = [], []
    lags_train, lags_val = [], []
    end_lags_train, end_lags_val = [], []
    for episode, split, annotated, detected, iou in rows:
        lag = "  -  "
        end_lag = "  -  "
        if annotated and detected:
            lag = f"{detected[0][0] - annotated[0][0]:+d}"
            end_lag = f"{detected[-1][1] - annotated[-1][1]:+d}"
            (lags_val if split == "val " else lags_train).append(detected[0][0] - annotated[0][0])
            (end_lags_val if split == "val " else end_lags_train).append(
                detected[-1][1] - annotated[-1][1]
            )
        print(f"{episode:>3} {split:>5} {str(annotated):>28} {str(detected):>28} "
              f"{iou:>5.2f} {lag:>5} {end_lag:>5}")
        (ious_val if split == "val " else ious_train).append(iou)
    print(f"\nMean IoU: train {np.mean(ious_train):.3f} (n={len(ious_train)}) | "
          f"val {np.mean(ious_val):.3f} (n={len(ious_val)})")
    if ious_val:
        print(f"Validation mean IoU: {np.mean(ious_val):.3f}")
    if lags_train:
        print(f"ON lag (frames): train mean {np.mean(lags_train):+.1f} | val mean {np.mean(lags_val or [0]):+.1f}")
    if end_lags_train:
        print(f"OFF lag (frames): train mean {np.mean(end_lags_train):+.1f} | val mean {np.mean(end_lags_val or [0]):+.1f}  (+ = switched OFF late)")


def torch_device(name: str):
    import torch
    return torch.device(name if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
