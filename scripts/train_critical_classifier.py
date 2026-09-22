#!/usr/bin/env python3
"""Train the critical-phase binary classifier (z_rl + proprio -> P(critical)).

Data sources (can be mixed):
  1. Precomputed z_rl from manually annotated teleop data (primary):
       --precomputed_zrl <npz from precompute_critical_zrl.py>
     The npz already contains balanced positive/negative samples (positive =
     inside the operator's annotated intervals; negative = unmarked frames of
     annotated episodes only, so unannotated episodes never leak labels).
  2. Live 'c'-boundary log (optional supplement):
       --record_path <train_rlt_stage2_pi05.py --critical_record_path output>
     z_rl/proprio samples with manual 'c' toggle events expanded into labels.

Usage:
    python scripts/train_critical_classifier.py \
        --precomputed_zrl outputs/critical_zrl.npz \
        --output outputs/critical_classifier.pt \
        --epochs 30 --batch_size 64
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot.policies.pi05_rlt.critical import CriticalPhaseClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precomputed_zrl", type=str, default=None,
                        help="NPZ from precompute_critical_zrl.py (primary source).")
    parser.add_argument("--record_path", type=str, default=None,
                        help="Live 'c'-boundary log from train_rlt_stage2_pi05.py (optional supplement).")
    parser.add_argument("--output", type=str, required=True,
                        help="Checkpoint output path (.pt).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--val_episodes", type=int, default=3,
                        help="Number of annotated episodes held out for validation (by episode, no leakage).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--record_repeat", type=int, default=1,
                        help="Repeat live record-log samples this many times when mixing with the "
                             "dataset (real-robot data is scarce; e.g. 20x balances ~300 samples "
                             "against ~15k dataset samples).")
    parser.add_argument("--tv_weight", type=float, default=0.05,
                        help="Total-variation penalty on the per-episode P(critical) sequence: "
                             "penalizes extra ON/OFF switches so each episode learns to have "
                             "exactly ONE critical segment.")
    parser.add_argument("--trials", type=int, default=1,
                        help="Train this many times with increasing seeds and keep the best "
                             "validation accuracy (helps escape bad initializations).")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


# ── Sampling helpers (shared with precompute_critical_zrl.py) ───────────────

def load_annotations(path: str) -> dict[int, list[tuple[int, int]]]:
    """{episode: [(start, end), ...]} with inclusive frame indices."""
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    annotations: dict[int, list[tuple[int, int]]] = {}
    for item in data.get("annotations", []):
        episode = int(item["episode"])
        start, end = int(item["start"]), int(item["end"])
        if start > end:
            raise ValueError(f"Annotation start > end for episode {episode}: {start} > {end}")
        annotations.setdefault(episode, []).append((start, end))
    return annotations


def sample_dataset_frames(
    dataset_path: str,
    annotations: dict[int, list[tuple[int, int]]],
    *,
    pos_per_interval: int,
    neg_per_episode: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Return (global_frame_index, label) pairs sampled from the dataset.

    Positives are drawn uniformly inside each annotated interval; negatives are
    drawn ONLY from annotated episodes' unmarked frames — an unannotated
    episode never silently contributes "non-critical" labels.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rng = random.Random(seed)
    ds = LeRobotDataset(repo_id=dataset_path)
    episode_ids = np.asarray(ds.hf_dataset["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(ds.hf_dataset["frame_index"], dtype=np.int64)
    if len(episode_ids) != ds.num_frames:
        raise ValueError("hf_dataset episode_index length does not match num_frames")

    episode_offsets: dict[int, np.ndarray] = {}
    for episode in np.unique(episode_ids):
        episode_offsets[int(episode)] = np.flatnonzero(episode_ids == int(episode))

    samples: list[tuple[int, int]] = []
    for episode, intervals in sorted(annotations.items()):
        indices = episode_offsets.get(episode)
        if indices is None:
            raise ValueError(f"Annotation references unknown episode {episode}")
        frames = frame_ids[indices]
        marked = np.zeros(len(frames), dtype=bool)
        for start, end in intervals:
            marked |= (frames >= start) & (frames <= end)
        unmarked_indices = indices[~marked]
        positive_count = 0
        for start, end in intervals:
            interval_mask = (frames >= start) & (frames <= end)
            pool = indices[interval_mask]
            count = min(pos_per_interval, len(pool))
            positive_count += count
            for global_idx in rng.sample(list(pool), count):
                samples.append((int(global_idx), 1))
        negative_count = min(neg_per_episode, len(unmarked_indices))
        for global_idx in rng.sample(list(unmarked_indices), negative_count):
            samples.append((int(global_idx), 0))
        logger.info(
            "Episode %s: %d intervals, %d marked frames -> %d positive / %d negative samples",
            episode, len(intervals), int(marked.sum()), positive_count, negative_count,
        )
    return samples


# ── Source 1: precomputed z_rl NPZ ──────────────────────────────────────────

def load_precomputed(path: str, *, val_episodes: int) -> tuple[dict, np.ndarray]:
    """Load the NPZ and derive a validation mask by held-out episode."""
    data = np.load(path, allow_pickle=False)
    z_rl = data["z_rl"].astype(np.float32)
    proprio = data["proprio"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    episodes = data["episode"].astype(np.int64)
    frames = data["frame"].astype(np.int64)
    annotations = json.loads(str(data["annotations"]))
    if isinstance(annotations, dict):
        # precompute_critical_zrl.py stores {episode: [[start, end], ...]}
        annotated_episodes = sorted(int(episode) for episode in annotations)
    else:
        # list of {"episode": ..., "start": ..., "end": ...} items
        annotated_episodes = sorted(int(item["episode"]) for item in annotations.get("annotations", []))
    val_episodes_set = set(annotated_episodes[:val_episodes])
    val_mask = np.isin(episodes, list(val_episodes_set))
    if val_mask.sum() == 0:
        raise ValueError(
            f"No samples fall in the held-out episodes {sorted(val_episodes_set)}; "
            "increase --val_episodes or annotate more episodes."
        )
    logger.info(
        "Precomputed source: %d samples (%d positive), val episodes %s (%d val samples)",
        len(labels), int(labels.sum()), sorted(val_episodes_set), int(val_mask.sum()),
    )
    return {
        "z_rl": z_rl,
        "proprio": proprio,
        "labels": labels,
        "episode": episodes.astype(np.int64),
        "frame": frames.astype(np.int64),
    }, val_mask


# ── Source 2: live 'c'-boundary record log ──────────────────────────────────

def load_record_log(path: str) -> tuple[dict, np.ndarray]:
    """Read a CriticalPhaseRecorder log -> (features, labels).

    Toggle events define per-episode critical intervals (sorted by step);
    samples inside an interval get label 1, outside get 0. Always training data
    (validation is a no-op mask).
    """
    samples: list[dict] = []
    toggles: list[tuple[int, int, bool]] = []
    with open(path, "rb") as file:
        while True:
            try:
                record = pickle.load(file)
            except EOFError:
                break
            if record["kind"] == "sample":
                samples.append(record)
            elif record["kind"] == "toggle":
                toggles.append((record["episode"], record["step"], record["critical"]))

    active: dict[int, list[tuple[int, int]]] = {}
    state: dict[int, bool] = {}
    interval_start: dict[int, int] = {}
    for episode, step, critical in sorted(toggles, key=lambda item: (item[0], item[1])):
        if critical and not state.get(episode, False):
            interval_start[episode] = step
            state[episode] = True
        elif not critical and state.get(episode, False):
            active.setdefault(episode, []).append((interval_start[episode], step))
            state[episode] = False
    for episode, is_active in state.items():
        if is_active:
            logger.warning("Episode %s ends with the critical phase still active; interval kept open.", episode)
            active.setdefault(episode, []).append((interval_start[episode], 2**31 - 1))

    z_rl, proprio, labels = [], [], []
    for sample in samples:
        episode, step = sample["episode"], sample["step"]
        marked = any(start <= step <= end for start, end in active.get(episode, []))
        z_rl.append(np.asarray(sample["z_rl"], dtype=np.float32))
        proprio.append(np.asarray(sample["proprio"], dtype=np.float32))
        labels.append(1 if marked else 0)
    logger.info(
        "Record log: %d labeled samples (%d positive, %d negative) from %d toggle boundaries",
        len(labels), sum(labels), len(labels) - sum(labels), len(toggles),
    )
    if not z_rl:
        raise ValueError("No samples found in record log")
    return {
        "z_rl": np.stack(z_rl),
        "proprio": np.stack(proprio),
        "labels": np.asarray(labels, dtype=np.int64),
        "episode": np.asarray([sample["episode"] for sample in samples], dtype=np.int64),
        "frame": np.asarray([sample["step"] for sample in samples], dtype=np.int64),
    }, np.zeros(len(labels), dtype=bool)


# ── Training ─────────────────────────────────────────────────────────────────

def train(
    features: dict,
    val_mask: np.ndarray,
    args: argparse.Namespace,
    *,
    seed: int | None = None,
) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = args.seed if seed is None else seed
    z_dim = features["z_rl"].shape[1]
    proprio_dim = features["proprio"].shape[1]
    model = CriticalPhaseClassifier(
        z_dim=z_dim,
        proprio_dim=proprio_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    proprio_mean = features["proprio"].mean(axis=0).astype(np.float32)
    proprio_std = features["proprio"].std(axis=0).astype(np.float32) + 1e-6

    z_t = torch.from_numpy(features["z_rl"]).to(device)
    p_t = (
        torch.from_numpy(features["proprio"]).to(device)
        - torch.from_numpy(proprio_mean).to(device)
    ) / torch.from_numpy(proprio_std).to(device)
    labels_t = torch.from_numpy(features["labels"]).to(device)

    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Both train and validation sets must be non-empty")

    # Group training samples by episode, sorted by frame, so each forward pass
    # sees one episode's probability SEQUENCE. A total-variation penalty on the
    # sequence discourages extra ON/OFF switches: each episode should contain
    # exactly ONE critical segment (the task prior).
    episodes = np.unique(features["episode"])
    groups = []
    for episode in episodes:
        mask = features["episode"] == episode
        order = np.argsort(features["frame"][mask])
        groups.append(np.flatnonzero(mask)[order])
    train_groups = [group for group in groups if not val_mask[group].any()]
    if not train_groups:
        raise ValueError("No training episodes available")

    best_val_acc = 0.0
    best_state = None
    rng = np.random.RandomState(seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_groups)
        total_loss = 0.0
        correct = 0
        for group in train_groups:
            logits = model(z_t[group], p_t[group])
            ce_loss = F.cross_entropy(logits, labels_t[group])
            loss = ce_loss
            if args.tv_weight > 0 and len(group) > 1:
                probabilities = torch.softmax(logits, dim=-1)[:, 1]
                tv = (probabilities[1:] - probabilities[:-1]).abs().mean()
                loss = ce_loss + args.tv_weight * tv
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(group)
            correct += int((logits.argmax(dim=-1) == labels_t[group]).sum().item())
        train_acc = correct / len(train_idx)

        model.eval()
        with torch.no_grad():
            val_correct = 0
            for start in range(0, len(val_idx), args.batch_size):
                batch = val_idx[start:start + args.batch_size]
                val_correct += int(
                    (model(z_t[batch], p_t[batch]).argmax(dim=-1) == labels_t[batch]).sum().item()
                )
        val_acc = val_correct / len(val_idx)
        logger.info(
            "Epoch %2d/%d | loss %.4f | train acc %.3f | val acc %.3f",
            epoch, args.epochs, total_loss / len(train_idx), train_acc, val_acc,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return {
        "classifier_state_dict": best_state,
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "z_dim": z_dim,
        "proprio_dim": proprio_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "val_acc": float(best_val_acc),
        "seed": seed,
        "schema_version": 1,
    }


def main() -> None:
    args = parse_args()
    if not args.precomputed_zrl and not args.record_path:
        raise ValueError("Provide at least one of --precomputed_zrl or --record_path")

    features_list, val_masks = [], []
    if args.precomputed_zrl:
        features, val_mask = load_precomputed(args.precomputed_zrl, val_episodes=args.val_episodes)
        features_list.append(features)
        val_masks.append(val_mask)
    if args.record_path:
        features, val_mask = load_record_log(args.record_path)
        if args.record_repeat > 1:
            features = {
                key: np.repeat(features[key], args.record_repeat, axis=0)
                for key in features
            }
            val_mask = np.repeat(val_mask, args.record_repeat)
            logger.info("Record-log samples repeated x%d -> %d samples",
                        args.record_repeat, len(features["labels"]))
        features_list.append(features)
        val_masks.append(val_mask)

    features = {
        key: np.concatenate([item[key] for item in features_list])
        for key in ("z_rl", "proprio", "labels", "episode", "frame")
    }
    val_mask = np.concatenate(val_masks)
    if features["z_rl"].shape[1] != features_list[0]["z_rl"].shape[1]:
        raise ValueError("z_rl dims disagree between sources")

    logger.info(
        "Total: %d samples (%d positive, %d negative)",
        len(features["labels"]), int(features["labels"].sum()),
        len(features["labels"]) - int(features["labels"].sum()),
    )
    best_checkpoint = None
    for trial in range(1, args.trials + 1):
        trial_seed = args.seed + (trial - 1) * 7
        logger.info("Trial %d/%d (seed %d)", trial, args.trials, trial_seed)
        checkpoint = train(features, val_mask, args=args, seed=trial_seed)
        if best_checkpoint is None or checkpoint["val_acc"] > best_checkpoint["val_acc"]:
            best_checkpoint = checkpoint
            logger.info("  -> new best val acc %.3f (seed %d)", checkpoint["val_acc"], trial_seed)
        else:
            logger.info("  -> val acc %.3f (kept previous best)", checkpoint["val_acc"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_checkpoint, output)
    logger.info("Saved classifier checkpoint to %s (best val acc %.3f, seed %d)",
                output, best_checkpoint["val_acc"], best_checkpoint["seed"])


if __name__ == "__main__":
    main()
