#!/usr/bin/env python3
"""Render visual checks for the critical-phase classifier (no robot needed).

Two views are produced from the same inputs, a precomputed z_rl npz
(``precompute_critical_zrl.py``) and a trained checkpoint
(``train_critical_classifier.py``):

  timeline  one panel per episode: the classifier probability over the episode,
            with the hand-annotated critical interval shaded and the 0.5 line
            drawn for reference.
  sheet     one row per annotated interval: the frames around the annotated
            start boundary plus the frame just before the annotated end
            boundary, bordered green inside the interval and red outside, with
            the classifier probability printed under each frame.

The probabilities are the raw softmax output of the classifier on every
precomputed sample, so they are the same numbers the runtime detector sees
before its moving average and hysteresis.

Note that the classifier is usually trained on the same episodes that are
visualized, so a clean separation confirms the pipeline and the labels but not
the ability to generalize to unseen episodes.

Usage:
    python scripts/visualize_critical_classifier.py \
        --precomputed_zrl outputs/critical_zrl_bolt.npz \
        --classifier outputs/critical_classifier_bolt.pt \
        --out_dir outputs/critical_vis --mode both \
        --dataset /path/to/lerobot/insert_bolt --annotations annotations.json \
        --episodes 0-11 --cam top
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lerobot" / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Sheet tiles, in pixels.
TILE_W, TILE_H, LABEL_H, PAD = 320, 240, 26, 6
# Timeline panels, in pixels.
PANEL_COLS, PANEL_W, PANEL_H, PANEL_PAD = 3, 430, 132, 16
PANEL_X, PANEL_Y = 36, 18


def parse_episodes(spec: str) -> list[int]:
    """Parse ``0-11,20`` into a sorted list of episode indices."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def classifier_probabilities(detector, z_rl, proprio, batch: int = 1000) -> np.ndarray:
    """Softmax probability of the critical class for every precomputed sample."""
    import torch

    probs = np.zeros(len(z_rl), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(z_rl), batch):
            z = torch.as_tensor(z_rl[i : i + batch], device=detector.device)
            p = torch.as_tensor(proprio[i : i + batch], device=detector.device)
            p = (p - detector.proprio_mean) / detector.proprio_std
            logits = detector.classifier(z, p)
            probs[i : i + batch] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs


def load_annotations(path: Path | None, npz) -> list[dict]:
    """Annotated critical intervals, from the JSON file if given, else from the npz.

    Two annotation layouts are accepted: a list of ``{"episode", "start", "end"}``
    records (the format ``annotate_dataset.py`` writes), and a mapping from
    episode index to a list of ``[start, end]`` pairs.

    The npz labels cannot be used to recover the intervals: sampling keeps only a
    few hundred frames per interval, so a label of 1 covers a subset of the
    interval and the end frame can carry a label of 0.
    """
    def normalize(raw) -> list[dict]:
        if isinstance(raw, dict) and "annotations" in raw:
            raw = raw["annotations"]
        if isinstance(raw, dict):
            return [{"episode": int(episode), "start": int(pair[0]), "end": int(pair[1])}
                    for episode, pairs in raw.items() for pair in pairs]
        return [{"episode": int(a["episode"]), "start": int(a["start"]), "end": int(a["end"])}
                for a in raw]

    if path is not None and path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        annotations = normalize(raw)
        logger.info("intervals from %s: %d", path.name, len(annotations))
        return annotations

    if "annotations" in npz.files:
        blob = npz["annotations"]
        try:
            parsed = json.loads(blob.item()) if getattr(blob, "shape", ()) == () else [
                json.loads(entry) if isinstance(entry, str) else entry for entry in blob
            ]
            annotations = normalize(parsed)
        except Exception as error:  # noqa: BLE001 - an unreadable blob is reported as "no annotations"
            logger.warning("could not parse the annotations stored in the npz: %s", error)
            annotations = []
        logger.info("intervals from the npz: %d", len(annotations))
        return annotations

    logger.warning("no annotated intervals found; --annotations is required for the sheet view")
    return []


def longest_run(series: np.ndarray, frame: np.ndarray, threshold: float) -> tuple[int, int]:
    """Longest run of consecutive samples above ``threshold``, as (start, end) frames."""
    on = series > threshold
    best_len, best_start, best_end = 0, -1, -1
    i = 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(on) and on[j + 1]:
            j += 1
        if j - i + 1 > best_len:
            best_len, best_start, best_end = j - i + 1, int(frame[i]), int(frame[j])
        i = j + 1
    return best_start, best_end


def render_timeline(episodes, episode, frame, probs, labels, ann_by_ep, out_dir: Path) -> None:
    """Grid of per-episode probability curves, one panel per episode."""
    rows = (len(episodes) + PANEL_COLS - 1) // PANEL_COLS
    width = PANEL_COLS * PANEL_W + PANEL_PAD * 2
    height = rows * PANEL_H + PANEL_PAD * 2
    img = Image.new("RGB", (width, height), (20, 20, 24))
    draw = ImageDraw.Draw(img)
    plot_w, plot_h = PANEL_W - 44, PANEL_H - 44

    with open(out_dir / "prob_timeline.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["episode", "frame", "prob"])
        for ep, fr, pr in zip(episode, frame, probs):
            writer.writerow([int(ep), int(fr), f"{pr:.4f}"])

    lines = ["episode  n  prob_in  prob_out  annotated_interval  detected_interval(>0.5)"]
    for i, ep in enumerate(episodes):
        cx, cy = PANEL_PAD + (i % PANEL_COLS) * PANEL_W, PANEL_PAD + (i // PANEL_COLS) * PANEL_H
        mask = episode == ep
        order = np.argsort(frame[mask])
        frames_ep, probs_ep = frame[mask][order], probs[mask][order]
        n = len(probs_ep)
        start, end = ann_by_ep.get(ep, (-1, -1))
        inside = (frames_ep >= start) & (frames_ep <= end) if start >= 0 else np.zeros(n, dtype=bool)
        prob_in = float(probs_ep[inside].mean()) if inside.any() else float("nan")
        prob_out = float(probs_ep[~inside].mean()) if (~inside).any() else float("nan")

        draw.rectangle([cx + PANEL_X, cy + PANEL_Y, cx + PANEL_X + plot_w, cy + PANEL_Y + plot_h],
                       outline=(70, 70, 80))
        if start >= 0 and n > 1:
            x0 = cx + PANEL_X + plot_w * (np.searchsorted(frames_ep, start) / max(n - 1, 1))
            x1 = cx + PANEL_X + plot_w * (np.searchsorted(frames_ep, end) / max(n - 1, 1))
            draw.rectangle([x0, cy + PANEL_Y, x1, cy + PANEL_Y + plot_h], fill=(28, 52, 34))
        y_mid = cy + PANEL_Y + plot_h * 0.5
        draw.line([cx + PANEL_X, y_mid, cx + PANEL_X + plot_w, y_mid], fill=(120, 120, 60))
        points = [(cx + PANEL_X + plot_w * k / max(n - 1, 1),
                   cy + PANEL_Y + plot_h * (1 - float(v))) for k, v in enumerate(probs_ep)]
        if len(points) > 1:
            draw.line(points, fill=(90, 200, 255), width=2)
        draw.text((cx + PANEL_X, cy + 4),
                  f"ep{ep} n={n} in={prob_in:.2f} out={prob_out:.2f}", fill=(232, 232, 232))
        draw.text((cx + 4, cy + PANEL_Y + plot_h - 4), "0", fill=(150, 150, 150))
        draw.text((cx + 10, cy + PANEL_Y - 2), "1", fill=(150, 150, 150))

        detected = longest_run(probs_ep, frames_ep, 0.5)
        lines.append(f"{ep:>3} {n:>4} {prob_in:>7.3f} {prob_out:>7.3f} "
                     f"{str((start, end)):>17} {str(detected):>25}")

    path = out_dir / "prob_timeline.png"
    img.save(path, optimize=True)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s and summary.txt (green band = annotated interval, blue curve = "
                "classifier probability, yellow line = 0.5)", path)

    if labels is not None and labels.any():
        logger.info("sample accuracy at 0.5: %.4f (in-interval samples only cover part of "
                    "each interval by construction)", ((probs >= 0.5).astype(int) == labels).mean())


def build_episode_file_map(dataset: Path, cam: str) -> dict[int, tuple[Path, int]]:
    """episode index -> (video file, frame offset of that episode inside the file)."""
    import pandas as pd

    files = sorted(glob.glob(str(dataset / "meta" / "episodes" / "chunk-*" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {dataset}/meta/episodes/")
    meta = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    chunk_col = f"videos/observation.images.{cam}/chunk_index"
    file_col = f"videos/observation.images.{cam}/file_index"
    for col in ("episode_index", "length", chunk_col, file_col):
        if col not in meta.columns:
            raise KeyError(f"episode metadata is missing the column {col}")

    root = dataset / "videos" / f"observation.images.{cam}"
    mapping: dict[int, tuple[Path, int]] = {}
    running: dict[tuple[int, int], int] = defaultdict(int)
    for _, row in meta.sort_values("episode_index").iterrows():
        key = (int(row[chunk_col]), int(row[file_col]))
        mapping[int(row["episode_index"])] = (
            root / f"chunk-{key[0]:03d}" / f"file-{key[1]:03d}.mp4", running[key])
        running[key] += int(row["length"])
    return mapping


def grab_frames(video_path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decode only the requested frame indices of one video file."""
    import av

    got: dict[int, np.ndarray] = {}
    if not wanted:
        return got
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                got[index] = frame.to_ndarray(format="rgb24")
                if len(got) == len(wanted):
                    break
    return got


def tile(img: np.ndarray, caption: str, border: tuple[int, int, int]) -> Image.Image:
    """One frame with a colored border and a caption underneath."""
    canvas = Image.new("RGB", (TILE_W + 2 * PAD, TILE_H + LABEL_H + 2 * PAD), (28, 28, 28))
    canvas.paste(Image.fromarray(img).resize((TILE_W, TILE_H)), (PAD, PAD))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([PAD - 2, PAD - 2, PAD + TILE_W + 1, PAD + TILE_H + 1], outline=border, width=3)
    draw.text((PAD, PAD + TILE_H + 4), caption, fill=(235, 235, 235))
    return canvas


def render_sheets(items, episode, frame, probs, labels, dataset: Path, cam: str,
                  out_dir: Path, per_sheet: int, context: int) -> int:
    """One sheet of frame rows per group of annotated intervals."""
    episode_map = build_episode_file_map(dataset, cam)

    needed: dict[int, set[int]] = defaultdict(set)
    for item in items:
        for boundary in (item["start"], item["end"]):
            needed[item["episode"]].update(range(boundary - context, boundary + context + 1))

    cache: dict[tuple[int, int], np.ndarray] = {}
    for ep, frames in sorted(needed.items()):
        if ep not in episode_map:
            logger.warning("episode %d has no video metadata, skipped", ep)
            continue
        video_path, offset = episode_map[ep]
        if not video_path.exists():
            logger.warning("missing video %s, skipped", video_path)
            continue
        local = {offset + f for f in frames if f >= 0}
        for f, rgb in grab_frames(video_path, local).items():
            cache[(ep, f - offset)] = rgb
    logger.info("decoded %d frames", len(cache))

    tiles_per_row = 2 * context + 2
    sheet_index = 0
    for batch_start in range(0, len(items), per_sheet):
        batch = items[batch_start : batch_start + per_sheet]
        sheet = Image.new("RGB", ((TILE_W + 2 * PAD) * tiles_per_row,
                                  (TILE_H + LABEL_H + 2 * PAD) * len(batch)), (18, 18, 18))
        for row, item in enumerate(batch):
            start, end = item["start"], item["end"]
            sequence = (list(range(start - context, start + context + 1))
                        + list(range(end - context, end + context + 1)))
            for column, f in enumerate(sequence[:tiles_per_row]):
                rgb = cache.get((item["episode"], f))
                if rgb is None:
                    continue
                inside = start <= f <= end
                probability = item["prob_start"] if column <= context else item["prob_end"]
                caption = (f"ep{item['episode']} f{f} p={probability:.2f}"
                           f"{' IN' if inside else ' out'}")
                sheet.paste(tile(rgb, caption, (80, 220, 120) if inside else (235, 90, 90)),
                            ((TILE_W + 2 * PAD) * column, (TILE_H + LABEL_H + 2 * PAD) * row))
        path = out_dir / f"sheet_{sheet_index:02d}.png"
        sheet.save(path, optimize=True)
        logger.info("wrote %s (episodes %s)", path, sorted({item["episode"] for item in batch}))
        sheet_index += 1

    csv_path = out_dir / "boundaries.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "episode", "start", "end", "prob_at_start", "prob_at_end",
            "prob_in_interval_mean", "prob_out_interval_mean", "n_sampled"])
        writer.writeheader()
        for item in items:
            mask = episode == item["episode"]
            frames_ep, probs_ep = frame[mask], probs[mask]
            inside = (frames_ep >= item["start"]) & (frames_ep <= item["end"])
            writer.writerow({
                "episode": item["episode"], "start": item["start"], "end": item["end"],
                "prob_at_start": f"{item['prob_start']:.4f}",
                "prob_at_end": f"{item['prob_end']:.4f}",
                "prob_in_interval_mean": f"{probs_ep[inside].mean():.4f}" if inside.any() else "",
                "prob_out_interval_mean": f"{probs_ep[~inside].mean():.4f}" if (~inside).any() else "",
                "n_sampled": int(mask.sum()),
            })
    logger.info("wrote %s", csv_path)
    return sheet_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--precomputed_zrl", type=Path, required=True,
                        help="npz from precompute_critical_zrl.py")
    parser.add_argument("--classifier", type=Path, required=True, help="classifier checkpoint")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("timeline", "sheet", "both"), default="both")
    parser.add_argument("--annotations", type=Path, default=None,
                        help="annotation JSON; defaults to the intervals stored in the npz")
    parser.add_argument("--episodes", type=str, default="0-11",
                        help="episodes to draw in the sheet view, e.g. 0-11,20")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="LeRobot dataset root, required for the sheet view")
    parser.add_argument("--cam", type=str, default="top", help="camera key for the sheet frames")
    parser.add_argument("--per_sheet", type=int, default=6, help="rows per sheet image")
    parser.add_argument("--context", type=int, default=1,
                        help="frames on each side of the start boundary")
    parser.add_argument("--threshold_on", type=float, default=0.75)
    parser.add_argument("--threshold_off", type=float, default=0.45)
    parser.add_argument("--smooth_steps", type=int, default=4)
    parser.add_argument("--min_on_chunks", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    import torch
    from lerobot.policies.pi05_rlt.critical import CriticalPhaseDetector

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "sheet" and args.dataset is None:
        parser.error("--dataset is required for --mode sheet")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.precomputed_zrl, allow_pickle=True)
    episode, frame = data["episode"].astype(int), data["frame"].astype(int)
    labels = data["labels"].astype(int) if "labels" in data.files else None

    if "prob" in data.files:
        probs = np.asarray(data["prob"], dtype=np.float32)
        logger.info("using the probabilities stored in the npz (%d samples)", len(probs))
    else:
        detector = CriticalPhaseDetector(
            str(args.classifier), device,
            threshold_on=args.threshold_on, threshold_off=args.threshold_off,
            smooth_steps=args.smooth_steps, min_on_chunks=args.min_on_chunks)
        probs = classifier_probabilities(detector, data["z_rl"], data["proprio"], args.batch_size)
        logger.info("computed probabilities for %d samples on %s", len(probs), device)

    anns = load_annotations(args.annotations, data)
    ann_by_ep = {a["episode"]: (a["start"], a["end"]) for a in anns}
    if labels is not None:
        positive, negative = probs[labels == 1], probs[labels == 0]
        logger.info("in-interval mean %.3f, out-of-interval mean %.3f, episodes %d",
                    positive.mean(), negative.mean(), len(np.unique(episode)))

    wanted = set(parse_episodes(args.episodes))
    episodes = [ep for ep in sorted(set(episode.tolist())) if ep in wanted] or sorted(set(episode.tolist()))

    if args.mode in ("timeline", "both"):
        render_timeline(episodes, episode, frame, probs, labels, ann_by_ep, args.out_dir)

    if args.mode in ("sheet", "both"):
        if not anns:
            logger.error("no annotated intervals, the sheet view cannot be drawn")
            return 1
        items = []
        for annotation in anns:
            ep, start, end = annotation["episode"], annotation["start"], annotation["end"]
            if ep not in wanted:
                continue
            mask = episode == ep
            order = np.argsort(frame[mask])
            frames_ep, probs_ep = frame[mask][order], probs[mask][order]
            if len(frames_ep) == 0:
                continue
            at_start = int(np.clip(np.searchsorted(frames_ep, start), 0, len(frames_ep) - 1))
            at_end = int(np.clip(np.searchsorted(frames_ep, end), 0, len(frames_ep) - 1))
            items.append({"episode": ep, "start": start, "end": end,
                          "prob_start": float(probs_ep[at_start]),
                          "prob_end": float(probs_ep[at_end])})
        if not items:
            logger.error("no intervals matched --episodes %s", args.episodes)
            return 1
        logger.info("drawing %d intervals (episodes %s)", len(items), args.episodes)
        render_sheets(items, episode, frame, probs, labels, args.dataset, args.cam,
                      args.out_dir, args.per_sheet, args.context)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
