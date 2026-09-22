#!/usr/bin/env python3
"""Interactive critical-phase annotation tool for a LeRobot teleop dataset.

Decodes the dataset's videos with PyAV (handles AV1/libsvtav1, which OpenCV's
VideoCapture often cannot). Two annotation modes, auto-selected:

1. GUI mode (default when a display is present; tkinter, no OpenCV GUI needed):
   plays the episode's top-camera video and marks intervals with the keyboard:
       s        mark interval START at the current frame
       e        mark interval END at the current frame (appends [start, end])
       u        undo the last interval
       space    pause / resume
       left/right   step one frame (auto-pause)
       Shift+left/right   jump ±30 frames
       n        skip to the next episode
       q        save annotations and quit
2. Contact-sheet mode (automatic fallback for headless/remote sessions):
   renders each episode into a montage PNG (thumbnail per frame with its frame
   number) into --sheet_dir, then collects intervals interactively on the
   terminal:
       100 300   add interval [100, 300]
       undo      remove the last interval
       skip      skip this episode (nothing saved)
       (blank)   finish this episode, continue with the next
       quit      save annotations and exit

Annotations are saved as JSON with inclusive, episode-local frame indices
(LeRobot frame_index semantics, 0-based per episode). The file is incremental:
already-annotated episodes are skipped, new intervals are appended.

Usage:
    python scripts/annotate_dataset.py \
        --dataset_path /path/to/my_dataset \
        --annotations_out annotations.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--annotations_out", type=str, required=True,
                        help="JSON output path (incremental: existing intervals are kept).")
    parser.add_argument("--camera", type=str, default="observation.images.top",
                        help="Camera used for contact-sheet mode (GUI mode shows all cameras side by side).")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Dataset frame rate (used to compute episode offsets in shared videos).")
    parser.add_argument("--force_sheet", action="store_true",
                        help="Force contact-sheet mode even if a display is available.")
    parser.add_argument("--sheet_dir", type=str, default="annotation_sheets",
                        help="Directory for contact-sheet PNGs (headless mode).")
    parser.add_argument("--sheet_cols", type=int, default=12,
                        help="Thumbnails per row in contact sheets.")
    parser.add_argument("--thumb_width", type=int, default=160,
                        help="Thumbnail width in contact sheets.")
    return parser.parse_args()


def load_existing(path: Path) -> dict[int, list[tuple[int, int]]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    annotations: dict[int, list[tuple[int, int]]] = {}
    for item in data.get("annotations", []):
        annotations.setdefault(int(item["episode"]), []).append(
            (int(item["start"]), int(item["end"]))
        )
    return annotations


def save_annotations(path: Path, annotations: dict[int, list[tuple[int, int]]]) -> None:
    payload = {
        "annotations": [
            {"episode": episode, "start": start, "end": end}
            for episode in sorted(annotations)
            for start, end in annotations[episode]
        ]
    }
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.flush()
    tmp.replace(path)
    logger.info("Saved %d intervals across %d episodes to %s",
                len(payload["annotations"]), len(annotations), path)


def episode_video_paths(
    dataset_path: Path,
    fps: float,
    cameras: tuple[str, ...] = ("observation.images.top", "observation.images.wrist"),
) -> list[tuple[int, dict[str, tuple[Path, int]], int]]:
    """Return [(episode_index, {camera: (video_path, start_frame)}, length)].

    Newer LeRobot metadata (0.5.x) stores per-episode video references as
    ``videos/<feature>/chunk_index`` + ``file_index``; several episodes share
    one video file, so the episode's position inside it is
    ``from_timestamp * fps`` (a continuous recording). The timestamps are
    RELATIVE TO EACH FILE (every file restarts at 0), and cameras can roll to
    a new file at different times (e.g. top every 688 s but wrist every 344 s),
    so each camera gets its OWN start_frame. Missing cameras (no parquet
    columns / missing file) are omitted from the dict.
    """
    from datasets import Dataset

    episodes_dir = dataset_path / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episodes parquet under {episodes_dir}")
    episodes = Dataset.from_parquet([str(path) for path in parquet_files])

    videos_dir = dataset_path / "videos"
    result: list[tuple[int, dict[str, tuple[Path, int]], int]] = []
    for row in episodes:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        paths: dict[str, tuple[Path, int]] = {}
        for cam in cameras:
            try:
                chunk_index = int(row[f"videos/{cam}/chunk_index"])
                file_index = int(row[f"videos/{cam}/file_index"])
                from_timestamp = float(row[f"videos/{cam}/from_timestamp"])
            except (KeyError, TypeError):
                continue
            video_path = (
                videos_dir / cam / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            )
            if video_path.exists():
                start_frame = int(round(from_timestamp * fps))
                paths[cam] = (video_path, start_frame)
        if not paths:
            raise FileNotFoundError(
                f"Episode {episode_index}: no video files found for cameras {cameras}"
            )
        result.append((episode_index, paths, length))
    return sorted(result, key=lambda item: item[0])



def _seek_to_frame(container, stream, target_frame: int, fps: float) -> None:
    """Seek to the key frame at or before ``target_frame`` (video-global index).

    PyAV's ``Container.seek`` interprets the offset in ``stream.time_base``
    units when a stream is given (NOT microseconds), so convert frame -> ticks.
    """
    offset = int(max(0, target_frame) * (1.0 / fps) / float(stream.time_base))
    container.seek(offset, stream=stream)


def iter_episode_frames(video_path: Path, start_frame: int, length: int, fps: float):
    """Yield (episode_local_frame, bgr24 ndarray) for an episode's frames (PyAV)."""
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    # Seek a little before the episode start (keyframe-aligned), then decode
    # forward, skipping frames until the episode's first frame.
    _seek_to_frame(container, stream, max(0, start_frame - 10), fps)
    try:
        for frame in container.decode(stream):
            if frame.pts is not None:
                seconds = float(frame.pts * stream.time_base)
                frame_no = int(round(seconds * fps))
            else:
                frame_no = -1  # unknown; caller falls back to counting
            if frame_no >= start_frame + length:
                break
            if frame_no >= start_frame:
                yield frame_no - start_frame, frame.to_ndarray(format="bgr24")
    finally:
        container.close()


class EpisodeFrameReader:
    """Random-access frame reader for one episode's video segment (PyAV).

    Sequential playback decodes forward without reopening; jumps (backward or
    far forward) reopen the container and seek, so pausing and ±30-frame
    stepping stay fast.
    """

    def __init__(self, video_path: Path, start_frame: int, length: int, fps: float):
        self._video_path = str(video_path)
        self._start_frame = start_frame
        self._length = length
        self._fps = fps
        self._container = None
        self._stream = None
        self._decoder = None
        self._current_frame = -1  # last frame index handed out (video-global)
        self._last_image = None   # cached ndarray of _current_frame (for repeated reads)

    def _open(self, episode_frame: int) -> None:
        import av

        if self._container is not None:
            self._container.close()
        self._container = av.open(self._video_path)
        self._stream = self._container.streams.video[0]
        _seek_to_frame(
            self._container, self._stream, max(0, self._start_frame + episode_frame - 10), self._fps,
        )
        self._decoder = self._container.decode(self._stream)
        self._current_frame = self._start_frame + episode_frame - 11
        self._last_image = None

    def read(self, episode_frame: int):
        """Return bgr24 ndarray for the episode-local frame, or None past the end."""
        target = self._start_frame + episode_frame
        if target < 0 or target >= self._start_frame + self._length:
            return None
        if (
            self._container is None
            or target < self._current_frame
            or target - self._current_frame > 90
        ):
            self._open(episode_frame)
        frame = None
        while self._current_frame < target:
            try:
                frame = next(self._decoder)
            except StopIteration:
                return None
            if frame.pts is not None:
                frame_no = int(round(float(frame.pts * self._stream.time_base) * self._fps))
            else:
                frame_no = self._current_frame + 1
            self._current_frame = frame_no
        if self._current_frame == target:
            if frame is None:
                # Repeated read of the current frame (GUI tick + keypress both
                # refresh the same frame): the decoder already handed it out on
                # the first call, so serve the cached image.
                return self._last_image
            self._last_image = frame.to_ndarray(format="bgr24")
            return self._last_image
        return None

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None


def gui_available() -> bool:
    """Tkinter-based GUI works whenever a display is present (no OpenCV GUI needed)."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


class _TkAnnotator:
    """tkinter playback window showing every camera side by side.

    Keys: space pause, s/e intervals, u undo, ←/→ step frames,
    Shift+←/→ jump ±30, n next episode, q quit.
    """

    DISPLAY_W = 480   # displayed width per camera (source 640x480)
    DISPLAY_H = 360

    def __init__(self, readers: dict[str, EpisodeFrameReader], episode: int, length: int):
        import tkinter as tk

        from PIL import Image, ImageTk

        self.readers = readers
        self.cameras = list(readers.keys())
        self.episode = episode
        self.length = length
        self.frame = 0
        self.paused = False
        self.start_mark: int | None = None
        self.intervals: list[tuple[int, int]] = []
        self.quit_all = False
        self._photos: dict[str, ImageTk.PhotoImage] = {}
        self._Image = Image
        self._ImageTk = ImageTk

        n_cameras = max(1, len(self.cameras))
        width = n_cameras * self.DISPLAY_W
        height = self.DISPLAY_H + 24
        self.root = tk.Tk()
        self.root.title(
            f"annotate episode {episode} | cameras: {', '.join(self.cameras)} "
            f"(s=start e=end u=undo space=pause ←/→=frame Shift+←/→=±30 n=next q=quit)"
        )
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="black")
        self.canvas.pack()
        self.status = tk.Label(self.root, text="", font=("TkDefaultFont", 12), anchor="w")
        self.status.pack(fill="x")
        self.root.bind("<KeyPress>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_frame()
        self.root.after(30, self._tick)

    def _show_frame(self) -> None:
        self.canvas.delete("all")
        for index, camera in enumerate(self.cameras):
            x0 = index * self.DISPLAY_W
            reader = self.readers[camera]
            bgr = reader.read(self.frame)
            if bgr is None:
                self.canvas.create_text(
                    x0 + self.DISPLAY_W // 2, self.DISPLAY_H // 2,
                    text=f"{camera}: no frame", fill="#888888",
                )
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = self._Image.fromarray(rgb).resize(
                    (self.DISPLAY_W, self.DISPLAY_H), self._Image.BILINEAR,
                )
                self._photos[camera] = self._ImageTk.PhotoImage(image)
                self.canvas.create_image(x0, 0, anchor="nw", image=self._photos[camera])
                for start, end in self.intervals:
                    if start <= self.frame <= end:
                        color = "#ff6666" if self.start_mark is None else "#ffcc66"
                        self.canvas.create_rectangle(
                            x0 + 2, 2, x0 + self.DISPLAY_W - 2, self.DISPLAY_H - 2,
                            outline=color, width=6,
                        )
                        break
                if self.start_mark is not None and self.frame >= self.start_mark:
                    self.canvas.create_rectangle(
                        x0 + 2, 2, x0 + self.DISPLAY_W - 2, self.DISPLAY_H - 2,
                        outline="#66ff66", width=3,
                    )
            self.canvas.create_text(
                x0 + 8, self.DISPLAY_H + 4, anchor="nw",
                text=camera.replace("observation.images.", ""),
                fill="#ffffff", font=("TkDefaultFont", 10),
            )
        status = "PAUSED" if self.paused else "PLAYING"
        self.status.config(
            text=f"ep {self.episode}  frame {self.frame}/{self.length}  [{status}]  "
                 f"intervals: {self.intervals}  (n = next episode, q = quit, "
                 f"auto-advances at the last frame)"
        )

    def _tick(self) -> None:
        if not self.paused and self.frame + 1 < self.length:
            self.frame += 1
            self._show_frame()
            self.root.after(30, self._tick)
        elif not self.paused:
            # Playback reached the last frame: auto-save and move to the next
            # episode (the main loop persists any marked intervals).
            print(f"  [ep {self.episode}] end of episode reached; moving to next episode")
            self.root.destroy()
        else:
            self.root.after(30, self._tick)

    def _on_key(self, event) -> None:
        key = event.keysym.lower()
        if key in ("q", "escape"):
            self.quit_all = True
            self.root.destroy()
        elif key == "space":
            self.paused = not self.paused
            self._show_frame()
        elif key == "n":
            self.root.destroy()
        elif key == "s":
            self.start_mark = self.frame
            print(f"  [ep {self.episode}] interval START at frame {self.frame}")
            self._show_frame()
        elif key == "e":
            if self.start_mark is None:
                print(f"  [ep {self.episode}] press 's' before 'e'")
            else:
                end_frame = max(self.start_mark, self.frame)
                self.intervals.append((self.start_mark, end_frame))
                print(f"  [ep {self.episode}] interval END at frame {self.frame} -> "
                      f"[{self.start_mark}, {end_frame}]")
                self.start_mark = None
            self._show_frame()
        elif key == "u":
            if self.intervals:
                removed = self.intervals.pop()
                print(f"  [ep {self.episode}] removed interval {removed}")
            else:
                print(f"  [ep {self.episode}] nothing to undo")
            self._show_frame()
        elif key in ("left", "right"):
            # Arrow keys step one frame (auto-pause); Shift+arrow jumps ±30.
            step = 30 if (event.state & 0x0001) else 1
            delta = step if key == "right" else -step
            self.paused = True
            self.frame = max(0, min(self.frame + delta, self.length - 1))
            self._show_frame()

    def _on_close(self) -> None:
        self.quit_all = True
        self.root.destroy()

    def run(self) -> tuple[list[tuple[int, int]], bool]:
        self.root.mainloop()
        return self.intervals, self.quit_all


def annotate_episode_gui(
    episode: int,
    video_infos: dict[str, tuple[Path, int]],
    length: int,
    fps: float,
) -> tuple[list[tuple[int, int]], bool]:
    """Play the episode in a tkinter window (all cameras side by side).

    Each camera has its own start_frame inside its own video file, because
    cameras can roll to new files at different times.
    """
    readers = {
        camera: EpisodeFrameReader(path, start_frame, length, fps)
        for camera, (path, start_frame) in video_infos.items()
    }
    print(f"\n=== Episode {episode} | {length} frames | cameras: {', '.join(video_infos)} ===")
    print("Keys: s=interval START  e=interval END  u=undo  space=pause  "
          "←/→=1 frame  Shift+←/→=±30  n=next  q=quit")
    try:
        annotator = _TkAnnotator(readers, episode, length)
        return annotator.run()
    finally:
        for reader in readers.values():
            reader.close()


def render_contact_sheet(
    video_path: Path,
    start_frame: int,
    length: int,
    fps: float,
    *,
    cols: int,
    thumb_width: int,
) -> np.ndarray:
    """Render one episode as a montage of thumbnails labelled with frame numbers."""
    thumb_height = int(thumb_width * 3 / 4)
    label_height = 22
    cell_w, cell_h = thumb_width, thumb_height + label_height
    rows = (length + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)
    for episode_frame, image in iter_episode_frames(video_path, start_frame, length, fps):
        col = episode_frame % cols
        row = episode_frame // cols
        thumb = cv2.resize(image, (thumb_width, thumb_height))
        y0 = row * cell_h + label_height
        x0 = col * cell_w
        sheet[y0:y0 + thumb_height, x0:x0 + thumb_width] = thumb
        cv2.putText(sheet, f"{episode_frame}", (x0 + 2, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return sheet


def annotate_episode_sheet(
    episode: int,
    video_path: Path,
    start_frame: int,
    length: int,
    fps: float,
    *,
    sheet_dir: Path,
    cols: int,
    thumb_width: int,
) -> tuple[list[tuple[int, int]], bool]:
    """Render a contact sheet, then collect intervals from terminal input.

    Returns (intervals, quit_all).
    """
    sheet_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = sheet_dir / f"episode_{episode:04d}.png"
    sheet = render_contact_sheet(video_path, start_frame, length, fps,
                                 cols=cols, thumb_width=thumb_width)
    cv2.imwrite(str(sheet_path), sheet)
    print(f"\n=== Episode {episode} | {length} frames | sheet: {sheet_path} ===")
    print("Open the sheet image and enter critical-phase intervals.")
    print("  '<start> <end>'  add interval | 'undo' | 'skip' | 'quit' | blank = next episode")
    intervals: list[tuple[int, int]] = []
    while True:
        try:
            line = input(f"  ep {episode} interval: ").strip().lower()
        except EOFError:
            return intervals, True
        if not line:
            return intervals, False
        if line in ("q", "quit"):
            return intervals, True
        if line == "skip":
            return intervals, False
        if line == "undo":
            if intervals:
                removed = intervals.pop()
                print(f"  removed {removed}")
            else:
                print("  nothing to undo")
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                print("  invalid numbers")
                continue
            if not (0 <= start <= end < length):
                print(f"  out of range [0, {length - 1}]")
                continue
            intervals.append((start, end))
            print(f"  added [{start}, {end}] (total {len(intervals)})")
        else:
            print("  expected: '<start> <end>'")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    out_path = Path(args.annotations_out)
    annotations = load_existing(out_path)
    if annotations:
        logger.info("Loaded %d previously annotated episodes; they will be skipped.", len(annotations))

    episodes = episode_video_paths(dataset_path, args.fps)
    logger.info("Found %d episodes in %s", len(episodes), dataset_path)

    use_gui = not args.force_sheet and gui_available()
    logger.info("Annotation mode: %s", "GUI playback (tkinter)" if use_gui else "contact-sheet (headless)")

    try:
        for episode, video_infos, length in episodes:
            if episode in annotations:
                logger.info("Skipping episode %d (already annotated: %s)", episode, annotations[episode])
                continue
            if use_gui:
                intervals, should_quit = annotate_episode_gui(
                    episode, video_infos, length, args.fps,
                )
            else:
                sheet_info = video_infos.get(args.camera)
                if sheet_info is None:
                    logger.warning(
                        "Episode %d: camera %s unavailable (have %s); skipping.",
                        episode, args.camera, sorted(video_infos),
                    )
                    continue
                sheet_video, sheet_start = sheet_info
                intervals, should_quit = annotate_episode_sheet(
                    episode, sheet_video, sheet_start, length, args.fps,
                    sheet_dir=Path(args.sheet_dir),
                    cols=args.sheet_cols,
                    thumb_width=args.thumb_width,
                )
            if intervals:
                annotations[episode] = intervals
                save_annotations(out_path, annotations)
            else:
                logger.info("Episode %d: no intervals marked, not saved.", episode)
            if should_quit:
                print("Quit requested.")
                break
    finally:
        if annotations:
            save_annotations(out_path, annotations)
    print(f"\nDone. Annotations: {out_path}")


if __name__ == "__main__":
    main()
