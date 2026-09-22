"""Shared runtime core for RL-Token-Pi05 online RL (used by both entry points).

Byte-identical components extracted from scripts/train_rlt_stage2_pi05.py and
scripts/eval_rlt_pi05_so101.py, kept here as the single source of truth:

- RerunVisualizationManager          background Rerun visualization thread
- TransitionSource/Transition/...    boundary snapshots and transition building
- Pi05ActionCache/LeaderSample/...   action cache and leader sampling
- ExecutionWindow/BoundaryContext/.. execution windows and takeover state
- quantile_normalize/_load_json/...  stats loading and quantile normalization
- parse_camera_map/parse_max_relative_target  config parsing

Any change here affects both entry scripts at once. If log attribution ever
needs to differ, pass a caller-provided logger instead of the module-level one.
"""
from __future__ import annotations

import enum
import json
import logging
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# SO-101 order used by the legacy evaluator and by the training checkpoints.
# Keep this explicit because LeRobot feature dictionaries are not an ordering
# contract: a robot/dataset can legally expose the same names in another order.
SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _canonical_joint_name(name: str) -> str:
    """Normalize LeRobot action feature names to bare SO-101 joint names."""
    value = str(name).strip()
    if value.startswith("action."):
        value = value[len("action."):]
    if value.endswith(".pos"):
        value = value[:-4]
    return value


def action_order_indices(source_names, target_names) -> tuple[int, ...]:
    """Return indices that reorder policy outputs into robot feature order.

    ``source_names`` describes the order used by the trained policy, while
    ``target_names`` is the order expected by the robot/dataset boundary.
    Duplicate, missing, or unknown joints fail closed before any hardware move.
    """
    source = tuple(_canonical_joint_name(name) for name in source_names)
    target = tuple(_canonical_joint_name(name) for name in target_names)
    allowed = set(SO101_JOINT_NAMES)
    if len(source) != len(target):
        raise ValueError(f"Action order length mismatch: source={len(source)} target={len(target)}")
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError(f"Action order contains duplicate joints: source={source}, target={target}")
    if set(source) != allowed:
        raise ValueError(f"Policy action order must contain SO-101 joints {SO101_JOINT_NAMES}, got {source}")
    if set(target) != allowed:
        raise ValueError(f"Robot action order must contain SO-101 joints {SO101_JOINT_NAMES}, got {target}")
    positions = {name: index for index, name in enumerate(source)}
    return tuple(positions[name] for name in target)


def reorder_action_tensor(action: torch.Tensor, *, source_names, target_names) -> torch.Tensor:
    """Reorder the final action dimension without changing batch/chunk axes."""
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"action must be a torch.Tensor, got {type(action).__name__}")
    indices = action_order_indices(source_names, target_names)
    if action.shape[-1] != len(indices):
        raise ValueError(
            f"Action tensor last dimension {action.shape[-1]} does not match action order {len(indices)}"
        )
    index = torch.tensor(indices, dtype=torch.long, device=action.device)
    return action.index_select(-1, index)

# ── Replay collection-phase ids (shared by Transition machinery) ──────────
COLLECTION_PHASE_UNKNOWN = 0
COLLECTION_PHASE_WARMUP = 1
COLLECTION_PHASE_ONLINE = 2


class RerunVisualizationManager:
    """Best-effort, non-blocking Rerun logging for live robot data."""

    def __init__(self, log_data, compress_images: bool, fps: float):
        self._log_data = log_data
        self._compress_images = compress_images
        self._period_s = 1.0 / fps
        self._condition = threading.Condition()
        self._observation: dict | None = None
        self._action: dict | None = None
        self._running = False
        self._disabled = False
        self._thread: threading.Thread | None = None

    @staticmethod
    def _copy_values(values: dict) -> dict:
        copied = {}
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            elif isinstance(value, torch.Tensor):
                copied[key] = value.detach().cpu().numpy().copy()
            elif isinstance(value, (int, float, np.integer, np.floating)):
                copied[key] = value
        return copied

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="rerun-visualization", daemon=True)
        self._thread.start()

    def publish_observation(self, observation: dict) -> None:
        if self._disabled:
            return
        with self._condition:
            self._observation = self._copy_values(observation)
            self._condition.notify()

    def publish_action(self, action: dict) -> None:
        if self._disabled:
            return
        with self._condition:
            self._action = self._copy_values(action)
            self._condition.notify()

    def _run(self) -> None:
        next_log_time = 0.0
        while self._running:
            with self._condition:
                while self._running and self._observation is None and self._action is None:
                    self._condition.wait(timeout=0.1)
                if not self._running:
                    return
                wait_s = next_log_time - time.monotonic()
                if wait_s > 0:
                    self._condition.wait(timeout=wait_s)
                    continue
                observation, action = self._observation, self._action
                self._observation, self._action = None, None
            try:
                self._log_data(
                    observation=observation,
                    action=action,
                    compress_images=self._compress_images,
                )
                next_log_time = time.monotonic() + self._period_s
            except Exception as exc:
                self._disabled = True
                logger.warning("Rerun visualization disabled after logging failure: %s", exc)
                return

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=0.5)


class TransitionSource(enum.IntEnum):
    BASE = 0
    RL = 1
    HUMAN = 2
    MIXED = 3


def collection_phase_to_id(phase: str) -> int:
    name = str(phase).split(":", 1)[0].lower()
    if name == "warmup":
        return COLLECTION_PHASE_WARMUP
    if name == "online":
        return COLLECTION_PHASE_ONLINE
    return COLLECTION_PHASE_UNKNOWN


def _ensure_array(value, *, dtype=None) -> np.ndarray:
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


@dataclass
class Transition:
    z_rl: np.ndarray           # (rlt_hidden_dim,)
    proprio: np.ndarray        # (state_dim,)
    ref_chunk: np.ndarray      # (n_action_steps_rl, action_dim)
    action_chunk: np.ndarray   # (n_action_steps_rl, action_dim)
    rewards: np.ndarray        # (n_action_steps_rl,)
    done: bool
    next_z_rl: np.ndarray
    next_proprio: np.ndarray
    next_ref_chunk: np.ndarray
    source: int
    source_chunk: np.ndarray   # uint8 (n_action_steps_rl,)
    valid_mask: np.ndarray     # bool (n_action_steps_rl,)
    executed_steps: int
    collection_phase: str
    success: int = 0
    intervention_flag: bool = False
    episode_id: int = 0
    step_id: int = 0

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {
            "z_rl": _ensure_array(self.z_rl, dtype=np.float16),
            "proprio": _ensure_array(self.proprio, dtype=np.float32),
            "ref_chunk": _ensure_array(self.ref_chunk, dtype=np.float16),
            "action_chunk": _ensure_array(self.action_chunk, dtype=np.float16),
            "rewards": _ensure_array(self.rewards, dtype=np.float32),
            "done": _ensure_array(self.done, dtype=np.bool_),
            "next_z_rl": _ensure_array(self.next_z_rl, dtype=np.float16),
            "next_proprio": _ensure_array(self.next_proprio, dtype=np.float32),
            "next_ref_chunk": _ensure_array(self.next_ref_chunk, dtype=np.float16),
            "source": _ensure_array(self.source, dtype=np.uint8),
            "source_chunk": _ensure_array(self.source_chunk, dtype=np.uint8),
            "valid_mask": _ensure_array(self.valid_mask, dtype=np.bool_),
            "executed_steps": _ensure_array(self.executed_steps, dtype=np.int16),
            "collection_phase_id": _ensure_array(
                collection_phase_to_id(self.collection_phase), dtype=np.uint8,
            ),
            "success": _ensure_array(self.success, dtype=np.int8),
            "intervention_flag": _ensure_array(self.intervention_flag, dtype=np.bool_),
            "episode_id": _ensure_array(self.episode_id, dtype=np.int32),
            "step_id": _ensure_array(self.step_id, dtype=np.int32),
        }


@dataclass
class Pi05ActionCache:
    """Full Pi0.5 raw chunks plus the model-space actions used by RLT."""

    raw_actions: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    z_rl: torch.Tensor | None = None
    next_index: int = 0

    def exhausted(self) -> bool:
        return self.raw_actions is None or self.next_index >= self.raw_actions.shape[1]

    def remaining_raw_actions(self) -> torch.Tensor | None:
        if self.exhausted() or self.raw_actions is None:
            return None
        return self.raw_actions[:, self.next_index:].clone()

    def refresh(self, raw_actions: torch.Tensor, z_rl: torch.Tensor, action_dim: int) -> None:
        self.raw_actions = raw_actions
        self.actions = raw_actions[:, :, :action_dim]
        self.z_rl = z_rl
        self.next_index = 0

    def clear(self) -> None:
        self.raw_actions = None
        self.actions = None
        self.z_rl = None
        self.next_index = 0

    def take(self, count: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        if self.exhausted() or self.actions is None or self.z_rl is None:
            raise RuntimeError("PI0.5 action cache is empty.")
        if count < 1 or self.next_index + count > self.actions.shape[1]:
            raise ValueError("Requested Pi0.5 action horizon is not available in the cache.")
        action_index = self.next_index
        result = self.actions[:, action_index : action_index + count]
        self.next_index += count
        return result, action_index, self.z_rl


@dataclass(frozen=True)
class LeaderSample:
    sequence: int
    timestamp: float
    action: dict[str, float]
    tensor: torch.Tensor
    healthy: bool = True

    def is_fresh(self, max_age_s: float, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        return self.healthy and current - self.timestamp <= max_age_s


@dataclass
class BoundaryContext:
    z_rl: torch.Tensor
    proprio: torch.Tensor
    ref_chunk: torch.Tensor
    collection_phase: str
    episode_id: int
    step_id: int


@dataclass
class ExecutionWindow:
    """Actions actually accepted by hardware under one observation boundary."""

    boundary: BoundaryContext
    actions: list[torch.Tensor] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)
    intervention_flags: list[bool] = field(default_factory=list)

    def append(self, actual_action: torch.Tensor, source: int, intervention: bool = False) -> None:
        if actual_action.ndim != 1:
            raise ValueError(f"Executed action must be rank 1, got {tuple(actual_action.shape)}")
        self.actions.append(actual_action.detach().clone())
        self.sources.append(int(source))
        self.intervention_flags.append(bool(intervention))

    @property
    def executed_steps(self) -> int:
        return len(self.actions)


def _resolve_chunk_source(sources: list[int], intervention_flags: list[bool]) -> tuple[int, bool]:
    intervention = any(intervention_flags)
    source_set = set(sources)
    has_human = int(TransitionSource.HUMAN) in source_set
    has_policy = any(s in source_set for s in (
        int(TransitionSource.BASE), int(TransitionSource.RL), int(TransitionSource.MIXED),
    ))
    if int(TransitionSource.MIXED) in source_set or (has_human and has_policy):
        return int(TransitionSource.MIXED), intervention
    if has_human or intervention:
        return int(TransitionSource.HUMAN), intervention
    return int(sources[0]), intervention


def quantile_normalize(tensor: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    denom = q99 - q01
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return 2.0 * (tensor - q01) / denom - 1.0


def _load_json(path: Path) -> dict:
    with open(path) as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _validate_quantiles(stats: dict) -> dict[str, list[float]]:
    result = {}
    for key in ("observation.state", "action"):
        if key not in stats or not isinstance(stats[key], dict):
            raise ValueError(f"Normalization stats missing {key!r}")
        q01 = np.asarray(stats[key].get("q01"), dtype=np.float32)
        q99 = np.asarray(stats[key].get("q99"), dtype=np.float32)
        if q01.shape != (6,) or q99.shape != (6,):
            raise ValueError(f"{key} quantiles must each be exactly 6D")
        if not np.isfinite(q01).all() or not np.isfinite(q99).all():
            raise ValueError(f"{key} quantiles must be finite")
        if not np.all(q99 > q01):
            raise ValueError(f"{key} requires q99 > q01 elementwise")
        result[key] = {"q01": q01.tolist(), "q99": q99.tolist()}
    return result


def load_normalization_stats(pi05_path: Path, explicit_stats_path: str | None) -> tuple[dict, str]:
    if explicit_stats_path:
        path = Path(explicit_stats_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit stats path does not exist: {path}")
        raw = _load_json(path)
        if raw.get("use_relative_actions") is True:
            raise ValueError("Stage 2 requires absolute 6D actions; use_relative_actions must be false")
        return _validate_quantiles(raw), str(path.resolve())

    processor_path = pi05_path / "policy_preprocessor.json"
    if not processor_path.is_file():
        raise FileNotFoundError("Checkpoint processor state is unavailable; pass --stats_path explicitly")
    processor = _load_json(processor_path)
    normalizers = [
        step for step in processor.get("steps", [])
        if step.get("registry_name") == "normalizer_processor"
    ]
    if len(normalizers) != 1:
        raise ValueError("Checkpoint must declare exactly one normalizer_processor")
    normalizer_config = normalizers[0].get("config", {})
    norm_map = normalizer_config.get("norm_map", {})
    if norm_map.get("STATE") != "QUANTILES" or norm_map.get("ACTION") != "QUANTILES":
        raise ValueError("Checkpoint STATE and ACTION normalization must both be QUANTILES")
    if normalizer_config.get("use_relative_actions") is True:
        raise ValueError("Checkpoint use_relative_actions must be false")

    candidates = sorted(pi05_path.glob("*normalizer*.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            "Checkpoint declares normalization but has no normalizer safetensors; pass --stats_path explicitly"
        )
    from safetensors.torch import load_file
    tensors = {}
    for candidate in candidates:
        tensors.update(load_file(str(candidate)))
    aliases = {
        "observation.state": ("observation.state", "state"),
        "action": ("action",),
    }
    extracted = {}
    for feature, prefixes in aliases.items():
        for prefix in prefixes:
            q01 = next((v for k, v in tensors.items() if k in (f"{prefix}.q01", f"{prefix}/q01")), None)
            q99 = next((v for k, v in tensors.items() if k in (f"{prefix}.q99", f"{prefix}/q99")), None)
            if q01 is not None and q99 is not None:
                extracted[feature] = {"q01": q01.cpu().tolist(), "q99": q99.cpu().tolist()}
                break
    return _validate_quantiles(extracted), ",".join(str(p.resolve()) for p in candidates)


def parse_camera_map(
    camera_map_json: str,
    required_features: list[str],
    configured_cameras: set[str],
    pad_missing: bool = False,
) -> dict[str, str]:
    if not camera_map_json:
        raise ValueError(
            "--camera_map is required. Map every physical camera to every image feature required by the checkpoint."
        )
    try:
        mapping = json.loads(camera_map_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--camera_map must be valid JSON: {exc}") from exc
    if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        raise ValueError("--camera_map must be a JSON object of physical camera names to checkpoint feature names.")
    physical_names = set(mapping)
    targets = list(mapping.values())
    if physical_names != configured_cameras:
        raise ValueError(
            f"Camera map keys {sorted(physical_names)} must exactly match configured cameras {sorted(configured_cameras)}."
        )
    if len(set(targets)) != len(targets):
        raise ValueError("Each physical camera must map to a distinct checkpoint image feature.")
    unknown = sorted(set(targets) - set(required_features))
    if unknown:
        raise ValueError(f"Camera map targets unknown checkpoint features: {unknown}")
    mapped_set = set(targets)
    missing_set = set(required_features) - mapped_set
    if missing_set:
        if not pad_missing:
            missing = sorted(missing_set)
            raise ValueError(
                f"Camera mapping does not satisfy checkpoint contract. Missing={missing}, "
                f"checkpoint_features={required_features}. "
                f"Supply the missing cameras or pass --pad_missing_cameras for test-only runs."
            )
        logger.warning(
            "TEST-ONLY MODE: unmapped camera features %s will be filled with zero-pixel tensors.",
            sorted(missing_set),
        )
    return mapping


def parse_max_relative_target(value: str | None) -> float | dict[str, float] | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError("--max_relative_target must be a positive scalar or a JSON object.") from exc
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        if not np.isfinite(parsed) or parsed <= 0:
            raise ValueError("--max_relative_target scalar must be finite and positive.")
        return float(parsed)
    if isinstance(parsed, dict) and parsed and all(
        isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
        and np.isfinite(v) and v > 0 for k, v in parsed.items()
    ):
        return {key: float(limit) for key, limit in parsed.items()}
    raise ValueError("--max_relative_target JSON must be a non-empty map of motor names to positive finite limits.")


__all__ = [
    "SO101_JOINT_NAMES",
    "action_order_indices",
    "reorder_action_tensor",
    "RerunVisualizationManager",
    "TransitionSource",
    "collection_phase_to_id",
    "_ensure_array",
    "Transition",
    "Pi05ActionCache",
    "LeaderSample",
    "BoundaryContext",
    "ExecutionWindow",
    "_resolve_chunk_source",
    "quantile_normalize",
    "_load_json",
    "_validate_quantiles",
    "load_normalization_stats",
    "parse_camera_map",
    "parse_max_relative_target",
]


def write_warmup_interval_records(
    out_file,
    ep: int,
    frames_in_interval: list,
    frame_data: dict,
    *,
    horizon: int,
    stride: int,
    action_dim: int,
) -> int:
    """Build stride-2 replay windows inside one annotated interval and write warmup
    journal records (pickle stream, schema identical to Stage-2 online transitions
    with ``collection_phase_id=WARMUP``). Shared by ``extract_annotated_warmup.py``
    and ``extract_warmup_from_cache.py``.

    ``frame_data`` maps ``(episode, frame) -> (z_rl fp16, proprio fp32, action fp32)``.
    Returns the number of records written.
    """
    f = out_file  # verbatim body below references ``f``
    n_records = 0
    for pos in range(0, len(frames_in_interval), stride):
        window_frames = frames_in_interval[pos:pos + horizon]
        if len(window_frames) < 1:
            break
        t0f = window_frames[0]
        t1f = frames_in_interval[min(pos + horizon, len(frames_in_interval) - 1)]
        z0, p0, _ = frame_data[(ep, t0f)]
        z1, p1, _ = frame_data[(ep, t1f)]

        action_chunk = np.zeros((horizon, action_dim), dtype=np.float32)
        next_ref_chunk = np.zeros((horizon, action_dim), dtype=np.float32)
        valid_mask = np.zeros(horizon, dtype=np.bool_)
        executed = 0
        for i, fr in enumerate(window_frames):
            action_chunk[i] = frame_data[(ep, fr)][2]
            valid_mask[i] = True
            executed += 1
        # next window's demo actions as next_ref_chunk (may be all-zero padding)
        next_frames = frames_in_interval[pos + horizon:pos + 2 * horizon]
        for i, fr in enumerate(next_frames):
            next_ref_chunk[i] = frame_data[(ep, fr)][2]

        record = {
            "z_rl": z0.astype(np.float16),
            "proprio": p0.astype(np.float32),
            "ref_chunk": action_chunk.astype(np.float16),
            "action_chunk": action_chunk.astype(np.float16),
            "rewards": np.zeros(horizon, dtype=np.float32),
            "done": np.asarray(False, dtype=np.bool_),
            "next_z_rl": z1.astype(np.float16),
            "next_proprio": p1.astype(np.float32),
            "next_ref_chunk": next_ref_chunk.astype(np.float16),
            "source": np.asarray(TransitionSource.BASE, dtype=np.uint8),
            "source_chunk": np.full(horizon, TransitionSource.BASE, dtype=np.uint8),
            "valid_mask": valid_mask,
            "executed_steps": np.asarray(executed, dtype=np.int16),
            "collection_phase_id": np.asarray(COLLECTION_PHASE_WARMUP, dtype=np.uint8),
            "success": np.asarray(0, dtype=np.int8),
            "intervention_flag": np.asarray(False, dtype=np.bool_),
            "episode_id": np.asarray(ep, dtype=np.int32),
            "step_id": np.asarray(t0f, dtype=np.int32),
        }
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_records += 1
    return n_records
