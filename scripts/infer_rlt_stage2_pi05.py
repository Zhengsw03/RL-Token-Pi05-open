#!/usr/bin/env python3
"""RLT Stage 2 deployment / inference script (SO-101 + π0.5, inference only).

Pipeline: frozen π0.5 VLA -> RLT encoder (z_rl) -> critical-phase classifier
          (automatic critical-phase detection) -> trained RLT Actor (takes over
          execution inside the critical phase) -> SO-101 follower.
          The leader arm is used for human intervention (HIL) and is neither
          mirrored nor driven by default.

Key behaviour:
  - No training: no replay buffer/journal, no critic/optimizer, no weight
    updates, no training checkpoints.
  - Keys match the training script exactly: a = Actor toggle, c = manual
    critical-phase toggle, e = emergency stop, f = finish the episode early,
    g = skip the episode, q = quit, and y/n/d/q scoring at each episode end.
  - Episodes restart at 1; dataset recording uses episode_index from 0.
  - --record_dataset turns rollouts into a standard LeRobotDataset (same format
    as teleoperation data, directly reusable for training).
  - Rerun visualization follows the same path as lerobot-rollout
    (log_rerun_data + compress + static).
  - Camera preflight at startup plus retries for transient observation stalls;
    the leader is not mirrored by default.

Usage:
    python scripts/infer_rlt_stage2_pi05.py --config configs/stage2_pi05.json

Hardware: local machine with SO-101 (follower + leader arms) + cameras.
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import signal
import sys
import traceback
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Allow importing local helper modules when running from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rlt_core import *  # noqa: F401,F403  (shared runtime core, see rlt_core.py)
SHUTDOWN = False
EMERGENCY_STOP = False
ACTOR_ENABLED = False  # Toggle with 'a' key during episode
SKIP_EPISODE = False  # Skip to next episode with 'g' key
FINISH_EPISODE = False  # Finish current episode early with 'f' key (still asks y/n for terminal reward)
CRITICAL_PHASE_ACTIVE = False  # Toggle with 'c' (on = RLT replay collection); resets each episode
CRITICAL_MANUAL_OVERRIDE_AT = 0.0  # wall time of the last manual 'c' press (auto-detector cooldown)
HIL_FORCE_RESUME = False  # Press 's' during HIL to exit human control and resume the policy immediately
HIL_TOGGLE_REQUESTED = False  # Press 'h' to toggle HIL mode (Evo-RL style explicit intervention switch)
POSITION_TOGGLE_REQUESTED = False  # Press 'p': pre-episode manual positioning toggle (NOT HIL, never enters replay). Sticky: a press during the previous episode tail is honored at the next positioning phase.
POSITIONING_STATE = "off"  # Main-thread broadcast for console feedback: "off" | "waiting" | "active"




class ConsoleInputManager:
    """Own stdin in one thread so safety controls and reward prompts cannot race.

    With --auto_critical enabled, 'a' must not require a prior 'c' press:
    CRITICAL_PHASE_ACTIVE mirrors the detector state and flips between True and
    False (and with --auto_critical_once, which by default fires once per
    episode, it stays False for a long time afterwards). An 'a' press landing in
    a False window was rejected with "press 'c' first", as observed in the log:
        15:15:41  Auto critical phase STOPPED
        15:15:45  Actor mode requires the critical phase: press 'c' first
        15:15:49  Actor mode requires the critical phase: press 'c' first
        15:15:53  Actor mode ON          (only succeeded after pressing 'c')
    In automatic mode the takeover timing is decided by the detector anyway, so
    'a' only enables the Actor; whether it actually takes over is still decided
    by CRITICAL_PHASE_ACTIVE at every control step.
    """

    def __init__(self, stream=None, *, auto_critical: bool = False):
        self._stream = stream if stream is not None else sys.stdin
        self._responses: deque[str] = deque()
        self._condition = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_toggle_time = 0.0
        # Whether auto critical detection is on; decides if 'a' still requires a prior 'c'
        self._auto_critical = bool(auto_critical)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=0.2)

    def _loop(self) -> None:
        global EMERGENCY_STOP, SHUTDOWN, ACTOR_ENABLED, SKIP_EPISODE, FINISH_EPISODE, CRITICAL_PHASE_ACTIVE, HIL_FORCE_RESUME, HIL_TOGGLE_REQUESTED, CRITICAL_MANUAL_OVERRIDE_AT, POSITION_TOGGLE_REQUESTED
        while self._running and not SHUTDOWN:
            try:
                line = self._stream.readline()
            except Exception:
                return
            if not line:
                time.sleep(0.05)
                continue
            value = line.strip().lower()
            if value.startswith("e"):
                EMERGENCY_STOP = True
                logger.warning("[KEY] EMERGENCY STOP requested (press 'e')")
                print("[KEY] EMERGENCY STOP", flush=True)
                with self._condition:
                    self._condition.notify_all()
                return
            if value.startswith("q"):
                SHUTDOWN = True
                logger.info("[KEY] Graceful quit requested (press 'q')")
                print("[KEY] Quit requested", flush=True)
                with self._condition:
                    self._condition.notify_all()
                return
            if value.startswith("a"):
                # Actor mode is gated on the critical phase: pressing 'a' only
                # turns the RL actor on while CRITICAL_PHASE_ACTIVE ('c' state);
                # pressing 'a' again turns it off. The state is sticky across
                # episodes — it is never reset automatically, only by another
                # explicit 'a' press.
                if ACTOR_ENABLED:
                    ACTOR_ENABLED = False
                    logger.info("[KEY] Actor mode OFF (press 'a' to re-enable)")
                    print("[KEY] Actor OFF", flush=True)
                elif CRITICAL_PHASE_ACTIVE:
                    ACTOR_ENABLED = True
                    logger.info("[KEY] Actor mode ON (critical phase active)")
                    print("[KEY] Actor ON", flush=True)
                elif self._auto_critical:
                    # Automatic mode does not require a prior 'c': the detector
                    # decides the takeover timing and 'a' only enables the Actor.
                    # Requiring 'c' would clash with the mirrored detector state
                    # (the flag is False after a detector STOPPED, so 'a' was
                    # wrongly rejected).
                    ACTOR_ENABLED = True
                    logger.info(
                        "[KEY] Actor mode ON (auto-critical: takes over once the detector enters the critical phase)"
                    )
                    print("[KEY] Actor ON (auto-critical: waiting for the critical phase)", flush=True)
                else:
                    logger.info("[KEY] Actor mode requires the critical phase: press 'c' first, then 'a'")
                    print("[KEY] Actor needs the critical phase (press 'c' first)", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("g"):
                SKIP_EPISODE = True
                logger.info("[KEY] Skip episode requested (press 'g')")
                print("[KEY] Skip episode", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("f"):
                FINISH_EPISODE = True
                logger.info("[KEY] Finish episode early requested (press 'f'); the round ends after the current chunk.")
                print("[KEY] Finish episode early", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("c"):
                # Toggle the critical phase: press once to START recording the
                # critical steps, press again to STOP recording (the rest of the
                # episode runs without entering replay).
                CRITICAL_PHASE_ACTIVE = not CRITICAL_PHASE_ACTIVE
                CRITICAL_MANUAL_OVERRIDE_AT = time.time()
                if CRITICAL_PHASE_ACTIVE:
                    logger.info("[KEY] Critical phase STARTED; RLT replay collection is active")
                    print("[KEY] Critical ON", flush=True)
                else:
                    logger.info("[KEY] Critical phase STOPPED; RLT replay collection paused")
                    print("[KEY] Critical OFF", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("s"):
                # Force-exit HIL and resume the policy immediately (no stillness wait).
                if HIL_FORCE_RESUME is False:
                    HIL_FORCE_RESUME = True
                    logger.info("[KEY] HIL force-resume requested (press 's'); policy will resume at the next block.")
                    print("[KEY] HIL force-resume", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("h"):
                # Explicit HIL mode toggle (Evo-RL style): enter intervention if
                # autonomous, exit intervention if already in HIL. Debounce so a
                # quick double press does not toggle twice.
                now = time.monotonic()
                if now - self._last_toggle_time < 1.0:
                    logger.info("[KEY] HIL toggle debounced (1s cooldown); ignored.")
                    print("[KEY] HIL toggle debounced", flush=True)
                else:
                    self._last_toggle_time = now
                    if HIL_TOGGLE_REQUESTED is False:
                        HIL_TOGGLE_REQUESTED = True
                        logger.info("[KEY] HIL mode toggle requested (press 'h').")
                        print("[KEY] HIL toggle", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            if value.startswith("p"):
                # Pre-episode MANUAL POSITIONING (distinct from HIL): toggles the
                # operator's direct-follow control before a VLA episode starts.
                # Press once to take control and move the robot to the desired
                # start pose; press again to start the episode. Positioning data
                # is never written to replay (no StepRecords / transitions), so
                # it can never be mistaken for HIL intervention data. The request
                # is STICKY: if the system is not in the positioning phase right
                # now (e.g. mid-episode or at the y/n prompt), the press is
                # latched and honored at the next episode's positioning phase.
                POSITION_TOGGLE_REQUESTED = True
                logger.info(
                    "[KEY] Manual positioning toggle requested (press 'p'); "
                    "positioning state = %s. Latched and honored at the next "
                    "pre-episode positioning phase.",
                    POSITIONING_STATE,
                )
                print(f"[KEY] Positioning toggle [state={POSITIONING_STATE}]", flush=True)
                with self._condition:
                    self._condition.notify_all()
                continue
            with self._condition:
                self._responses.append(value)
                self._condition.notify_all()

    def prompt_episode_reward(self) -> tuple[float, bool, bool]:
        print("\nWas this episode successful? [y/n/d(iscard)/q(uit)]: ", end="", flush=True)
        while not SHUTDOWN and not EMERGENCY_STOP:
            with self._condition:
                if not self._responses:
                    self._condition.wait(timeout=0.1)
                    continue
                response = self._responses.popleft()
            if response in ("y", "yes", "1"):
                return 1.0, True, False
            if response in ("n", "no", "0"):
                return 0.0, True, False
            if response in ("d", "discard"):
                return 0.0, True, True
            if response in ("q", "quit"):
                return 0.0, False, False
            print("  Please enter 'y', 'n', 'd', or 'q': ", end="", flush=True)
        return 0.0, False, False

    def prompt_hil_resolution(self) -> str:
        """Resolve a hands-busy HIL pause after the leader has remained still."""
        print(
            "\nHuman control paused. [y] success / [n] failure / [r] resume / [q] quit: ",
            end="", flush=True,
        )
        while not SHUTDOWN and not EMERGENCY_STOP:
            with self._condition:
                if not self._responses:
                    self._condition.wait(timeout=0.1)
                    continue
                response = self._responses.popleft()
            if response in ("y", "yes", "1"):
                return "success"
            if response in ("n", "no", "0"):
                return "failure"
            if response in ("r", "resume"):
                return "resume"
            if response in ("q", "quit"):
                return "quit"
            print("  Please enter 'y', 'n', 'r', or 'q': ", end="", flush=True)
        return "quit"


def signal_handler(sig, frame):
    global SHUTDOWN
    logger.info("Shutdown signal received. Finishing current episode...")
    SHUTDOWN = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures for PI0.5 RLT Stage 2
# ─────────────────────────────────────────────────────────────────────────────


















@dataclass
class TakeoverState:
    """Small coordinator for lossless takeover requests and leader sample ordering."""

    pending: bool = False
    last_leader_sequence: int = -1

    def latch(self, requested: bool = True) -> None:
        self.pending = self.pending or bool(requested)

    def handoff_finished(self, succeeded: bool) -> None:
        if succeeded:
            self.pending = False

    def accept_leader_sequence(self, sequence: int) -> bool:
        if sequence <= self.last_leader_sequence:
            return False
        self.last_leader_sequence = sequence
        return True


# ─────────────────────────────────────────────────────────────────────────────
# TeleopManager: background 100Hz leader polling with takeover detection
# ─────────────────────────────────────────────────────────────────────────────

class TeleopManager:
    JOINT_NAMES = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]

    def __init__(
        self,
        leader,
        threshold: float = 0.5,
        trigger_frames: int = 2,
        release_frames: int = 60,
        poll_hz: float = 100.0,
    ):
        self.leader = leader
        self.threshold = threshold
        self.trigger_frames = trigger_frames
        self.release_frames = release_frames
        self.poll_dt = 1.0 / poll_hz
        self._lock = threading.Lock()
        self._bus_lock = threading.Lock()  # serializes leader bus reads (poll) and writes (mirror)
        self._latest_sample: LeaderSample | None = None
        self._sequence = 0
        self._takeover_requested = False
        self._resume_requested = False
        self._still_count = 0
        self._prev: dict[str, float] | None = None
        self._human_mode = False
        self._human_not_before = 0.0
        self._stillness_not_before = 0.0
        self._human_stillness_guard_s = 0.0
        self._paused = False
        self._idle = threading.Event()
        self._idle.set()
        self._running = False
        self._thread: threading.Thread | None = None
        self._reads = 0
        self._faults = 0

    @property
    def latest_tensor(self) -> torch.Tensor | None:
        sample = self.latest_sample
        return None if sample is None else sample.tensor

    @property
    def latest_action(self) -> dict[str, float] | None:
        sample = self.latest_sample
        return None if sample is None else sample.action

    @property
    def latest_sample(self) -> LeaderSample | None:
        with self._lock:
            return self._latest_sample

    @property
    def latest_sequence(self) -> int:
        with self._lock:
            return -1 if self._latest_sample is None else self._latest_sample.sequence

    def clear_takeover_request(self) -> None:
        """Discard a stale unconsumed takeover request (called at episode end so
        the leader mirror re-engages instead of leaving the leader torque-off)."""
        with self._lock:
            self._takeover_requested = False

    @property
    def takeover_request_pending(self) -> bool:
        """Non-consuming peek at the takeover request flag (used by the leader mirror)."""
        with self._lock:
            return self._takeover_requested

    def fresh_sample(self, max_age_s: float, *, after_sequence: int | None = None) -> LeaderSample | None:
        with self._lock:
            sample = self._latest_sample
            if sample is None or not sample.is_fresh(max_age_s):
                return None
            if after_sequence is not None and sample.sequence <= after_sequence:
                return None
            return sample

    def wait_for_fresh_sample(
        self,
        max_age_s: float,
        *,
        after_sequence: int,
        timeout_s: float = 1.0,
    ) -> LeaderSample | None:
        """Wait for a post-handoff sample instead of reusing the paused poller's cache."""
        deadline = time.monotonic() + timeout_s
        while self._running and not SHUTDOWN and not EMERGENCY_STOP:
            sample = self.fresh_sample(max_age_s, after_sequence=after_sequence)
            if sample is not None:
                return sample
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(self.poll_dt, 0.01))
        return None

    @property
    def stats(self) -> tuple[int, int]:
        return self._reads, self._faults

    def consume_takeover_request(self) -> bool:
        with self._lock:
            requested = self._takeover_requested
            self._takeover_requested = False
            return requested

    def consume_resume_request(self) -> bool:
        with self._lock:
            requested = self._resume_requested
            self._resume_requested = False
            return requested

    def leader_is_fresh(self, max_age_s: float) -> bool:
        return self.fresh_sample(max_age_s) is not None

    def enter_human_mode(self, minimum_dwell_s: float) -> None:
        with self._lock:
            now = time.monotonic()
            self._human_mode = True
            self._human_stillness_guard_s = minimum_dwell_s
            self._human_not_before = now + minimum_dwell_s
            self._stillness_not_before = now + minimum_dwell_s
            self._resume_requested = False
            self._still_count = 0
            self._prev = None

    def exit_human_mode(self) -> None:
        with self._lock:
            self._human_mode = False
            self._takeover_requested = False
            self._resume_requested = False
            self._still_count = 0
            self._prev = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def reset_episode(self):
        self.exit_human_mode()

    def pause(self) -> bool:
        self._idle.clear()
        with self._lock:
            self._paused = True
        return self._idle.wait(timeout=max(1.0, self.poll_dt * 5))

    def resume(self):
        with self._lock:
            self._paused = False
            self._still_count = 0
            self._prev = None

    def _loop(self):
        while self._running and not SHUTDOWN:
            with self._lock:
                paused = self._paused
            if paused:
                self._idle.set()
                time.sleep(self.poll_dt)
                continue
            self._idle.clear()
            try:
                with self._bus_lock:
                    action = self.leader.get_action()
                values = []
                for joint in self.JOINT_NAMES:
                    value = float(action[f"{joint}.pos"])
                    if not math.isfinite(value):
                        raise ValueError(f"non-finite {joint}.pos={value}")
                    values.append(value)
                with self._lock:
                    self._sequence += 1
                    timestamp = time.monotonic()
                    tensor = torch.tensor(values, dtype=torch.float32)
                    self._latest_sample = LeaderSample(
                        sequence=self._sequence,
                        timestamp=timestamp,
                        action=dict(action),
                        tensor=tensor,
                        healthy=True,
                    )
                    self._reads += 1
                    self._update_motion_locked(action)
            except Exception:
                with self._lock:
                    self._faults += 1
            time.sleep(self.poll_dt)

    def _update_motion_locked(self, action: dict[str, float]) -> None:
        # HIL entry and exit are EXPLICIT key toggles only ('h' enters, 'h'/'s'
        # exits). No automatic takeover and no auto-resume are derived from leader
        # motion, so this only tracks the previous sample (kept for future use).
        current = {f"{joint}.pos": float(action[f"{joint}.pos"]) for joint in self.JOINT_NAMES}
        self._prev = current


# ─────────────────────────────────────────────────────────────────────────────
# Human feedback
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# Quantile normalization helpers (π0.5 uses QUANTILES mode)
# ─────────────────────────────────────────────────────────────────────────────



def quantile_unnormalize(tensor: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    denom = q99 - q01
    return (tensor + 1.0) * denom / 2.0 + q01


def make_pi05_reference_chunk(
    actions: torch.Tensor,
    *,
    action_dim: int,
    start_index: int,
    num_steps: int,
    stride: int = 1,
) -> torch.Tensor:
    """Build a normalized RLT reference horizon from a raw PI0.5 action chunk.

    Paper Eq. (5) / Algorithm 1 define the reference as ``a~1:C`` — the first C
    contiguous VLA actions. Stride-2 subsampling applies only to replay
    transition collection, never to the reference chunk, so ``stride`` defaults
    to 1 (contiguous) and is kept only for tests of the stride helper.
    """
    indices = start_index + torch.arange(num_steps, device=actions.device) * stride
    indices = indices.clamp_max(actions.shape[1] - 1)
    return actions.index_select(1, indices)[:, :, :action_dim]

# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing for π0.5 (SigLIP expects 224x224, [-1, 1])
# ─────────────────────────────────────────────────────────────────────────────





def canonical_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare_output_dir(path: Path, *, dry_run: bool) -> None:
    """Fail closed on prior contents; dry-run performs no filesystem mutation."""
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"--output_dir is not a directory: {path}")
        contents = list(path.iterdir())
        if contents:
            raise FileExistsError(
                f"--output_dir already contains {len(contents)} item(s): {path}. "
                "Stage 2 resume is unsupported; choose a new empty directory."
            )
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)




def resolve_tokenizer_path(pi05_path: Path, explicit: str) -> str:
    if explicit:
        tokenizer = Path(explicit).expanduser()
        if not tokenizer.exists():
            raise FileNotFoundError(f"Explicit tokenizer path does not exist: {tokenizer}")
        return str(tokenizer.resolve())
    processor_path = pi05_path / "policy_preprocessor.json"
    if not processor_path.is_file():
        raise FileNotFoundError(
            f"No --tokenizer_path and checkpoint has no {processor_path.name}; refusing fallback."
        )
    processor = _load_json(processor_path)
    names = [
        step.get("config", {}).get("tokenizer_name")
        for step in processor.get("steps", [])
        if step.get("registry_name") == "tokenizer_processor"
    ]
    names = [name for name in names if isinstance(name, str) and name]
    if len(names) != 1:
        raise ValueError(f"Expected exactly one tokenizer_name in {processor_path}, got {names}")
    if names[0] == "google/gemma-2b":
        raise ValueError("PI0.5 tokenizer provenance must never fall back to google/gemma-2b")
    return names[0]






def reconstruct_stage1_provenance(checkpoint: dict, checkpoint_path: Path) -> dict:
    explicit = checkpoint.get("provenance")
    if isinstance(explicit, dict):
        return explicit
    effective = checkpoint.get("effective_args", {})
    precomputed = effective.get("precomputed_path") if isinstance(effective, dict) else None
    if not precomputed:
        return {"legacy": True, "rlt_checkpoint": str(checkpoint_path.resolve())}
    cache_path = Path(precomputed).expanduser()
    if not cache_path.is_absolute():
        cache_path = (checkpoint_path.parent / cache_path).resolve()
    meta_path = cache_path / "meta.json"
    if not meta_path.is_file():
        return {
            "legacy": True,
            "precomputed_path": str(cache_path),
            "cache_meta_missing": True,
        }
    return {
        "legacy": True,
        "precomputed_path": str(cache_path),
        "precomputed_meta": _load_json(meta_path),
    }

def load_actor_checkpoint(path: Path) -> dict:
    """Deployment loader for a trained Stage 2 actor.

    Unlike ``prepare_resume_checkpoint`` in the training script, deployment only
    needs the actor weights, so a replay journal next to the checkpoint is NOT
    required, and neither are the critic/optimizer (an earlier version demanded
    the full training state, which is pointless for deployment).
    The same-provenance check (whether VLA/RLT go together) happens separately
    once runtime_provenance is built.
    """
    if not path.is_file():
        raise FileNotFoundError(f"actor checkpoint does not exist: {path} (pass it with --actor_checkpoint)")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"actor_state_dict", "actor_contract", "rl_chunk_length", "episode"}
    missing = sorted(required - set(checkpoint))
    if missing or checkpoint.get("actor_contract") != "paper_full_output_v1" or checkpoint.get("rl_chunk_length") != 10:
        raise ValueError(
            "Checkpoint is not a paper-aligned full-output Stage 2 actor checkpoint "
            f"(missing={missing}, actor_contract={checkpoint.get('actor_contract')}, "
            f"rl_chunk_length={checkpoint.get('rl_chunk_length')})."
        )
    return checkpoint

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _strip_json_comments(text: str) -> str:
    """Strip // line comments and /* */ block comments from JSON (JSONC).

    Comments inside string literals are preserved, so paths like
    ``"https://..."`` are safe.
    """
    output: list[str] = []
    index = 0
    length = len(text)
    in_string = False
    while index < length:
        char = text[index]
        if in_string:
            output.append(char)
            if char == "\\" and index + 1 < length:
                output.append(text[index + 1])
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "/":
            while index < length and text[index] != "\n":
                index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "*":
            index += 2
            while index + 1 < length and not (text[index] == "*" and text[index + 1] == "/"):
                index += 1
            index += 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def save_critical_transition(
    frames_dir,
    camera_names,
    *,
    episode: int,
    step: int,
    direction: str,
    source: str,
    prob: float | None,
    obs,
) -> None:
    """Persist one critical-phase on/off transition: an event CSV row plus the
    current camera frames, so the auto-detector's (or manual 'c' toggle's)
    timing can be reviewed offline even if the terminal log is delayed.

    ``direction`` is "on" (critical phase started) or "off" (stopped);
    ``source`` is "auto" or "manual". Files are named
    ``ep<ep>_step<step>_<on|off>_<auto|manual>_<camera>.jpg``.
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    events_path = frames_dir / "critical_events.csv"
    if not events_path.exists():
        with open(events_path, "w", newline="") as file:
            csv.writer(file).writerow(
                ["episode", "step", "direction", "source", "prob", "wall_time"]
            )
    with open(events_path, "a", newline="") as file:
        csv.writer(file).writerow([
            int(episode),
            int(step),
            direction,
            source,
            "" if prob is None else f"{prob:.3f}",
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ])
    import cv2 as _cv2

    for cam in camera_names:
        image = obs.get(cam) if isinstance(obs, dict) else None
        if image is None:
            continue
        fname = (
            f"ep{int(episode):03d}_step{int(step):05d}_{direction}_{source}_{cam}.jpg"
        )
        try:
            # SO-101 camera frames are RGB; cv2.imwrite expects BGR, so convert
            # or the saved colors are channel-swapped (blue <-> yellow).
            frame = np.asarray(image)
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
            _cv2.imwrite(str(frames_dir / fname), frame)
        except Exception as exc:  # noqa: BLE001 - saving is best-effort
            logger.warning("Critical frame save failed (%s): %s", fname, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout recording (deploy only): LeRobotDataset, episodes from index 0
# ─────────────────────────────────────────────────────────────────────────────



class EpisodeDatasetRecorder:
    """Record deployment rollouts into a standard LeRobotDataset.

    - episode_index starts at 0 (a fresh dataset every run; never appended to an
      existing one).
    - features match teleoperation data: observation.images.<cam> (video),
      observation.state, action.
    - add_step() per step: current observation images (copied, because camera
      buffers get reused) + state + the command actually sent.
    - save_episode() per episode (clear_episode_buffer() when y/n scoring
      discards it).
    """

    def __init__(
        self,
        *,
        repo_id: str,
        root: Path,
        fps: float,
        camera_map: dict[str, str],
        camera_shapes: dict[str, tuple[int, int, int]],
        state_names: list[str],
        task: str,
        use_videos: bool = True,
        record_hil: bool = True,
        streaming_encoding: bool = True,
        encoder_threads: int = 2,
        dry_run: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.root = root
        self.fps = float(fps)
        # obs keys (physical camera names such as "top") -> dataset feature keys ("observation.images.top")
        self.camera_map = dict(camera_map)
        self.camera_shapes = {k: tuple(v) for k, v in camera_shapes.items()}
        self.state_names = list(state_names)
        self.task = task
        self.record_hil = bool(record_hil)
        self.streaming_encoding = bool(streaming_encoding)
        self.encoder_threads = int(encoder_threads)
        self.dry_run = dry_run
        self.dataset = None
        self.frames_in_episode = 0
        self.episodes_saved = 0
        self.skipped_steps_in_episode = 0
        self._add_ms_total = 0.0
        self._add_ms_max = 0.0
        if dry_run:
            return

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.state_names),),
                "names": list(self.state_names),
            },
            "action": {
                "dtype": "float32",
                "shape": (len(self.state_names),),
                "names": list(self.state_names),
            },
        }
        for obs_key, dataset_key in self.camera_map.items():
            # With use_videos=False the dtype must be declared as "image" (otherwise
            # LeRobotDataset.create fails outright); video mode uses "video" to match
            # teleoperation data.
            features[dataset_key] = {
                "dtype": "video" if use_videos else "image",
                "shape": self.camera_shapes[obs_key],
                "names": ["height", "width", "channels"],
            }
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=int(round(self.fps)),
            features=features,
            root=self.root,
            robot_type="so101_follower",
            use_videos=use_videos,
            image_writer_threads=max(1, len(self.camera_map)),
            streaming_encoding=self.streaming_encoding,
            encoder_threads=self.encoder_threads if self.streaming_encoding else None,
        )
        logger.info(
            "Rollout recording enabled: repo_id=%s root=%s fps=%d features=%s "
            "(episode_index starts at 0, streaming_encoding=%s, encoder_threads=%s)",
            self.repo_id, self.root, int(round(self.fps)), sorted(features),
            self.streaming_encoding, self.encoder_threads,
        )

    def add_step(self, obs: dict | None, action, source: str = "AUTO") -> None:
        """Record one step: observation images + state + the action actually sent.

        ``action`` may be:
          - torch.Tensor / np.ndarray of length 6, ordered like JOINT_NAMES (this is
            exactly what send_validated_action() returns);
          - a dict of the form {"<joint>.pos": float}.
        Note: image keys in follower.get_observation() are physical camera names
        (top/wrist) while dataset features use the checkpoint keys
        (observation.images.top/...), hence the mapping.
        """
        if self.dataset is None:
            return
        if obs is None:
            self.note_skipped_step()
            return
        frame: dict = {}
        for obs_key, dataset_key in self.camera_map.items():
            image = obs.get(obs_key)
            if image is None:
                image = obs.get(dataset_key)   # some call sites already mapped the key
            if image is None:
                raise KeyError(
                    f"recording is missing camera {obs_key!r} (dataset key {dataset_key!r}); "
                    f"the observation provides {sorted(obs)}"
                )
            frame[dataset_key] = np.array(image, copy=True)  # camera buffers are reused, so copy
        state = np.asarray([obs.get(f"{j}.pos") for j in self.state_names], dtype=np.float32)
        if isinstance(action, dict):
            command = np.asarray([action[f"{j}.pos"] for j in self.state_names], dtype=np.float32)
        else:
            raw = action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
            command = np.asarray(raw, dtype=np.float32).reshape(-1)
            if command.size != len(self.state_names):
                raise ValueError(
                    f"recorded action has the wrong dimension: {command.size} != "
                    f"{len(self.state_names)} (passed type {type(action).__name__})"
                )
        if not np.isfinite(state).all() or not np.isfinite(command).all():
            raise ValueError("non-finite value in a recorded frame (state/action)")
        frame["observation.state"] = state
        frame["action"] = command
        frame["task"] = self.task
        t0 = time.perf_counter()
        self.dataset.add_frame(frame)
        dt_ms = (time.perf_counter() - t0) * 1e3
        self._add_ms_total += dt_ms
        self._add_ms_max = max(self._add_ms_max, dt_ms)
        self.frames_in_episode += 1

    def note_skipped_step(self) -> None:
        """A step was not recorded (e.g. a skipped HIL segment): the episode has a
        trajectory gap and is dropped at the end."""
        self.skipped_steps_in_episode += 1

    def save_episode(self, *, keep: bool = True) -> None:
        if self.dataset is None:
            return
        gap = self.skipped_steps_in_episode > 0
        keep = keep and self.frames_in_episode > 0 and not gap
        if not keep:
            self.dataset.clear_episode_buffer()
            logger.info(
                "Episode not saved (keep=%s, frames=%d, skipped steps=%d%s)",
                keep, self.frames_in_episode, self.skipped_steps_in_episode,
                "; unrecorded HIL steps create a trajectory gap" if gap else "",
            )
        else:
            self.dataset.save_episode()
            avg_ms = self._add_ms_total / max(1, self.frames_in_episode)
            logger.info(
                "Saved episode_index=%d (%d frames) | %d total | add_frame %.1f ms/frame (max %.1f ms)",
                self.episodes_saved, self.frames_in_episode, self.episodes_saved + 1, avg_ms, self._add_ms_max,
            )
            if self._add_ms_max > 0.5 * (1000.0 / self.fps):
                logger.warning(
                    "Slowest recorded frame %.1f ms exceeds half the control period "
                    "%.1f ms -- if teleoperation feels sluggish, consider "
                    "--record_dataset false or a lower camera resolution",
                    self._add_ms_max, 1000.0 / self.fps,
                )
            self.episodes_saved += 1
        self.frames_in_episode = 0
        self.skipped_steps_in_episode = 0
        self._add_ms_total = 0.0
        self._add_ms_max = 0.0

    def finalize(self) -> None:
        if self.dataset is None:
            return
        try:
            self.dataset.finalize()
            logger.info("Recording finalized: %d episode(s) -> %s", self.episodes_saved, self.root)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Recording finalization failed: %s", exc)


def parse_args():
    parser = argparse.ArgumentParser(description="RLT Stage 2: Online RL on SO-101 with Intervention (π0.5)")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file; explicit CLI arguments override its values")
    # Model paths
    parser.add_argument("--pi05_path", type=str, required=True,
                        help="Path to pretrained π0.5 checkpoint")
    parser.add_argument("--rlt_checkpoint", type=str, required=True,
                        help="Path to Stage 1 RLT encoder-decoder checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="",
                        help="Local path to PaliGemma tokenizer")
    parser.add_argument("--stats_path", type=str, default=None,
                        help="Explicit stats.json override (default: checkpoint normalizer state)")
    parser.add_argument("--task", type=str, default="pick up the red cube and place it in the box",
                        help="Task description for the VLA")
    # Robot connection
    parser.add_argument("--follower_port", type=str, default="",
                        help="Follower serial port (required unless --dry_run)")
    parser.add_argument("--leader_port", type=str, default="",
                        help="Leader serial port (required unless --dry_run)")
    parser.add_argument("--follower_id", type=str, default="so101_follower")
    parser.add_argument("--leader_id", type=str, default="so101_leader")
    parser.add_argument("--camera_names", type=str, nargs="+", default=["top", "wrist"])
    parser.add_argument("--cameras", type=str,
                        default='{"top": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}, "wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}}')
    parser.add_argument("--camera_map", type=str, default="",
                        help="JSON mapping of physical camera name to checkpoint feature")
    parser.add_argument("--display_data", action=argparse.BooleanOptionalAction, default=True,
                        help="Show live follower observations and executed actions in Rerun "
                             "(on by default in the deployment script; disable with "
                             "--no-display_data)")
    parser.add_argument("--display_compressed_images", action="store_true",
                        help="Compress images before sending them to Rerun")
    parser.add_argument("--display_fps", type=float, default=10.0,
                        help="Maximum Rerun logging frequency")
    parser.add_argument("--display_session_name", type=str, default="rlt_deploy",
                        help="Rerun session name")
    parser.add_argument("--pad_missing_cameras", action="store_true",
                        help="Fill unmapped checkpoint image features with zero-pixel tensors (test-only)")
    parser.add_argument("--max_relative_target", type=str, default=None,
                        help="Optional positive degree scalar or JSON per-joint motor limits; omitted disables clipping")
    parser.add_argument("--vla_only", action="store_true",
                        help="Disable actor execution; frozen VLA + leader takeover only")
    parser.add_argument("--inference_only", action="store_true",
                        help="Pure inference deployment on the real robot: load a trained Stage 2 "
                             "checkpoint (--actor_checkpoint) and run frozen VLA + RL actor with auto "
                             "critical-phase detection and actor gating, but NO RL updates, NO "
                             "replay recording, NO reward prompts. Optional per-episode y/n scoring "
                             "with --inference_score.")
    parser.add_argument("--inference_score", action=argparse.BooleanOptionalAction, default=True,
                        help="In --inference_only mode, ask y/n/q after each episode just to count "
                             "success rate (never fed back into learning). On by default in the "
                             "deployment script.")
    # ── Rollout recording (deployment only) ────────────────────────────
    parser.add_argument("--record_dataset", action=argparse.BooleanOptionalAction, default=False,
                        help="Record deployment rollouts as a standard LeRobotDataset "
                             "(observation.images.* + observation.state + action), with "
                             "episode_index starting at 0.")
    parser.add_argument("--record_repo_id", type=str, default="local/so101_deploy_rollouts",
                        help="repo_id of the recorded dataset (an identifier only; never pushed to the Hub)")
    parser.add_argument("--record_root", type=str, default="",
                        help="dataset root; default <repo>/outputs/deploy_recordings/<last repo_id segment>")
    parser.add_argument("--record_task", type=str, default="",
                        help="task text written into the dataset; defaults to --task")
    parser.add_argument("--record_videos", action=argparse.BooleanOptionalAction, default=True,
                        help="encode images as video (default True, matching teleoperation data)")
    parser.add_argument("--record_hil_steps", action=argparse.BooleanOptionalAction, default=True,
                        help="also record human-in-the-loop (HIL) steps (default True; when "
                             "disabled, any episode with an intervention is discarded because "
                             "its trajectory would have a gap)")
    parser.add_argument("--record_streaming_encoding", action=argparse.BooleanOptionalAction, default=True,
                        help="encode video in a background thread (default True, so the control period is not slowed down)")
    parser.add_argument("--record_encoder_threads", type=int, default=2,
                        help="number of background video-encoding threads (used with --record_streaming_encoding)")
    parser.add_argument("--rtc_enabled", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Pi0.5 real-time chunking guidance")
    parser.add_argument("--rtc_execution_horizon", type=int, default=10,
                        help="Actions executed between Pi0.5 replans; must match RLT horizon")
    parser.add_argument("--rtc_prefix_attention_schedule", choices=["EXP", "LINEAR", "ONES", "ZEROS"], default="EXP",
                        help="Pi0.5 RTC prefix attention schedule")
    parser.add_argument("--rtc_max_guidance_weight", type=float, default=5.0,
                        help="Maximum Pi0.5 RTC guidance weight")
    parser.add_argument("--critical_frames_dir", type=str, default=None,
                        help="Directory for key frames saved at critical-phase on/off transitions "
                             "(default: <output_dir>/critical_frames).")
    parser.add_argument("--save_critical_frames", action=argparse.BooleanOptionalAction, default=True,
                        help="Save key frames at critical-phase on/off transitions "
                             "(default: on; use --no-save_critical_frames to disable).")
    parser.add_argument("--auto_critical", action="store_true",
                        help="Automatically toggle CRITICAL_PHASE_ACTIVE with a trained classifier "
                             "(requires --critical_classifier); manual 'c' always overrides for a cooldown.")
    parser.add_argument("--critical_classifier", type=str, default=None,
                        help="Path to a trained critical-phase classifier checkpoint "
                             "(scripts/train_critical_classifier.py).")
    parser.add_argument("--auto_critical_threshold_on", type=float, default=0.5,
                        help="Smoothed P(critical) above which the auto-detector turns the critical phase ON "
                             "(lower = earlier detection, at the cost of occasional false starts).")
    parser.add_argument("--auto_critical_threshold_off", type=float, default=0.35,
                        help="Smoothed P(critical) below which the auto-detector turns the critical phase OFF.")
    parser.add_argument("--auto_critical_smooth_steps", type=int, default=5,
                        help="Moving-average window (in chunks) for the auto-detector probability "
                             "(smaller = faster response, more jitter).")
    parser.add_argument("--auto_critical_min_on_chunks", type=int, default=2,
                        help="Minimum chunks the critical phase stays ON once triggered, before an "
                             "OFF is accepted (debounce against probability dips causing "
                             "false ON/OFF flicker).")
    parser.add_argument("--auto_critical_start_delay_steps", type=int, default=60,
                        help="Do not auto-ON during the first N steps of an episode (the reset/approach "
                             "phase is never the critical phase; avoids frame-0 false positives).")
    parser.add_argument("--actor_critical_delay_steps", type=int, default=15,
                        help="Minimum consecutive critical-phase CONTROL steps before the RL actor "
                             "may take control. Counted per boundary window (each window advances "
                             "the counter by n_action_steps_rl=10), so 15 = the actor switches at "
                             "the 2nd window after detection. Guards transient auto-critical false "
                             "positives from grabbing control away from VLA.")
    parser.add_argument("--auto_critical_once", action=argparse.BooleanOptionalAction, default=True,
                        help="Each episode has ONE critical segment: after the auto-detector has "
                             "turned ON and then OFF, it will not auto-ON again in the same episode "
                             "(manual 'c' still works).")
    parser.add_argument("--auto_critical_override_cooldown", type=float, default=3.0,
                        help="Seconds after a manual 'c' press during which the auto-detector is muted.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate paths without opening serial ports")
    parser.add_argument("--reset_between_episodes", action="store_true")
    parser.add_argument("--manual_positioning", action=argparse.BooleanOptionalAction, default=False,
                        help="Pre-episode manual positioning (straw fixed, no auto reset): press 'p' to "
                             "teleoperate the follower (leader direct-follow) to the desired start pose, "
                             "press 'p' again to start the VLA episode. Replaces the between-episode auto "
                             "reset. This is NOT HIL: positioning data never enters the replay buffer. "
                             "Configurable via config key \"manual_positioning\" (true/false); "
                             "--manual_positioning / --no-manual_positioning override the config either way.")
    parser.add_argument("--reset_steps", type=int, default=90)
    parser.add_argument("--reset_dt", type=float, default=0.08)
    parser.add_argument("--reset_tolerance_deg", type=float, default=5.0,
                        help="Acceptable final maximum reset error in degrees")
    parser.add_argument("--reset_max_passes", type=int, default=2,
                        help="Maximum closed-loop reset correction passes")
    parser.add_argument("--leader_hil_align_duration_s", type=float, default=5.0,
                        help="Seconds to smoothly align the powered leader to follower before takeover")
    parser.add_argument("--leader_hil_align_fps", type=float, default=50.0,
                        help="Leader alignment command rate")
    parser.add_argument("--leader_hil_align_tolerance", type=float, default=10.0,
                        help="Required final six-joint leader/follower alignment error")
    parser.add_argument("--intervention_threshold", type=float, default=1.0,
                        help="Per-sample leader movement in degrees required to request automatic takeover")
    parser.add_argument("--intervention_trigger_frames", type=int, default=2,
                        help="Consecutive moving leader samples required to latch takeover")
    parser.add_argument("--intervention_release_frames", type=int, default=100,
                        help="Still leader samples required after the last leader motion to resume policy")
    parser.add_argument("--leader_hil_min_dwell_s", type=float, default=2.0,
                        help="Minimum direct-follow duration and post-motion quiet guard before policy resume")
    # Training
    parser.add_argument("--max_episodes", type=int, default=200)
    parser.add_argument("--steps_per_episode", type=int, default=300)
    parser.add_argument("--control_fps", type=float, default=10.0)
    parser.add_argument("--output_dir", type=str, default="checkpoints/rlt_stage2_pi05")
    parser.add_argument("--resume_from", "--actor_checkpoint", type=str, default=None,
                        dest="resume_from",
                        help="Trained Stage 2 actor checkpoint to deploy. Required: it holds the policy "
                             "weights this script runs. Accepted as --actor_checkpoint as well, and as the "
                             "'resume_from' key in a config file. Deployment never resumes training.")
    parser.add_argument("--actor_max_relative_target", type=str, default=None,
                        help="Per-step per-joint angle limit (deg) applied ONLY to ACTOR (RL policy) "
                             "actions, relative to the last commanded pose. VLA/HUMAN actions are not "
                             "clamped. JSON map of joint names to positive limits or a scalar.")
    parser.add_argument("--actor_execute_mean", action="store_true",
                        help="Execute the actor's MEAN action deterministically (no Gaussian sampling). "
                             "Eliminates sampling jitter; the actor output then tracks its BC-anchored "
                             "mean (≈ reference) so execution is smooth like the VLA.")
    parser.add_argument("--actor_noise_std", type=float, default=0.0,
                        help="Exploration noise (normalized space) added on top of the mean when "
                             "actor_execute_mean is on (e.g. 0.01 = small exploration without full "
                             "sampling jitter). 0 = pure deterministic mean.")
    parser.add_argument("--actor_execution_scale", type=float, default=1.0,
                        help="Scale factor applied to ACTOR actions before execution (e.g. 0.5 = half "
                             "speed). Slower actor motion means the same task spans more control steps "
                             "and records more transitions. Applied to the executed action (recorded as "
                             "sent); the actor network still trains toward the full reference via BC.")
    parser.add_argument("--observation_retries", type=int, default=3,
                        help="Retries for a transient camera/observation fault (e.g. OpenCVCamera "
                             "'latest frame is too old') before the run aborts. No robot command is "
                             "sent during the retries; 0 = abort on the first fault.")
    parser.add_argument("--observation_retry_delay_s", type=float, default=0.25,
                        help="Delay between observation retries (seconds).")
    parser.add_argument("--camera_preflight_retries", type=int, default=6,
                        help="fresh-frame attempts per camera at startup (0 = skip the camera preflight)")
    parser.add_argument("--camera_preflight_delay_s", type=float, default=0.5,
                        help="delay between camera-preflight attempts (seconds)")
    parser.add_argument("--save_debug_obs", action="store_true",
                        help="Save per-episode debug observations (camera images at each replan + "
                             "per-step state/action/source CSV) under output_dir/debug_obs/ for inspection.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--actor_residual_scale", type=float, default=0.0,
                        help="Deprecated compatibility flag; paper-aligned Stage 2 requires 0.0.")
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--online_bc_weight", type=float, default=10.0,
                        help="BC regularization weight after learner warmup")
    parser.add_argument("--delta_weight", type=float, default=0.0,
                        help="Smoothness (delta) loss weight: penalizes the difference between the "
                             "actor's per-step action deltas and the reference/human action deltas "
                             "(computed in unnormalized degree space). 0 = paper-aligned RLT (no "
                             "smoothness). Start at 0.02 for real-robot jitter reduction; the "
                             "'delta' column in rl_metrics.log shows the raw delta penalty — aim "
                             "for delta_weight*delta to stay below ~1/3 of the Q term "
                             "(online_q_weight*actorQ) so RL is not suppressed.")
    parser.add_argument("--policy_fixed_std", type=float, default=0.05)
    parser.add_argument("--reference_dropout", type=float, default=0.5,
                        help="Probability of zeroing the reference chunk during actor training")
    parser.add_argument("--actor_hidden_dim", type=int, default=256,
                        help="Hidden width of the paper-aligned actor MLP")
    parser.add_argument("--actor_num_layers", type=int, default=2,
                        help="Number of MLP layers in the actor (2 or 3)")
    parser.add_argument("--critic_hidden_dim", type=int, default=256,
                        help="Hidden width of the paper-aligned critic MLP")
    parser.add_argument("--critic_num_layers", type=int, default=2,
                        help="Number of MLP layers in the critic (2 or 3)")
    parser.add_argument("--target_tau", type=float, default=0.005)
    parser.add_argument("--actor_execution_steps", type=int, default=0,
                        help="Steps the ACTOR executes before re-planning (receding horizon). "
                             "0 = full RL chunk (10 steps).")
    parser.add_argument("--replan_every_window", action=argparse.BooleanOptionalAction, default=True,
                        help="Replan a fresh 50-step VLA inference at EVERY 10-step window for "
                             "both the frozen VLA and the RL actor, so the reference chunk comes "
                             "from the current observation (paper Algorithm 1: a~t:t+C-1 ~ "
                             "pi_vla(st)). Cost: the full 10-step diffusion (~1-5 s on the 3B "
                             "model) runs every 0.33 s, so the robot moves in bursts. Disable "
                             "with --no-replan_every_window to reuse the cached 50-step chunk "
                             "and replan only when it is exhausted (lerobot-record cadence: "
                             "smooth, but references become chunk-continuation).")
    parser.add_argument("--leader_mirror", action=argparse.BooleanOptionalAction, default=False,
                        help="Continuously mirror the follower pose onto the leader arm. Off by "
                             "default: during deployment/inference the leader does not move at all "
                             "(--no-leader_mirror); enable --leader_mirror when a human should be "
                             "able to grab the leader for HIL at any moment. Disabling does not "
                             "affect HIL (takeover aligns via hil_offset).")
    parser.add_argument("--leader_mirror_torque_limit", type=float, default=0.5,
                        help="Leader mirror holding torque as a fraction of max (0-1). Lower = the "
                             "leader yields to a human more easily (easier HIL takeover).")
    parser.add_argument("--seed", type=int, default=42)
    # Intervention shaping is intentionally disabled in strict RL Token mode:
    # human corrections are demonstrations, not an extra reward source.
    parser.add_argument("--intervention_reward_bonus", type=float, default=0.0)
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=str, default=None)
    config_args, _ = config_probe.parse_known_args()
    if config_args.config:
        config_path = Path(config_args.config).expanduser()
        with open(config_path, encoding="utf-8") as file:
            # JSONC support: // line and /* block */ comments are allowed.
            config_values = json.loads(_strip_json_comments(file.read()))
        if not isinstance(config_values, dict):
            parser.error(f"Config must contain a JSON object: {config_path}")
        actions = {action.dest: action for action in parser._actions}
        unknown = sorted(set(config_values) - set(actions))
        if unknown:
            parser.error(f"Unknown config keys: {', '.join(unknown)}")
        for key in ("cameras", "camera_map"):
            if isinstance(config_values.get(key), dict):
                config_values[key] = json.dumps(config_values[key])
        for key in ("max_relative_target", "actor_max_relative_target"):
            if config_values.get(key) is not None and not isinstance(config_values[key], str):
                config_values[key] = json.dumps(config_values[key])
        parser.set_defaults(**config_values)
        for key in config_values:
            actions[key].required = False
    return parser.parse_args()






# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global SHUTDOWN, EMERGENCY_STOP, SKIP_EPISODE, FINISH_EPISODE, CRITICAL_PHASE_ACTIVE, HIL_FORCE_RESUME, HIL_TOGGLE_REQUESTED
    global POSITION_TOGGLE_REQUESTED, POSITIONING_STATE
    args = parse_args()
    # ── Deployment script: inference only ────────────────────────────────
    # This script never learns: no replay, no updates, no training checkpoints.
    # Episodes restart at 1, and dataset recording uses episode_index from 0.
    args.inference_only = True
    if not args.resume_from:
        raise ValueError(
            "the deployment script requires a trained Stage 2 actor checkpoint: pass "
            "--actor_checkpoint (also accepted as --resume_from, or as the config key 'resume_from')"
        )
    if args.control_fps <= 0:
        raise ValueError("--control_fps must be positive.")
    if args.reset_steps < 1:
        raise ValueError("--reset_steps must be at least 1.")
    if args.reset_dt < 0:
        raise ValueError("--reset_dt cannot be negative.")
    if args.reset_tolerance_deg < 0 or not math.isfinite(args.reset_tolerance_deg):
        raise ValueError("--reset_tolerance_deg must be finite and non-negative.")
    if args.reset_max_passes < 1:
        raise ValueError("--reset_max_passes must be at least 1.")
    if args.leader_hil_align_duration_s <= 0 or not math.isfinite(args.leader_hil_align_duration_s):
        raise ValueError("--leader_hil_align_duration_s must be finite and positive.")
    if args.leader_hil_align_fps <= 0 or not math.isfinite(args.leader_hil_align_fps):
        raise ValueError("--leader_hil_align_fps must be finite and positive.")
    if args.leader_hil_align_tolerance < 0 or not math.isfinite(args.leader_hil_align_tolerance):
        raise ValueError("--leader_hil_align_tolerance must be finite and non-negative.")
    if args.intervention_threshold <= 0 or not math.isfinite(args.intervention_threshold):
        raise ValueError("--intervention_threshold must be finite and positive.")
    if args.intervention_trigger_frames < 1 or args.intervention_release_frames < 1:
        raise ValueError("--intervention_trigger_frames and --intervention_release_frames must be positive.")
    if args.leader_hil_min_dwell_s < 0 or not math.isfinite(args.leader_hil_min_dwell_s):
        raise ValueError("--leader_hil_min_dwell_s must be finite and non-negative.")
    if args.rtc_execution_horizon < 0:
        raise ValueError("--rtc_execution_horizon must be non-negative (0 = disable RTC).")
    if args.rtc_max_guidance_weight <= 0 or not math.isfinite(args.rtc_max_guidance_weight):
        raise ValueError("--rtc_max_guidance_weight must be finite and positive.")
    if args.actor_residual_scale != 0:
        raise ValueError("Paper-aligned RLT uses a full-output Actor; --actor_residual_scale must be 0.")
    if args.intervention_reward_bonus != 0:
        raise ValueError(
            "Paper-aligned RLT uses terminal-only binary rewards; --intervention_reward_bonus must be 0."
        )
    if args.delta_weight < 0 or not math.isfinite(args.delta_weight):
        raise ValueError("--delta_weight must be >= 0 (0 = paper-aligned, no smoothness loss).")
    if args.inference_only and not args.resume_from:
        raise ValueError(
            "--inference_only requires an actor checkpoint (--actor_checkpoint / --resume_from / "
            "the 'resume_from' config key)."
        )
    # The deployment script also allows --vla_only (pure-VLA control run with the Actor disabled)
    if args.display_fps <= 0 or not math.isfinite(args.display_fps):
        raise ValueError("--display_fps must be finite and positive.")
    if not args.dry_run and (not args.follower_port or not args.leader_port):
        raise ValueError("--follower_port and --leader_port are required unless --dry_run")

    # ── Print effective training parameters ────────────────────────────────
    logger.info("=" * 78)
    logger.info("STAGE 2 DEPLOY / INFERENCE PARAMETERS (inference only, no learning)")
    logger.info("=" * 78)
    logger.info("Model / data:")
    logger.info("  pi05_path       = %s", args.pi05_path)
    logger.info("  rlt_checkpoint  = %s", args.rlt_checkpoint)
    logger.info("  actor_checkpoint= %s", args.resume_from)
    logger.info("  critical_clf    = %s", args.critical_classifier)
    logger.info("  tokenizer       = %s", args.tokenizer_path)
    logger.info("  stats_path      = %s", args.stats_path)
    logger.info("  task            = %s", args.task)
    logger.info("  output_dir      = %s", args.output_dir)
    logger.info("Episode / control:")
    logger.info("  control_fps     = %s | steps_per_episode = %s | max_episodes = %s",
                args.control_fps, args.steps_per_episode, args.max_episodes)
    logger.info("  actor_execution_steps = %s | actor_critical_delay_steps = %s",
                args.actor_execution_steps, args.actor_critical_delay_steps)
    logger.info("  rtc_enabled     = %s | rtc_execution_horizon = %s", args.rtc_enabled, args.rtc_execution_horizon)
    logger.info("Recording / visualization:")
    logger.info("  record_dataset  = %s | repo_id = %s | root = %s | videos = %s",
                args.record_dataset, args.record_repo_id,
                args.record_root or "(default outputs/deploy_recordings/<last repo_id segment>)", args.record_videos)
    logger.info("  display_data    = %s (Rerun) | display_fps = %s",
                args.display_data, args.display_fps)
    logger.info("Learning hyperparameters (all inert in deployment mode; used only to load/align the checkpoint structure):")
    logger.info("  actor_lr = %s | critic_lr = %s | discount(gamma) = %s | target_tau = %s",
                args.actor_lr, args.critic_lr, args.discount, args.target_tau)
    logger.info("  actor net = %s x %s | critic net = %s x %s",
                args.actor_hidden_dim, args.actor_num_layers, args.critic_hidden_dim, args.critic_num_layers)
    logger.info("Safety / HIL:")
    logger.info("  dry_run         = %s | max_relative_target = %s",
                args.dry_run, args.max_relative_target if args.max_relative_target else "disabled")
    logger.info("  actor_max_relative_target = %s (ACTOR-only per-step clamp)",
                args.actor_max_relative_target if args.actor_max_relative_target else "disabled")
    logger.info("  actor_noise_std = %s | actor_execute_mean = %s | control_fps = %s",
                args.actor_noise_std, args.actor_execute_mean, args.control_fps)
    logger.info("  leader_mirror = %s (False = leader does not move) | leader_mirror_torque_limit = %s | intervention_threshold = %s",
                args.leader_mirror, args.leader_mirror_torque_limit, args.intervention_threshold)
    logger.info("  leader_hil_min_dwell_s = %s | intervention_release_frames = %s",
                args.leader_hil_min_dwell_s, args.intervention_release_frames)
    logger.info("=" * 78)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir).expanduser()
    resume_source = Path(args.resume_from).expanduser() if args.resume_from else None
    resume_checkpoint = None
    # The deployment script never resumes, so a non-empty output_dir gets a numeric
    # suffix (keeping the previous logs/recordings) and prepare_output_dir's
    # "must be empty" check cannot abort a deployment.
    if output_dir.exists() and any(output_dir.iterdir()):
        _base, _suffix = output_dir, 0
        while output_dir.exists() and any(output_dir.iterdir()):
            _suffix += 1
            output_dir = Path(f"{_base}_v{_suffix}")
        logger.warning("output_dir %s is not empty -> using %s for this run", _base, output_dir)
        args.output_dir = str(output_dir)
    prepare_output_dir(output_dir, dry_run=args.dry_run)
    if resume_source is not None:
        resume_checkpoint = load_actor_checkpoint(resume_source)

    # ── Logging split: detailed logs go to <output_dir>/train.log (tail it in a
    # second terminal); the console stays clean (warnings/errors + explicit
    # prints below: episode counter, key echoes, reward prompts) so keyboard
    # input is never buried under INFO lines.
    if not args.dry_run:
        try:
            train_log_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
            train_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            _root_logger = logging.getLogger()
            _root_logger.addHandler(train_log_handler)
            for _handler in list(_root_logger.handlers):
                if isinstance(_handler, logging.StreamHandler) and not isinstance(_handler, logging.FileHandler):
                    _handler.setLevel(logging.WARNING)
            logger.info("Detailed logs -> %s (console shows warnings and prompts only)", output_dir / "train.log")
        except OSError as _exc:
            logger.warning("Could not open train.log: %s", _exc)
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(
        "Episode reset: enabled=%s | steps=%s | dt=%.2fs | tolerance=%.2f° | passes=%s",
        args.reset_between_episodes, args.reset_steps, args.reset_dt,
        args.reset_tolerance_deg, args.reset_max_passes,
    )
    logger.info("Press 'e' for EMERGENCY STOP, 'q' for graceful quit, 'f' to finish the episode early. Move the leader to request HIL.")

    # ── Load frozen π0.5 ─────────────────────────────────────────────
    logger.info(f"Loading π0.5 from {args.pi05_path}...")
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05_rlt.vla_compat import (
        extract_embeddings as extract_pi05_embeddings,
    )

    pi05_path = Path(args.pi05_path).expanduser()
    if not (pi05_path / "config.json").is_file() or not (pi05_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete PI0.5 checkpoint: {pi05_path}")
    tokenizer_name = resolve_tokenizer_path(pi05_path, args.tokenizer_path)
    logger.info("Resolved tokenizer provenance: %s", tokenizer_name)

    # Load policy via from_pretrained so input_features values are parsed into
    # PolicyFeature objects (with .type), then take config from the policy.
    # Going PI05Config(**config_dict) leaves them as plain dicts and breaks
    # downstream .image_features / .action_feature; PI05Config.from_pretrained
    # fails because config.json keeps a "type" field not in the dataclass.
    pi05_policy = PI05Policy.from_pretrained(pretrained_name_or_path=args.pi05_path)
    pi05_config = pi05_policy.config

    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    pi05_config.rtc_config = RTCConfig(
        enabled=args.rtc_enabled and args.rtc_execution_horizon > 0,
        prefix_attention_schedule=RTCAttentionSchedule(args.rtc_prefix_attention_schedule),
        execution_horizon=max(args.rtc_execution_horizon, 1),  # Min 1 for config validation
        max_guidance_weight=args.rtc_max_guidance_weight,
    )
    pi05_policy.config = pi05_config
    pi05_policy.init_rtc_processor()
    pi05_policy.to(device)
    pi05_policy.eval()
    for p in pi05_policy.parameters():
        p.requires_grad = False

    checkpoint_image_features = [
        k for k in pi05_config.input_features
        if k.startswith("observation.images.")
    ]
    logger.info("Checkpoint image features: %s", checkpoint_image_features)
    logger.info("π0.5 loaded and frozen with official processor pipeline.")

    # ── Load RLT encoder (frozen from Stage 1) ─────────────────────────
    logger.info(f"Loading RLT encoder from {args.rlt_checkpoint}...")
    from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
    from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTokenEncoder, RLTChunkActor

    rlt_config = PI05RLTConfig(
        mode="online_rl",
        discount=args.discount,
        bc_weight=args.online_bc_weight,
        target_tau=args.target_tau,
        state_dim=6,
        action_dim=6,
        action_stride=2,
        n_action_steps_rl=10,
    )
    if rlt_config.n_action_steps_rl != args.rtc_execution_horizon and args.rtc_execution_horizon != 0:
        raise ValueError(
            "--rtc_execution_horizon must equal PI05RLTConfig.n_action_steps_rl "
            f"({rlt_config.n_action_steps_rl}) or be 0 (disable RTC) so replay matches real execution."
        )
    if rlt_config.action_stride < 1:
        raise ValueError("--action_stride must be positive.")
    if not 0.0 <= args.reference_dropout <= 1.0:
        raise ValueError("--reference_dropout must be in [0, 1].")
    if args.auto_critical and not args.critical_classifier:
        raise ValueError("--auto_critical requires --critical_classifier (a trained checkpoint path).")
    if not 0.0 < args.auto_critical_threshold_off < args.auto_critical_threshold_on < 1.0:
        raise ValueError(
            f"--auto_critical thresholds must satisfy 0 < off < on < 1 "
            f"(got off={args.auto_critical_threshold_off}, on={args.auto_critical_threshold_on}). "
            "Off must stay strictly below on (hysteresis band); raise threshold_on "
            "as well if you need a higher threshold_off."
        )
    if args.auto_critical_smooth_steps < 1:
        raise ValueError("--auto_critical_smooth_steps must be at least 1.")
    if args.auto_critical_min_on_chunks < 0:
        raise ValueError("--auto_critical_min_on_chunks cannot be negative.")
    if args.auto_critical_start_delay_steps < 0:
        raise ValueError("--auto_critical_start_delay_steps cannot be negative.")
    if args.auto_critical_override_cooldown < 0:
        raise ValueError("--auto_critical_override_cooldown cannot be negative.")
    if args.policy_fixed_std <= 0 or not math.isfinite(args.policy_fixed_std):
        raise ValueError("--policy_fixed_std must be finite and positive.")
    if args.actor_hidden_dim <= 0 or args.critic_hidden_dim <= 0:
        raise ValueError("--actor_hidden_dim and --critic_hidden_dim must be positive.")
    if args.actor_num_layers < 1 or args.critic_num_layers < 1:
        raise ValueError("--actor_num_layers and --critic_num_layers must be at least 1.")

    rlt_checkpoint_path = Path(args.rlt_checkpoint).expanduser()
    if not rlt_checkpoint_path.is_file():
        raise FileNotFoundError(f"RLT checkpoint does not exist: {rlt_checkpoint_path}")
    ckpt = torch.load(rlt_checkpoint_path, map_location=device, weights_only=False)
    if ckpt.get("rlt_architecture") != "paper_v1" or not ckpt.get("image_only", False):
        raise ValueError(
            "Stage 2 requires a paper_v1 image-only Stage 1 checkpoint; retrain legacy checkpoints."
        )
    saved_config = ckpt.get("config", {})
    for field in (
        "vlm_hidden_dim", "rlt_hidden_dim", "rlt_encoder_layers", "rlt_num_heads",
    ):
        if field in saved_config and saved_config[field] != getattr(rlt_config, field):
            raise ValueError(
                f"RLT checkpoint {field}={saved_config[field]} does not match Stage 2 "
                f"config {getattr(rlt_config, field)}."
            )
    encoder = RLTokenEncoder(rlt_config).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    logger.info("RLT encoder loaded and frozen.")

    # ── Load normalization stats (QUANTILES mode for π0.5) ──────────────
    stats, stats_source = load_normalization_stats(pi05_path, args.stats_path)
    stage1_provenance = reconstruct_stage1_provenance(ckpt, rlt_checkpoint_path)
    runtime_provenance = {
        "pi05_path": str(pi05_path.resolve()),
        "rlt_checkpoint": str(rlt_checkpoint_path.resolve()),
        "tokenizer": tokenizer_name,
        "stats_source": stats_source,
        "task": args.task,
        "camera_map": args.camera_map,
        "actor_residual_scale": args.actor_residual_scale,
        "rl_chunk_length": rlt_config.n_action_steps_rl,
        "action_stride": rlt_config.action_stride,
        "reference_dropout": args.reference_dropout,
        "actor_hidden_dim": args.actor_hidden_dim,
        "actor_num_layers": args.actor_num_layers,
        "critic_hidden_dim": args.critic_hidden_dim,
        "critic_num_layers": args.critic_num_layers,
        "stage1": stage1_provenance,
    }
    runtime_provenance["digest"] = canonical_digest(runtime_provenance)
    checkpoint_digest = ckpt.get("runtime_compatibility_digest")
    if checkpoint_digest is not None and checkpoint_digest != runtime_provenance["digest"]:
        raise ValueError(
            "RLT checkpoint provenance is incompatible with this Stage 2 runtime: "
            f"checkpoint={checkpoint_digest}, runtime={runtime_provenance['digest']}"
        )

    # ── Actor checkpoint provenance check (VLA / RLT / stats must match training) ──
    if resume_checkpoint is not None:
        saved = resume_checkpoint.get("runtime_provenance") or {}
        keys = ("pi05_path", "rlt_checkpoint", "stats_source", "rl_chunk_length")
        mismatch = {k: {"actor_trained_with": saved.get(k), "now": runtime_provenance.get(k)}
                    for k in keys if saved.get(k) != runtime_provenance.get(k)}
        if mismatch:
            raise ValueError(
                "The actor checkpoint does not share provenance with the current "
                "VLA/RLT/stats; pairing the wrong models degrades results silently, "
                f"so startup is refused: {mismatch}"
            )
        logger.info("Actor checkpoint provenance check passed (trained at episode=%s)", resume_checkpoint.get("episode"))

    state_q01 = torch.tensor(stats["observation.state"]["q01"], dtype=torch.float32, device=device)
    state_q99 = torch.tensor(stats["observation.state"]["q99"], dtype=torch.float32, device=device)
    action_q01 = torch.tensor(stats["action"]["q01"], dtype=torch.float32, device=device)
    action_q99 = torch.tensor(stats["action"]["q99"], dtype=torch.float32, device=device)
    logger.info(f"State q01: {state_q01.tolist()}")
    logger.info(f"State q99: {state_q99.tolist()}")
    logger.info(f"Action q01: {action_q01.tolist()}")
    logger.info(f"Action q99: {action_q99.tolist()}")

    def normalize_state(state_deg: torch.Tensor) -> torch.Tensor:
        return quantile_normalize(state_deg, state_q01, state_q99)

    def normalize_action(action_deg: torch.Tensor) -> torch.Tensor:
        return quantile_normalize(action_deg, action_q01, action_q99)

    def unnormalize_action(action_norm: torch.Tensor) -> torch.Tensor:
        return quantile_unnormalize(action_norm, action_q01, action_q99)

    # ── Create the actor (inference needs the actor only, not critic/optimizer) ──
    actor = RLTChunkActor(
        rlt_config,
        hidden_dim=args.actor_hidden_dim,
        num_layers=args.actor_num_layers,
    ).to(device)
    actor.fixed_std = args.policy_fixed_std
    logger.info("Actor mode: paper full-output with reference pass-through and BC regularization")
    if resume_checkpoint is not None:
        actor.load_state_dict(resume_checkpoint["actor_state_dict"], strict=True)
        logger.info("Loaded trained Stage 2 actor from %s", args.resume_from)
    actor.eval()

    actor_params = sum(p.numel() for p in actor.parameters())
    # Deployment-recording handle: initialized before the try block so the finally
    # block can always reference it safely
    rollout_recorder: EpisodeDatasetRecorder | None = None
    logger.info("Actor: %s params, fixed std=%.4f", f"{actor_params:,}", args.policy_fixed_std)


    # The inference script has no replay buffer / journal / training checkpoint

    # Tokenizer is loaded and validated before any hardware objects are constructed.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    def tokenize_prompt(prompt_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="max_length",
            max_length=pi05_config.tokenizer_max_length,
            truncation=True,
        )
        return tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device).bool()

    # ── Validate camera and hardware contracts ──────────────────────────
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.teleoperators.so_leader.so_leader import SO101Leader

    try:
        cameras_dict = json.loads(args.cameras)
        if not isinstance(cameras_dict, dict):
            raise ValueError("--cameras must be a JSON object.")
        if set(args.camera_names) != set(cameras_dict):
            raise ValueError(
                f"--camera_names {args.camera_names} must exactly match --cameras keys {sorted(cameras_dict)}."
            )
        camera_feature_map = parse_camera_map(
            args.camera_map, checkpoint_image_features, set(cameras_dict), pad_missing=args.pad_missing_cameras,
        )
        max_relative_target = parse_max_relative_target(args.max_relative_target)
        actor_relative_limit = parse_max_relative_target(args.actor_max_relative_target)
    except ValueError as exc:
        logger.error("Startup preflight failed: %s", exc)
        return


    camera_configs = {}
    for cam_name, cam_cfg in cameras_dict.items():
        if cam_cfg.get("type") != "opencv":
            logger.error("Only OpenCV camera configs are supported; %s has type %r.", cam_name, cam_cfg.get("type"))
            return
        # fourcc / warmup_s must be passed through: with default parameters one
        # wrist camera reported opened=True but returned 0 frames; fourcc=MJPG plus
        # a 3 s warmup was stable in practice.
        camera_configs[cam_name] = OpenCVCameraConfig(
            index_or_path=cam_cfg["index_or_path"],
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30),
            fourcc=cam_cfg.get("fourcc"),
            warmup_s=cam_cfg.get("warmup_s", 1),
        )
    logger.info("Physical-to-checkpoint camera map: %s", camera_feature_map)

    if args.dry_run:
        logger.info("Dry-run preflight succeeded; no serial ports or cameras were opened.")
        return
    if max_relative_target is None:
        logger.warning(
            "Follower relative-target clipping is disabled. Commands still pass finite/shape validation, "
            "but no per-step degree clamp is applied."
        )

    # ── Connect to SO-101 ──────────────────────────────────────────────
    logger.info("Connecting to SO-101 arms...")
    console = ConsoleInputManager(auto_critical=bool(args.auto_critical))
    console.start()
    follower = None
    leader = None
    follower_connected = False
    leader_connected = False
    robot_faulted = False
    try:
        follower = SO101Follower(SO101FollowerConfig(
            port=args.follower_port,
            id=args.follower_id,
            cameras=camera_configs,
            max_relative_target=max_relative_target,
        ))
        leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))
        follower.connect()
        follower_connected = True
        leader.connect()
        leader_connected = True
        logger.info("Both arms connected; follower target delta limit: %s", max_relative_target)
    except Exception as e:
        logger.error("Failed to connect to robot: %s", e)
        if leader_connected:
            leader.disconnect()
        if follower_connected:
            follower.disconnect()
        return

    # ── Camera preflight: confirm every camera delivers fresh frames before the
    # ── arms move at all. A flaky wrist-camera USB link used to abort the whole
    # ── run during reset validation ("latest frame is too old 2959ms"), so the
    # ── cameras are checked separately here with actionable hints.
    if not args.dry_run:
        cameras = getattr(follower, "cameras", {}) or {}
        dead: list[str] = []
        for cam_index, (cam_name, cam) in enumerate(cameras.items(), 1):
            logger.info("[%d/%d] camera preflight %s: %s", cam_index, len(cameras), cam_name, cam)
            healthy = False
            for attempt in range(1, args.camera_preflight_retries + 1):
                try:
                    frame = cam.async_read()
                    logger.info(
                        "  [OK] %s delivers frames (shape=%s, attempt %d)",
                        cam_name, getattr(frame, "shape", "?"), attempt,
                    )
                    healthy = True
                    break
                except (TimeoutError, RuntimeError, OSError, ConnectionError) as exc:
                    logger.warning("  [WARN] %s frame attempt %d/%d failed: %s",
                                   cam_name, attempt, args.camera_preflight_retries, exc)
                    time.sleep(args.camera_preflight_delay_s)
            if not healthy:
                dead.append(cam_name)
        if dead:
            logger.error(
                "Camera preflight failed: %s returned no fresh frame -- the arms never "
                "moved. Check that (1) `v4l2-ctl --list-devices` / `lerobot-find-cameras` "
                "map index_or_path to the right device; (2) a different USB port/cable is "
                "used (avoid a hub or two cameras sharing one controller); (3) lowering "
                "cameras.<name>.fps from 30 to 15 reduces bandwidth. Then restart.",
                dead,
            )
            if leader_connected:
                leader.disconnect()
            if follower_connected:
                follower.disconnect()
            return

    teleop = None
    leader_mirror_stop: threading.Event | None = None
    leader_mirror_thread: threading.Thread | None = None
    leader_mirror_moving = False  # True while the leader mirror is actively driving the leader
    leader_mirror_heartbeat = 0.0  # last time the mirror thread beat (watchdog)
    visualization = None
    rerun_shutdown = None
    if args.display_data and not args.dry_run:
        try:
            from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
            import rerun as rr  # noqa: F401

            init_rerun(session_name=args.display_session_name)
            visualization = RerunVisualizationManager(
                log_rerun_data,
                compress_images=args.display_compressed_images,
                fps=args.display_fps,
            )
            visualization.start()
            logger.info(
                "Rerun visualization enabled (same path as lerobot-rollout): compress=%s, max %.1f FPS",
                args.display_compressed_images, args.display_fps,
            )
            rerun_shutdown = rr.rerun_shutdown
        except Exception as exc:
            logger.warning("Could not start Rerun visualization; continuing without it: %s", exc)
    try:
        # ── Configure TeleopManager before starting its polling thread ──────
        teleop = TeleopManager(
            leader,
            threshold=args.intervention_threshold,
            trigger_frames=args.intervention_trigger_frames,
            release_frames=args.intervention_release_frames,
            poll_hz=100.0,
        )

        # ── Joint name mapping ─────────────────────────────────────────────
        JOINT_NAMES = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]

        def action_dict_to_tensor(action: dict, source: str) -> torch.Tensor:
            missing = [f"{joint}.pos" for joint in JOINT_NAMES if f"{joint}.pos" not in action]
            if missing:
                raise ValueError(f"{source} action missing joints: {missing}")
            values = np.asarray([action[f"{joint}.pos"] for joint in JOINT_NAMES], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"{source} action contains non-finite values: {values.tolist()}")
            return torch.from_numpy(values)

        def leader_action_to_tensor(leader_action: dict) -> torch.Tensor:
            return action_dict_to_tensor(leader_action, "leader")

        def obs_to_state_tensor(obs: dict) -> torch.Tensor:
            state = {f"{joint}.pos": obs.get(f"{joint}.pos") for joint in JOINT_NAMES}
            return action_dict_to_tensor(state, "follower observation")

        def tensor_to_robot_action(action_tensor: torch.Tensor, source: str) -> dict:
            if not isinstance(action_tensor, torch.Tensor):
                raise TypeError(f"{source} target must be a torch.Tensor, got {type(action_tensor).__name__}")
            if tuple(action_tensor.shape) != (rlt_config.action_dim,):
                raise ValueError(
                    f"{source} target must have exact shape ({rlt_config.action_dim},), got {tuple(action_tensor.shape)}"
                )
            values = action_tensor.detach().float().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"{source} target contains non-finite values: {values.tolist()}")
            return {f"{JOINT_NAMES[i]}.pos": float(values[i]) for i in range(rlt_config.action_dim)}

        last_valid_target: dict[str, float] | None = None

        observation_faulted = False
        observation_fault_reason: str | None = None

        def read_follower_observation(context: str) -> dict | None:
            global SHUTDOWN
            nonlocal observation_faulted, observation_fault_reason, robot_faulted
            if observation_faulted:
                return None
            try:
                observation = follower.get_observation()
                if visualization is not None:
                    visualization.publish_observation(observation)
                return observation
            except (ConnectionError, OSError, RuntimeError, TimeoutError, ValueError, KeyError) as exc:
                # A single transient camera hiccup (OpenCVCamera latest frame too
                # old: 543ms) used to kill the whole session and waste many
                # episodes; retry a bounded number of times first. No robot command
                # is issued while retrying, and only a persistently missing
                # observation falls through to the original fatal path.
                for _attempt in range(1, args.observation_retries + 1):
                    time.sleep(args.observation_retry_delay_s)
                    try:
                        observation = follower.get_observation()
                        if visualization is not None:
                            visualization.publish_observation(observation)
                        logger.warning(
                            "Transient observation fault recovered after %d retries (%s): %s",
                            _attempt, context, exc,
                        )
                        return observation
                    except (ConnectionError, OSError, RuntimeError, TimeoutError, ValueError, KeyError) as retry_exc:
                        exc = retry_exc
                observation_faulted = True
                observation_fault_reason = f"{context}: {exc}"
                robot_faulted = True
                SHUTDOWN = True
                logger.error("FATAL follower observation fault (%s); stopping without further robot commands.", observation_fault_reason)
                return None

        def send_validated_action(action_tensor: torch.Tensor, source: str) -> torch.Tensor | None:
            nonlocal last_valid_target, robot_faulted
            if robot_faulted or observation_faulted or EMERGENCY_STOP:
                return None
            target = tensor_to_robot_action(action_tensor, source)
            for retry in range(3):
                try:
                    actual_target = follower.send_action(target)
                    if not isinstance(actual_target, dict):
                        raise RuntimeError(f"Follower returned invalid sent action: {actual_target!r}")
                    actual_tensor = action_dict_to_tensor(actual_target, f"{source} sent action")
                    last_valid_target = actual_target
                    max_delta = float((actual_tensor - action_tensor.detach().cpu()).abs().max())
                    if max_delta > 1e-4:
                        logger.warning("%s action clamped; max requested-vs-sent delta %.3f degrees.", source, max_delta)
                    if visualization is not None:
                        # The visualization manager consumes dict(action name -> degrees), same as lerobot's
                        visualization.publish_action(actual_target)
                    return actual_tensor
                except (ConnectionError, OSError, RuntimeError, ValueError, KeyError) as exc:
                    if retry == 2:
                        robot_faulted = True
                        logger.error("Follower command fault while sending %s: %s", source, exc)
                        return None
                    time.sleep(0.05)
            return None

        # Capture the leader/master pose once before background polling begins.
        startup_leader_reset_target: torch.Tensor | None = None
        try:
            startup_leader_reset_target = leader_action_to_tensor(leader.get_action())
            logger.info(
                "Captured startup leader/master reset target: %s",
                startup_leader_reset_target.tolist(),
            )
        except (ConnectionError, OSError, ValueError, KeyError) as exc:
            logger.warning("Could not capture startup leader/master reset target; automatic reset disabled: %s", exc)

        teleop.start()
        logger.info(
            "TeleopManager started: movement > %.2f° for %s samples requests HIL; "
            "%s still samples resume policy.",
            args.intervention_threshold, args.intervention_trigger_frames, args.intervention_release_frames,
        )

        def reset_follower_to_initial_pose() -> bool:
            nonlocal robot_faulted
            if not args.reset_between_episodes:
                logger.info("Reset skipped: --reset_between_episodes is disabled.")
                return False
            if startup_leader_reset_target is None:
                logger.warning("Reset skipped: startup leader/master reset target was unavailable.")
                return False
            if robot_faulted or observation_faulted or EMERGENCY_STOP:
                logger.warning("Reset skipped: robot/observation fault or emergency stop is active.")
                return False

            target = startup_leader_reset_target
            for reset_pass in range(1, args.reset_max_passes + 1):
                current_obs = read_follower_observation(f"follower reset pass {reset_pass}")
                if current_obs is None:
                    return False
                current = obs_to_state_tensor(current_obs)
                initial_error = float((target - current).abs().max())
                logger.info(
                    "Reset pass %s/%s | max error %.2f degrees | %s steps at %.2fs",
                    reset_pass, args.reset_max_passes, initial_error, args.reset_steps, args.reset_dt,
                )
                if initial_error <= args.reset_tolerance_deg:
                    logger.info("Reset already within %.2f degree tolerance.", args.reset_tolerance_deg)
                    return True

                for i in range(1, args.reset_steps + 1):
                    interpolated = current + (target - current) * (i / args.reset_steps)
                    if send_validated_action(interpolated, "RESET") is None:
                        return False
                    if last_valid_target is not None:
                        mirror_follower_to_leader(last_valid_target)
                    time.sleep(args.reset_dt)

                final_obs = read_follower_observation(f"follower reset verification {reset_pass}")
                if final_obs is None:
                    return False
                final_state = obs_to_state_tensor(final_obs)
                final_error = float((target - final_state).abs().max())
                if final_error <= args.reset_tolerance_deg:
                    logger.info("Reset completed | final max error %.2f degrees.", final_error)
                    return True
                logger.warning(
                    "Reset pass %s incomplete | final max error %.2f > tolerance %.2f.",
                    reset_pass, final_error, args.reset_tolerance_deg,
                )

            logger.warning("Reset failed after %s passes; refusing to claim the initial pose was restored.", args.reset_max_passes)
            return False

        def hold_follower_at_current_pose(context: str) -> None:
            """Episode-end hold for manual-positioning / no-reset setups.

            Re-commands the follower to its current (last commanded) pose so the
            arm is EXPLICITLY locked in place while the operator judges the
            episode and then teleoperates the next start pose ('p'). Best-effort:
            a failed hold only logs a warning and never escalates to a fault.
            """
            if last_valid_target is None:
                return
            try:
                hold_target = action_dict_to_tensor(last_valid_target, "hold")
            except ValueError as exc:
                logger.warning("Hold command skipped (%s): %s", context, exc)
                return
            actual = send_validated_action(hold_target, "HOLD")
            if actual is not None:
                logger.info(
                    "Holding follower at current pose (%s); waiting for 'p' teleop to the next start pose.",
                    context,
                )

        # ── π0.5 official policy + processors ─────────────────────────────
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference

        preprocessor, _postprocessor = make_pre_post_processors(
            pi05_config,
            pretrained_path=str(pi05_path),
        )

        def run_pi05_inference(
            obs: dict,
            state_tensor: torch.Tensor,
            *,
            prev_chunk_left_over: torch.Tensor | None = None,
            inference_delay: int = 0,
            raw_actions_for_embedding: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # SO-101 follower returns per-joint floats and camera frames keyed by
            # physical camera name (e.g. "top", "wrist"). π0.5 expects features
            # keyed as in its config: "observation.images.top", etc. Use
            # camera_feature_map to remap image keys, and wrap scalar joint
            # values as ndarrays so prepare_observation_for_inference's
            # torch.from_numpy() doesn't choke on bare Python floats.
            raw_obs = {}
            for k, v in obs.items():
                if k in camera_feature_map:
                    raw_obs[camera_feature_map[k]] = v
                elif isinstance(v, (int, float)):
                    raw_obs[k] = np.asarray(float(v), dtype=np.float32)
                else:
                    raw_obs[k] = v
            raw_obs["observation.state"] = state_tensor.cpu().numpy()
            prepared = prepare_observation_for_inference(raw_obs, device, task=args.task)
            processed = preprocessor(prepared)

            with torch.no_grad():
                if raw_actions_for_embedding is not None:
                    # Replay materialization path: the boundary's actions were
                    # already cached by the live inference. z_rl only depends on
                    # the prefix embedding (computed from these same actions), so
                    # skip the expensive diffusion (10 denoising LM steps) and
                    # recompute z_rl from the cached actions directly.
                    raw_actions_full = raw_actions_for_embedding.to(device=device, dtype=torch.float32)
                    if raw_actions_full.ndim == 2:
                        raw_actions_full = raw_actions_full.unsqueeze(0)
                else:
                    pi05_kwargs = {}
                    if args.rtc_execution_horizon > 0:
                        pi05_kwargs = {
                            "prev_chunk_left_over": prev_chunk_left_over,
                            "inference_delay": inference_delay,
                            "execution_horizon": args.rtc_execution_horizon,
                        }
                    raw_actions_full = pi05_policy.predict_action_chunk(
                        processed,
                        **pi05_kwargs,
                    ).float()

                # Extract image-only embeddings using same processed batch.
                # _preprocess_images lives on the policy wrapper (PI05Policy),
                # while extract_embeddings lives on the underlying PI05Pytorch.
                # Pad actions to max_action_dim (32) for extract_embeddings
                B, T, D = raw_actions_full.shape
                padded_actions = torch.zeros(B, T, pi05_config.max_action_dim, device=device, dtype=raw_actions_full.dtype)
                padded_actions[:, :, :D] = raw_actions_full
                images, img_masks = pi05_policy._preprocess_images(processed)
                tokens = processed["observation.language.tokens"]
                token_masks = processed["observation.language.attention_mask"]
                prefix_out, _, prefix_mask = extract_pi05_embeddings(
                    pi05_policy, images, img_masks, tokens, token_masks, padded_actions,
                    chunk_size=pi05_config.chunk_size,
                    max_action_dim=pi05_config.max_action_dim,
                    image_only=True,
                )
                z_rl = encoder(prefix_out.float(), mask=prefix_mask)

            return raw_actions_full, z_rl


        # ── Deploy loop ─────────────────────────────────────────────────────
        logger.info("\nStarting deployment inference, up to %s episode(s) (press q to quit)...", args.max_episodes)
        logger.info(
            "Actor takeover condition: critical phase sustained for %s steps AND 'a' enabled "
            "(deployment mode has no update-count threshold)",
            args.actor_critical_delay_steps,
        )
        logger.info(
            "Automatic HIL: move any leader joint to align it to the follower and enter direct-follow; "
            "keep the leader still to resume policy.\n"
        )
        logger.info("Press 'c' when the critical phase begins; only then is RLT replay recorded.")
        logger.info("Press 'a' during episode to toggle Actor mode ON/OFF (starts OFF).")
        logger.info("Press 'g' during episode to skip to next episode.")

        episode_rewards: list[float] = []
        episode_interventions: list[int] = []
        inference_scores: list[bool] = []  # y/n success counts (deployment mode, never learned from)
        # Deployment restarts episode counting at 1 and does not inherit the episode
        # number stored in the checkpoint (dataset recording uses episode_index from 0).
        start_episode = 1
        if resume_checkpoint is not None:
            logger.info(
                "Loaded the actor checkpoint (trained at episode=%s, updates=%s); deployment starts counting from a new episode.",
                resume_checkpoint.get("episode"), resume_checkpoint.get("total_updates"),
            )
        # Rolling RL metrics (Q estimates, TD error, ...) for periodic logging.

        # ── Tail-friendly metrics log (separate file, e.g. `tail -f rl_metrics.log`) ──
        # Columns: batch Q is the mean over the 256-sample batch (dominated by
        # mid-episode bootstrap rows -> stays low); q*_max is the max over the
        # batch (≈ terminal/rewarded rows -> should approach γ^(C-1) ≈ 0.91);
        # q1_rew is the mean Q on the rewarded (terminal success) rows only.
        # actor_on = 1 means the RL Actor actually took over execution in the latest
        # episode (intervention); 0 means pure VLA / no intervention.
        # Non-finite values (no data) are written as empty strings instead of nan so
        # the TSV curves stay parseable.
        target_dt = 1.0 / args.control_fps

        # ── Asynchronous training worker ─────────────────────────────
        # Paper Algorithm 1 performs rollouts and learning asynchronously.
        # The main thread keeps collecting real-robot episodes; this worker
        # continuously samples replay and updates actor/critic in the background.

        # Reset to the captured leader/master pose before the first episode.
        human_active = False
        positioning_active = False  # True only during the pre-episode positioning phase (NOT HIL)
        takeover = None
        # Leader mirror: a background thread drives the powered leader to the
        # follower's commanded pose so the master mirrors the slave during
        # autonomous execution and resets (takeover then starts from an aligned
        # leader). It runs continuously (also during π0.5 inference pauses) and
        # releases leader torque the moment a takeover is requested, so a human
        # grab never fights the mirror motor. Bus access is serialized with the
        # 100 Hz teleop poller via teleop._bus_lock.
        leader_mirror_lock = threading.Lock()
        leader_mirror_target: dict[str, float] | None = None
        leader_mirror_stop = threading.Event()
        leader_mirror_enabled = False
        leader_hold_active = False  # True while the leader holds its own pose (HIL mode)

        def mirror_follower_to_leader(target: dict) -> None:
            """Publish the follower's commanded pose for the mirror thread.

            With --no-leader_mirror this returns immediately: the leader never moves
            and stays free. HIL takeover still works (it aligns poses via hil_offset
            and does not depend on continuous mirroring).
            """
            nonlocal leader_mirror_target
            if not args.leader_mirror:
                return
            with leader_mirror_lock:
                leader_mirror_target = dict(target)

        def _leader_soft_torque_on() -> None:
            """Enable leader torque at the configured soft limit (safe to call repeatedly)."""
            if not leader_mirror_enabled and not leader_hold_active:
                leader.enable_torque()
                try:
                    torque_limit = int(max(0.0, min(1.0, args.leader_mirror_torque_limit)) * 1000)
                    for motor in leader.bus.motors:
                        leader.bus.write("Torque_Limit", motor, torque_limit)
                except Exception as exc:
                    logger.warning(
                        "Leader Torque_Limit write failed (mirror continues at full torque; "
                        "takeover may feel stiff): %s", exc,
                    )

        def leader_mirror_loop() -> None:
            nonlocal leader_mirror_enabled, leader_mirror_moving, leader_mirror_heartbeat, leader_hold_active
            mirror_hz = max(args.control_fps, 20.0)
            mirror_dt = 1.0 / mirror_hz
            ramp_alpha = 0.35    # exponential smoothing toward the commanded pose
            written: torch.Tensor | None = None   # last pose actually commanded (smoothed, clamped)
            leader_limits: dict[str, tuple[float, float]] | None = None
            last_status_log = 0.0
            while not leader_mirror_stop.is_set() and not SHUTDOWN and not EMERGENCY_STOP:
                try:
                    now = time.monotonic()
                    leader_mirror_heartbeat = now
                    if (
                        human_active
                        or positioning_active
                        or (takeover is not None and takeover.pending)
                        or teleop.takeover_request_pending
                        or HIL_TOGGLE_REQUESTED
                    ):
                        # HIL HOLD MODE: keep the soft torque on and command the
                        # leader to its own current pose (goal = present). The arm
                        # never drops (torque stays on) and never fights the human
                        # (the goal follows wherever the hand pushes it).
                        try:
                            with teleop._bus_lock:
                                _leader_soft_torque_on()
                                present = leader_action_to_tensor(leader.get_action())
                                leader.send_feedback(tensor_to_robot_action(present, "leader HIL hold"))
                        except (ConnectionError, OSError, ValueError, KeyError) as exc:
                            logger.warning("Leader HIL hold command failed (continuing): %s", exc)
                        leader_mirror_enabled = False
                        leader_hold_active = True
                        leader_mirror_moving = False
                        time.sleep(mirror_dt)
                        continue
                    with leader_mirror_lock:
                        target = leader_mirror_target
                    if target is None:
                        leader_mirror_moving = False
                        time.sleep(mirror_dt)
                        continue
                    try:
                        with teleop._bus_lock:
                            if leader_hold_active:
                                # Leaving HIL: torque is already on (hold mode); switch
                                # back to mirroring from the leader's current pose.
                                leader_hold_active = False
                                leader_mirror_enabled = False
                                written = None
                            if not leader_mirror_enabled:
                                _leader_soft_torque_on()
                                leader_mirror_enabled = True
                                # Start the ramp from the leader's current pose so a
                                # large first target does not jerk the leader.
                                written = leader_action_to_tensor(leader.get_action())
                            goal_t = action_dict_to_tensor(target, "leader mirror target")
                            if written is None:
                                written = goal_t
                            else:
                                written = written + (goal_t - written) * ramp_alpha
                            # Clamp the commanded pose to the leader's reachable
                            # range: commanding an unreachable pose would make the
                            # leader sit at its limit and the deviation check would
                            # mistake that saturation for a human grab.
                            if leader_limits is None:
                                try:
                                    cal = getattr(leader.bus, "calibration", None) or {}
                                    lims = {}
                                    for motor in leader.bus.motors:
                                        c = cal.get(motor)
                                        if c is None:
                                            lims = {}
                                            break
                                        res = leader.bus.model_resolution_table[leader.bus.motors[motor].model] - 1
                                        half = (c.range_max - c.range_min) / 2.0 * 360.0 / res
                                        lims[motor] = (-half, half)
                                    if len(lims) == len(leader.bus.motors):
                                        leader_limits = lims
                                        logger.info(
                                            "Leader mirror reachable range (deg): %s",
                                            {k: (round(v[0], 1), round(v[1], 1)) for k, v in lims.items()},
                                        )
                                except Exception as exc:
                                    logger.warning("Leader mirror limit clamp unavailable: %s", exc)
                                    leader_limits = None
                            if leader_limits is not None:
                                vals = written.tolist()
                                for idx, name in enumerate(JOINT_NAMES):
                                    lim = leader_limits.get(name)
                                    if lim is None:
                                        break
                                    vals[idx] = min(lim[1], max(lim[0], vals[idx]))
                                written = torch.tensor(vals, dtype=written.dtype)
                            leader.send_feedback(tensor_to_robot_action(written, "leader mirror"))
                            leader_mirror_moving = True
                    except (ConnectionError, OSError, ValueError, KeyError) as exc:
                        logger.warning("Leader mirror command failed (continuing): %s", exc)
                        time.sleep(mirror_dt)
                        continue
                    if now - last_status_log >= 10.0:
                        last_status_log = now
                        logger.info(
                            "Mirror status: enabled=%s hold=%s moving=%s",
                            leader_mirror_enabled, leader_hold_active, leader_mirror_moving,
                        )
                except Exception as exc:
                    logger.exception("Leader mirror loop iteration failed (continuing): %s", exc)
                time.sleep(mirror_dt)

        leader_mirror_thread = threading.Thread(
            target=leader_mirror_loop,
            name="leader-mirror",
            daemon=True,
        )
        leader_mirror_thread.start()
        logger.info("Leader mirror thread started (master mirrors slave at %.0f Hz).", max(args.control_fps, 20.0))

        if not EMERGENCY_STOP and not robot_faulted and not args.manual_positioning:
            reset_follower_to_initial_pose()

        # ── Rollout recording (deployment only): episode_index starts at 0 ──
        if args.record_dataset:
            default_root = Path(__file__).resolve().parent.parent / "outputs" / "deploy_recordings"
            record_name = args.record_repo_id.split("/")[-1] or "rollouts"
            record_root = Path(args.record_root).expanduser() if args.record_root else default_root / record_name
            # A non-empty directory gets a numeric suffix so every run records from 0
            if not args.dry_run:
                base_root, suffix = record_root, 0
                while record_root.exists() and any(record_root.iterdir()):
                    suffix += 1
                    record_root = Path(f"{base_root}_v{suffix}")
                if suffix:
                    logger.warning("Recording directory %s already exists; using %s", base_root, record_root)
            # Physical camera names (keys in follower.get_observation()) -> checkpoint
            # feature keys, e.g. {"top": "observation.images.top", "wrist": "observation.images.wrist"}
            camera_key_map = {name: camera_feature_map[name] for name in sorted(camera_feature_map)}
            camera_shapes = {
                name: (int(cameras_dict[name].get("height", 480)),
                       int(cameras_dict[name].get("width", 640)), 3)
                for name in camera_key_map
            }
            rollout_recorder = EpisodeDatasetRecorder(
                repo_id=args.record_repo_id,
                root=record_root,
                fps=args.control_fps,
                camera_map=camera_key_map,
                camera_shapes=camera_shapes,
                state_names=list(JOINT_NAMES),
                task=args.record_task or args.task,
                use_videos=bool(args.record_videos),
                record_hil=bool(args.record_hil_steps),
                streaming_encoding=bool(args.record_streaming_encoding),
                encoder_threads=int(args.record_encoder_threads),
                dry_run=bool(args.dry_run),
            )
            logger.info("Recording to: %s (obs key -> dataset key: %s, shapes: %s, hil=%s)",
                        record_root, camera_key_map, camera_shapes, args.record_hil_steps)

        # Critical-phase classifier: optional background logging of z_rl/proprio
        # samples with manual 'c' boundaries, and optional auto-detection that
        # toggles CRITICAL_PHASE_ACTIVE from a trained checkpoint.
        # Key frames at critical-phase on/off transitions (auto or manual),
        # saved so the detector's timing can be reviewed offline.
        critical_frames_dir: Path | None = None
        if args.save_critical_frames:
            critical_frames_dir = (
                Path(args.critical_frames_dir).expanduser()
                if args.critical_frames_dir else output_dir / "critical_frames"
            )
            critical_frames_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Critical transition key frames will be saved to %s", critical_frames_dir)
        critical_detector = None
        if args.auto_critical:
            from lerobot.policies.pi05_rlt.critical import CriticalPhaseDetector
            critical_detector = CriticalPhaseDetector(
                args.critical_classifier,
                device,
                threshold_on=args.auto_critical_threshold_on,
                threshold_off=args.auto_critical_threshold_off,
                smooth_steps=args.auto_critical_smooth_steps,
                min_on_chunks=args.auto_critical_min_on_chunks,
            )
            logger.info(
                "Auto critical-phase detection active (on>=%.2f, off<=%.2f, smooth=%d chunks).",
                args.auto_critical_threshold_on, args.auto_critical_threshold_off,
                args.auto_critical_smooth_steps,
            )

        for episode in range(start_episode, args.max_episodes + 1):
            if SHUTDOWN or EMERGENCY_STOP:
                break

            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {episode}/{args.max_episodes}")
            logger.info(f"{'='*60}")
            # Clean console: only the episode counter + key echoes + prompts.
            print(f"\nEpisode {episode}/{args.max_episodes}", flush=True)

            # Debug observation dump for inspection (images + per-step state/action).
            debug_dir = None
            debug_steps: list[dict] = []
            if args.save_debug_obs and not args.dry_run:
                debug_dir = output_dir / "debug_obs" / f"episode_{episode:04d}"
                try:
                    debug_dir.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    debug_dir = None
                if debug_dir is not None:
                    logger.info("Debug observations will be saved to %s", debug_dir)

            teleop.reset_episode()
            SKIP_EPISODE = False  # Reset skip flag at start of episode
            FINISH_EPISODE = False  # Reset early-finish flag at start of episode
            CRITICAL_PHASE_ACTIVE = False
            # NOTE: POSITION_TOGGLE_REQUESTED is intentionally NOT cleared here.
            # A 'p' pressed during the previous episode's tail (y/n prompt,
            # replay finalization, running episode) is STICKY and is honored by
            # the next positioning phase below, so an operator press is never
            # silently swallowed.
            _last_critical_seen = False  # per-episode baseline; 'c' boundaries are detected by changes
            _last_z_rl = None  # cached z_rl from the previous chunk (early auto-detection)
            _auto_fired_once = False  # single critical segment per episode: auto ON happened
            _critical_sustained_steps = 0  # consecutive critical steps (actor takeover guard)
            intervention_steps = 0
            human_active = False
            hil_terminal_requested = False

            # ── Execution state ────────────────────────────────────────────
            action_cache = Pi05ActionCache()
            takeover = TakeoverState()

            # Actor per-step clamp accounting (reported in the episode summary).
            actor_clamp_max_delta = 0.0

            # Per-step counters (critical-phase gating + debug logging).
            episode_step_counter = 0
            critical_step_counter: int | None = None

            # ── Pre-episode manual positioning (NOT HIL) ─────────────────────
            # When the straw is fixed to the follower, the start pose is no
            # longer a fixed reset target. With --manual_positioning the
            # operator teleoperates the follower to the desired start pose
            # before the VLA episode starts:
            #   press 'p' once  -> enter direct-follow (leader controls follower)
            #   press 'p' again -> finish positioning, start the VLA episode
            #   (press 'p' twice with no movement = confirm current pose)
            # This phase deliberately runs OUTSIDE the episode data collection:
            # it touches no StepRecords, no critical-phase counters, no
            # transitions, no intervention flags and no reward, so it can never
            # be confused with (or leak into) HIL intervention replay data.
            positioning_active = False
            if args.manual_positioning:
                POSITIONING_STATE = "waiting"
                if not POSITION_TOGGLE_REQUESTED:
                    logger.info(
                        "Episode %s: follower is HOLDING at the end-of-episode pose. "
                        "Press 'p' to take manual control and move it to the desired "
                        "start pose; press 'p' again to start the VLA episode. "
                        "Positioning is NOT HIL and never enters replay. "
                        "('e'=estop, 'q'=quit)",
                        episode,
                    )
                    # Phase 0: wait for the first 'p' (operator may estop/quit meanwhile).
                    while not SHUTDOWN and not EMERGENCY_STOP and not POSITION_TOGGLE_REQUESTED:
                        time.sleep(0.05)
                    if SHUTDOWN or EMERGENCY_STOP:
                        POSITIONING_STATE = "off"
                        break
                else:
                    logger.info(
                        "Episode %s: a 'p' press from the previous episode tail was "
                        "latched; entering positioning immediately.",
                        episode,
                    )
                POSITION_TOGGLE_REQUESTED = False

                # Phase 1: direct-follow until the second 'p'.
                leader_now = teleop.latest_sample
                pos_obs = read_follower_observation("positioning entry")
                if pos_obs is None or leader_now is None or not leader_now.is_fresh(max_age_s=0.25):
                    logger.error(
                        "Positioning entry failed (no fresh leader/follower sample); "
                        "starting the episode without positioning."
                    )
                    POSITIONING_STATE = "off"
                else:
                    try:
                        follower_state = obs_to_state_tensor(pos_obs).to(device)
                        leader_pose = leader_action_to_tensor(leader_now.action)
                    except ValueError as exc:
                        logger.error("Positioning entry sample invalid: %s", exc)
                        POSITIONING_STATE = "off"
                    else:
                        pos_offset = follower_state.detach().cpu() - leader_pose.detach().cpu()
                        pos_last_sequence = leader_now.sequence - 1
                        positioning_active = True
                        POSITIONING_STATE = "active"
                        pos_step_start = time.time()
                        pos_status_cycle = 0
                        logger.info(
                            "Positioning active: follower mirrors leader + offset %s; "
                            "press 'p' to finish.",
                            pos_offset.tolist(),
                        )
                        while positioning_active and not SHUTDOWN and not EMERGENCY_STOP:
                            if POSITION_TOGGLE_REQUESTED:
                                POSITION_TOGGLE_REQUESTED = False
                                positioning_active = False
                                break
                            sample = teleop.wait_for_fresh_sample(
                                max_age_s=0.25,
                                after_sequence=pos_last_sequence,
                                timeout_s=max(0.1, target_dt * 2),
                            )
                            if sample is None:
                                time.sleep(0.005)
                                continue
                            pos_last_sequence = sample.sequence
                            try:
                                target_deg = leader_action_to_tensor(sample.action) + pos_offset
                            except ValueError as exc:
                                logger.error("Invalid positioning leader target: %s", exc)
                                break
                            actual_target = send_validated_action(target_deg, "POSITION")
                            if actual_target is None:
                                SHUTDOWN = True
                                break
                            pos_status_cycle += 1
                            if pos_status_cycle % 50 == 0:
                                logger.info(
                                    "Positioning live: leader seq=%d | leader[deg]=%s | "
                                    "sent[deg]=%s",
                                    sample.sequence,
                                    [round(float(v), 2) for v in leader_action_to_tensor(sample.action).tolist()],
                                    [round(float(v), 2) for v in actual_target.detach().cpu().tolist()],
                                )
                            remaining_dt = target_dt - (time.time() - pos_step_start)
                            if remaining_dt > 0:
                                time.sleep(remaining_dt)
                            pos_step_start = time.time()
                        positioning_active = False
                        POSITIONING_STATE = "off"
                        # Re-anchor the leader mirror to the positioned follower pose
                        # so the master mirrors the slave again after positioning.
                        if last_valid_target is not None:
                            mirror_follower_to_leader(last_valid_target)
                        if SHUTDOWN or EMERGENCY_STOP:
                            break
                        logger.info(
                            "Positioning finished; starting the VLA episode from the positioned pose."
                        )

            # Calculate steps per block: use actor_execution_steps if set, otherwise full chunk
            steps_per_block = rlt_config.n_action_steps_rl
            if args.actor_execution_steps > 0:
                steps_per_block = min(args.actor_execution_steps, rlt_config.n_action_steps_rl)
            max_rollout_blocks = args.steps_per_episode // steps_per_block
            if max_rollout_blocks < 1:
                raise ValueError(
                    "--steps_per_episode must be at least steps_per_block so every replay row is real."
                )
            if args.steps_per_episode % rlt_config.n_action_steps_rl:
                logger.warning(
                    "Ignoring final %d control steps so every replay transition has a full real horizon.",
                    args.steps_per_episode % rlt_config.n_action_steps_rl,
                )
            # Synchronized HIL takeover offset (follower - leader, degrees). Set at
            # each takeover; HUMAN targets are leader_pose + hil_offset so the
            # follower mirrors the human from its current pose without a freeze.
            hil_offset = torch.zeros(rlt_config.action_dim)

            for step in range(max_rollout_blocks):
                if SHUTDOWN or EMERGENCY_STOP or SKIP_EPISODE or FINISH_EPISODE:
                    break
                step_start = time.time()
                takeover.latch(teleop.consume_takeover_request())

                if not human_active and (takeover.pending or HIL_TOGGLE_REQUESTED):
                    hil_toggle_entry = HIL_TOGGLE_REQUESTED
                    HIL_TOGGLE_REQUESTED = False
                    if hil_toggle_entry:
                        logger.info("HIL mode toggle: entering human control (explicit switch).")
                    if not teleop.leader_is_fresh(max_age_s=0.25):
                        logger.warning("HIL request remains pending until a fresh leader sample arrives.")
                    else:
                        logger.info("HIL takeover pending and leader fresh; entering human control.")
                        # Synchronized HIL entry: the leader and follower move
                        # together from the first instant. Instead of freezing the
                        # follower while aligning the powered leader, record the
                        # takeover offset follower - leader and enter direct-follow
                        # immediately: every HUMAN target is leader_pose + offset,
                        # so the follower mirrors the human from its current pose
                        # with no jump and no freeze.
                        leader_now = teleop.latest_sample
                        if leader_now is None or not leader_now.is_fresh(max_age_s=0.25):
                            logger.error("No fresh leader sample at HIL handoff; refusing synchronized takeover.")
                            takeover.handoff_finished(False)
                            continue
                        handoff_obs = read_follower_observation("HIL handoff")
                        if handoff_obs is None:
                            break
                        leader_pose = leader_action_to_tensor(leader_now.action)
                        hil_offset = obs_to_state_tensor(handoff_obs).detach().cpu() - leader_pose.detach().cpu()
                        takeover.last_leader_sequence = leader_now.sequence - 1
                        # Entering HIL: the mirror thread switches to hold mode
                        # (goal = present, soft torque) so the leader neither drops
                        # nor fights the human. No torque-off here.
                        human_active = True
                        teleop.enter_human_mode(args.leader_hil_min_dwell_s)
                        action_cache.clear()
                        takeover.handoff_finished(True)
                        logger.info(
                            "Automatic HIL active (synchronized, follower mirrors leader + offset %s)",
                            hil_offset.tolist(),
                        )

                force_resume = HIL_FORCE_RESUME or HIL_TOGGLE_REQUESTED
                if force_resume:
                    HIL_FORCE_RESUME = False
                    HIL_TOGGLE_REQUESTED = False
                    logger.info("HIL force-resume by operator ('s'/'h' key); exiting human control.")
                if human_active and (teleop.consume_resume_request() or force_resume):
                    human_active = False
                    teleop.exit_human_mode()
                    action_cache.clear()
                    # Interventions are corrections: exit HIL and continue the
                    # episode. The terminal reward is judged at the normal
                    # episode-end prompt (y/n/d/q), not at HIL exit.
                    logger.info("HIL exited; policy replans and resumes (episode continues).")

                if hil_terminal_requested:
                    break

                # ── HUMAN direct-follow: one independent window per chunk ──
                if human_active:

                    # A manual 'c' pressed during HIL is also a transition:
                    # record its key frame (best effort) for offline review.
                    if CRITICAL_PHASE_ACTIVE != _last_critical_seen:
                        direction = "on" if CRITICAL_PHASE_ACTIVE else "off"
                        if critical_frames_dir is not None:
                            hil_obs = read_follower_observation("critical frame (HIL)")
                            save_critical_transition(
                                critical_frames_dir,
                                list(camera_feature_map),
                                episode=episode,
                                step=episode_step_counter,
                                direction=direction,
                                source="manual",
                                prob=None,
                                obs=hil_obs,
                            )
                            logger.info(
                                "[KEY] Critical phase transition saved (HIL): ep%d step%d %s (manual)",
                                episode, episode_step_counter, direction,
                            )
                        _last_critical_seen = CRITICAL_PHASE_ACTIVE

                    human_actions: list[torch.Tensor] = []
                    for _ in range(rlt_config.n_action_steps_rl):
                        sample = teleop.wait_for_fresh_sample(
                            max_age_s=0.25,
                            after_sequence=takeover.last_leader_sequence,
                            timeout_s=max(0.1, target_dt * 2),
                        )
                        if sample is None or not takeover.accept_leader_sequence(sample.sequence):
                            logger.warning("Stopping HUMAN chunk: no fresh unexecuted leader sample arrived.")
                            break
                        try:
                            # Synchronized HIL: follower tracks leader + takeover
                            # offset so both arms move together with no jump.
                            target_deg = leader_action_to_tensor(sample.action) + hil_offset
                        except ValueError as exc:
                            logger.error("Invalid leader target: %s", exc)
                            SHUTDOWN = True
                            break
                        if CRITICAL_PHASE_ACTIVE and critical_step_counter is None:
                            critical_step_counter = 0
                        actual_target = send_validated_action(target_deg, "HUMAN")
                        if actual_target is None:
                            SHUTDOWN = True
                            break
                        # Deployment recording: also record every human-in-the-loop (HIL)
                        # step, otherwise the trajectory has a gap. To avoid slowing the
                        # human takeover down, the extra observation read happens only when
                        # recording is on and record_hil is set.
                        if rollout_recorder is not None and rollout_recorder.record_hil:
                            step_obs = read_follower_observation("human step recording")
                            if step_obs is not None:
                                rollout_recorder.add_step(step_obs, actual_target, source="HUMAN")
                        elif rollout_recorder is not None:
                            rollout_recorder.note_skipped_step()
                        normalized_action = normalize_action(actual_target.to(device))
                        human_actions.append(normalized_action)
                        intervention_steps += 1
                        if CRITICAL_PHASE_ACTIVE:
                            if critical_step_counter is None:
                                # CRITICAL_PHASE_ACTIVE is toggled by the console
                                # thread ('c') and may flip mid-chunk, so the
                                # top-of-iteration init may not have run yet.
                                critical_step_counter = 0
                            critical_step_counter += 1
                        episode_step_counter += 1
                        remaining_dt = target_dt - (time.time() - step_start)
                        if remaining_dt > 0:
                            time.sleep(remaining_dt)
                        step_start = time.time()

                    continue

                # ── Autonomous boundary inference and execution ──
                _auto_triggered = False  # per-boundary: did the early auto-detector fire?
                obs = read_follower_observation("autonomous pre-action")
                if obs is None:
                    break
                if debug_dir is not None and CRITICAL_PHASE_ACTIVE:
                    try:
                        import cv2 as _cv2
                        for _cam in camera_feature_map:
                            _img = obs.get(_cam)
                            if _img is not None:
                                # Camera frames are RGB; cv2.imwrite expects BGR.
                                _frame = np.asarray(_img)
                                if _frame.ndim == 3 and _frame.shape[2] == 3:
                                    _frame = _cv2.cvtColor(_frame, _cv2.COLOR_RGB2BGR)
                                _cv2.imwrite(str(debug_dir / f"obs_{episode_step_counter:05d}_{_cam}.jpg"), _frame)
                    except Exception as _exc:
                        logger.warning("Debug image save failed: %s", _exc)
                try:
                    state_tensor = obs_to_state_tensor(obs).to(device)
                    # Early detection BEFORE the (blocking) inference: use the
                    # previous chunk's cached z_rl + current proprio so the
                    # auto-detector fires before the inference stall, removing
                    # ~0.5-1 s of detection lag (the boundary inference waits
                    # for the diffusion to finish).
                    # Gating: skip the start-of-episode delay window, and once
                    # the single critical segment has run its course (auto ON ->
                    # OFF), do not auto-ON again in this episode.
                    _auto_detection_armed = (
                        critical_detector is not None
                        and episode_step_counter >= args.auto_critical_start_delay_steps
                        and not (
                            args.auto_critical_once
                            and _auto_fired_once
                            and not CRITICAL_PHASE_ACTIVE
                        )
                        and time.time() - CRITICAL_MANUAL_OVERRIDE_AT >= args.auto_critical_override_cooldown
                    )
                    if _auto_detection_armed and _last_z_rl is not None:
                        desired = critical_detector.update(
                            _last_z_rl,
                            normalize_state(state_tensor)[:rlt_config.state_dim].detach().cpu().numpy(),
                        )
                        if desired is not None and desired != CRITICAL_PHASE_ACTIVE:
                            CRITICAL_PHASE_ACTIVE = desired
                            _auto_triggered = True
                            if desired:
                                _auto_fired_once = True
                            logger.info(
                                "[KEY] Auto critical phase %s (early, P(critical)=%.3f)",
                                "STARTED" if desired else "STOPPED",
                                critical_detector.last_probability,
                            )
                    # --replan_every_window (default ON): force a fresh 50-step
                    # plan at EVERY 10-step window for BOTH the frozen VLA and
                    # the RL actor, so the reference chunk is always sampled
                    # from the CURRENT observation (paper Algorithm 1:
                    # a~t:t+C-1 ~ pi_vla(st)). Cost: each replan runs the full
                    # 10-step flow-matching diffusion of the 3B model (~1-5 s),
                    # so the robot moves 0.33 s then waits for the next plan —
                    # the stop-and-go cadence. Pass
                    # --no-replan_every_window to keep the cached 50-step chunk
                    # and replan only when it is exhausted (lerobot-record
                    # cadence: smooth, but references become chunk-continuation).
                    if args.replan_every_window:
                        action_cache.clear()
                    # Replan only when the cached VLA chunk can no longer serve
                    # the next 10-step window (≈ every 50 control steps = 1.67 s
                    # at 30 Hz) — i.e. full-chunk execution, exactly like
                    # lerobot-record. Replanning at every window (every 0.33 s)
                    # forced each boundary through the full 10-step flow-matching
                    # diffusion of the 3B model (~1-5 s), so pure-VLA motion froze
                    # between chunks (0.33 s of motion followed by 1-5 s of waiting). While the
                    # cached chunk still covers the window, reuse its raw actions
                    # and only recompute z_rl for the CURRENT observation
                    # (extract_embeddings = 1 cheap LM forward; the image-only
                    # prefix hidden states depend on the images, not on the
                    # action values, so any aligned sub-chunk works).
                    # The leftover-reuse path relies on extract_embeddings accepting
                    # action blocks shorter than chunk_size (the vla_compat noise and
                    # the embed_suffix attention-mask length were fixed accordingly),
                    # so Stage 2 keeps its original logic here.
                    needs_replan = (
                        action_cache.raw_actions is None
                        or action_cache.next_index + rlt_config.n_action_steps_rl
                        > action_cache.raw_actions.shape[1]
                    )
                    if needs_replan:
                        raw_actions, z_rl = run_pi05_inference(
                            obs, state_tensor, prev_chunk_left_over=None, inference_delay=0,
                        )
                        action_cache.refresh(raw_actions.detach(), z_rl.detach(), rlt_config.action_dim)
                    else:
                        cached_leftover = action_cache.remaining_raw_actions()
                        _, z_rl = run_pi05_inference(
                            obs, state_tensor,
                            raw_actions_for_embedding=cached_leftover,
                        )
                except (KeyError, ValueError, RuntimeError) as exc:
                    # Print the full stack: the message alone does not locate a tensor
                    # size mismatch (a "size of tensor a (50) must match tensor b (40)"
                    # error was hidden that way once).
                    logger.error("π0.5 autonomous boundary inference failed: %s", exc)
                    logger.error("Full traceback:\n%s", traceback.format_exc())
                    SHUTDOWN = True
                    break
                vla_actions, action_index, _ = action_cache.take(rlt_config.n_action_steps_rl)
                _last_z_rl = z_rl.detach().cpu().numpy().squeeze()

                # ── Critical-phase classifier bookkeeping (auto-detection + logging) ──
                auto_triggered = _auto_triggered
                if (
                    _auto_detection_armed
                    and critical_detector is not None
                    and not auto_triggered
                    and time.time() - CRITICAL_MANUAL_OVERRIDE_AT >= args.auto_critical_override_cooldown
                ):
                    # First chunk of the episode (no cached z_rl yet): fall back
                    # to a post-inference check with the fresh z_rl.
                    desired = critical_detector.update(
                        _last_z_rl,
                        normalize_state(state_tensor)[:rlt_config.state_dim].detach().cpu().numpy(),
                    )
                    if desired is not None and desired != CRITICAL_PHASE_ACTIVE:
                        CRITICAL_PHASE_ACTIVE = desired
                        auto_triggered = True
                        if desired:
                            _auto_fired_once = True
                        logger.info(
                            "[KEY] Auto critical phase %s (P(critical)=%.3f)",
                            "STARTED" if desired else "STOPPED",
                            critical_detector.last_probability,
                        )
                if CRITICAL_PHASE_ACTIVE != _last_critical_seen:
                    direction = "on" if CRITICAL_PHASE_ACTIVE else "off"
                    source = "auto" if auto_triggered else "manual"
                    if critical_frames_dir is not None:
                        save_critical_transition(
                            critical_frames_dir,
                            list(camera_feature_map),
                            episode=episode,
                            step=episode_step_counter,
                            direction=direction,
                            source=source,
                            prob=critical_detector.last_probability if auto_triggered else None,
                            obs=obs,
                        )
                        logger.info(
                            "[KEY] Critical phase transition saved: ep%d step%d %s (%s)",
                            episode, episode_step_counter, direction, source,
                        )
                    _last_critical_seen = CRITICAL_PHASE_ACTIVE

                # Paper Eq. (5): reference = first C contiguous VLA actions so the
                # executed chunk and the BC target share the same time steps.
                ref_actions = make_pi05_reference_chunk(
                    action_cache.raw_actions,
                    action_dim=rlt_config.action_dim,
                    start_index=action_index,
                    num_steps=rlt_config.n_action_steps_rl,
                )
                proprio_norm = normalize_state(state_tensor)[:rlt_config.state_dim].detach()
                # Consecutive critical-phase CONTROL steps: the actor may only
                # take control after --actor_critical_delay_steps control steps
                # of sustained critical phase, so transient auto-detector false
                # positives cannot grab control from VLA. The counter advances by
                # one window (n_action_steps_rl = 10 control steps) per boundary.
                _critical_sustained_steps = (
                    _critical_sustained_steps + rlt_config.n_action_steps_rl if CRITICAL_PHASE_ACTIVE else 0
                )
                if (
                    not CRITICAL_PHASE_ACTIVE
                    or _critical_sustained_steps < args.actor_critical_delay_steps
                    or args.vla_only
                    or not ACTOR_ENABLED
                ):
                    requested_actions = vla_actions
                    action_source = "VLA"
                else:
                    with torch.no_grad():
                        action_mean, _actor_std = actor(
                            z_rl, proprio_norm.unsqueeze(0), ref_actions,
                        )
                        # Deployment: deterministic execution (use the mean); exploration
                        # noise is added only when actor_noise_std > 0.
                        requested_actions = action_mean
                        if args.actor_noise_std > 0:
                            requested_actions = requested_actions + (
                                torch.randn_like(requested_actions) * args.actor_noise_std
                            )
                        if args.actor_execution_scale != 1.0:
                            requested_actions = requested_actions * args.actor_execution_scale
                    action_source = "ACTOR"
                    # Log residual correction magnitude
                    if actor.is_residual:
                        residual = (requested_actions - ref_actions).abs().mean().item()
                        logger.debug("Actor residual correction: %.4f (scale=%.3f)", residual, actor._residual_scale)

                window_executed_steps = 0
                takeover_during_chunk = False
                # Determine how many steps to execute before re-planning
                steps_to_execute = rlt_config.n_action_steps_rl
                if action_source == "ACTOR" and args.actor_execution_steps > 0:
                    steps_to_execute = min(args.actor_execution_steps, rlt_config.n_action_steps_rl)
                for action_offset in range(steps_to_execute):
                    takeover.latch(teleop.consume_takeover_request())
                    if takeover.pending:
                        takeover_during_chunk = True
                        # Don't clear cache — keep raw actions for RTC
                        # guidance on resume; just invalidate the cursor
                        # so the next autonomous block replans.
                        break
                    if CRITICAL_PHASE_ACTIVE and critical_step_counter is None:
                        critical_step_counter = 0
                    target_deg = unnormalize_action(requested_actions[0, action_offset])
                    # ACTOR-only per-step angle limit, relative to the last
                    # commanded pose. VLA / HUMAN / RESET are never clamped here
                    # (the driver-level max_relative_target is the only all-source
                    # clamp, and it is normally disabled).
                    if action_source == "ACTOR" and actor_relative_limit is not None and last_valid_target is not None:
                        ref = action_dict_to_tensor(last_valid_target, "actor clamp reference").to(target_deg.device)
                        clamped = target_deg.clone()
                        for _joint_idx, _joint_name in enumerate(JOINT_NAMES):
                            _lim = actor_relative_limit.get(_joint_name)
                            if _lim is None:
                                continue
                            _lo = ref[_joint_idx] - _lim
                            _hi = ref[_joint_idx] + _lim
                            clamped[_joint_idx] = torch.clamp(target_deg[_joint_idx], _lo, _hi)
                        if not torch.equal(clamped, target_deg):
                            _clamp_delta = float((clamped - target_deg).abs().max())
                            actor_clamp_max_delta = max(actor_clamp_max_delta, _clamp_delta)
                            logger.debug("Actor action clamped by per-step limit (max delta %.1f deg).",
                                         _clamp_delta)
                        target_deg = clamped
                    actual_target = send_validated_action(target_deg, action_source)
                    if actual_target is None:
                        SHUTDOWN = True
                        break
                    # Deployment recording: one frame per execution step, with its own
                    # observation. ``obs`` was read once at the chunk boundary and would
                    # otherwise be written for all steps of the chunk, so the recorded video
                    # would repeat the same image until the next re-plan (chunk-length
                    # duplicates) and the actions would not line up with their images.
                    if rollout_recorder is not None and not SHUTDOWN:
                        step_obs = read_follower_observation("deployment step recording")
                        rollout_recorder.add_step(step_obs if step_obs is not None else obs, actual_target)
                    if debug_dir is not None and CRITICAL_PHASE_ACTIVE:
                        debug_steps.append({
                            "step": episode_step_counter,
                            "source": action_source,
                            "state": obs_to_state_tensor(obs).tolist(),
                            "target": target_deg.detach().cpu().tolist(),
                            "sent": actual_target.detach().cpu().tolist(),
                        })
                    if last_valid_target is not None:
                        mirror_follower_to_leader(last_valid_target)
                    window_executed_steps += 1
                    if CRITICAL_PHASE_ACTIVE:
                        if critical_step_counter is None:
                            # CRITICAL_PHASE_ACTIVE is toggled by the console
                            # thread ('c') and may flip mid-chunk, so the
                            # top-of-iteration init may not have run yet.
                            critical_step_counter = 0
                        critical_step_counter += 1
                    episode_step_counter += 1
                    remaining_dt = target_dt - (time.time() - step_start)
                    if remaining_dt > 0:
                        time.sleep(remaining_dt)
                    step_start = time.time()
                if SHUTDOWN:
                    break
                # Receding-horizon ACTOR execution (partial windows) clears the
                # cache so the next boundary forces a fresh plan. In the default
                # --replan_every_window mode the top-of-boundary clear already
                # forces a fresh plan at every window; in --no-replan_every_window
                # (full-chunk) mode this is what keeps a partial actor window from
                # reusing the old plan's suffix.
                if action_source == "ACTOR" and args.actor_execution_steps > 0 and window_executed_steps < rlt_config.n_action_steps_rl:
                    action_cache.clear()

                if takeover_during_chunk:
                    # The request stays latched; the next iteration performs the handoff.
                    continue

                # Execute this chunk in full; if VLA inference slows us down and the
                # chunk cannot keep up, end the episode early.
                if window_executed_steps < steps_to_execute:
                    break

                if step % 10 == 0:
                    logger.info(
                        "  Step %s/%s | Source: %s | Interventions so far: %s",
                        step, max_rollout_blocks, action_source, intervention_steps,
                    )
            # ── End of rollout blocks ───────────────────────────────────────
            # Reset per-episode HIL/mirror state so the leader mirror re-engages
            # (torque on, hold) during the reward prompt, replay finalization and
            # the between-episode reset. A stale unconsumed takeover request or an
            # episode ending mid-HIL would otherwise leave the leader torque-off
            # and it would drop.
            if not SHUTDOWN and not EMERGENCY_STOP:
                human_active = False
                takeover = TakeoverState()
                teleop.clear_takeover_request()
                # When no auto reset will happen (manual positioning on, or
                # reset explicitly disabled), explicitly lock the follower at its
                # current pose and wait for the operator's 'p' teleop instead of
                # leaving the arm to hold implicitly.
                if args.manual_positioning or not args.reset_between_episodes:
                    hold_follower_at_current_pose("episode end")
            # Write per-step debug records (state / target / sent / source).
            if debug_dir is not None and debug_steps:
                try:
                    with open(debug_dir / "steps.csv", "w") as _f:
                        _f.write("step,source,state,target,sent\n")
                        for _rec in debug_steps:
                            _f.write(f"{_rec['step']},{_rec['source']},"
                                     f"{_rec['state']},{_rec['target']},{_rec['sent']}\n")
                except OSError as _exc:
                    logger.warning("Debug steps CSV write failed: %s", _exc)
            # Materialize deferred takeover/HUMAN boundaries only after the

            # ── End of episode ──────────────────────────────────────────────
            if SKIP_EPISODE:
                SKIP_EPISODE = False
                logger.info("Episode skipped by user (press 'g').")
                if not EMERGENCY_STOP and not robot_faulted and not args.manual_positioning:
                    reset_follower_to_initial_pose()
                continue
            if FINISH_EPISODE:
                FINISH_EPISODE = False
                logger.info("Episode %s finished early by user (press 'f').", episode)

            # ── Inference-only: no replay recording, no reward scoring, no checkpoint ──
            if args.inference_only:
                keep_episode = True
                if (
                    args.inference_score
                    and not SHUTDOWN and not EMERGENCY_STOP and not robot_faulted
                ):
                    # Reuse the console stdin thread (y/n/d/q) so the key listener never competes for input
                    score_reward, should_continue, score_discard = console.prompt_episode_reward()
                    keep_episode = not score_discard
                    if not should_continue:
                        if rollout_recorder is not None:
                            rollout_recorder.save_episode(keep=keep_episode)
                        break
                    if not score_discard:
                        inference_scores.append(score_reward > 0)
                        logger.info(
                            "Inference success rate: %.0f%% (%s/%s)",
                            100 * sum(inference_scores) / len(inference_scores),
                            sum(inference_scores), len(inference_scores),
                        )
                # Deployment recording: flush this episode at the end (discarded ones are cleared)
                if rollout_recorder is not None:
                    rollout_recorder.save_episode(keep=keep_episode)
                if not EMERGENCY_STOP and not robot_faulted and not args.manual_positioning:
                    reset_follower_to_initial_pose()
                continue

        # ── Cleanup ─────────────────────────────────────────────────────────
        # Stop the async trainer before saving checkpoints so state_dicts are stable.
        leader_mirror_stop.set()
        leader_mirror_thread.join(timeout=2.0)

        teleop.stop()
        if leader_connected:
            try:
                leader.disable_torque()
            except Exception as exc:
                logger.warning("Leader torque release during cleanup failed: %s", exc)

        if EMERGENCY_STOP and not robot_faulted and last_valid_target is not None:
            logger.warning("EMERGENCY STOP — no additional trajectory commands will be issued.")
        elif robot_faulted:
            logger.warning("Robot session faulted; skipping hold/reset commands during cleanup.")
        elif args.reset_between_episodes and not args.manual_positioning:
            logger.info("Returning follower to the captured initial follower pose.")
            reset_follower_to_initial_pose()

        logger.info("Disconnecting from robot...")
        if leader_connected:
            try:
                leader.disconnect()
            except Exception as exc:
                logger.warning("Leader disconnect failed: %s", exc)
        if follower_connected:
            try:
                follower.disconnect()
            except Exception as exc:
                logger.warning("Follower disconnect failed: %s", exc)

        with open(output_dir / "deploy_summary.json", "w") as f:
            json.dump({
                "episodes": len(inference_scores),
                "scored_success": int(sum(inference_scores)),
                "episode_rewards": episode_rewards,
                "episode_interventions": episode_interventions,
                "recorded_episodes": None if rollout_recorder is None else rollout_recorder.episodes_saved,
                "record_root": None if rollout_recorder is None else str(rollout_recorder.root),
            }, f, indent=2)

        total_episodes = len(episode_rewards)
        total_interventions = sum(episode_interventions)
        logger.info("\nDeploy run complete.")
        logger.info("  Episodes: %s", total_episodes)
        logger.info("  Total intervention steps: %s", total_interventions)
        if inference_scores:
            logger.info(
                "  Final success rate (y/n scoring): %.0f%% (%s/%s)",
                100 * sum(inference_scores) / len(inference_scores),
                sum(inference_scores), len(inference_scores),
            )
        if rollout_recorder is not None:
            logger.info("  Recorded episodes: %s -> %s",
                        rollout_recorder.episodes_saved, rollout_recorder.root)
        logger.info("  Logs: %s", output_dir)
        print(f"\nDeploy run complete! Episodes: {total_episodes} | "
              f"Success: "
              f"{100 * sum(inference_scores) / len(inference_scores) if inference_scores else 0:.0f}% "
              f"({sum(inference_scores)}/{len(inference_scores)})",
              flush=True)
        if rollout_recorder is not None:
            rollout_recorder.finalize()
    finally:
        if rollout_recorder is not None:
            try:
                rollout_recorder.finalize()   # write meta/video even on an abnormal exit
            except Exception as exc:
                logger.warning("Recording finalization failed: %s", exc)
        if visualization is not None:
            try:
                visualization.stop()
            except Exception as exc:
                logger.warning("Rerun visualization cleanup failed: %s", exc)
        if rerun_shutdown is not None:
            try:
                rerun_shutdown()
            except Exception as exc:
                logger.warning("Rerun shutdown failed: %s", exc)
        if teleop is not None:
            try:
                teleop.stop()
            except Exception as exc:
                logger.warning("Teleop cleanup failed: %s", exc)
        if leader_mirror_stop is not None:
            leader_mirror_stop.set()
            if leader_mirror_thread is not None:
                try:
                    leader_mirror_thread.join(timeout=2.0)
                except Exception as exc:
                    logger.warning("Leader mirror cleanup failed: %s", exc)
        console.stop()
        if leader_connected and leader is not None:
            try:
                leader.disable_torque()
            except Exception as exc:
                logger.warning("Leader torque release during final cleanup failed: %s", exc)
            try:
                leader.disconnect()
            except Exception as exc:
                logger.warning("Leader final disconnect failed: %s", exc)
        if follower_connected and follower is not None:
            try:
                follower.disconnect()
            except Exception as exc:
                logger.warning("Follower final disconnect failed: %s", exc)


if __name__ == "__main__":
    main()
