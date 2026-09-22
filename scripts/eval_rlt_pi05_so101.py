#!/usr/bin/env python3
"""Evaluate a residual PI0.5 RLT Actor on an SO-101 without training."""

import argparse
import json
import logging
import math
import signal
import sys
import threading
import time
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






def signal_handler(sig, frame):
    global SHUTDOWN
    logger.info("Shutdown signal received. Finishing current episode...")
    SHUTDOWN = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures for PI0.5 RLT evaluation
# ─────────────────────────────────────────────────────────────────────────────















































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
        self._latest_sample: LeaderSample | None = None
        self._sequence = 0
        self._takeover_requested = False
        self._resume_requested = False
        self._motion_count = 0
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
            self._motion_count = 0
            self._prev = None

    def exit_human_mode(self) -> None:
        with self._lock:
            self._human_mode = False
            self._takeover_requested = False
            self._resume_requested = False
            self._motion_count = 0
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
            self._motion_count = 0
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
        current = {f"{joint}.pos": float(action[f"{joint}.pos"]) for joint in self.JOINT_NAMES}
        if self._prev is None:
            self._prev = current
            return
        moved = any(abs(current[key] - self._prev[key]) > self.threshold for key in current)
        self._prev = current
        if moved:
            now = time.monotonic()
            self._motion_count += 1
            self._still_count = 0
            self._resume_requested = False
            if self._human_mode:
                self._stillness_not_before = now + self._human_stillness_guard_s
            if not self._human_mode and self._motion_count >= self.trigger_frames:
                self._takeover_requested = True
            return
        self._motion_count = 0
        if self._human_mode and time.monotonic() >= self._stillness_not_before:
            self._still_count += 1
            if self._still_count >= self.release_frames:
                self._resume_requested = True


# ─────────────────────────────────────────────────────────────────────────────
# Human feedback
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# Quantile normalization helpers (π0.5 uses QUANTILES mode)
# ─────────────────────────────────────────────────────────────────────────────





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
    to 1 (contiguous).
    """
    indices = start_index + torch.arange(num_steps, device=actions.device) * stride
    indices = indices.clamp_max(actions.shape[1] - 1)
    return actions.index_select(1, indices)[:, :, :action_dim]








# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing for π0.5 (SigLIP expects 224x224, [-1, 1])
# ─────────────────────────────────────────────────────────────────────────────



































# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate paper-aligned PI0.5 RLT Actor on SO-101")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file; explicit CLI arguments override its values")
    parser.add_argument("--pi05_path", required=True)
    parser.add_argument("--rlt_checkpoint", required=True)
    parser.add_argument("--actor_checkpoint", required=True)
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--stats_path", default=None)
    parser.add_argument("--task", required=True)
    parser.add_argument("--follower_port", required=True)
    parser.add_argument("--follower_id", default="so101_follower")
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--camera_map", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps_per_episode", type=int, default=300)
    parser.add_argument("--control_fps", type=float, default=30.0)
    parser.add_argument("--max_relative_target", default=None,
                        help="Optional physical joint-target delta limit; omitted disables clipping")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtc_execution_horizon", type=int, default=10)
    parser.add_argument("--actor_execution_steps", type=int, default=0,
                        help="Steps to execute before re-planning (receding horizon). "
                             "Lower = smoother but slower. 0 = execute full chunk.")
    parser.add_argument("--replan_every_window", action=argparse.BooleanOptionalAction, default=True,
                        help="Replan a fresh 50-step VLA inference at EVERY 10-step window so the "
                             "actor reference comes from the current observation (paper Algorithm 1). "
                             "Cost: the full 10-step diffusion runs every 0.33 s. Disable with "
                             "--no-replan_every_window to reuse the cached 50-step chunk (smooth, "
                             "lerobot-record cadence, chunk-continuation references).")
    parser.add_argument("--actor_hidden_dim", type=int, default=256,
                        help="Hidden width of the paper-aligned actor MLP")
    parser.add_argument("--actor_num_layers", type=int, default=2,
                        help="Number of MLP layers in the actor (2 or 3)")
    parser.add_argument("--critic_hidden_dim", type=int, default=256,
                        help="Hidden width of the paper-aligned critic MLP")
    parser.add_argument("--critic_num_layers", type=int, default=2,
                        help="Number of MLP layers in the critic (2 or 3)")
    parser.add_argument("--rtc_prefix_attention_schedule", choices=["EXP", "LINEAR", "ONES", "ZEROS"], default="EXP")
    parser.add_argument("--rtc_max_guidance_weight", type=float, default=5.0)
    parser.add_argument("--display_data", action="store_true")
    parser.add_argument("--display_compressed_images", action="store_true")
    parser.add_argument("--display_fps", type=float, default=10.0)
    parser.add_argument("--score_episodes", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=str, default=None)
    config_args, _ = config_probe.parse_known_args()
    if config_args.config:
        config_path = Path(config_args.config).expanduser()
        with open(config_path, encoding="utf-8") as file:
            config_values = json.load(file)
        if not isinstance(config_values, dict):
            parser.error(f"Config must contain a JSON object: {config_path}")
        actions = {action.dest: action for action in parser._actions}
        unknown = sorted(set(config_values) - set(actions))
        if unknown:
            parser.error(f"Unknown config keys: {', '.join(unknown)}")
        for key in ("cameras", "camera_map"):
            if isinstance(config_values.get(key), dict):
                config_values[key] = json.dumps(config_values[key])
        if config_values.get("max_relative_target") is not None and not isinstance(
            config_values["max_relative_target"], str
        ):
            config_values["max_relative_target"] = json.dumps(config_values["max_relative_target"])
        parser.set_defaults(**config_values)
        for key in config_values:
            actions[key].required = False
    return parser.parse_args()






# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.episodes < 1 or args.steps_per_episode < 1 or args.control_fps <= 0 or args.display_fps <= 0:
        raise ValueError("episodes, steps_per_episode, control_fps, and display_fps must be positive")
    if args.actor_execution_steps < 0:
        raise ValueError("--actor_execution_steps must be non-negative (0 = full actor chunk)")
    if args.rtc_execution_horizon not in (0, 10):
        raise ValueError("--rtc_execution_horizon must be 0 or match the 10-step RLT chunk")
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05_rlt.vla_compat import (
        extract_embeddings as extract_pi05_embeddings,
    )
    from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
    from lerobot.policies.pi05_rlt.modeling_pi05_rlt import RLTokenEncoder, RLTChunkActor
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower

    device = torch.device(args.device)
    pi05_path = Path(args.pi05_path).expanduser()
    actor_path = Path(args.actor_checkpoint).expanduser()
    if not (pi05_path / "config.json").is_file() or not (pi05_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete PI0.5 checkpoint: {pi05_path}")
    actor_checkpoint = torch.load(actor_path, map_location=device, weights_only=False)
    if (
        actor_checkpoint.get("schema_version") != 3
        or actor_checkpoint.get("actor_contract") != "paper_full_output_v1"
        or actor_checkpoint.get("rl_chunk_length") != 10
    ):
        raise ValueError("--actor_checkpoint must be a paper-aligned full-output Actor checkpoint")
    provenance = actor_checkpoint.get("runtime_provenance", {})
    if provenance.get("task") != args.task or provenance.get("camera_map") != args.camera_map:
        raise ValueError("Actor checkpoint task or camera map does not match this evaluation")
    if provenance.get("actor_residual_scale", 0.0) != 0:
        raise ValueError("Paper-aligned Actor checkpoint must use full-output mode")
    if "action_stride" in provenance and provenance["action_stride"] != 2:
        raise ValueError("Actor checkpoint action_stride must be 2 for paper-aligned stride-2 replay")
    if "actor_hidden_dim" in provenance and provenance["actor_hidden_dim"] != args.actor_hidden_dim:
        raise ValueError("Actor checkpoint actor_hidden_dim does not match this evaluation")
    if "actor_num_layers" in provenance and provenance["actor_num_layers"] != args.actor_num_layers:
        raise ValueError("Actor checkpoint actor_num_layers does not match this evaluation")
    if "critic_hidden_dim" in provenance and provenance["critic_hidden_dim"] != args.critic_hidden_dim:
        raise ValueError("Actor checkpoint critic_hidden_dim does not match this evaluation")
    if "critic_num_layers" in provenance and provenance["critic_num_layers"] != args.critic_num_layers:
        raise ValueError("Actor checkpoint critic_num_layers does not match this evaluation")
    # Match Stage 2 loading: from_pretrained parses input/output feature entries
    # into PolicyFeature objects, while PI05Config(**json) can leave raw dicts.
    policy = PI05Policy.from_pretrained(pretrained_name_or_path=str(pi05_path))
    pi05_config = policy.config
    pi05_config.rtc_config = RTCConfig(
        enabled=args.rtc_execution_horizon > 0,
        prefix_attention_schedule=RTCAttentionSchedule(args.rtc_prefix_attention_schedule),
        execution_horizon=max(args.rtc_execution_horizon, 1),
        max_guidance_weight=args.rtc_max_guidance_weight,
    )
    policy.config = pi05_config
    policy = policy.to(device).eval()
    policy.init_rtc_processor()
    for parameter in policy.parameters(): parameter.requires_grad = False
    stage1 = torch.load(Path(args.rlt_checkpoint).expanduser(), map_location=device, weights_only=False)
    if stage1.get("rlt_architecture") != "paper_v1" or not stage1.get("image_only", False):
        raise ValueError("--rlt_checkpoint must be an image-only paper_v1 Stage 1 checkpoint")
    rlt_config = PI05RLTConfig(
        mode="online_rl", state_dim=6, action_dim=6,
        action_stride=2, n_action_steps_rl=10,
    )
    encoder = RLTokenEncoder(rlt_config).to(device).eval()
    encoder.load_state_dict(stage1["encoder_state_dict"], strict=True)
    actor = RLTChunkActor(
        rlt_config,
        hidden_dim=args.actor_hidden_dim,
        num_layers=args.actor_num_layers,
    ).to(device).eval()
    actor.load_state_dict(actor_checkpoint["actor_state_dict"], strict=True)
    preprocessor, postprocessor = make_pre_post_processors(
        pi05_config,
        pretrained_path=str(pi05_path),
    )
    cameras_raw = json.loads(args.cameras)
    feature_names = [key for key in pi05_config.input_features if key.startswith("observation.images.")]
    camera_map = parse_camera_map(args.camera_map, feature_names, set(cameras_raw))

    # The actor was trained on quantile-normalized proprioception; the raw
    # follower state must be normalized with the same stats before it is passed
    # to the actor (the preprocessor normalizes the state used by π0.5 itself).
    stats, stats_source = load_normalization_stats(pi05_path, args.stats_path)
    state_q01 = torch.tensor(stats["observation.state"]["q01"], dtype=torch.float32, device=device)
    state_q99 = torch.tensor(stats["observation.state"]["q99"], dtype=torch.float32, device=device)
    logger.info("Evaluation normalization stats: %s", stats_source)

    def normalize_state(state_deg: torch.Tensor) -> torch.Tensor:
        return quantile_normalize(state_deg, state_q01, state_q99)

    if args.dry_run:
        logger.info("Evaluation dry-run succeeded; no hardware opened.")
        return
    camera_configs = {name: OpenCVCameraConfig(index_or_path=cfg["index_or_path"], width=cfg.get("width", 640), height=cfg.get("height", 480), fps=cfg.get("fps", 30)) for name, cfg in cameras_raw.items()}
    follower = SO101Follower(SO101FollowerConfig(port=args.follower_port, id=args.follower_id, cameras=camera_configs, max_relative_target=parse_max_relative_target(args.max_relative_target)))
    visualization = None
    try:
        if args.display_data:
            from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
            init_rerun(session_name="eval_rlt_pi05")
            visualization = RerunVisualizationManager(log_rerun_data, args.display_compressed_images, args.display_fps)
            visualization.start()
        follower.connect()
        # Keep the policy/robot order in one shared source of truth.  LeRobot's
        # feature dictionaries may expose the same names in another order.
        joint_names = list(SO101_JOINT_NAMES)
        def run_pi05_inference(
            obs_dict: dict,
            state_tensor: torch.Tensor,
            *,
            prev_chunk_left_over: torch.Tensor | None = None,
            raw_actions_for_embedding: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # SO-101 follower returns per-joint floats and camera frames keyed by
            # physical camera name (e.g. "top", "wrist"). π0.5 expects features
            # keyed as in its config: "observation.images.*". Remap image keys and
            # wrap scalar joints as ndarrays so prepare_observation_for_inference's
            # torch.from_numpy() does not choke on bare Python floats.
            raw_obs = {}
            for k, v in obs_dict.items():
                if k in camera_map:
                    raw_obs[camera_map[k]] = v
                elif isinstance(v, (int, float)):
                    raw_obs[k] = np.asarray(float(v), dtype=np.float32)
                else:
                    raw_obs[k] = v
            raw_obs["observation.state"] = state_tensor.cpu().numpy()
            prepared = prepare_observation_for_inference(raw_obs, device, task=args.task)
            processed = preprocessor(prepared)
            with torch.no_grad():
                if raw_actions_for_embedding is not None:
                    # Cache-reuse path: the chunk was already diffused by the
                    # live inference; skip the expensive diffusion and only
                    # recompute z_rl for the current observation (the image-only
                    # prefix hidden states do not depend on the action values).
                    raw_actions_full = raw_actions_for_embedding.to(device=device, dtype=torch.float32)
                    if raw_actions_full.ndim == 2:
                        raw_actions_full = raw_actions_full.unsqueeze(0)
                else:
                    pi05_kwargs = {}
                    if args.rtc_execution_horizon > 0:
                        pi05_kwargs = {
                            "prev_chunk_left_over": prev_chunk_left_over,
                            "inference_delay": 0,
                            "execution_horizon": args.rtc_execution_horizon,
                        }
                    raw_actions_full = policy.predict_action_chunk(
                        processed,
                        **pi05_kwargs,
                    ).float()
                # Pad actions to max_action_dim (32) for extract_embeddings
                B, T, D = raw_actions_full.shape
                padded_actions = torch.zeros(B, T, pi05_config.max_action_dim, device=device, dtype=raw_actions_full.dtype)
                padded_actions[:, :, :D] = raw_actions_full
                # _preprocess_images lives on the PI05Policy wrapper, not on the
                # inner PI05Pytorch model.
                images, img_masks = policy._preprocess_images(processed)
                tokens = processed["observation.language.tokens"]
                token_masks = processed["observation.language.attention_mask"]
                prefix_out, _, prefix_mask = extract_pi05_embeddings(
                    policy, images, img_masks, tokens, token_masks, padded_actions,
                    chunk_size=pi05_config.chunk_size,
                    max_action_dim=pi05_config.max_action_dim,
                    image_only=True,
                )
                z_rl = encoder(prefix_out.float(), mask=prefix_mask)
            return raw_actions_full, z_rl
        def send(action_tensor: torch.Tensor) -> None:
            target = {f"{name}.pos": float(action_tensor[i]) for i, name in enumerate(joint_names)}
            actual = follower.send_action(target)
            if visualization: visualization.publish_action(actual)
        successes = []
        execution_steps = (
            rlt_config.n_action_steps_rl
            if args.actor_execution_steps <= 0
            else args.actor_execution_steps
        )
        if execution_steps > rlt_config.n_action_steps_rl:
            raise ValueError(
                "--actor_execution_steps cannot exceed the actor chunk length "
                f"({rlt_config.n_action_steps_rl})"
            )
        for episode in range(1, args.episodes + 1):
            action_cache = Pi05ActionCache()
            for _ in range(args.steps_per_episode // execution_steps):
                obs = follower.get_observation()
                if visualization: visualization.publish_observation(obs)
                current_state = torch.tensor([obs[f"{name}.pos"] for name in joint_names], dtype=torch.float32, device=device).unsqueeze(0)
                # Replan only when the cached VLA chunk can no longer serve the
                # next window (≈ every 50 control steps = 1.67 s at 30 Hz) —
                # full-chunk execution exactly like lerobot-record. Replanning
                # every window (every 0.33 s) ran the full 10-step diffusion of
                # the 3B model per window (~1-5 s), freezing the robot between
                # chunks. While the cache covers the window, reuse its raw
                # actions and only recompute z_rl for the current observation
                # (1 cheap LM forward; image-only prefix states do not depend on
                # the action values).
                # --replan_every_window (default ON): force a fresh 50-step plan
                # at every 10-step window so the actor reference comes from the
                # current observation (paper Algorithm 1). --no-replan_every_window
                # keeps the cached chunk and replans only when exhausted.
                if args.replan_every_window:
                    action_cache.clear()
                needs_replan = (
                    action_cache.raw_actions is None
                    or action_cache.next_index + execution_steps
                    > action_cache.raw_actions.shape[1]
                )
                if needs_replan:
                    raw_actions, z_rl = run_pi05_inference(
                        obs,
                        current_state,
                        prev_chunk_left_over=None,
                    )
                    action_cache.refresh(raw_actions.detach(), z_rl.detach(), rlt_config.action_dim)
                else:
                    cached_leftover = action_cache.remaining_raw_actions()
                    _, z_rl = run_pi05_inference(
                        obs,
                        current_state,
                        raw_actions_for_embedding=cached_leftover,
                    )
                _, action_index, _ = action_cache.take(execution_steps)
                # Paper Eq. (5): reference = C contiguous VLA actions matching
                # the executed steps (sub-chunk of the current plan at
                # action_index), consistent with the chunk-continuation
                # references the actor was trained on.
                raw_ref = make_pi05_reference_chunk(
                    action_cache.raw_actions,
                    action_dim=rlt_config.action_dim,
                    start_index=action_index,
                    num_steps=rlt_config.n_action_steps_rl,
                )
                # The actor was trained on quantile-normalized proprioception.
                actor_actions, _ = actor(z_rl, normalize_state(current_state), raw_ref)
                # The postprocessor takes the raw action tensor directly and returns
                # the unnormalized action chunk.
                final_actions = postprocessor(actor_actions)
                for action in final_actions[0, :execution_steps]:
                    send(action.detach().cpu())
                    time.sleep(1 / args.control_fps)
                # Receding-horizon actor execution (partial windows) forces a
                # fresh plan on the next iteration by clearing the cache, so the
                # remaining suffix of the old plan is never reused.
                if execution_steps < rlt_config.n_action_steps_rl:
                    action_cache.clear()
            if args.score_episodes:
                answer = input(f"Episode {episode} successful? [y/n/q]: ").strip().lower()
                if answer.startswith("q"): break
                successes.append(answer.startswith("y"))
                logger.info("Success rate: %.0f%% (%s/%s)", 100 * sum(successes) / len(successes), sum(successes), len(successes))
    finally:
        if visualization: visualization.stop()
        if follower.is_connected: follower.disconnect()

if __name__ == "__main__":
    main()
