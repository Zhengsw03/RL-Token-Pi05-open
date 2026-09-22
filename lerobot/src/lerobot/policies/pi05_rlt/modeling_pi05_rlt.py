"""π0.5 with RL Token (RLT) — core model implementation.

Implements the RLT paper (Physical Intelligence) for π0.5:
  Stage 1: Train encoder-decoder to extract z_rl from frozen π0.5 embeddings
  Stage 2: Train lightweight actor-critic using z_rl for online RL

Architecture (paper Eq. (1)-(2)):
  Frozen π0.5 → append learned <rl> token → encoder → z_rl → Actor/Critic
  Stage 1 decoder teacher-forces stopped-gradient VLA tokens with causal attention.
"""

import copy
import logging
from collections import deque
from functools import lru_cache

import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pi05_rlt.vla_compat import (
    extract_embeddings as extract_pi05_embeddings,
)
from lerobot.policies.pi05_rlt.vla_compat import resolve_pi05_utils
from lerobot.utils.constants import ACTION

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RLT Encoder: append one learned RL token to the frozen VLA token sequence
# ─────────────────────────────────────────────────────────────────────────────


class RLTokenEncoder(nn.Module):
    """Paper Eq. (1): encode ``[z_1, ..., z_M, e_rl]`` and return position M+1."""

    def __init__(self, config: PI05RLTConfig):
        super().__init__()
        d = config.rlt_hidden_dim
        self.vlm_proj = (
            nn.Linear(config.vlm_hidden_dim, d)
            if config.vlm_hidden_dim != d
            else nn.Identity()
        )
        self.rl_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.rlt_num_heads,
            dim_feedforward=d * 4,
            dropout=config.rlt_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.rlt_encoder_layers,
            norm=nn.LayerNorm(d),
        )

    def forward(self, vlm_embeddings: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return the encoded appended RL token.

        Args:
            vlm_embeddings: ``(B, L, vlm_hidden_dim)`` frozen VLA tokens.
            mask: ``(B, L)`` boolean validity mask. The appended RL token is
                always valid.
        """
        if vlm_embeddings.ndim != 3:
            raise ValueError(
                f"vlm_embeddings must have shape (B, L, D), got {tuple(vlm_embeddings.shape)}"
            )
        batch_size, seq_len, _ = vlm_embeddings.shape
        tokens = self.vlm_proj(vlm_embeddings)
        rl_token = self.rl_token.expand(batch_size, -1, -1)
        augmented = torch.cat([tokens, rl_token], dim=1)

        # mask: True=valid → src_key_padding_mask: True=IGNORE (PyTorch convention)
        key_padding_mask = None
        if mask is not None:
            if mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"mask must have shape {(batch_size, seq_len)}, got {tuple(mask.shape)}"
                )
            rl_valid = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            key_padding_mask = torch.cat([~mask.bool(), rl_valid], dim=1)

        encoded = self.transformer(augmented, src_key_padding_mask=key_padding_mask)
        return encoded[:, -1]


# ─────────────────────────────────────────────────────────────────────────────
# RLT Decoder: teacher-forced causal reconstruction from z_rl
# ─────────────────────────────────────────────────────────────────────────────


class RLTokenDecoder(nn.Module):
    """Paper Eq. (2): predict z_i from ``[z_rl, stopgrad(z_1:i-1)]``."""

    def __init__(self, config: PI05RLTConfig):
        super().__init__()
        d = config.rlt_hidden_dim
        self.target_proj = (
            nn.Linear(config.vlm_hidden_dim, d)
            if config.vlm_hidden_dim != d
            else nn.Identity()
        )
        self.z_proj = nn.Linear(d, d) if config.vlm_hidden_dim != d else nn.Identity()
        self.pos_embed = nn.Parameter(torch.zeros(1, 2048, d))
        nn.init.normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.rlt_num_heads,
            dim_feedforward=d * 4,
            dropout=config.rlt_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer,
            num_layers=config.rlt_decoder_layers,
            norm=nn.LayerNorm(d),
        )
        self.pred_head = (
            nn.Linear(d, config.vlm_hidden_dim)
            if d != config.vlm_hidden_dim
            else nn.Identity()
        )

    def forward(
        self,
        z_rl: Tensor,
        targets: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Teacher-force stopped-gradient targets and reconstruct the full sequence."""
        if targets.ndim != 3:
            raise ValueError(f"targets must have shape (B, L, D), got {tuple(targets.shape)}")
        batch_size, target_len, _ = targets.shape
        if z_rl.shape[0] != batch_size:
            raise ValueError("z_rl and targets must have the same batch size")
        if target_len > self.pos_embed.shape[1]:
            raise ValueError(
                f"target length {target_len} exceeds decoder limit {self.pos_embed.shape[1]}"
            )

        shifted = targets.detach()[:, :-1]
        decoder_inputs = torch.cat(
            [self.z_proj(z_rl).unsqueeze(1), self.target_proj(shifted)],
            dim=1,
        )
        decoder_inputs = decoder_inputs + self.pos_embed[:, :target_len]

        causal_mask = torch.triu(
            torch.ones(target_len, target_len, dtype=torch.bool, device=targets.device),
            diagonal=1,
        )
        # mask: True=valid → src_key_padding_mask: True=IGNORE
        key_padding_mask = None
        if mask is not None:
            if mask.shape != (batch_size, target_len):
                raise ValueError(
                    f"mask must have shape {(batch_size, target_len)}, got {tuple(mask.shape)}"
                )
            # Position 0 contains z_rl (always valid); position i>0 contains target i-1.
            input_invalid = torch.cat(
                [torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device),
                 ~mask[:, :-1].bool()],
                dim=1,
            )
            key_padding_mask = input_invalid

        decoded = self.decoder(
            decoder_inputs,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        return self.pred_head(decoded)


# ─────────────────────────────────────────────────────────────────────────────
# Actor/Critic MLP: paper full-output heads (z_rl + state + ref/action chunk)
# ─────────────────────────────────────────────────────────────────────────────


def _build_mlp(input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> nn.Sequential:
    """Plain ReLU MLP: ``[Linear, ReLU] * (num_layers - 1) + Linear(output_dim)``.

    The exact module layout matches the RL-Token-Pi05 Stage-2 trained
    checkpoints (state dict keys ``net.*`` for the actor, ``q1.*``/``q2.*``
    for the critic), so ``load_state_dict(..., strict=True)`` works on them.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class RLTChunkActor(nn.Module):
    """Paper full-output Gaussian chunk actor: (z_rl, proprio, ref_chunk) -> action mean.

    Lightweight MLP matching the RL-Token-Pi05 Stage-2 trained architecture
    (``actor_contract = paper_full_output_v1``): concatenate ``z_rl``,
    proprioception and the flattened reference chunk, then
    ``[Linear, ReLU] * (num_layers - 1) + Linear(chunk * action_dim)``.
    Returns a Gaussian with fixed std. Always full output — no residual mode
    (``is_residual`` stays False, per the paper contract).
    """

    def __init__(
        self,
        config: PI05RLTConfig,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        # ``getattr`` with the paper defaults keeps duck-typed configs working
        # (legacy tests construct heads from SimpleNamespace instances).
        self.hidden_dim = int(
            hidden_dim if hidden_dim is not None else getattr(config, "actor_hidden_dim", 256)
        )
        self.num_layers = int(
            num_layers if num_layers is not None else getattr(config, "actor_num_layers", 2)
        )
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {self.num_layers}")

        z_dim = int(getattr(config, "rlt_hidden_dim", 256))
        state_dim = int(getattr(config, "state_dim", 6))
        action_dim = int(getattr(config, "action_dim", 6))
        chunk = int(getattr(config, "n_action_steps_rl", 10))

        self.action_dim = action_dim
        self.chunk = chunk
        self.fixed_std = float(getattr(config, "policy_fixed_std", 0.05))
        self.is_residual = False  # paper contract: full output, never residual

        input_dim = z_dim + state_dim + chunk * action_dim
        output_dim = chunk * action_dim
        self.net = _build_mlp(input_dim, self.hidden_dim, self.num_layers, output_dim)

    def _flatten_chunk(self, chunk: Tensor, batch: int) -> Tensor:
        """Flatten a (B, chunk, action_dim) / (B, chunk*action_dim) / (chunk, action_dim)
        tensor to (B, chunk*action_dim), validating every dimension so an MLP input
        mismatch fails with a clear message instead of an opaque matmul error."""
        flat = self.chunk * self.action_dim
        if chunk.ndim == 3:
            flat_chunk = chunk.reshape(chunk.shape[0], -1)
        elif chunk.ndim == 2 and chunk.shape[1] == flat:
            flat_chunk = chunk
        elif chunk.ndim == 2:
            # (chunk, action_dim) without a batch dimension: treat as one sample.
            flat_chunk = chunk.reshape(1, -1)
        else:
            raise ValueError(
                f"ref_chunk must have rank 2 or 3, got rank {chunk.ndim} with shape {tuple(chunk.shape)}"
            )
        if flat_chunk.shape[-1] != flat:
            raise ValueError(
                f"ref_chunk last dim {flat_chunk.shape[-1]} != expected {flat} "
                f"(chunk={self.chunk}, action_dim={self.action_dim})"
            )
        if flat_chunk.shape[0] != batch:
            raise ValueError(
                f"ref_chunk batch {flat_chunk.shape[0]} != input batch {batch}; "
                "add a batch dimension to ref_chunk"
            )
        return flat_chunk

    def forward(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        ref_chunk: Tensor,
        training: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if z_rl.ndim == 1:
            z_rl = z_rl.unsqueeze(0)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        ref_flat = self._flatten_chunk(ref_chunk, z_rl.shape[0])
        x = torch.cat([z_rl, proprio, ref_flat], dim=-1)
        mean_flat = self.net(x)
        mean = mean_flat.reshape(-1, self.chunk, self.action_dim)
        std = torch.full_like(mean, self.fixed_std)
        return mean, std


class RLTTwinCritic(nn.Module):
    """Twin critic for full chunk actions: (z_rl, proprio, action_chunk) -> Q1, Q2.

    Same lightweight MLP family as the actor (paper-aligned, keys ``q1.*`` /
    ``q2.*`` matching the RL-Token-Pi05 Stage-2 trained checkpoints).
    """

    def __init__(
        self,
        config: PI05RLTConfig,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        # ``getattr`` with the paper defaults keeps duck-typed configs working
        # (legacy tests construct heads from SimpleNamespace instances).
        self.hidden_dim = int(
            hidden_dim if hidden_dim is not None else getattr(config, "critic_hidden_dim", 256)
        )
        self.num_layers = int(
            num_layers if num_layers is not None else getattr(config, "critic_num_layers", 2)
        )
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {self.num_layers}")

        z_dim = int(getattr(config, "rlt_hidden_dim", 256))
        state_dim = int(getattr(config, "state_dim", 6))
        action_dim = int(getattr(config, "action_dim", 6))
        chunk = int(getattr(config, "n_action_steps_rl", 10))

        self.action_dim = action_dim
        self.chunk = chunk
        input_dim = z_dim + state_dim + chunk * action_dim
        self.q1 = _build_mlp(input_dim, self.hidden_dim, self.num_layers, 1)
        self.q2 = _build_mlp(input_dim, self.hidden_dim, self.num_layers, 1)

    def _flatten_chunk(self, chunk: Tensor, batch: int) -> Tensor:
        """Flatten a (B, chunk, action_dim) / (B, chunk*action_dim) / (chunk, action_dim)
        tensor to (B, chunk*action_dim), validating every dimension so an MLP input
        mismatch fails with a clear message instead of an opaque matmul error."""
        flat = self.chunk * self.action_dim
        if chunk.ndim == 3:
            flat_chunk = chunk.reshape(chunk.shape[0], -1)
        elif chunk.ndim == 2 and chunk.shape[1] == flat:
            flat_chunk = chunk
        elif chunk.ndim == 2:
            # (chunk, action_dim) without a batch dimension: treat as one sample.
            flat_chunk = chunk.reshape(1, -1)
        else:
            raise ValueError(
                f"action_chunk must have rank 2 or 3, got rank {chunk.ndim} with shape {tuple(chunk.shape)}"
            )
        if flat_chunk.shape[-1] != flat:
            raise ValueError(
                f"action_chunk last dim {flat_chunk.shape[-1]} != expected {flat} "
                f"(chunk={self.chunk}, action_dim={self.action_dim})"
            )
        if flat_chunk.shape[0] != batch:
            raise ValueError(
                f"action_chunk batch {flat_chunk.shape[0]} != input batch {batch}; "
                "add a batch dimension to action_chunk"
            )
        return flat_chunk

    def forward(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        action_chunk: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if z_rl.ndim == 1:
            z_rl = z_rl.unsqueeze(0)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        action_flat = self._flatten_chunk(action_chunk, z_rl.shape[0])
        x = torch.cat([z_rl, proprio, action_flat], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        action_chunk: Tensor,
    ) -> Tensor:
        q1, q2 = self(z_rl, proprio, action_chunk)
        return torch.minimum(q1, q2)


# ─────────────────────────────────────────────────────────────────────────────
# PI05RLTPolicy: wraps frozen π0.5 + trainable RLT components
# ─────────────────────────────────────────────────────────────────────────────


class PI05RLTPolicy(PreTrainedPolicy):
    """Full RLT policy wrapping frozen π0.5 + trainable RLT components.

    Supports 3 modes:
      - "rlt_training": Stage 1 — train encoder/decoder on demo data
      - "online_rl": Stage 2 — train actor/critic with online RL
      - "inference": Deploy on robot
    """

    config_class = PI05RLTConfig
    name = "pi05_rlt"

    def __init__(self, config: PI05RLTConfig, **kwargs):
        super().__init__(config)
        self.config = config

        # ── RLT encoder + decoder ────────────────────────────────────────
        self.rlt_encoder = RLTokenEncoder(config)
        self.rlt_decoder = RLTokenDecoder(config)

        # ── Actor + Critic ───────────────────────────────────────────────
        self.actor = RLTChunkActor(config)
        self.critic = RLTTwinCritic(config)
        self.critic_target = RLTTwinCritic(config)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False
        # Target actor kept for checkpoint parity with the Stage-2 artifacts
        # (schema v3 stores actor_target_state_dict). Frozen; not used by the
        # TD3-style helpers above, which follow the trained policy's convention.
        self.actor_target = copy.deepcopy(self.actor)
        for p in self.actor_target.parameters():
            p.requires_grad = False

        # ── Frozen π0.5 (loaded lazily) ──────────────────────────────────
        # ``_frozen_vla`` is stored OUTSIDE the nn.Module tree on purpose
        # (plain object attribute, see load_frozen_vla): the ~2.4B π0.5
        # weights must never be dumped into this policy's state_dict /
        # model.safetensors. Checkpoints only round-trip the compact RLT
        # components; the backbone is referenced via ``pi05_pretrained_path``.
        self._vla_loaded = False
        self.__dict__["_frozen_vla"] = None

        # ── Action queue for deployment ──────────────────────────────────
        self._init_deploy_state()
        self.reset()

    def _init_deploy_state(self) -> None:
        """Deployment state (Stage-2 semantics at window granularity)."""
        self._critical_active = False
        self._critical_sustained_steps = 0
        self._actor_enabled = False       # sticky across episodes ('a' key)
        self._auto_fired_once = False     # per episode
        self._window_steps = 0            # control steps since episode start
        self._manual_override_at = float("-inf")
        self._detector = None
        self.__dict__["_last_z_rl"] = None

    def reset(self):
        """Clear the action queue and per-episode deployment state.

        Mirrors the Stage-2 episode boundary: critical phase / sustained
        counter / once-flag / step counter reset; the detector's probability
        window and ``actor_enabled`` persist (sticky manual gate).
        Call :meth:`set_actor_enabled` to change the gate.
        """
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps_rl)}
        self._critical_active = False
        self._critical_sustained_steps = 0
        self._auto_fired_once = False
        self._window_steps = 0
        self._manual_override_at = float("-inf")
        self.__dict__["_last_z_rl"] = None

    # ── Deployment controls (used by record / lerobot-rollout) ──────────

    @property
    def critical_active(self) -> bool:
        return self._critical_active

    @property
    def actor_enabled(self) -> bool:
        return self._actor_enabled

    def set_actor_enabled(self, enabled: bool) -> None:
        """Manual actor gate ('a' key), sticky across episodes."""
        self._actor_enabled = bool(enabled)

    def set_critical_manual(self, active: bool) -> None:
        """Manual critical toggle ('c' key). Always wins over auto-detection
        for ``auto_critical_override_cooldown`` seconds."""
        import time as _time

        self._critical_active = bool(active)
        self._manual_override_at = _time.time()

    def _ensure_detector(self) -> None:
        if self._detector is None and self.config.deploy_auto_critical:
            from lerobot.policies.pi05_rlt.critical import CriticalPhaseDetector

            self._detector = CriticalPhaseDetector(
                self.config.critical_classifier_path,
                next(self.parameters()).device,
                threshold_on=self.config.auto_critical_threshold_on,
                threshold_off=self.config.auto_critical_threshold_off,
                smooth_steps=self.config.auto_critical_smooth_steps,
                min_on_chunks=self.config.auto_critical_min_on_chunks,
            )

    def _detection_armed(self) -> bool:
        import time as _time

        if not self.config.deploy_auto_critical or self._detector is None:
            return False
        if self._window_steps < self.config.auto_critical_start_delay_steps:
            return False
        if (
            self.config.auto_critical_once
            and self._auto_fired_once
            and not self._critical_active
        ):
            return False
        if _time.time() - self._manual_override_at < self.config.auto_critical_override_cooldown:
            return False
        return True

    @torch.no_grad()
    def _update_auto_critical(self, z_rl: Tensor | np.ndarray | None, proprio_np) -> bool:
        """Run the auto detector once; returns True when the state flipped."""
        if not self._detection_armed() or z_rl is None:
            return False
        if isinstance(z_rl, Tensor):
            z_np = z_rl.detach().cpu().numpy().squeeze()
        else:
            z_np = np.asarray(z_rl).squeeze()
        desired = self._detector.update(z_np, proprio_np)
        # Log the gate state every window: without this, a never-firing
        # classifier is invisible on the robot (symptom: pure-VLA behaviour,
        # skipped straw grasp, gripper cycling at the cup).
        prob = getattr(self._detector, "_probabilities", None)
        prob_last = prob[-1] if prob else float("nan")
        prob_avg = float(np.mean(prob)) if prob else float("nan")
        logger.info(
            "[gate] window=%d raw_p=%.3f avg_p=%.3f on=%.2f off=%.2f critical=%s sustained=%d",
            self._window_steps,
            prob_last,
            prob_avg,
            self._detector.threshold_on,
            self._detector.threshold_off,
            self._critical_active,
            self._critical_sustained_steps,
        )
        if desired is not None and desired != self._critical_active:
            self._critical_active = bool(desired)
            if desired:
                self._auto_fired_once = True
            return True
        return False

    def _actor_takes_over(self) -> bool:
        """Stage-2 actor takeover matrix (window granularity).

        Actor output is used only when the critical phase is active AND it has
        been sustained for ``actor_critical_delay_steps`` control steps AND the
        manual gate is enabled (and not in vla_only mode).
        """
        cfg = self.config
        if cfg.deploy_vla_only or not self._actor_enabled:
            return False
        if not self._critical_active:
            return False
        return self._critical_sustained_steps >= cfg.actor_critical_delay_steps

    def _fill_window(self, batch: dict[str, Tensor]) -> None:
        """One window (n_action_steps_rl control steps): detect, infer, gate."""
        from lerobot.utils.constants import OBS_STATE

        self._window_steps += self.config.n_action_steps_rl

        # Early detection on the boundary uses the PREVIOUS window's cached
        # z_rl + current proprio (before the blocking π0.5 inference).
        state_raw = batch[OBS_STATE]
        state_np = (
            state_raw[:, -1, :] if state_raw.ndim > 2 else state_raw
        )[:, : self.config.state_dim].detach().cpu().numpy().squeeze()
        flipped_early = self._update_auto_critical(self._last_z_rl, state_np)

        ref_actions_sub, z_rl, state_rl = self._infer_window(batch)
        z_np = z_rl.detach().cpu().numpy().squeeze()
        self.__dict__["_last_z_rl"] = z_np
        self.__dict__["_last_ref_actions"] = ref_actions_sub.detach().cpu()

        # First window of the episode has no cached z_rl: post-inference check.
        if not flipped_early:
            self._update_auto_critical(z_rl, state_np)

        # Sustained-critical bookkeeping (advances by one window per boundary).
        if self._critical_active:
            self._critical_sustained_steps += self.config.n_action_steps_rl
        else:
            self._critical_sustained_steps = 0

        if self._actor_takes_over():
            action_mean, _ = self.actor(z_rl, state_rl, ref_actions_sub)
            chunk = action_mean
        else:
            chunk = ref_actions_sub  # pure VLA reference execution
        self._queues[ACTION].extend(chunk.transpose(0, 1))

    def load_frozen_vla(self, pi05_model):
        """Attach a frozen π0.5 backbone (PI05Pytorch, or a PI05Policy wrapper).

        The backbone is deliberately NOT registered as a submodule, so it is
        excluded from ``state_dict()`` and from ``save_pretrained`` output.
        Attach the backbone after moving the policy to its device; the
        backbone keeps its own device.
        """
        if pi05_model is None:
            raise ValueError("pi05_model must not be None")
        if hasattr(pi05_model, "model") and hasattr(pi05_model, "predict_action_chunk"):
            # A PI05Policy wrapper was passed: keep its underlying PI05Pytorch.
            pi05_model = pi05_model.model
        # The backbone is no longer required to provide extract_embeddings: recent
        # pi05 builds removed it and _extract_vla_embeddings_impl() rebuilds the
        # behaviour from public APIs. Only the required pieces are checked here.
        if not hasattr(pi05_model, "embed_prefix") or not hasattr(pi05_model, "embed_suffix"):
            raise TypeError(
                "pi05_model must be a PI05Pytorch instance (or a PI05Policy wrapper); "
                "got %r" % type(pi05_model).__name__
            )
        # Bypass nn.Module.__setattr__ registration on purpose (see docstring).
        self.__dict__["_frozen_vla"] = pi05_model
        self._frozen_vla.eval()
        for p in self._frozen_vla.parameters():
            p.requires_grad = False
        self._vla_loaded = True

    def attach_frozen_vla_from_config(self) -> None:
        """Load and attach the frozen π0.5 backbone from ``config.pi05_pretrained_path``.

        Makes the policy self-contained for lerobot toolchains
        (``from_pretrained`` → robot inference) when the config records where
        the π0.5 ``pretrained_model`` directory lives. Requires ``transformers``
        and the workspace ``lerobot.policies.pi05`` at call time.
        """
        if self._vla_loaded:
            return
        path = getattr(self.config, "pi05_pretrained_path", "")
        if not path:
            raise ValueError(
                "config.pi05_pretrained_path is empty; pass an explicit backbone "
                "to load_frozen_vla() instead."
            )
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        pi05_policy = PI05Policy.from_pretrained(path)
        self.load_frozen_vla(pi05_policy)
        # PI05Policy.from_pretrained moves the backbone to ITS checkpoint
        # config's device (training machine: cuda) while this wrapper follows
        # the deployment config (often cpu until the rollout script moves the
        # whole policy). A device mismatch surfaces as
        # "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)"
        # inside the SigLIP tower on the first select_action call. Force the
        # backbone onto the wrapper's device so the policy is self-consistent
        # before the rollout script (or any caller) applies its own .to(device).
        wrapper_dev = next(self.parameters()).device
        self._frozen_vla.to(device=wrapper_dev)
        logger.info("Frozen π0.5 backbone moved to wrapper device: %s", wrapper_dev)
        logger.info("Frozen π0.5 attached from %s", path)

    def _ensure_vla(self):
        if not self._vla_loaded and getattr(self.config, "pi05_pretrained_path", ""):
            self.attach_frozen_vla_from_config()
        if not self._vla_loaded:
            raise RuntimeError(
                "Frozen π0.5 not loaded. Either set config.pi05_pretrained_path and "
                "call policy.attach_frozen_vla_from_config(), or call "
                "policy.load_frozen_vla(pi05_model) first."
            )

    # ── Embedding extraction from frozen π0.5 ────────────────────────────

    @staticmethod
    def _pi05_utils():
        """Compatibility-layer entry point: see ``lerobot.policies.pi05_rlt.vla_compat.resolve_pi05_utils``."""
        return resolve_pi05_utils()

    def _extract_vla_embeddings_impl(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise=None,
        time=None,
        image_only: bool = False,
    ):
        """Extract embeddings from the frozen π0.5 (implemented in vla_compat; works across old and new lerobot)."""
        return extract_pi05_embeddings(
            self._frozen_vla,
            images, img_masks, tokens, masks, actions,
            chunk_size=self.config.chunk_size,
            max_action_dim=self.config.max_action_dim,
            noise=noise,
            time=time,
            image_only=image_only,
        )

    def extract_vla_embeddings(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run frozen π0.5 forward and return (prefix_out, expert_out) embeddings.

        Returns:
            prefix_out: (B, L_prefix, 2048) — PaliGemma final hidden states
            expert_out: (B, chunk_size, 1024) — Gemma Expert final hidden states
        """
        self._ensure_vla()
        return self._extract_vla_embeddings_impl(
            images, img_masks, tokens, masks, actions,
            noise=noise, time=time,
        )

    # ── Image-only prefix extraction (per paper, matching openpi-RLT) ────

    @torch.no_grad()
    def extract_prefix_image_only(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Extract image-only prefix embeddings + mask for Stage 1 RLT training.

        Per RLT paper: "for tasks with a fixed language instruction we drop
        language embeddings in this step." Also drops discretized state tokens
        that were injected into the language prompt (SO-101 engineering).

        Returns:
            prefix_img: (B, num_img_tokens, 2048) — image prefix only
            mask: (B, num_img_tokens) — bool: True=valid token, False=padding/masked camera
        """
        self._ensure_vla()
        prefix_img, _, prefix_mask = self._extract_vla_embeddings_impl(
            images, img_masks, tokens, masks, actions,
            image_only=True,
        )
        return prefix_img.float(), prefix_mask

    # ── Stage 1: RLT Training (encoder-decoder) ────────────────────────

    def forward_rlt_training(
        self,
        vlm_embeddings: Tensor,     # (B, L, 2048) — image-only prefix
        mask: Tensor | None = None,  # (B, L) — bool: True=valid, False=padding
    ) -> dict[str, Tensor]:
        """Stage 1 forward: encoder → z_rl → decoder → masked reconstruction loss.

        Per paper Eq. (2): stop-grad target, per-token MSE with mask.
        """
        # Detach targets (stop-gradient as per RLT paper Eq. 2)
        vlm_target = vlm_embeddings.detach()

        # Encode Eq. (1): append e_rl and take the final encoder position.
        z_rl = self.rlt_encoder(vlm_target, mask=mask)

        # Decode Eq. (2): [z_rl, stopgrad(z_1:M-1)] with causal attention.
        vlm_recon = self.rlt_decoder(z_rl, vlm_target, mask=mask)

        # Masked reconstruction loss
        sq_error = (vlm_recon - vlm_target).pow(2)  # (B, L, D)
        if mask is not None:
            # Mean over valid (unmasked) positions only
            mask_expanded = mask.float().unsqueeze(-1)  # (B, L, 1)
            loss = (sq_error * mask_expanded).sum() / (mask_expanded.sum() * vlm_target.shape[-1]).clamp(min=1.0)
        else:
            loss = sq_error.mean()

        return {
            "loss": loss,
            "vlm_recon_loss": loss,
            "z_rl": z_rl,
        }

    # ── Stage 2: Online RL (actor-critic) ───────────────────────────────

    def forward_critic_loss(
        self,
        z_rl: Tensor,
        state: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_z_rl: Tensor,
        next_state: Tensor,
        next_ref_actions: Tensor,
        dones: Tensor,
    ) -> dict[str, Tensor]:
        """Compute TD3-style twin critic loss."""
        with torch.no_grad():
            next_action_mean, _ = self.actor(next_z_rl, next_state, next_ref_actions)
            target_q = self.critic_target.q_min(next_z_rl, next_state, next_action_mean)
            td_target = rewards + self.config.discount * (1.0 - dones) * target_q

        q1, q2 = self.critic(z_rl, state, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        return {
            "critic_loss": critic_loss,
            "q1_mean": q1.mean(),
            "q2_mean": q2.mean(),
            "td_target_mean": td_target.mean(),
        }

    def forward_actor_loss(
        self,
        z_rl: Tensor,
        state: Tensor,
        ref_actions: Tensor,
    ) -> dict[str, Tensor]:
        """Compute actor loss: maximize Q + BC regularization."""
        action_mean, _ = self.actor(z_rl, state, ref_actions, training=True)

        q_value = self.critic.q_min(z_rl.detach(), state, action_mean)
        q_loss = -q_value.mean()

        bc_loss = F.mse_loss(action_mean, ref_actions)

        actor_loss = q_loss + self.config.bc_weight * bc_loss

        return {
            "actor_loss": actor_loss,
            "q_loss": q_loss,
            "bc_loss": bc_loss,
        }

    def update_target_critic(self):
        """Soft update target critic via EMA."""
        tau = self.config.target_tau
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

    # ── Inference ─────────────────────────────────────────────────────

    def _prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the frozen π0.5 model.

        Handles resize with padding to (224, 224) and normalization to [-1, 1].
        """
        *_, resize_with_pad_torch, _ = self._pi05_utils()

        images = []
        img_masks = []
        device = next(self.parameters()).device
        for key in self.config.image_features:
            if key not in batch:
                continue
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            img = img.to(device).float()

            # Ensure [B, C, H, W]
            is_channels_first = img.shape[1] == 3
            if not is_channels_first:
                img = img.permute(0, 3, 1, 2)

            # Resize to π0.5's expected resolution (224x224)
            if img.shape[2:4] != self.config.image_resolution:
                # resize_with_pad_torch expects [B, H, W, C]
                img_hwc = img.permute(0, 2, 3, 1)
                img_hwc = resize_with_pad_torch(img_hwc, *self.config.image_resolution)
                img = img_hwc.permute(0, 3, 1, 2)

            # Normalize from [0,1] to [-1,1] as expected by SigLIP
            img = img * 2.0 - 1.0

            images.append(img)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        if len(images) == 0 and len(self.config.image_features) > 0:
            # Fail loudly instead of running the frozen VLA blind: missing
            # camera observations silently degrades the VLA plan and the z_rl
            # embedding to language/proprioception-only outputs (real-robot
            # symptom: skipped grasp, gripper cycling at the cup). An empty
            # config.image_features list is allowed only for unit tests with
            # stubbed VLA backbones; the exporter now inherits the camera
            # feature contract from the source PI0.5 config.
            raise ValueError(
                "PI05RLTPolicy: camera features "
                f"{list(self.config.image_features)} are missing from the batch "
                f"(batch keys={sorted(batch.keys())})."
            )

        return images, img_masks

    @torch.no_grad()
    def _infer_window(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Frozen π0.5 forward for one window: VLA reference + z_rl + state.

        Returns:
            ref_actions_sub: (B, n_action_steps_rl, action_dim) contiguous VLA reference
            z_rl: (B, rlt_hidden_dim)
            state_rl: (B, state_dim) normalized proprioception (actor/detector input)
        """
        self._ensure_vla()
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
        *_, pad_vector = self._pi05_utils()

        # Prepare inputs
        images, img_masks = self._prepare_images(batch)
        # Language tokens/masks arrive on CPU from the robot rollout path (the
        # deployment preprocessor has no DeviceProcessorStep and lerobot-rollout
        # keeps observations on CPU). Move them to the policy device — images
        # are already moved inside _prepare_images, and state moves to
        # z_rl.device below. Without this, the frozen SigLIP tower raises
        # "index is on cpu, different from other tensors on cuda:0".
        device = next(self.parameters()).device
        tokens = batch[OBS_LANGUAGE_TOKENS].to(device)
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device)

        # Get VLA reference actions (full 50-step chunk)
        ref_actions_full = self._frozen_vla.sample_actions(images, img_masks, tokens, masks)
        ref_actions_full = ref_actions_full[:, :, :self.config.action_dim]
        self.__dict__["_last_vla_full"] = ref_actions_full.detach().float().cpu()
        self.__dict__["_last_img_sig"] = [
            (tuple(img.shape), float(img.float().sum()), float(img.float().abs().sum())) for img in images
        ]
        self.__dict__["_last_tokens"] = tokens.detach().cpu()

        # Reference chunk: the FIRST n_action_steps_rl CONTIGUOUS VLA actions
        # (make_pi05_reference_chunk semantics, stride=1). This must mirror
        # Stage-2 training and scripts/eval_rlt_pi05_so101.py exactly: the
        # stride-2 subsampling in the config applies to REPLAY-TRANSITION
        # collection (build_stride2_transitions) only, never to the reference
        # chunk fed to the actor or executed by the frozen VLA. Striding here
        # stretches the executed window to 2x its trained temporal span and
        # drops every other fine-grained action (e.g. gripper micro-moves),
        # which on hardware reads as a skipped grasp with a cycling gripper.
        ref_actions_sub = ref_actions_full[:, : self.config.n_action_steps_rl, :]

        # Get dummy actions for embedding extraction (use ref as proxy)
        dummy_actions = pad_vector(ref_actions_full, self.config.max_action_dim)

        # Extract image-only prefix + mask for encoder
        prefix_img, prefix_mask = self.extract_prefix_image_only(
            images, img_masks, tokens, masks, dummy_actions,
        )

        # Encode to z_rl (image-only prefix, per paper)
        z_rl = self.rlt_encoder(prefix_img, mask=prefix_mask)

        # Get normalized state for actor / critical detector
        state_raw = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state_rl = state_raw[:, :self.config.state_dim].to(z_rl.device)
        return ref_actions_sub, z_rl, state_rl

    @torch.no_grad()
    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Legacy full inference: frozen π0.5 → z_rl → actor → actions.

        Always actor-refined (no deployment gating). Returns (B, C, action_dim).
        """
        ref_actions_sub, z_rl, state_rl = self._infer_window(batch)
        action_mean, _ = self.actor(z_rl, state_rl, ref_actions_sub)
        self.__dict__["_last_ref_actions"] = ref_actions_sub.detach().cpu()
        self.__dict__["_last_z_rl"] = z_rl.detach().cpu()
        self.__dict__["_last_state_rl"] = state_rl.detach().float().cpu()
        return action_mean

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict a chunk of actions for a given observation."""
        return self._get_action_chunk(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action for robot execution.

        Manages an action queue: only calls the full pipeline when the
        queue is empty, otherwise pops the next pre-computed action. With
        ``config.deploy_auto_critical`` the queue is filled per window by
        :meth:`_fill_window` (auto-critical + actor takeover semantics);
        otherwise the legacy actor-refined chunk is used.
        """
        self.eval()

        if len(self._queues[ACTION]) == 0:
            if self.config.deploy_auto_critical:
                self._ensure_detector()
                self._fill_window(batch)
            else:
                actions = self._get_action_chunk(batch)
                self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    # ── Standard forward (dispatches based on mode) ─────────────────────

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.config.mode == "rlt_training":
            mask = batch.get("prefix_mask")
            result = self.forward_rlt_training(
                batch["vlm_embeddings"],
                mask=mask,
            )
            return result["loss"], {k: v.item() if isinstance(v, Tensor) and v.ndim == 0 else v for k, v in result.items() if k != "z_rl"}
        else:
            raise ValueError(
                f"forward() not supported for mode='{self.config.mode}'. "
                "Use forward_critic_loss/forward_actor_loss for online_rl."
            )

    def get_optim_params(self) -> dict:
        if self.config.mode == "rlt_training":
            return list(self.rlt_encoder.parameters()) + list(self.rlt_decoder.parameters())
        elif self.config.mode == "online_rl":
            return list(self.actor.parameters()) + list(self.critic.parameters())
        else:
            return []
