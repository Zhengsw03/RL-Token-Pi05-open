"""Configuration for π0.5 with RL Token (RLT) policy — paper-aligned, SO-101.

RLT (Physical Intelligence) trains a compact RL token ``z_rl`` on top of
frozen π0.5 embeddings (Stage 1), then a lightweight full-output Gaussian
Actor + Twin Critic via online RL (Stage 2). This is the canonical
configuration of the RL-Token-Pi05 pipeline on SO-101:

- ``z_rl`` dim 2048 (matches the frozen VLA token dim, per the paper)
- ``state_dim = action_dim = 6`` (5 joints + gripper)
- RL chunk = ``n_action_steps_rl = 10`` steps, subsampled with
  ``action_stride = 2`` from π0.5's 50-step reference chunk
- paper full-output MLP actor/critic with fixed std ``policy_fixed_std``
  (no residual mode)

.. note::
   The earlier reference-projection actor/critic fields (``actor_hidden_dims``
   list, ``ref_dropout``, ``actor_residual_scale``) were removed: the paper
   pipeline always trains the full-output MLP heads and the SmolVLA-oriented
   variant is not maintained anymore.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("pi05_rlt")
@dataclass
class PI05RLTConfig(PreTrainedConfig):
    # ── π0.5 base model ──────────────────────────────────────────────────
    pi05_pretrained_path: str = ""  # Path to pretrained π0.5 checkpoint
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "bfloat16"

    # These mirror PI05Config defaults for the frozen π0.5
    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    image_resolution: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 200
    n_obs_steps: int = 1

    # ── RLT encoder-decoder (trained contract: keep dims aligned with the
    #    Stage 1 checkpoints produced by train_rlt_stage1_pi05.py) ────────
    rlt_architecture: str = "paper_v1"
    vlm_hidden_dim: int = 2048      # Gemma 2B hidden size (VLA token dim)
    expert_hidden_dim: int = 1024   # Gemma 300M hidden size
    rlt_hidden_dim: int = 2048      # z_rl dimension (match VLA dim, per paper)
    rlt_encoder_layers: int = 4
    rlt_decoder_layers: int = 4
    rlt_num_heads: int = 8
    rlt_dropout: float = 0.1

    # ── Paper full-output actor MLP ──────────────────────────────────────
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2

    # ── Twin critic MLP ──────────────────────────────────────────────────
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2

    # Fixed exploration std of the Gaussian actor (paper: small fixed std)
    policy_fixed_std: float = 0.05

    # ── SO-101 action space (matches trained π0.5 checkpoint) ────────────
    action_dim: int = 6            # 5 joints + gripper
    state_dim: int = 6
    n_action_steps_rl: int = 10    # RL chunk executed per policy call
    action_stride: int = 2         # subsample stride over π0.5's 50-step chunk
    # Canonical SO-101 feature names ("{joint}.pos") for the trained action
    # order. Declared so export_rlt_pretrained can persist it into config.json
    # and rollout boundaries can validate/reorder before touching hardware.
    action_feature_names: list[str] | None = None
    # Export-contract fields (written by scripts/export_rlt_pretrained.py).
    # The local modeling is hard-wired to the paper contract (full-output MLP
    # actor, no residual head), so these are descriptive round-trip fields:
    # they must exist so a saved config.json loads back via from_pretrained.
    actor_critic_style: str = "paper_mlp"
    actor_residual_scale: float = 0.0

    # ── RL hyperparameters used by the policy helpers (Stage 2) ──────────
    discount: float = 0.99
    bc_weight: float = 0.1
    target_tau: float = 0.005

    # ── Deployment behavior (lerobot record/rollout) ─────────────────────
    # All fields default OFF so plain loading keeps the legacy behavior
    # (always actor-refined chunk on each window). When enabled, the policy
    # replicates the Stage-2 online-RL deployment semantics at window
    # granularity: auto-critical detection on chunk boundaries (VLA <-> actor
    # switching) and a sticky manual actor gate.
    deploy_auto_critical: bool = False
    critical_classifier_path: str = ""      # legacy .pt checkpoint (CriticalPhaseDetector format)
    auto_critical_threshold_on: float = 0.75
    auto_critical_threshold_off: float = 0.45
    auto_critical_smooth_steps: int = 8
    auto_critical_min_on_chunks: int = 3
    auto_critical_start_delay_steps: int = 0   # control steps before detection arms
    auto_critical_once: bool = True            # only one auto ON per episode
    auto_critical_override_cooldown: float = 3.0  # s after a manual 'c'
    actor_critical_delay_steps: int = 0        # sustained critical steps before actor takes over
    deploy_vla_only: bool = False              # never let the actor take over

    # ── Training mode ────────────────────────────────────────────────────
    mode: str = "rlt_training"

    # ── Stage 1 optimizer/schedule (used by get_optimizer_preset) ────────
    rlt_lr: float = 1e-4
    rlt_weight_decay: float = 1e-5
    rlt_warmup_steps: int = 200
    rlt_total_steps: int = 5000

    # ── Normalization ────────────────────────────────────────────────────
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # Paths for loading stage checkpoints
    rlt_encoder_path: str = ""
    rlt_decoder_path: str = ""
    actor_path: str = ""
    critic_path: str = ""

    def __post_init__(self):
        super().__post_init__()
        if self.mode not in ("rlt_training", "online_rl", "inference"):
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of: rlt_training, online_rl, inference")
        if self.rlt_architecture != "paper_v1":
            raise ValueError(
                f"Unsupported rlt_architecture '{self.rlt_architecture}'. Expected 'paper_v1'."
            )
        if self.rlt_hidden_dim % self.rlt_num_heads != 0:
            raise ValueError("rlt_hidden_dim must be divisible by rlt_num_heads")
        if self.action_dim > self.max_action_dim:
            raise ValueError(f"action_dim ({self.action_dim}) cannot exceed max_action_dim ({self.max_action_dim})")
        if self.state_dim > self.max_state_dim:
            raise ValueError(f"state_dim ({self.state_dim}) cannot exceed max_state_dim ({self.max_state_dim})")
        if self.action_dim <= 0 or self.state_dim <= 0:
            raise ValueError("action_dim and state_dim must be positive")
        if self.n_action_steps_rl < 1:
            raise ValueError("n_action_steps_rl must be at least 1")
        if self.action_stride < 1:
            raise ValueError("action_stride must be at least 1")
        if self.actor_num_layers < 1 or self.critic_num_layers < 1:
            raise ValueError("actor_num_layers and critic_num_layers must be at least 1")
        if self.actor_hidden_dim <= 0 or self.critic_hidden_dim <= 0:
            raise ValueError("actor_hidden_dim and critic_hidden_dim must be positive")
        if self.policy_fixed_std <= 0:
            raise ValueError("policy_fixed_std must be positive")
        if self.deploy_auto_critical:
            if not self.critical_classifier_path:
                raise ValueError(
                    "deploy_auto_critical requires critical_classifier_path "
                    "(a trained CriticalPhaseDetector .pt checkpoint)"
                )
            if not (0.0 <= self.auto_critical_threshold_off < self.auto_critical_threshold_on <= 1.0):
                raise ValueError(
                    "auto_critical thresholds must satisfy 0 <= off < on <= 1 "
                    f"(got off={self.auto_critical_threshold_off}, on={self.auto_critical_threshold_on})"
                )
            if self.auto_critical_smooth_steps < 1 or self.auto_critical_min_on_chunks < 0:
                raise ValueError("auto_critical_smooth_steps/min_on_chunks out of range")
            if self.auto_critical_override_cooldown < 0:
                raise ValueError("auto_critical_override_cooldown must be >= 0")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.rlt_lr,
            weight_decay=self.rlt_weight_decay,
            grad_clip_norm=1.0,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.rlt_lr,
            decay_lr=0.0,
            num_warmup_steps=self.rlt_warmup_steps,
            num_decay_steps=self.rlt_total_steps,
        )

    def validate_features(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.n_action_steps_rl))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def n_action_steps(self) -> int:
        return self.n_action_steps_rl
