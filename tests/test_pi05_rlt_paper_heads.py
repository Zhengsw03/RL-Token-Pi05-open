"""Paper full-output actor/critic heads live in the lerobot ``pi05_rlt`` package.

These tests pin the contract that makes the package the single source of
truth for the RL-Token-Pi05 Stage-2 heads:

- module/state-dict layout must match the trained checkpoints
  (``actor_contract = paper_full_output_v1``, keys ``net.*`` / ``q1.*``/``q2.*``),
- shapes follow ``PI05RLTConfig`` dims (z_rl, state, action, RL chunk),
- the fixed Gaussian std comes from ``policy_fixed_std``.

They use tiny dims for speed; the real 2048-dim trained checkpoints are
verified separately by ``scripts/verify_rlt_checkpoints.py``.
"""
import sys
from pathlib import Path

import torch

LEROBOT_SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import (
    RLTChunkActor,
    RLTTwinCritic,
)


def make_config(**overrides) -> PI05RLTConfig:
    values = {
        "rlt_hidden_dim": 32,  # z_rl
        "state_dim": 6,
        "action_dim": 6,
        "n_action_steps_rl": 10,
        "actor_hidden_dim": 16,
        "actor_num_layers": 2,
        "critic_hidden_dim": 16,
        "critic_num_layers": 2,
        "policy_fixed_std": 0.05,
        "mode": "online_rl",
    }
    values.update(overrides)
    return PI05RLTConfig(**values)


def test_paper_actor_state_dict_layout_matches_trained_contract():
    actor = RLTChunkActor(make_config())
    keys = list(actor.state_dict())
    # num_layers=2 -> Linear(0), ReLU(1), Linear(2): only net.0/net.2 have params.
    assert keys == ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"]
    assert actor.is_residual is False
    # net.0 input = z(32) + state(6) + chunk(10*6); net.2 output = chunk(10) * action(6)
    assert tuple(actor.net[0].weight.shape) == (16, 32 + 6 + 60)
    assert tuple(actor.net[2].weight.shape) == (60, 16)


def test_paper_critic_state_dict_layout_matches_trained_contract():
    critic = RLTTwinCritic(make_config())
    keys = list(critic.state_dict())
    assert keys == [
        "q1.0.weight", "q1.0.bias", "q1.2.weight", "q1.2.bias",
        "q2.0.weight", "q2.0.bias", "q2.2.weight", "q2.2.bias",
    ]


def test_paper_actor_forward_shapes_and_fixed_std():
    cfg = make_config()
    actor = RLTChunkActor(cfg).eval()
    z = torch.randn(2, 32)
    state = torch.randn(2, 6)
    ref = torch.randn(2, 10, 6)
    with torch.no_grad():
        mean, std = actor(z, state, ref)
    assert tuple(mean.shape) == (2, 10, 6)
    assert torch.allclose(std, torch.full_like(std, cfg.policy_fixed_std))
    # explicit overrides (mirrors legacy script construction) still work
    actor2 = RLTChunkActor(make_config(), hidden_dim=24, num_layers=3)
    assert actor2.hidden_dim == 24 and actor2.num_layers == 3
    assert tuple(actor2.net[0].weight.shape) == (24, 32 + 6 + 60)


def test_paper_critic_forward_and_q_min():
    critic = RLTTwinCritic(make_config()).eval()
    z = torch.randn(2, 32)
    state = torch.randn(2, 6)
    actions = torch.randn(2, 10, 6)
    with torch.no_grad():
        q1, q2 = critic(z, state, actions)
        qmin = critic.q_min(z, state, actions)
    assert tuple(q1.shape) == (2, 1)
    assert torch.allclose(qmin, torch.minimum(q1, q2))


def test_legacy_style_singleton_inputs_accepted():
    """Training code sometimes feeds (chunk, action_dim) refs without a batch dim."""
    actor = RLTChunkActor(make_config()).eval()
    z = torch.randn(32)
    state = torch.randn(6)
    ref = torch.randn(10, 6)
    with torch.no_grad():
        mean, _ = actor(z, state, ref)
    assert tuple(mean.shape) == (1, 10, 6)


def test_config_validates_paper_hyperparameters():
    import pytest

    with pytest.raises(ValueError):
        make_config(actor_num_layers=0)
    with pytest.raises(ValueError):
        make_config(actor_hidden_dim=0)
    with pytest.raises(ValueError):
        make_config(policy_fixed_std=0.0)
    with pytest.raises(ValueError):
        make_config(n_action_steps_rl=0)
    with pytest.raises(ValueError):
        make_config(action_stride=0)


def test_config_defaults_match_trained_contract():
    """Defaults must stay aligned with real Stage-1/2 checkpoints (2048-dim z_rl)."""
    cfg = PI05RLTConfig(mode="online_rl")
    assert cfg.rlt_hidden_dim == 2048
    assert cfg.vlm_hidden_dim == 2048
    assert cfg.rlt_encoder_layers == 4
    assert cfg.rlt_num_heads == 8
    assert (cfg.state_dim, cfg.action_dim) == (6, 6)
    assert cfg.n_action_steps_rl == 10
    assert cfg.action_stride == 2
    assert cfg.policy_fixed_std == 0.05
