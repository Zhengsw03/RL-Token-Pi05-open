"""PI05RLTPolicy pretrained round-trip: RLT-only checkpoint, VLA excluded.

Guards the "from_pretrained self-consistency" contract of the lerobot pi05_rlt package:

- the frozen π0.5 backbone attached via load_frozen_vla() must NEVER appear in
  ``state_dict()`` / ``model.safetensors`` (checkpoints stay compact and the
  backbone is referenced by ``pi05_pretrained_path`` in the config);
- ``save_pretrained`` → ``from_pretrained`` must round-trip every RLT weight
  and preserve the config (mode, pi05_pretrained_path, dims);
- clear errors when no backbone is available and no path is configured.

Runs on CPU with tiny dims.
"""
import sys
import tempfile
from pathlib import Path

import torch

LEROBOT_SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy


def make_config(**overrides) -> PI05RLTConfig:
    values = {
        "mode": "online_rl",
        "vlm_hidden_dim": 8,
        "rlt_hidden_dim": 8,
        "rlt_num_heads": 2,
        "rlt_encoder_layers": 1,
        "rlt_decoder_layers": 1,
        "rlt_dropout": 0.0,
        "actor_hidden_dim": 16,
        "critic_hidden_dim": 16,
    }
    values.update(overrides)
    return PI05RLTConfig(**values)


class DummyBackbone(torch.nn.Module):
    """Minimal stand-in exposing the PI05Pytorch surface used by the policy."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)

    def extract_embeddings(self, *args, **kwargs):
        raise NotImplementedError("dummy")

    def sample_actions(self, *args, **kwargs):
        raise NotImplementedError("dummy")


def test_frozen_vla_is_excluded_from_state_dict_and_save():
    policy = PI05RLTPolicy(make_config())
    backbone = DummyBackbone()
    policy.load_frozen_vla(backbone)
    assert policy._vla_loaded
    keys = list(policy.state_dict())
    assert not any("frozen_vla" in k for k in keys), keys
    # load_frozen_vla also accepts a PI05Policy-like wrapper (uses .model)
    wrapper = torch.nn.Module()
    wrapper.model = backbone
    wrapper.predict_action_chunk = lambda *a, **k: None
    policy2 = PI05RLTPolicy(make_config())
    policy2.load_frozen_vla(wrapper)
    assert not any("frozen_vla" in k for k in policy2.state_dict())


def test_save_pretrained_round_trip_preserves_weights_and_config():
    cfg = make_config(pi05_pretrained_path="/some/pi05/pretrained_model", actor_hidden_dim=16)
    policy = PI05RLTPolicy(cfg)
    # Perturb weights so a failed reload is easy to detect.
    with torch.no_grad():
        for p in policy.rlt_encoder.parameters():
            p.normal_()
        policy.actor.net[0].weight.normal_()
        policy.critic.q1[0].weight.normal_()

    with tempfile.TemporaryDirectory() as directory:
        policy.save_pretrained(Path(directory))
        files = sorted(p.name for p in Path(directory).iterdir())
        assert "config.json" in files and "model.safetensors" in files, files

        reloaded = PI05RLTPolicy.from_pretrained(directory)
        # Config preserved.
        assert reloaded.config.mode == "online_rl"
        assert reloaded.config.pi05_pretrained_path == "/some/pi05/pretrained_model"
        assert reloaded.config.rlt_hidden_dim == 8
        assert reloaded.config.actor_hidden_dim == 16
        # Every RLT weight round-trips exactly.
        for name, param in policy.named_parameters():
            assert name in dict(reloaded.named_parameters()), name
            # from_pretrained moves the policy to config.device (cuda when
            # available) while the original may live on CPU; compare on CPU.
            torch.testing.assert_close(
                reloaded.get_parameter(name).detach().cpu(), param.detach().cpu(), msg=f"weight mismatch: {name}"
            )
        # Backbone is not loaded back (it is referenced by path, not saved).
        assert not reloaded._vla_loaded


def test_vla_errors_are_actionable():
    policy = PI05RLTPolicy(make_config())
    try:
        policy._ensure_vla()
    except RuntimeError as exc:
        assert "pi05_pretrained_path" in str(exc)
    else:
        raise AssertionError("_ensure_vla should fail without a backbone")
    try:
        policy.attach_frozen_vla_from_config()
    except ValueError as exc:
        assert "pi05_pretrained_path" in str(exc)
    else:
        raise AssertionError("attach with empty path should raise ValueError")
    try:
        policy.load_frozen_vla(None)
    except ValueError:
        pass
    else:
        raise AssertionError("load_frozen_vla(None) should raise ValueError")
    try:
        policy.load_frozen_vla(torch.nn.Linear(2, 2))  # no extract_embeddings
    except TypeError:
        pass
    else:
        raise AssertionError("load_frozen_vla of a plain module should raise TypeError")


def test_dummy_attach_does_not_break_forward_rlt_training():
    """Attaching a backbone must not affect the RLT component graph/loss."""
    policy = PI05RLTPolicy(make_config(mode="rlt_training"))
    policy.load_frozen_vla(DummyBackbone())
    vlm = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    loss, info = policy({"vlm_embeddings": vlm, "prefix_mask": mask})
    assert loss.ndim == 0 and loss.isfinite()
    assert "vlm_recon_loss" in info
