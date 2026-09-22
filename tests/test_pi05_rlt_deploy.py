"""pi05_rlt deployment semantics (Stage-2 behavior at window granularity).

With ``config.deploy_auto_critical`` the policy's ``select_action`` fills its
10-step queue per window with EITHER the pure VLA reference or the actor chunk,
mirroring the Stage-2 online deployment matrix:

- actor takes over only when: critical phase active AND sustained >=
  ``actor_critical_delay_steps`` AND manual actor gate enabled (and not
  ``deploy_vla_only``);
- auto-critical detection runs on window boundaries (early: previous z_rl;
  first window: post-inference), manual 'c' overrides auto for a cooldown;
- per-episode ``reset()`` clears critical/sustained/once state but keeps the
  sticky actor gate.

A FakeVLA (deterministic sample_actions/extract_embeddings) stands in for the
frozen π0.5 so the tests run on CPU with tiny dims.
"""
import sys
from pathlib import Path

import numpy as np
import torch

LEROBOT_SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pi05_rlt.critical import CriticalPhaseClassifier
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy

C, A, S, Z, L = 10, 6, 6, 32, 8  # chunk, action, state, z/rlt dim, fake seq len


class FakeVLA(torch.nn.Module):
    """Deterministic π0.5 stand-in; bias sign flips critical probability."""

    def __init__(self, ref_const: float = 1.0):
        super().__init__()
        self.ref_const = float(ref_const)
        self.bias = 1.0

    def sample_actions(self, images, img_masks, tokens, masks):
        B = tokens.shape[0]
        rows = torch.arange(50, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).expand(B, 50, A)
        return rows * 0.01 + self.ref_const

    def extract_embeddings(self, images, img_masks, tokens, masks, actions, image_only=True):
        B = tokens.shape[0]
        emb = torch.full((B, L, Z), 0.1 * self.bias, dtype=torch.float32)
        mask = torch.ones(B, L, dtype=torch.bool)
        return emb, None, mask

    def parameters(self):  # pragma: no cover - torch.nn.Module contract
        return super().parameters()


def make_config(**overrides) -> PI05RLTConfig:
    values = dict(
        mode="online_rl",
        vlm_hidden_dim=Z, rlt_hidden_dim=Z, rlt_num_heads=2,
        rlt_encoder_layers=1, rlt_decoder_layers=1, rlt_dropout=0.0,
        n_action_steps_rl=C, action_stride=2, state_dim=S, action_dim=A,
    )
    values.update(overrides)
    return PI05RLTConfig(**values)


def make_batch(state: float = 0.0) -> dict:
    return {
        "observation.state": torch.tensor([[state] * S], dtype=torch.float32),
        "observation.language.tokens": torch.zeros(1, 4, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(1, 4, dtype=torch.bool),
    }


def make_policy(vla: FakeVLA, **cfg_overrides) -> PI05RLTPolicy:
    policy = PI05RLTPolicy(make_config(**cfg_overrides))
    policy.load_frozen_vla(vla)
    policy.eval()
    return policy


def drain(policy, n: int, batch) -> list:
    return [policy.select_action(batch) for _ in range(n)]


def expected_ref_rows(vla: FakeVLA, n=C) -> torch.Tensor:
    full = vla.sample_actions([], [], torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.bool))
    return full[0, :n]  # contiguous reference chunk (make_pi05_reference_chunk semantics)


def test_legacy_path_unchanged_when_deploy_off():
    vla = FakeVLA()
    policy = make_policy(vla)
    batch = make_batch()
    chunk = policy._get_action_chunk(batch)  # actor chunk (legacy path)
    pops = drain(policy, C, batch)
    assert len(pops) == C
    for i, a in enumerate(pops):
        torch.testing.assert_close(a, chunk[0, i].unsqueeze(0))
    # pop C+1 -> refills with an identical chunk (deterministic VLA)
    again = drain(policy, 1, batch)[0]
    torch.testing.assert_close(again, chunk[0, 0].unsqueeze(0))


def test_actor_takeover_matrix_via_manual_critical(tmp_path):
    classifier_path = _write_always_classifier(tmp_path, 1.0)
    vla = FakeVLA()
    policy = make_policy(vla, deploy_auto_critical=True, critical_classifier_path=str(classifier_path),
                         actor_critical_delay_steps=0)
    batch = make_batch()
    ref = expected_ref_rows(vla)

    # gate off -> pure VLA reference
    wins = drain(policy, C, batch)
    for i, a in enumerate(wins):
        torch.testing.assert_close(a[0], ref[i], msg=f"gate-off row {i}")
    assert not policy.actor_enabled

    # manual critical ON + gate ON -> actor chunk
    policy.set_actor_enabled(True)
    policy.set_critical_manual(True)
    assert policy.critical_active
    chunk = policy._get_action_chunk(batch)  # deterministic -> same actor chunk
    pops = drain(policy, C, batch)
    for i, a in enumerate(pops):
        torch.testing.assert_close(a[0], chunk[0, i], msg=f"actor row {i}")

    # critical OFF (manual) -> back to VLA even with gate on
    policy.set_critical_manual(False)
    wins = drain(policy, C, batch)
    for i, a in enumerate(wins):
        torch.testing.assert_close(a[0], ref[i], msg=f"gate-on-critical-off row {i}")


def test_delay_sticky_gate_and_episode_reset(tmp_path):
    classifier_path = _write_always_classifier(tmp_path, 1.0)
    vla = FakeVLA()
    policy = make_policy(vla, deploy_auto_critical=True, critical_classifier_path=str(classifier_path),
                         actor_critical_delay_steps=20)
    batch = make_batch()
    ref = expected_ref_rows(vla)
    policy.set_actor_enabled(True)
    policy.set_critical_manual(True)

    # window 1: sustained = 10 < 20 -> VLA
    w1 = drain(policy, C, batch)
    for i, a in enumerate(w1):
        torch.testing.assert_close(a[0], ref[i])
    # window 2: sustained = 20 -> actor
    chunk2 = policy._get_action_chunk(batch)
    w2 = drain(policy, C, batch)
    for i, a in enumerate(w2):
        torch.testing.assert_close(a[0], chunk2[0, i])

    # episode reset: critical cleared, actor gate stays
    policy.reset()
    assert not policy.critical_active
    assert policy.actor_enabled
    assert policy._critical_sustained_steps == 0
    # after reset, critical off -> VLA again
    w3 = drain(policy, C, batch)
    for i, a in enumerate(w3):
        torch.testing.assert_close(a[0], ref[i])


def test_auto_detector_flips_critical_and_actor_takes_over(tmp_path):
    """Classifier trained on z sign; FakeVLA.bias flips -> critical ON -> actor."""
    classifier_path = _write_sign_classifier(tmp_path, z_dim=Z)
    vla = FakeVLA()
    policy = make_policy(vla, deploy_auto_critical=True, critical_classifier_path=str(classifier_path),
                         auto_critical_threshold_on=0.7, auto_critical_threshold_off=0.3,
                         auto_critical_smooth_steps=3, auto_critical_min_on_chunks=0,
                         actor_critical_delay_steps=20)
    policy.set_actor_enabled(True)
    # window 1: safe state -> VLA, z cached
    batch = make_batch(state=2.0)
    w1 = drain(policy, C, batch)
    ref = expected_ref_rows(vla)
    for i, a in enumerate(w1):
        torch.testing.assert_close(a[0], ref[i])
    assert not policy.critical_active

    # window 2: critical state -> detection flips critical ON (sustained 10 < 20)
    batch_crit = make_batch(state=-2.0)
    w2 = drain(policy, C, batch_crit)
    for i, a in enumerate(w2):
        torch.testing.assert_close(a[0], ref[i])
    assert policy.critical_active

    # window 3: critical sustained (20 >= 20) -> actor chunk
    chunk3 = policy._get_action_chunk(batch_crit)
    w3 = drain(policy, C, batch_crit)
    for i, a in enumerate(w3):
        torch.testing.assert_close(a[0], chunk3[0, i])


def _write_always_classifier(tmp_path, sign: float):
    """A classifier that always votes critical (separability via constant input)."""
    model = CriticalPhaseClassifier(z_dim=Z, proprio_dim=S, hidden_dim=16, num_layers=1)
    # force logits[1] >> logits[0] for the constant input the policy will feed
    with torch.no_grad():
        model.net[-1].weight.zero_()
        model.net[-1].bias.zero_()
        # input z is ~constant 0.1*sign over all dims + normalized proprio; set
        # first-layer so the sign reaches the head strongly enough.
        torch.nn.init.zeros_(model.net[0].weight)
        model.net[0].bias.zero_()
    path = tmp_path / "cls.pt"
    torch.save({
        "classifier_state_dict": model.state_dict(),
        "proprio_mean": np.zeros(S, dtype=np.float32),
        "proprio_std": np.ones(S, dtype=np.float32),
        "z_dim": Z, "proprio_dim": S, "hidden_dim": 16, "num_layers": 1,
    }, path)
    return path


def _write_sign_classifier(tmp_path, z_dim: int):
    """Train a classifier on proprio sign: state[:, 0] < 0 -> critical.

    The batch state is controlled per window in the test (encoder z is not
    directly controllable), so the detector flips deterministically.
    """
    torch.manual_seed(0)
    model = CriticalPhaseClassifier(z_dim=z_dim, proprio_dim=S, hidden_dim=16, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    z = torch.randn(128, z_dim)
    p_pos = torch.cat([torch.full((64, 1), 2.0), torch.randn(64, S - 1)], dim=1)
    p_neg = torch.cat([torch.full((64, 1), -2.0), torch.randn(64, S - 1)], dim=1)
    labels = torch.cat([torch.zeros(64, dtype=torch.long), torch.ones(64, dtype=torch.long)])
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(z, torch.cat([p_pos, p_neg])), labels)
        loss.backward()
        opt.step()
    with torch.no_grad():
        p_crit = torch.softmax(model(z[:1], torch.full((1, S), -2.0)), -1)[0, 1]
        p_safe = torch.softmax(model(z[:1], torch.full((1, S), 2.0)), -1)[0, 1]
        assert p_crit > 0.9 and p_safe < 0.1, (p_crit.item(), p_safe.item())
    path = tmp_path / "cls_sign.pt"
    torch.save({
        "classifier_state_dict": model.state_dict(),
        "proprio_mean": np.zeros(S, dtype=np.float32),
        "proprio_std": np.ones(S, dtype=np.float32),
        "z_dim": z_dim, "proprio_dim": S, "hidden_dim": 16, "num_layers": 2,
    }, path)
    return path
