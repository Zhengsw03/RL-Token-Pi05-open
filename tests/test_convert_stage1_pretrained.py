"""convert_stage1_pretrained round trip.

A lerobot-trained Stage-1 checkpoint (``pretrained_model/`` dir) converted to
the legacy ``.pt`` bundle must:
- carry the fields the Stage-2 loader validates (rlt_architecture=paper_v1,
  image_only, config dims);
- strict-load back into the same package modules with identical weights.

Runs with tiny dims on CPU.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

LEROBOT_SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "convert_stage1_pretrained", Path(__file__).parents[1] / "scripts" / "convert_stage1_pretrained.py"
)
convert_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_mod)

from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import (
    PI05RLTPolicy,
    RLTokenDecoder,
    RLTokenEncoder,
)


def _make_trained_policy(seed: int = 0):
    cfg = PI05RLTConfig(
        mode="rlt_training",
        vlm_hidden_dim=16,
        rlt_hidden_dim=16,
        rlt_num_heads=2,
        rlt_encoder_layers=1,
        rlt_decoder_layers=1,
        rlt_dropout=0.0,
    )
    policy = PI05RLTPolicy(cfg)
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in policy.rlt_encoder.parameters():
            p.normal_()
        for p in policy.rlt_decoder.parameters():
            p.normal_()
    return policy


def test_convert_round_trip_preserves_weights_and_loader_fields(tmp_path):
    policy = _make_trained_policy()
    pretrained_dir = tmp_path / "pretrained_model"
    policy.save_pretrained(pretrained_dir)

    out_pt = tmp_path / "stage1.pt"
    convert_mod.convert_stage1_pretrained(pretrained_dir, out_pt)

    ckpt = torch.load(out_pt, map_location="cpu", weights_only=False)
    # Stage-2 loader contract fields.
    assert ckpt["rlt_architecture"] == "paper_v1"
    assert ckpt["image_only"] is True
    cfg = ckpt["config"]
    assert cfg["vlm_hidden_dim"] == 16 and cfg["rlt_hidden_dim"] == 16
    assert cfg["rlt_encoder_layers"] == 1 and cfg["rlt_num_heads"] == 2

    # Strict reload into the same package modules the Stage-2 trainer uses.
    enc = RLTokenEncoder(PI05RLTConfig(mode="online_rl", vlm_hidden_dim=16, rlt_hidden_dim=16,
                                       rlt_num_heads=2, rlt_encoder_layers=1, rlt_decoder_layers=1))
    dec = RLTokenDecoder(PI05RLTConfig(mode="online_rl", vlm_hidden_dim=16, rlt_hidden_dim=16,
                                       rlt_num_heads=2, rlt_encoder_layers=1, rlt_decoder_layers=1))
    enc.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    dec.load_state_dict(ckpt["decoder_state_dict"], strict=True)

    # Weights identical to the source policy modules.
    for name, param in policy.rlt_encoder.state_dict().items():
        torch.testing.assert_close(enc.state_dict()[name], param, msg=f"encoder {name}")
    for name, param in policy.rlt_decoder.state_dict().items():
        torch.testing.assert_close(dec.state_dict()[name], param, msg=f"decoder {name}")

    # Forward smoke on the reloaded modules.
    d = 16
    vlm = torch.randn(2, 6, d)
    mask = torch.ones(2, 6, dtype=torch.bool)
    with torch.no_grad():
        z = enc(vlm, mask=mask)
        recon = dec(z, vlm, mask=mask)
    assert tuple(z.shape) == (2, d)
    assert tuple(recon.shape) == (2, 6, d)
    assert torch.isfinite(recon).all()


def test_convert_rejects_non_stage1_mode(tmp_path):
    policy = PI05RLTPolicy(PI05RLTConfig(
        mode="online_rl", vlm_hidden_dim=16, rlt_hidden_dim=16, rlt_num_heads=2,
        rlt_encoder_layers=1, rlt_decoder_layers=1))
    pretrained_dir = tmp_path / "pretrained_model"
    policy.save_pretrained(pretrained_dir)
    with pytest.raises(SystemExit):
        convert_mod.convert_stage1_pretrained(pretrained_dir, tmp_path / "x.pt")
