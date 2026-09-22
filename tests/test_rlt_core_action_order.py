import importlib.util
import sys
from pathlib import Path

import pytest
import torch


CORE = Path(__file__).parents[1] / "scripts" / "rlt_core.py"
spec = importlib.util.spec_from_file_location("rlt_core_action_order", CORE)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def test_identity_order_is_preserved():
    names = [f"{name}.pos" for name in core.SO101_JOINT_NAMES]
    action = torch.arange(6, dtype=torch.float32)
    out = core.reorder_action_tensor(action, source_names=names, target_names=names)
    torch.testing.assert_close(out, action)


def test_policy_action_is_reordered_to_robot_feature_order():
    source = [f"{name}.pos" for name in core.SO101_JOINT_NAMES]
    target = ["gripper.pos", "wrist_roll.pos", "wrist_flex.pos", "elbow_flex.pos", "shoulder_lift.pos", "shoulder_pan.pos"]
    action = torch.arange(6, dtype=torch.float32)
    out = core.reorder_action_tensor(action, source_names=source, target_names=target)
    torch.testing.assert_close(out, torch.tensor([5, 4, 3, 2, 1, 0], dtype=torch.float32))


@pytest.mark.parametrize("bad", [
    ["shoulder_pan.pos"] * 6,
    ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "other.pos"],
])
def test_invalid_robot_order_is_rejected(bad):
    source = [f"{name}.pos" for name in core.SO101_JOINT_NAMES]
    with pytest.raises(ValueError):
        core.reorder_action_tensor(torch.zeros(6), source_names=source, target_names=bad)
