"""Unit tests for export_rlt_pretrained.py pure helpers.

Pins the file-filtering logic that turns a π0.5 pretrained_model directory
into a self-contained exported dir (processor artifacts only — policy weights
are never copied from the π0.5 dir).
"""
import importlib.util
import json
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

_EXPORT = Path(__file__).parents[1] / "scripts" / "export_rlt_pretrained.py"
spec = importlib.util.spec_from_file_location("export_rlt_pretrained", _EXPORT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def _make_pi05_dir(tmp_path: Path) -> Path:
    src = tmp_path / "pi05"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "model.safetensors").write_bytes(b"w")
    (src / "train_config.json").write_text("{}")
    (src / "policy_preprocessor.json").write_text('{"name": "policy_preprocessor"}')
    (src / "policy_postprocessor.json").write_text('{"name": "policy_postprocessor"}')
    (src / "policy_preprocessor_step_2_normalizer_processor.safetensors").write_bytes(b"n")
    (src / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(b"u")
    (src / "processor.json").write_text("{}")  # generic name without "policy_" prefix
    (src / "notes.txt").write_text("not a processor")
    (src / "sub").mkdir()  # unrelated directories are ignored
    (src / "tokenizer").mkdir()  # the tokenizer artifact declared by policy_preprocessor.json
    (src / "tokenizer" / "tokenizer_config.json").write_text("{}")
    return src


def test_copy_processor_artifacts_filters_correctly(tmp_path):
    src = _make_pi05_dir(tmp_path)
    dst = tmp_path / "out"
    dst.mkdir()
    copied = export.copy_processor_artifacts(src, dst)
    assert sorted(copied) == sorted([
        "processor.json",  # generic processor config (newer lerobot layout)
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "tokenizer/",  # required artifact: resolved next to policy_preprocessor.json
    ])
    # policy weights / generic configs / text files must NOT be copied
    names = {p.name for p in dst.iterdir()}
    assert names == {c.rstrip("/") for c in copied}
    assert (dst / "tokenizer" / "tokenizer_config.json").is_file()
    assert "model.safetensors" not in names and "config.json" not in names
    assert "train_config.json" not in names and "notes.txt" not in names
    assert "sub" not in names  # unrelated directories stay excluded


def test_copy_processor_artifacts_requires_the_tokenizer_dir(tmp_path):
    """A π0.5 dir without tokenizer/ would export a dir that record/rollout cannot load."""
    src = _make_pi05_dir(tmp_path)
    for child in (src / "tokenizer").iterdir():
        child.unlink()
    (src / "tokenizer").rmdir()
    dst = tmp_path / "out_tokenizer"
    dst.mkdir()
    try:
        export.copy_processor_artifacts(src, dst)
    except SystemExit as exc:
        assert "Tokenizer directory not found" in str(exc)
    else:
        raise AssertionError("expected SystemExit when the tokenizer dir is missing")


def test_copy_processor_artifacts_fails_when_nothing_to_copy(tmp_path):
    src = tmp_path / "empty_pi05"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"w")  # only weights, no processors
    dst = tmp_path / "out2"
    dst.mkdir()
    try:
        export.copy_processor_artifacts(src, dst)
    except SystemExit as exc:
        assert "No processor artifacts" in str(exc)
    else:
        raise AssertionError("expected SystemExit when the π0.5 dir has no processor files")


def test_export_config_keeps_runtime_contract():
    stage1 = {
        "config": {
            "vlm_hidden_dim": 2048,
            "rlt_hidden_dim": 2048,
            "rlt_encoder_layers": 2,
            "rlt_decoder_layers": 2,
            "rlt_num_heads": 8,
            "rlt_dropout": 0.1,
            "state_dim": 6,
            "action_dim": 6,
            "action_stride": 2,
            "n_action_steps_rl": 10,
            "actor_critic_style": "paper_mlp",
            "actor_hidden_dim": 256,
            "critic_hidden_dim": 256,
            "policy_fixed_std": 0.05,
        }
    }
    stage2 = {
        "schema_version": 3,
        "actor_contract": "paper_full_output_v1",
        "rl_chunk_length": 10,
        "actor_state_dict": {
            "net.0.weight": torch.zeros(256, 2114),
            "net.2.weight": torch.zeros(60, 256),
        },
        "critic_state_dict": {
            "q1.0.weight": torch.zeros(256, 2114),
            "q1.2.weight": torch.zeros(1, 256),
        },
        "runtime_provenance": {"action_stride": 2},
    }
    kwargs = export.build_export_config_kwargs(stage1, stage2, device="cpu")
    assert kwargs["state_dim"] == 6
    assert kwargs["action_dim"] == 6
    assert kwargs["n_action_steps_rl"] == 10
    assert kwargs["action_stride"] == 2
    assert kwargs["actor_critic_style"] == "paper_mlp"
    assert kwargs["actor_hidden_dim"] == 256
    assert kwargs["critic_hidden_dim"] == 256
    assert kwargs["action_feature_names"] == [
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
    ]


def test_export_shape_validation_rejects_wrong_config():
    stage2 = {
        "actor_state_dict": {"net.0.weight": torch.zeros(256, 2114)},
        "critic_state_dict": {"q1.0.weight": torch.zeros(256, 2114)},
    }
    kwargs = {
        "rlt_hidden_dim": 2048,
        "state_dim": 6,
        "action_dim": 6,
        "n_action_steps_rl": 10,
        "actor_hidden_dim": 256,
        "critic_hidden_dim": 256,
    }
    export.validate_export_shapes(kwargs, stage2)
    kwargs["action_stride"] = 1
    kwargs["rlt_hidden_dim"] = 1024
    with pytest.raises(ValueError, match="input width"):
        export.validate_export_shapes(kwargs, stage2)


def test_deployment_metadata_pins_so101_order():
    stage1 = {"step": 123}
    stage2 = {
        "actor_contract": "paper_full_output_v1",
        "rl_chunk_length": 10,
        "runtime_provenance": {"action_stride": 2},
        "episode": 7,
    }
    metadata = export.build_deployment_metadata(stage1, stage2, "/pi05")
    assert metadata["action_names"] == [
        "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
    ]
    assert metadata["rl_chunk_length"] == 10
    assert metadata["action_stride"] == 2


def test_saved_native_config_requires_action_order(tmp_path):
    names = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
             "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"action_feature_names": names}), encoding="utf-8")
    export.validate_saved_action_order(path, names)

    path.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="PI05RLTConfig"):
        export.validate_saved_action_order(path, names)
