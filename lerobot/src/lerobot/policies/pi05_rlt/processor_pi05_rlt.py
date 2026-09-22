"""Processor for π0.5-RLT policy.

Two regimes:

- ``mode="rlt_training"`` (Stage 1): the policy consumes *precomputed* frozen
  π0.5 prefix embeddings (``vlm_embeddings`` + ``prefix_mask``), so the
  preprocessor is a light pass-through (rename → batch dimension → device).
  No normalization/tokenization is applied: the embeddings already live in
  the VLA hidden space and the Stage-1 loss is a masked MSE on them.

- ``mode="online_rl"/"inference"`` (Stage 2 / robot): delegates to π0.5's
  processor since the frozen π0.5 backbone expects identical input formatting
  (images + language tokens including discretized state).
"""

from typing import Any

import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.processor import DeviceProcessorStep, PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import batch_to_transition
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

# Stage-1 batch keys (dataset feature names of the precomputed π0.5 prefix).
RLT_STAGE1_KEYS = ("vlm_embeddings", "prefix_mask")


def _stage1_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """Like ``batch_to_transition`` but keeps the Stage-1 embedding tensors.

    ``batch_to_transition`` drops any key that is not an observation/action/
    complementary key, so the RLT tensors are routed into the observation group
    (kept verbatim by ``transition_to_batch``) instead.
    """
    transition = batch_to_transition(batch)
    observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
    for key in RLT_STAGE1_KEYS:
        if key in batch:
            observation[key] = batch[key]
    transition[TransitionKey.OBSERVATION] = observation
    return transition


def make_pi05_rlt_pre_post_processors(
    config: PI05RLTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Constructs pre/post-processor pipelines for π0.5-RLT.

    See module docstring for the mode-dependent behavior.
    """
    if config.mode == "rlt_training":
        # Stage 1: pass-through for precomputed embeddings (no π0.5 needed).
        preprocessor = PolicyProcessorPipeline(
            steps=[DeviceProcessorStep(device=config.device)],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=_stage1_batch_to_transition,
        )
        postprocessor = PolicyProcessorPipeline(
            steps=[DeviceProcessorStep(device="cpu")],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        )
        return preprocessor, postprocessor

    # Create a PI05Config from our config for the delegate call
    pi05_config = PI05Config(
        paligemma_variant=config.paligemma_variant,
        action_expert_variant=config.action_expert_variant,
        dtype=config.dtype,
        chunk_size=config.chunk_size,
        n_action_steps=config.n_action_steps_rl,
        max_state_dim=config.max_state_dim,
        max_action_dim=config.max_action_dim,
        image_resolution=config.image_resolution,
        tokenizer_max_length=config.tokenizer_max_length,
        normalization_mapping=config.normalization_mapping,
        device=config.device,
    )
    # Populate input/output features from our config
    pi05_config.input_features = config.input_features
    pi05_config.output_features = config.output_features
    pi05_config.validate_features()

    return make_pi05_pre_post_processors(pi05_config, dataset_stats)
