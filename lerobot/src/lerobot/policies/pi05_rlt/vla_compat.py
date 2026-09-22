# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Frozen π0.5 backbone compatibility layer: embedding extraction for RLT code
on both older and newer lerobot generations.

Background
----------
After the upstream π0.5 refactor:

* ``PI05Pytorch.extract_embeddings()`` was removed, so this module rebuilds the
  same semantics from public APIs (``embed_prefix`` / ``embed_suffix``);
* ``make_att_2d_masks`` / ``prepare_attention_masks_4d`` / ``resize_with_pad_torch``
  / ``pad_vector`` moved from ``policies.pi05.modeling_pi05`` to
  ``policies.common.vla_utils``, which :func:`resolve_pi05_utils` adapts to
  automatically;
* ``PI05Policy`` no longer exposes ``extract_embeddings``, so callers should pass
  ``PI05Policy(...).model`` (that is, ``PI05Pytorch``).

The module is shared by ``modeling_pi05_rlt.PI05RLTPolicy`` and by this project's
``precompute_pi05_embeddings.py`` / ``train_rlt_stage1_pi05.py`` so the same
compatibility logic does not sprawl across files (the next upstream π0.5 change
only needs this one file updated).

Public API
----------
* :func:`resolve_pi05_utils` -> ``(make_att_2d_masks, prepare_attention_masks_4d, resize_with_pad_torch, pad_vector)``
* :func:`resize_with_pad` -> thin wrapper around image padding-resize (no dependency on modeling_pi05)
* :func:`unwrap_pi05_backbone` -> get ``PI05Pytorch`` out of a ``PI05Policy`` or a bare backbone
* :func:`extract_embeddings` -> equivalent to the old ``PI05Pytorch.extract_embeddings``
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from torch import Tensor


@lru_cache(maxsize=1)
def resolve_pi05_utils():
    """Resolve the π0.5 helper functions once and return a 4-tuple.

    Returns:
        (make_att_2d_masks, prepare_attention_masks_4d | None, resize_with_pad_torch, pad_vector)

        Older lerobot builds implement 4D mask expansion as the instance method
        ``PI05Pytorch._prepare_attention_masks_4d``; in that case the second item is
        ``None`` and callers must fall back to that instance method.
    """
    try:
        from lerobot.policies.common.vla_utils import (  # newer location
            make_att_2d_masks,
            pad_vector,
            prepare_attention_masks_4d,
            resize_with_pad_torch,
        )

        return (make_att_2d_masks, prepare_attention_masks_4d, resize_with_pad_torch, pad_vector)
    except ImportError:
        from lerobot.policies.pi05.modeling_pi05 import (  # older location
            make_att_2d_masks,
            pad_vector,
            resize_with_pad_torch,
        )

        return (make_att_2d_masks, None, resize_with_pad_torch, pad_vector)


def resize_with_pad(images: Tensor, height: int, width: int, mode: str = "bilinear") -> Tensor:
    """Padding-resize ``images`` (given as [B, H, W, C] or [*b, c, h, w]) keeping
    the aspect ratio.

    A pure forward to the implementation of the installed lerobot version, so
    callers do not need to know which module it lives in.
    """
    *_, resize_with_pad_torch, _ = resolve_pi05_utils()
    return resize_with_pad_torch(images, height, width, mode=mode)


def unwrap_pi05_backbone(model: Any) -> Any:
    """Return the callable backbone (``PI05Pytorch``).

    ``PI05Policy`` is a ``PreTrainedPolicy`` wrapper whose real forward backbone is
    ``.model``; recent ``PI05Policy`` no longer forwards ``extract_embeddings``, so
    unwrapping is required.
    """
    if model is None:
        raise ValueError("pi05 model must not be None")
    if hasattr(model, "model") and not hasattr(model, "embed_prefix"):
        return model.model
    return model


def extract_embeddings(
    vla: Any,
    images: list[Tensor],
    img_masks: list[Tensor],
    tokens: Tensor,
    masks: Tensor,
    actions: Tensor,
    *,
    chunk_size: int,
    max_action_dim: int,
    noise: Tensor | None = None,
    time: Tensor | None = None,
    image_only: bool = False,
):
    """Equivalent to the old ``PI05Pytorch.extract_embeddings``.

    Args:
        vla: a ``PI05Pytorch`` instance (or a ``PI05Policy``, which is unwrapped).
        image_only: when True, return only the image-token prefix and mask (the RLT
            paper's approach: drop language embeddings for tasks with a fixed
            instruction).

    Returns:
        ``image_only=False``: ``(prefix_out, suffix_out)``
        ``image_only=True`` : ``(prefix_img, suffix_out, prefix_img_pad)``
    """
    vla = unwrap_pi05_backbone(vla)

    # Prefer the legacy method when the backbone provides it: behaviour then matches
    # historical results exactly.
    legacy = getattr(vla, "extract_embeddings", None)
    if legacy is not None:
        return legacy(
            images, img_masks, tokens, masks, actions,
            noise=noise, time=time, image_only=image_only,
        )

    make_att_2d_masks, prepare_attention_masks_4d, _, _ = resolve_pi05_utils()

    if noise is None:
        noise = vla.sample_noise((actions.shape[0], chunk_size, max_action_dim), actions.device)
    if time is None:
        time = vla.sample_time(actions.shape[0], actions.device)

    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions

    prefix_embs, prefix_pad_masks, prefix_att_masks = vla.embed_prefix(images, img_masks, tokens, masks)
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = vla.embed_suffix(x_t, time)

    if (
        vla.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
        == torch.bfloat16
    ):
        prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d_masks_4d = (
        prepare_attention_masks_4d(att_2d_masks)
        if prepare_attention_masks_4d is not None
        else vla._prepare_attention_masks_4d(att_2d_masks)
    )

    outputs_embeds, _ = vla.paligemma_with_expert.forward(
        attention_mask=att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    prefix_out = outputs_embeds[0]
    suffix_out = outputs_embeds[1][:, -chunk_size:]

    if image_only:
        # embed_prefix places images before language, so the image tokens are the
        # first num_img_tokens entries.
        num_img_tokens = prefix_embs.shape[1] - tokens.shape[1]
        return (
            prefix_out[:, :num_img_tokens].to(torch.float32),
            suffix_out.to(torch.float32),
            prefix_pad_masks[:, :num_img_tokens].to(torch.bool),
        )

    return prefix_out, suffix_out
