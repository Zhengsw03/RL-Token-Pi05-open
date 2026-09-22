"""Critical-phase classifier and runtime detector for RL-Token Stage 2.

Migrated from the RL-Token-Pi05 project (``scripts/critical_networks.py``,
v1.14) so that the auto-critical deployment behavior lives with the
``pi05_rlt`` policy in the lerobot package. Pure torch/numpy — no lerobot
imports needed to run inference.

- ``CriticalPhaseClassifier``: (z_rl, proprio) -> P(critical) MLP
- ``CriticalPhaseDetector``: runtime wrapper — one ``update`` per autonomous
  chunk, moving-averaged probability with hysteresis thresholds decides the
  desired critical-phase state (``None`` = no change).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn


class CriticalPhaseClassifier(nn.Module):
    """MLP classifier on the RL token + proprioception: class 1 = critical.

    Inputs:
      z_rl:     (B, z_dim) float32 — RL token from the frozen Stage 1 encoder
      proprio:  (B, proprio_dim) float32 (already normalized at runtime)
    """

    def __init__(
        self,
        z_dim: int = 256,
        proprio_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        layers: list[nn.Module] = []
        in_dim = z_dim + proprio_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z_rl: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_rl, proprio], dim=-1))


class CriticalPhaseDetector:
    """Runtime wrapper around :class:`CriticalPhaseClassifier`.

    One ``update`` per autonomous chunk; smoothed probability with hysteresis
    thresholds decides the desired critical-phase state (``None`` = no change).
    Manual-override cooldown is enforced by the caller (training script) via
    the ``CRITICAL_MANUAL_OVERRIDE_AT`` wall-clock stamp.
    """

    def __init__(
        self,
        checkpoint_path,
        device,
        *,
        threshold_on: float = 0.6,
        threshold_off: float = 0.4,
        smooth_steps: int = 10,
        min_on_chunks: int = 3,
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.z_dim = int(checkpoint.get("z_dim", 256))
        self.proprio_dim = int(checkpoint.get("proprio_dim", 6))
        self.classifier = CriticalPhaseClassifier(
            z_dim=self.z_dim,
            proprio_dim=self.proprio_dim,
            hidden_dim=int(checkpoint.get("hidden_dim", 256)),
            num_layers=int(checkpoint.get("num_layers", 2)),
        ).to(device)
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.classifier.eval()
        self.device = device
        self.proprio_mean = torch.as_tensor(
            checkpoint["proprio_mean"], dtype=torch.float32, device=device,
        )
        self.proprio_std = torch.as_tensor(
            checkpoint["proprio_std"], dtype=torch.float32, device=device,
        ).clamp_min(1e-6)
        if self.proprio_mean.numel() != self.proprio_dim:
            raise ValueError(
                f"Classifier proprio_mean has {self.proprio_mean.numel()} dims, "
                f"expected {self.proprio_dim}"
            )
        self.threshold_on = threshold_on
        self.threshold_off = threshold_off
        self.smooth_steps = max(1, smooth_steps)
        self.min_on_chunks = max(0, min_on_chunks)
        self._probabilities: deque[float] = deque(maxlen=self.smooth_steps)
        self._update_count = 0
        self._on_since_update: int | None = None  # update count when ON started

    def reset(self) -> None:
        self._probabilities.clear()

    def update(self, z_rl, proprio) -> bool | None:
        """Return True/False when the smoothed probability crosses a threshold, else None.

        ON uses the moving average (debounced). OFF triggers on either the
        average OR the single latest sample dropping below ``threshold_off``:
        when the critical phase ends the probability collapses, and the
        average would lag by ~smooth_steps chunks, delaying the OFF by seconds.
        Hysteresis (off < on) prevents flip-flopping: after a fast OFF, the
        average must climb back above ``threshold_on`` to turn ON again.
        """
        z_t = torch.as_tensor(
            np.asarray(z_rl, dtype=np.float32), device=self.device,
        ).reshape(1, -1)
        if z_t.shape[1] != self.z_dim:
            raise ValueError(
                f"Classifier expects z_rl dim {self.z_dim}, got {z_t.shape[1]}"
            )
        p_t = (
            torch.as_tensor(np.asarray(proprio, dtype=np.float32), device=self.device)
            - self.proprio_mean
        ) / self.proprio_std
        p_t = p_t.reshape(1, -1)
        with torch.no_grad():
            logits = self.classifier(z_t, p_t)
            probability = torch.softmax(logits, dim=-1)[0, 1].item()
        self._probabilities.append(probability)
        self._update_count += 1
        # Fast ON: the mean of the LAST TWO samples (the full moving average
        # would lag ~smooth_steps chunks at the boundary because stale
        # low-probability samples keep the average below threshold_on).
        if len(self._probabilities) >= 2:
            recent2 = (self._probabilities[-1] + self._probabilities[-2]) / 2
        else:
            recent2 = probability
        if recent2 >= self.threshold_on:
            self._on_since_update = self._update_count
            return True
        # Fast OFF: single latest sample (or the full average) below threshold.
        if probability <= self.threshold_off or sum(self._probabilities) / len(self._probabilities) <= self.threshold_off:
            # Debounce: once ON, hold at least ``min_on_chunks`` updates so a
            # brief probability dip cannot flip OFF -> ON -> OFF repeatedly
            # (which reads as false "ON" triggers).
            if (
                self._on_since_update is not None
                and self._update_count - self._on_since_update < self.min_on_chunks
            ):
                return None
            self._on_since_update = None
            return False
        return None

    @property
    def last_probability(self) -> float | None:
        return self._probabilities[-1] if self._probabilities else None
