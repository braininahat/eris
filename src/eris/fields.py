"""2-D and 3-D scalar-field operations on entropy stacks.

Conventions:
- ``H`` (single layer) is shaped ``(gy, gx)``.
- ``H_stack`` (per image, across depth) is ``(L, gy, gx)``.
- ``H_volume`` (per layer, across video frames) is ``(t, gy, gx)``.
- ``H_4d`` (per video, across both depth and time) is ``(L, t, gy, gx)``.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage


def gradient_2d(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spatial ``(Hy, Hx)`` for a layer's ``(gy, gx)`` field."""
    if H.ndim != 2:
        raise ValueError(f"H must be 2-D (gy, gx); got {H.shape}")
    gy, gx = np.gradient(H)
    return gy, gx


def laplacian_2d(H: np.ndarray) -> np.ndarray:
    """Discrete 2-D Laplacian ``∇²H``."""
    if H.ndim != 2:
        raise ValueError(f"H must be 2-D (gy, gx); got {H.shape}")
    return scipy.ndimage.laplace(H)


def speed_grid_2d(H: np.ndarray) -> np.ndarray:
    """Per-cell gradient magnitude ``|∇H|``."""
    gy, gx = gradient_2d(H)
    return np.hypot(gy, gx)


def depth_derivatives(H_stack: np.ndarray) -> dict[str, np.ndarray]:
    """First/second derivatives along the depth axis ``L``.

    Args:
        H_stack: ``(L, gy, gx)``.
    Returns:
        Dict of ``(L, gy, gx)`` arrays:
        ``dH_dL`` (first), ``d2H_dL2`` (second), ``grad_mag`` (per-layer
        spatial gradient magnitude), ``d_grad_dL`` (rate of change of the
        gradient magnitude with depth).
    """
    if H_stack.ndim != 3:
        raise ValueError(f"H_stack must be (L, gy, gx); got {H_stack.shape}")
    dH_dL = np.gradient(H_stack, axis=0)
    d2H_dL2 = np.gradient(dH_dL, axis=0)
    gy = np.gradient(H_stack, axis=1)
    gx = np.gradient(H_stack, axis=2)
    grad_mag = np.hypot(gy, gx)
    d_grad_dL = np.gradient(grad_mag, axis=0)
    return {
        "dH_dL": dH_dL,
        "d2H_dL2": d2H_dL2,
        "grad_mag": grad_mag,
        "d_grad_dL": d_grad_dL,
    }


def gradient_3d(H_volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatio-temporal gradient ``(Ht, Hy, Hx)`` of a 3-D field.

    Args:
        H_volume: ``(t, gy, gx)``.
    """
    if H_volume.ndim != 3:
        raise ValueError(f"H_volume must be (t, gy, gx); got {H_volume.shape}")
    Ht = np.gradient(H_volume, axis=0)
    Hy = np.gradient(H_volume, axis=1)
    Hx = np.gradient(H_volume, axis=2)
    return Ht, Hy, Hx


def laplacian_3d(H_volume: np.ndarray) -> np.ndarray:
    """3-D discrete Laplacian on ``(t, gy, gx)``."""
    if H_volume.ndim != 3:
        raise ValueError(f"H_volume must be (t, gy, gx); got {H_volume.shape}")
    return scipy.ndimage.laplace(H_volume)


def transition_layer(H_stack: np.ndarray) -> int:
    """Argmax across layers of ΔH-std (a scalar per layer).

    Returns the layer index ``L`` of the largest within-layer ΔH spread,
    a proxy for the phase-transition point. ``H_stack`` is ``(L, gy, gx)``.
    """
    if H_stack.ndim != 3:
        raise ValueError(f"H_stack must be (L, gy, gx); got {H_stack.shape}")
    dH = np.diff(H_stack, axis=0)               # (L-1, gy, gx)
    dh_std = dH.reshape(dH.shape[0], -1).std(axis=1)
    # Layer index in original L: index 0 of dH = L_1 → L_2 transition.
    return int(np.argmax(dh_std)) + 1


def per_layer_stats(H_stack: np.ndarray) -> dict[str, np.ndarray]:
    """Layer-wise scalar summaries used in ``phase_curves`` figures."""
    if H_stack.ndim != 3:
        raise ValueError(f"H_stack must be (L, gy, gx); got {H_stack.shape}")
    H_flat = H_stack.reshape(H_stack.shape[0], -1)
    grad_mag = np.array([speed_grid_2d(H).mean() for H in H_stack])
    dH = np.diff(H_stack, axis=0)
    dH_std = np.concatenate([[0.0], dH.reshape(dH.shape[0], -1).std(axis=1)])
    return {
        "H_mean": H_flat.mean(axis=1),
        "H_std": H_flat.std(axis=1),
        "abs_grad_H_mean": grad_mag,
        "dH_std": dH_std,
    }
