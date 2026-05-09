"""Generic per-patch entropy-stack extraction for HuggingFace ViT models.

The smoke-test ``viz_one.py`` hard-coded ViT-B/16 (12 blocks, 768 hidden,
14×14 patch grid). Step A of the plan needs the same pipeline run on
ViT-L/14, DINO-v2, and CLIP-ViT-B/16 — all of which differ in some
combination of (n_layers, hidden_dim, patch_grid_size, CLS-token presence,
final-LayerNorm position).

This module abstracts the forward pass + per-patch entropy field
construction over those differences. Returns a per-image ``H_stack`` of
shape ``(n_layers, gy, gx)`` regardless of the underlying architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel

from .estimators import vasicek_entropy


# ── per-architecture spec ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArchSpec:
    """How to forward this model and where to find its residual streams."""
    model_name: str
    has_cls: bool
    grid_size: tuple[int, int]   # (gy, gx) AFTER stripping CLS
    n_layers: int                # number of transformer-block outputs to analyse
    hidden_dim: int


# ── architecture registry ─────────────────────────────────────────────────────


def _vit_forward(model, processor, images, device):
    """ViT / DINO-v2-style forward returning all hidden states."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states


def _clip_forward(model, processor, images, device):
    """CLIP needs its vision encoder explicitly."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.vision_model(
            pixel_values=inputs["pixel_values"], output_hidden_states=True
        )
    return out.hidden_states


def load_model(arch: str, device: str = "cuda"):
    """Load (model, processor, forward_fn, ArchSpec) for an arch nickname."""
    if arch in ("vit_b16", "google/vit-base-patch16-224"):
        name = "google/vit-base-patch16-224"
        model = AutoModel.from_pretrained(name).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(name)
        spec = ArchSpec(model_name=name, has_cls=True,
                        grid_size=(14, 14), n_layers=12, hidden_dim=768)
        return model, proc, _vit_forward, spec
    if arch in ("vit_l16", "google/vit-large-patch16-224"):
        name = "google/vit-large-patch16-224"
        model = AutoModel.from_pretrained(name).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(name)
        spec = ArchSpec(model_name=name, has_cls=True,
                        grid_size=(14, 14), n_layers=24, hidden_dim=1024)
        return model, proc, _vit_forward, spec
    if arch in ("dinov2_b", "facebook/dinov2-base"):
        name = "facebook/dinov2-base"
        model = AutoModel.from_pretrained(name).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(name)
        # DINO-v2 base: patch=14, image=224 → 16x16 grid; has CLS.
        spec = ArchSpec(model_name=name, has_cls=True,
                        grid_size=(16, 16), n_layers=12, hidden_dim=768)
        return model, proc, _vit_forward, spec
    if arch in ("clip_b16", "openai/clip-vit-base-patch16"):
        name = "openai/clip-vit-base-patch16"
        model = CLIPModel.from_pretrained(name).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(name)
        spec = ArchSpec(model_name=name, has_cls=True,
                        grid_size=(14, 14), n_layers=12, hidden_dim=768)
        return model, proc, _clip_forward, spec
    raise ValueError(f"unknown arch: {arch}")


# ── entropy field per layer per image ─────────────────────────────────────────


def _per_patch_entropy_field(
    hidden: torch.Tensor, spec: ArchSpec, estimator: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """``hidden`` of shape ``(1, n_tokens, hidden_dim)`` → ``(gy, gx)``."""
    h = hidden.detach().cpu().float().numpy()
    if h.ndim != 3 or h.shape[0] != 1:
        raise ValueError(f"hidden must be (1, n_tokens, hidden_dim); got {h.shape}")
    if spec.has_cls:
        h = h[:, 1:, :]                                 # drop CLS
    gy, gx = spec.grid_size
    n_patches = gy * gx
    if h.shape[1] != n_patches:
        raise ValueError(
            f"after CLS-strip got {h.shape[1]} patches; spec expects {n_patches} "
            f"({gy}×{gx}) for {spec.model_name}"
        )
    if h.shape[2] != spec.hidden_dim:
        raise ValueError(
            f"hidden_dim mismatch: got {h.shape[2]}, spec {spec.hidden_dim}"
        )
    h = h[0]                                            # (n_patches, hidden)
    # per-patch standardisation
    mu = h.mean(axis=1, keepdims=True)
    sd = h.std(axis=1, keepdims=True) + 1e-9
    h = (h - mu) / sd
    H = estimator(h)                                    # (n_patches,)
    return H.reshape(gy, gx)


def extract_entropy_stack(
    arch: str,
    images: Iterable[Image.Image],
    device: str = "cuda",
    estimator: Callable[[np.ndarray], np.ndarray] = vasicek_entropy,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, ArchSpec]:
    """Run ``arch`` over ``images`` and return the per-image H_stack.

    Args:
        arch: registry nickname or HF model id (see ``load_model``).
        images: PIL images in any size; ``AutoImageProcessor`` handles
            resize / normalisation per arch.
        device: cuda/cpu.
        estimator: per-patch entropy estimator.
        progress: callback ``(idx, n_total)`` for progress reporting.

    Returns:
        ``(stacks, spec)`` where ``stacks`` has shape
        ``(n_images, n_layers, gy, gx)`` (np.float32) and ``spec`` is the
        ArchSpec used.
    """
    model, processor, forward_fn, spec = load_model(arch, device=device)
    images = list(images)
    n = len(images)
    out = np.empty((n, spec.n_layers, spec.grid_size[0], spec.grid_size[1]),
                   dtype=np.float32)
    for i, img in enumerate(images):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img)).convert("RGB")
        else:
            img = img.convert("RGB")
        hidden_states = forward_fn(model, processor, img, device)
        # Convention: hidden_states[0] = patch-embedding pre-block-1; we analyse
        # 1..n_layers (post each transformer block).
        for L in range(spec.n_layers):
            out[i, L] = _per_patch_entropy_field(
                hidden_states[L + 1], spec, estimator
            )
        if progress is not None:
            progress(i + 1, n)
    return out, spec
