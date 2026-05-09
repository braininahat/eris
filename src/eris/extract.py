"""Generic per-patch entropy-stack extraction for HuggingFace ViT models.

The smoke-test ``viz_one.py`` hard-coded ViT-B/16 (12 blocks, 768 hidden,
14×14 patch grid). Step A of the plan needs the same pipeline run on
ViT-L/14, DINO-v2, and CLIP-ViT-B/16 — all of which differ in some
combination of (n_layers, hidden_dim, patch_grid_size, CLS-token presence,
final-LayerNorm position).

This module abstracts the forward pass + per-patch entropy field
construction over those differences. Returns a per-image ``H_stack`` of
shape ``(n_layers, gy, gx)`` regardless of the underlying architecture.

For native-3D video architectures (V-JEPA 2), see
``extract_entropy_volume`` which returns a per-clip entropy stack of
shape ``(n_layers, gt, gy, gx)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoVideoProcessor,
    CLIPModel,
    VJEPA2Model,
)

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


@dataclass(frozen=True)
class VideoArchSpec:
    """Native-3D video model. Tokens form a (gt, gy, gx) grid; no CLS."""
    model_name: str
    grid_size_3d: tuple[int, int, int]   # (gt, gy, gx)
    n_layers: int
    hidden_dim: int
    n_frames_in: int                     # frames the processor expects
    image_size: int                      # spatial size after crop


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


# ── video-native architectures (V-JEPA 2) ────────────────────────────────────


def _vjepa2_forward(
    model, processor, video_frames: np.ndarray, device: str,
    dtype: torch.dtype = torch.float16,
):
    """Forward a single clip; ``video_frames`` is uint8 (T, H, W, 3).

    V-JEPA 2 has no CLS token; output token order is (gt, gy, gx) flattened
    with gt slowest (matching ``conv3d`` + ``flatten(2)`` in the embedding).
    """
    inputs = processor(videos=[video_frames], return_tensors="pt")
    inputs = {
        k: (v.to(device, dtype=dtype) if v.dtype.is_floating_point else v.to(device))
        for k, v in inputs.items()
    }
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states                        # tuple of (1, gt*gy*gx, hidden)


def load_video_model(arch: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    """Load (model, processor, forward_fn, VideoArchSpec) for a video-native arch."""
    if arch in ("vjepa2_l", "facebook/vjepa2-vitl-fpc64-256"):
        name = "facebook/vjepa2-vitl-fpc64-256"
        model = VJEPA2Model.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        proc = AutoVideoProcessor.from_pretrained(name)
        # 64 frames, tubelet=2, 256², patch=16  →  32 × 16 × 16 = 8192 tokens
        spec = VideoArchSpec(model_name=name, grid_size_3d=(32, 16, 16),
                             n_layers=24, hidden_dim=1024,
                             n_frames_in=64, image_size=256)
        return model, proc, _vjepa2_forward, spec
    if arch in ("vjepa2_g", "facebook/vjepa2-vitg-fpc64-256"):
        name = "facebook/vjepa2-vitg-fpc64-256"
        model = VJEPA2Model.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        proc = AutoVideoProcessor.from_pretrained(name)
        spec = VideoArchSpec(model_name=name, grid_size_3d=(32, 16, 16),
                             n_layers=40, hidden_dim=1408,
                             n_frames_in=64, image_size=256)
        return model, proc, _vjepa2_forward, spec
    raise ValueError(f"unknown video arch: {arch}")


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


def _per_token_entropy_volume(
    hidden: torch.Tensor, spec: VideoArchSpec,
    estimator: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """``hidden`` of shape ``(1, gt*gy*gx, hidden)`` → ``(gt, gy, gx)`` entropy field."""
    h = hidden.detach().cpu().float().numpy()
    if h.ndim != 3 or h.shape[0] != 1:
        raise ValueError(f"hidden must be (1, n_tokens, hidden); got {h.shape}")
    gt, gy, gx = spec.grid_size_3d
    n_tokens = gt * gy * gx
    if h.shape[1] != n_tokens:
        raise ValueError(
            f"got {h.shape[1]} tokens; spec expects {n_tokens} ({gt}×{gy}×{gx})"
        )
    if h.shape[2] != spec.hidden_dim:
        raise ValueError(f"hidden_dim mismatch: got {h.shape[2]}, spec {spec.hidden_dim}")
    h = h[0]                                            # (n_tokens, hidden)
    mu = h.mean(axis=1, keepdims=True)
    sd = h.std(axis=1, keepdims=True) + 1e-9
    h = (h - mu) / sd
    H = estimator(h)                                    # (n_tokens,)
    return H.reshape(gt, gy, gx)


def extract_entropy_volume(
    arch: str,
    video_frames: np.ndarray,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    estimator: Callable[[np.ndarray], np.ndarray] = vasicek_entropy,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, VideoArchSpec]:
    """Run a video-native arch over a single clip; return (n_layers, gt, gy, gx).

    Args:
        arch: registry nickname (``vjepa2_l``, ``vjepa2_g``).
        video_frames: uint8 array of shape (T, H, W, 3). The processor
            will resize/centre-crop to the model's expected spatial size
            and (if T != ``spec.n_frames_in``) sample-or-pad along time.
            For best results, supply exactly ``spec.n_frames_in`` frames
            already cropped / resized to ``spec.image_size``.
        device: cuda/cpu.
        dtype: model weight dtype (fp16 by default to fit ViT-L/G on a 4090).
        estimator: per-token entropy estimator.
        progress: callback ``(layer_idx, n_layers)`` for progress reporting.

    Returns:
        ``(volumes, spec)`` where ``volumes`` has shape
        ``(n_layers, gt, gy, gx)`` (np.float32) and ``spec`` is a
        VideoArchSpec.
    """
    model, processor, forward_fn, spec = load_video_model(arch, device=device, dtype=dtype)
    hidden_states = forward_fn(model, processor, video_frames, device, dtype=dtype)
    gt, gy, gx = spec.grid_size_3d
    out = np.empty((spec.n_layers, gt, gy, gx), dtype=np.float32)
    for L in range(spec.n_layers):
        out[L] = _per_token_entropy_volume(hidden_states[L + 1], spec, estimator)
        if progress is not None:
            progress(L + 1, spec.n_layers)
    return out, spec


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
