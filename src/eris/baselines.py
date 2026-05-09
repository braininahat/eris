"""Step-C baselines — three per-patch signals to compare against the
entropy field on the same images.

All three return arrays of shape ``(n_images, n_layers, gy, gx)`` so they
can drop into the same ``per_layer_stats`` / ``transition_layer`` plumbing
as the entropy field.

1. ``l2_norm_field`` — ``||residual_stream[L, patch]||_2`` per (layer,
   patch). No estimator. The simplest possible per-patch scalar.
2. ``attention_rollout_field`` — at each layer, mean attention over heads,
   then per-query-patch entropy of the resulting key distribution. Per-
   layer (not rolled-up cumulatively) — "rollout" in the looser sense of
   "per-layer attention summary" (cumulative attention rollout would
   collapse the depth profile we're studying).
3. ``random_init_entropy_field`` — same Vasicek-on-standardised-residual
   pipeline as ``extract_entropy_stack``, but with a randomly-initialised
   ViT-B/16. Tells us whether the L4→L5 phase transition is a property
   of the supervised checkpoint or of the architecture itself.

ViT-B/16 is the only model these baselines support — Step C is scoped to
that architecture.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTConfig, ViTModel

from .extract import _per_patch_entropy_field, ArchSpec
from .estimators import vasicek_entropy


_VIT_B16 = "google/vit-base-patch16-224"
_VIT_B16_SPEC = ArchSpec(
    model_name=_VIT_B16, has_cls=True, grid_size=(14, 14),
    n_layers=12, hidden_dim=768,
)


def _load_processor() -> AutoImageProcessor:
    return AutoImageProcessor.from_pretrained(_VIT_B16)


def _prep_image(img) -> Image.Image:
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    return img.convert("RGB")


# ── 1. per-patch L2 norm of the residual stream ───────────────────────────────


def l2_norm_field(
    model_name: str,
    images: Iterable,
    device: str = "cuda",
) -> np.ndarray:
    """``||h[L, p]||_2`` for every (image, layer, patch). No estimator.

    Returns ``(n_images, n_layers, gy, gx)`` of float32. Standardisation
    is intentionally NOT applied: we want the scalar that's *available
    without the entropy machinery*.
    """
    if model_name not in (_VIT_B16, "vit_b16"):
        raise ValueError(f"Step C is ViT-B/16 only; got {model_name}")
    spec = _VIT_B16_SPEC
    model = ViTModel.from_pretrained(_VIT_B16).to(device).eval()
    processor = _load_processor()
    images = [_prep_image(im) for im in images]
    n = len(images)
    gy, gx = spec.grid_size
    out = np.empty((n, spec.n_layers, gy, gx), dtype=np.float32)
    for i, img in enumerate(images):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            res = model(**inputs, output_hidden_states=True)
        for L in range(spec.n_layers):
            h = res.hidden_states[L + 1].detach().cpu().float().numpy()  # (1, T, D)
            if spec.has_cls:
                h = h[:, 1:, :]
            norms = np.linalg.norm(h[0], axis=1)               # (n_patches,)
            out[i, L] = norms.reshape(gy, gx)
    return out


# ── 2. per-patch attention entropy at each layer ──────────────────────────────


def attention_rollout_field(
    model_name: str,
    images: Iterable,
    device: str = "cuda",
) -> np.ndarray:
    """Per-query-patch attention entropy at each layer.

    At layer L: average attention over heads → ``A[L]`` of shape
    ``(n_tokens, n_tokens)`` (rows = queries, cols = keys, row-stochastic
    after the average since each head is row-stochastic). For each query
    patch q, compute ``H_q = -Σ_k A[L, q, k] log A[L, q, k]``. Reshape
    over the 14×14 patch grid (CLS dropped).

    "rollout" here means the per-layer attention summary, not the
    cumulative rollout-product (which would obscure the depth profile we
    care about).

    Returns ``(n_images, n_layers, gy, gx)`` in nats.
    """
    if model_name not in (_VIT_B16, "vit_b16"):
        raise ValueError(f"Step C is ViT-B/16 only; got {model_name}")
    spec = _VIT_B16_SPEC
    cfg = ViTConfig.from_pretrained(_VIT_B16)
    cfg.output_attentions = True
    model = ViTModel.from_pretrained(_VIT_B16, config=cfg).to(device).eval()
    processor = _load_processor()
    images = [_prep_image(im) for im in images]
    n = len(images)
    gy, gx = spec.grid_size
    n_patches = gy * gx
    out = np.empty((n, spec.n_layers, gy, gx), dtype=np.float32)
    for i, img in enumerate(images):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            res = model(**inputs, output_attentions=True)
        # res.attentions: tuple of length n_layers, each (1, heads, T, T)
        for L in range(spec.n_layers):
            A = res.attentions[L].detach().cpu().float().numpy()[0]      # (h, T, T)
            A_mean = A.mean(axis=0)                                       # (T, T)
            # Drop CLS row (queries) and CLS column (key) so the entropy is
            # over the patch-key distribution only. Re-normalise rows.
            if spec.has_cls:
                A_pp = A_mean[1:, 1:]                                     # (P, P)
            else:
                A_pp = A_mean
            row_sum = A_pp.sum(axis=1, keepdims=True) + 1e-12
            P = A_pp / row_sum                                            # row-stochastic
            P = np.clip(P, 1e-12, 1.0)
            H = -(P * np.log(P)).sum(axis=1)                              # (n_patches,)
            out[i, L] = H.reshape(gy, gx)
    return out


# ── 3. random-init ViT entropy field ──────────────────────────────────────────


def random_init_entropy_field(
    model_name: str,
    images: Iterable,
    device: str = "cuda",
    seed: int = 0,
) -> np.ndarray:
    """Same per-patch entropy pipeline as ``extract_entropy_stack`` but
    with a randomly-initialised ViT-B/16.

    Sets ``torch.manual_seed(seed)`` before constructing the model so the
    weights are reproducible. The processor is still loaded from the HF
    config (deterministic — it's just resize/normalise).

    Returns ``(n_images, n_layers, gy, gx)`` in nats.
    """
    if model_name not in (_VIT_B16, "vit_b16"):
        raise ValueError(f"Step C is ViT-B/16 only; got {model_name}")
    spec = _VIT_B16_SPEC
    torch.manual_seed(seed)
    cfg = ViTConfig.from_pretrained(_VIT_B16)
    model = ViTModel(cfg).to(device).eval()                                 # random init
    processor = _load_processor()
    images = [_prep_image(im) for im in images]
    n = len(images)
    gy, gx = spec.grid_size
    out = np.empty((n, spec.n_layers, gy, gx), dtype=np.float32)
    for i, img in enumerate(images):
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            res = model(**inputs, output_hidden_states=True)
        for L in range(spec.n_layers):
            out[i, L] = _per_patch_entropy_field(
                res.hidden_states[L + 1], spec, vasicek_entropy
            )
    return out


# ── External outlier definition: input-gradient saliency ──────────────────────


def saliency_patch_grid(
    images: Iterable,
    device: str = "cuda",
    grid: tuple[int, int] = (14, 14),
) -> np.ndarray:
    """Input-pixel gradient saliency, max-pooled to the 14×14 ViT-B/16 grid.

    Forward each PIL image through the ImageNet-classifier ViT-B/16, take
    ``∂(logit_pred) / ∂(pixel)`` as a (3, 224, 224) tensor, take elementwise
    abs, max-pool over channels then over 16×16 spatial tiles → (14, 14)
    saliency map per image.

    Returns ``(n_images, gy, gx)`` of float32.
    """
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(_VIT_B16).to(device).eval()
    processor = _load_processor()
    images = [_prep_image(im) for im in images]
    gy, gx = grid
    n = len(images)
    out = np.empty((n, gy, gx), dtype=np.float32)
    for i, img in enumerate(images):
        inputs = processor(images=img, return_tensors="pt").to(device)
        pix = inputs["pixel_values"].clone().detach()
        pix.requires_grad_(True)
        logits = model(pixel_values=pix).logits
        pred = int(logits.argmax(dim=-1).item())
        target = logits[0, pred]
        model.zero_grad(set_to_none=True)
        target.backward()
        g = pix.grad.detach().cpu().float().numpy()[0]                  # (3, 224, 224)
        sal = np.max(np.abs(g), axis=0)                                 # (224, 224)
        # Max-pool to 14×14 (16×16 tiles).
        ph, pw = sal.shape
        ty, tx = ph // gy, pw // gx
        sal = sal[: ty * gy, : tx * gx].reshape(gy, ty, gx, tx)
        out[i] = sal.max(axis=(1, 3))
    return out
