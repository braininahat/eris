# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # umap_embeddings — UMAP in residual-stream space, not output space
#
# `umap_h_stacks.py` projected the *derived* entropy field (1 scalar per
# patch). That collapses a 768-D residual to one number; geometry is mostly
# lost. Here we UMAP the **raw per-patch residual vectors** at each layer,
# colour by their per-patch entropy, and ask:
#
# 1. Do post-transition outlier patches (low H, high deviation from
#    median) form a **distinct geometric cluster** in residual space?
# 2. Is the cluster shape **standardisation-dependent** — i.e. is the
#    separation a magnitude effect (lost after per-patch standardise)
#    or a distribution-shape effect (preserved after standardise)?
# 3. Do the cluster shapes confirm the L4→L5 transition geometrically
#    (compact pre-transition, bimodal post-transition)?
#
# Pool patches across 200 tiny-imagenet val images (same indices as
# Step A) at four layers per arch: L=1 (pre), L_post (right after
# transition), L_mid (a few layers further), L_final.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from datasets import load_dataset
from PIL import Image

from eris.extract import load_model
from eris.estimators import vasicek_entropy

REPO = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
RESULTS = REPO / "results" / "umap_embeddings"
RESULTS.mkdir(parents=True, exist_ok=True)
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Two archs to keep compute bounded.  Layer indices are 0-based; we'll
# render L0 (after first block = pre-transition baseline), L_{post-1}
# (transition layer), L_post (right after), L_final.
TARGETS = [
    ("vit_b16",  [0, 4, 5, 11]),   # L1, L5 (pre), L6 (post), L12
    ("dinov2_b", [0, 6, 7, 11]),   # L1, L7 (pre), L8 (post), L12
]
N_IMAGES = 200

# %%
indices = np.load(REPO / "results" / "cross_arch" / "vit_b16" / "image_indices.npy")
indices = indices[:N_IMAGES]
print(f"using {len(indices)} images from Step A's sample")

ds = load_dataset("Maysee/tiny-imagenet", split="valid")
images_pil = [ds[int(i)]["image"].convert("RGB") for i in indices]
print(f"loaded {len(images_pil)} PIL images")


# %%
def per_patch_residuals(arch: str, images: list, layer_indices: list[int]) -> dict[int, np.ndarray]:
    """Run forward, return ``{layer: (n_imgs * n_patches, hidden_dim)}`` for
    each requested layer (0-based; layer 0 = post-block-1 = first transformer
    block output)."""
    model, processor, forward_fn, spec = load_model(arch, device=DEVICE)
    out: dict[int, list] = {L: [] for L in layer_indices}
    for img in images:
        hidden_states = forward_fn(model, processor, img, DEVICE)
        for L in layer_indices:
            h = hidden_states[L + 1]                       # +1: skip patch-embed
            h = h.detach().cpu().float().numpy()           # (1, n_tokens, d)
            if spec.has_cls:
                h = h[:, 1:, :]
            n_patches = spec.grid_size[0] * spec.grid_size[1]
            assert h.shape[1] == n_patches, (h.shape, n_patches)
            out[L].append(h[0])                            # (n_patches, d)
    return {L: np.concatenate(v, axis=0) for L, v in out.items()}


# %%
def patch_entropies(residuals: np.ndarray) -> np.ndarray:
    """Per-patch standardised differential entropy on (N, d) → (N,)."""
    mu = residuals.mean(axis=1, keepdims=True)
    sd = residuals.std(axis=1, keepdims=True) + 1e-9
    z = (residuals - mu) / sd
    return vasicek_entropy(z)


def umap_2d(X: np.ndarray, n_neighbors: int = 30) -> np.ndarray:
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        random_state=SEED, init="random",
                        low_memory=True, metric="euclidean")
    return reducer.fit_transform(X.astype(np.float32))


# %% [markdown]
# ## Render UMAP at the four target layers per arch, two normalisations

# %%
for arch, layer_idxs in TARGETS:
    print(f"\n=== {arch} — extracting residuals ===")
    res_per_layer = per_patch_residuals(arch, images_pil, layer_idxs)

    # 2 rows (raw / standardised) × 4 cols (4 layers).
    fig, axes = plt.subplots(2, len(layer_idxs), figsize=(4.5 * len(layer_idxs), 9))
    label_for_norm = {"raw": "raw residual", "std": "per-patch standardised"}
    for col, L in enumerate(layer_idxs):
        h_raw = res_per_layer[L]                            # (N, d)
        H_per_patch = patch_entropies(h_raw)                # (N,)
        # save raw embeddings for posterity
        np.savez(RESULTS / f"{arch}_L{L+1}_residuals.npz",
                 residuals=h_raw, H=H_per_patch)

        for row, mode in enumerate(("raw", "std")):
            X = h_raw if mode == "raw" else (
                (h_raw - h_raw.mean(axis=1, keepdims=True))
                / (h_raw.std(axis=1, keepdims=True) + 1e-9)
            )
            emb = umap_2d(X, n_neighbors=30)

            ax = axes[row, col]
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=H_per_patch,
                            cmap="viridis", s=2.5, alpha=0.55,
                            vmin=np.percentile(H_per_patch, 1),
                            vmax=np.percentile(H_per_patch, 99))
            ax.set_title(f"L={L+1}  ·  {label_for_norm[mode]}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if col == len(layer_idxs) - 1:
                cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
                cbar.set_label("per-patch H (nats)", fontsize=8)

    fig.suptitle(f"{arch} — UMAP of per-patch residuals  "
                 f"(N = {res_per_layer[layer_idxs[0]].shape[0]} patches "
                 f"from {N_IMAGES} tiny-imagenet val images)", y=0.995)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{arch}_umap_residuals.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {arch}_umap_residuals.png")

# %% [markdown]
# ## Quantitative — do low-H "outlier" patches sit far from the bulk?
#
# For each (arch, layer, normalisation) compute the silhouette-like
# separation: for each patch, distance to the nearest 50 other patches
# (mean), grouped by whether that patch is in the bottom-1% of H or not.
# If outlier patches form a geometric cluster, the within-group distances
# should be smaller and between-group distances larger.

# %%
import pandas as pd
from sklearn.neighbors import NearestNeighbors

rows = []
for arch, layer_idxs in TARGETS:
    for L in layer_idxs:
        npz = np.load(RESULTS / f"{arch}_L{L+1}_residuals.npz")
        h_raw = npz["residuals"]
        H = npz["H"]
        for mode in ("raw", "std"):
            X = h_raw if mode == "raw" else (
                (h_raw - h_raw.mean(axis=1, keepdims=True))
                / (h_raw.std(axis=1, keepdims=True) + 1e-9)
            )
            outlier_mask = H < np.percentile(H, 1)
            nn = NearestNeighbors(n_neighbors=51).fit(X)  # +1 for self
            dists, _ = nn.kneighbors(X)
            mean_nn_dist = dists[:, 1:].mean(axis=1)
            d_outlier = mean_nn_dist[outlier_mask].mean() if outlier_mask.any() else np.nan
            d_bulk = mean_nn_dist[~outlier_mask].mean()
            rows.append({
                "arch": arch, "layer": L + 1, "norm": mode,
                "n_outliers": int(outlier_mask.sum()),
                "mean_nn_dist_outlier": float(d_outlier),
                "mean_nn_dist_bulk": float(d_bulk),
                "ratio_outlier_over_bulk": float(d_outlier / d_bulk),
            })

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "outlier_geometry.csv", index=False)
print(df.round(3).to_string(index=False))
