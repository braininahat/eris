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
# # umap_h_stacks — UMAP of per-image entropy fields per architecture
#
# For each architecture, take the cached `(n_images, n_layers, gy, gx)`
# entropy stack from Step A and ask: does the per-image entropy field
# separate tiny-imagenet semantic classes when projected to 2-D?
#
# Two views per arch:
# 1. **Whole-stack UMAP** — flatten each image's entropy field across
#    all layers (`(n_layers × gy × gx)`-dim feature vector per image),
#    UMAP→2-D, colour by tiny-imagenet class.
# 2. **Per-layer-slab UMAP** — at the post-transition layer L_post and
#    the final layer L_final, flatten only that layer's `(gy, gx)` field
#    and UMAP→2-D. Tests whether semantic separation is concentrated in
#    specific layers.
#
# Same 200 images across all 4 archs (Step A's `image_indices.npy`).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from datasets import load_dataset

REPO = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
RESULTS = REPO / "results" / "umap"
RESULTS.mkdir(parents=True, exist_ok=True)
SEED = 0

ARCHS = [
    ("vit_b16",  "ViT-B/16 (supervised)",  5, 12),
    ("vit_l16",  "ViT-L/16 (supervised)",  5, 24),
    ("dinov2_b", "DINO-v2 (SSL)",          8, 12),
    ("clip_b16", "CLIP-ViT-B/16 (CL)",     7, 12),
]

# %%
indices = np.load(REPO / "results" / "cross_arch" / "vit_b16" / "image_indices.npy")
print(f"loaded {len(indices)} image indices from Step A")

ds = load_dataset("Maysee/tiny-imagenet", split="valid")
labels = np.array([ds[int(i)]["label"] for i in indices])
class_names = ds.features["label"].names
print(f"unique classes in sample: {len(set(labels))}/{len(class_names)}")


# %%
def umap_2d(X: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """Standardise X column-wise then UMAP→2."""
    Xs = (X - X.mean(axis=0, keepdims=True))
    sd = Xs.std(axis=0, keepdims=True)
    sd[sd == 0] = 1
    Xs = Xs / sd
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        random_state=SEED, init="random")
    return reducer.fit_transform(Xs)


# %% [markdown]
# ## 1. Whole-stack UMAP per arch (flatten across all layers)

# %%
fig = plt.figure(figsize=(16, 4.5))
for col, (arch, title, _Lpost, _Lmax) in enumerate(ARCHS):
    H = np.load(REPO / "results" / "cross_arch" / arch / "H_stack.npz")["H"]
    n_imgs, n_layers, gy, gx = H.shape
    X = H.reshape(n_imgs, -1)                                 # (n_imgs, L*gy*gx)
    emb = umap_2d(X, n_neighbors=min(15, n_imgs - 1))
    np.save(RESULTS / f"{arch}_whole_stack_emb.npy", emb)

    ax = fig.add_subplot(1, 4, col + 1)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab20",
                    s=18, alpha=0.85, edgecolor="black", linewidth=0.2)
    ax.set_title(f"{title}\nwhole-stack UMAP ({n_layers}×{gy}×{gx} → 2)",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("UMAP of per-image entropy stacks  —  colour = tiny-imagenet class",
             y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "whole_stack_umap.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"saved {RESULTS / 'whole_stack_umap.png'}")

# %% [markdown]
# ## 2. Per-layer UMAP — post-transition vs final layer
#
# At the post-transition layer (`L_post`, taken from Step A's per-arch
# transition CSV) and the final layer (`L_max`), flatten only that
# layer's `(gy, gx)` field per image and UMAP. Useful comparison: does
# semantic information concentrate at the transition layer or only at
# the final?

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for col, (arch, title, L_post, L_max) in enumerate(ARCHS):
    H = np.load(REPO / "results" / "cross_arch" / arch / "H_stack.npz")["H"]
    n_imgs, n_layers, gy, gx = H.shape
    # 0-indexed: layer L_post in 1-indexed convention = L_post-1 here
    L_post_idx = L_post - 1
    L_max_idx = n_layers - 1
    for row, (L_idx, label_str) in enumerate(
        [(L_post_idx, f"L={L_post} (post-transition)"),
         (L_max_idx,  f"L={n_layers} (final)")]
    ):
        Xl = H[:, L_idx, :, :].reshape(n_imgs, -1)
        emb = umap_2d(Xl, n_neighbors=min(15, n_imgs - 1))
        ax = axes[row, col]
        ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab20", s=14,
                   alpha=0.85, edgecolor="black", linewidth=0.2)
        ax.set_title(f"{title}\n{label_str}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("UMAP of per-image entropy field at one layer  —  colour = class",
             y=1.0)
fig.tight_layout()
fig.savefig(RESULTS / "per_layer_umap.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"saved {RESULTS / 'per_layer_umap.png'}")

# %% [markdown]
# ## 3. Quantitative — k-NN classification on the UMAP embeddings
#
# Sanity: if classes truly separate in 2-D UMAP space, k-NN classifier
# accuracy ≫ chance. Reports per-arch / per-view accuracy.

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import pandas as pd
rows = []
for arch, title, L_post, L_max in ARCHS:
    H = np.load(REPO / "results" / "cross_arch" / arch / "H_stack.npz")["H"]
    n_imgs, n_layers, gy, gx = H.shape
    chance = 1.0 / len(set(labels))
    # Whole stack
    X = H.reshape(n_imgs, -1)
    emb = umap_2d(X, n_neighbors=min(15, n_imgs - 1))
    knn = KNeighborsClassifier(n_neighbors=5)
    acc = cross_val_score(knn, emb, labels, cv=5).mean()
    rows.append({"arch": arch, "view": "whole_stack", "knn5_acc": acc, "chance": chance})
    # Post-transition layer
    Xl = H[:, L_post - 1, :, :].reshape(n_imgs, -1)
    emb = umap_2d(Xl, n_neighbors=min(15, n_imgs - 1))
    acc = cross_val_score(KNeighborsClassifier(n_neighbors=5), emb, labels, cv=5).mean()
    rows.append({"arch": arch, "view": f"L{L_post}_post_trans",
                 "knn5_acc": acc, "chance": chance})
    # Final layer
    Xl = H[:, -1, :, :].reshape(n_imgs, -1)
    emb = umap_2d(Xl, n_neighbors=min(15, n_imgs - 1))
    acc = cross_val_score(KNeighborsClassifier(n_neighbors=5), emb, labels, cv=5).mean()
    rows.append({"arch": arch, "view": f"L{n_layers}_final",
                 "knn5_acc": acc, "chance": chance})

agreement = pd.DataFrame(rows)
agreement.to_csv(RESULTS / "knn_accuracy.csv", index=False)
print(agreement.round(3).to_string(index=False))
