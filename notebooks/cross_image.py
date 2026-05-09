# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # cross_image — summary across all 8 stimuli
#
# Reads each image's `entropies.npz`, builds:
# - `results/cross_image_summary.png` — rows = images, cols = original |
#   `H@L1` | `H@L6` | `H@L12` | `Σ|∇H| dL` | `Σ|∇²H| dL`. Color range
#   fixed per row (so within-image comparisons are valid; across-image
#   magnitudes can differ).
# - `results/per_image_metrics.csv` — one row per (image, layer) with
#   `H_mean, H_std, abs_grad_H_mean, abs_laplacian_H_mean, dH_mean, dH_std`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
from PIL import Image

try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:
    NB_DIR = Path.cwd()
REPO = NB_DIR if (NB_DIR / "pyproject.toml").exists() else NB_DIR.parent
RESULTS = REPO / "results"
IMAGES = REPO / "images"
N_BLOCKS = 12

names = sorted(d.name for d in RESULTS.iterdir() if (d / "entropies.npz").exists())
print(f"images with entropies: {len(names)}")
print(f"  {names}")

# %%
H_per_image = {}
metrics_rows = []
for name in names:
    Hs = np.load(RESULTS / name / "entropies.npz")["H"]
    H_per_image[name] = Hs
    for L in range(N_BLOCKS):
        H = Hs[L]
        gy, gx = np.gradient(H)
        lap = scipy.ndimage.laplace(H)
        dH = (Hs[L] - Hs[L - 1]) if L > 0 else np.zeros_like(H)
        metrics_rows.append({
            "image": name, "layer": L + 1,
            "H_mean": float(H.mean()),
            "H_std": float(H.std()),
            "abs_grad_H_mean": float(np.hypot(gy, gx).mean()),
            "abs_laplacian_H_mean": float(np.abs(lap).mean()),
            "dH_mean": float(dH.mean()),
            "dH_std": float(dH.std()),
        })

metrics = pd.DataFrame(metrics_rows)
metrics.to_csv(RESULTS / "per_image_metrics.csv", index=False)
print(f"saved {RESULTS / 'per_image_metrics.csv'}  ({len(metrics)} rows)")
metrics.head(13)

# %% [markdown]
# ## Cross-image grid

# %%
fig, axes = plt.subplots(len(names), 6, figsize=(18, 2.6 * len(names)))
for r, name in enumerate(names):
    Hs = H_per_image[name]
    img_pil = Image.open(IMAGES / f"{name}.jpg").convert("RGB")

    # Per-image fixed vmin/vmax for the H columns.
    vmin, vmax = float(Hs.min()), float(Hs.max())

    # Per-image cumulative gradient and laplacian magnitudes.
    grad_mag_sum = np.zeros_like(Hs[0])
    lap_abs_sum = np.zeros_like(Hs[0])
    for L in range(N_BLOCKS):
        gy, gx = np.gradient(Hs[L])
        grad_mag_sum += np.hypot(gy, gx)
        lap_abs_sum += np.abs(scipy.ndimage.laplace(Hs[L]))

    # Col 0: original image
    axes[r, 0].imshow(img_pil)
    axes[r, 0].set_ylabel(name, fontsize=11, rotation=0, ha="right",
                          va="center", labelpad=20)
    axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    if r == 0: axes[r, 0].set_title("original", fontsize=11)

    # Col 1-3: H at L1, L6, L12
    for c, L in enumerate([0, 5, 11]):
        axes[r, 1 + c].imshow(Hs[L], vmin=vmin, vmax=vmax, cmap="magma")
        axes[r, 1 + c].set_xticks([]); axes[r, 1 + c].set_yticks([])
        if r == 0: axes[r, 1 + c].set_title(f"H @ L{L+1}", fontsize=11)

    # Col 4: cumulative gradient magnitude
    im = axes[r, 4].imshow(grad_mag_sum, cmap="viridis")
    axes[r, 4].set_xticks([]); axes[r, 4].set_yticks([])
    if r == 0: axes[r, 4].set_title("Σ |∇H|", fontsize=11)

    # Col 5: cumulative laplacian magnitude
    im = axes[r, 5].imshow(lap_abs_sum, cmap="viridis")
    axes[r, 5].set_xticks([]); axes[r, 5].set_yticks([])
    if r == 0: axes[r, 5].set_title("Σ |∇²H|", fontsize=11)

fig.suptitle("Cross-image entropy-field summary  (rows = images;  H cols share row vmin/vmax)",
             y=1.0)
fig.tight_layout()
fig.savefig(RESULTS / "cross_image_summary.png", dpi=180, bbox_inches="tight")
print(f"saved {RESULTS / 'cross_image_summary.png'}")

# %% [markdown]
# ## Layer-wise statistics — paired with the input image strip

# %%
# Build a stable colour mapping image-name → colour. tab10 has 10 distinct
# entries; we have ≤8 images.
_tab10 = plt.get_cmap("tab10")
colors = {n: _tab10(i % 10) for i, n in enumerate(names)}

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 4, height_ratios=[0.6, 1.0, 1.0], hspace=0.35,
                      wspace=0.25)

# Top strip: every input image, labelled, in the same colour as its curve.
for c, name in enumerate(names):
    ax = fig.add_subplot(gs[0, c % 4]) if c < 4 else None
# Actually: 8 images in a single row would be too narrow at this width.
# Use a separate strip subplot below.

# Re-do as 4 rows: image strip (1 row of 8 panels), then 2x2 metrics.
fig.clear()
gs = fig.add_gridspec(
    3, 8,
    height_ratios=[0.6, 1.0, 1.0],
    hspace=0.30, wspace=0.18,
)
for c, name in enumerate(names):
    ax = fig.add_subplot(gs[0, c])
    ax.imshow(Image.open(IMAGES / f"{name}.jpg").convert("RGB"))
    ax.set_title(name, fontsize=9, color=colors[name])
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(colors[name]); spine.set_linewidth(2.0)

ax_mean = fig.add_subplot(gs[1, :4])
ax_std = fig.add_subplot(gs[1, 4:])
ax_grad = fig.add_subplot(gs[2, :4])
ax_dH = fig.add_subplot(gs[2, 4:])

for name in names:
    sub = metrics[metrics["image"] == name].sort_values("layer")
    c = colors[name]
    ax_mean.plot(sub["layer"], sub["H_mean"], marker="o", color=c, label=name)
    ax_std.plot(sub["layer"], sub["H_std"], marker="o", color=c, label=name)
    ax_grad.plot(sub["layer"], sub["abs_grad_H_mean"], marker="o", color=c, label=name)
    ax_dH.plot(sub["layer"], sub["dH_std"], marker="o", color=c, label=name)

ax_mean.set_title("H_mean per layer (analytic Gaussian = 1.4189)")
ax_mean.axhline(0.5 * np.log(2 * np.pi * np.e), color="k", lw=0.8, ls="--",
                label="Gaussian baseline")
ax_std.set_title("H_std per layer (within-layer spread)")
ax_grad.set_title("mean |∇H| per layer")
ax_dH.set_title("ΔH std per layer (cross-layer change)")
for ax in (ax_mean, ax_std, ax_grad, ax_dH):
    ax.set_xlabel("layer")
ax_mean.legend(fontsize=8, loc="lower left", ncol=2)
fig.suptitle("Depth curves with input-image strip (colour-matched to the curves)",
             y=1.0)
fig.savefig(RESULTS / "depth_curves.png", dpi=160, bbox_inches="tight")
print(f"saved {RESULTS / 'depth_curves.png'}")
