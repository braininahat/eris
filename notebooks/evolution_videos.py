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
# # evolution_videos — animated depth evolution paired with input image
#
# Two outputs per run:
#
# - `results/{image}/evolution.gif` — 4-panel per-image animation, frames
#   L=1..12: (input image static | H heatmap | H + ∇H quiver | streamlines).
# - `results/cross_evolution.gif` — single GIF showing all 8 H fields
#   simultaneously, animated across layers, with input thumbnails as a
#   static strip at the top so you can see the L4→L5 phase transition
#   happening across every image at once.

# %%
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from PIL import Image

try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:
    NB_DIR = Path.cwd()
REPO = NB_DIR if (NB_DIR / "pyproject.toml").exists() else NB_DIR.parent
RESULTS = REPO / "results"
IMAGES = REPO / "images"
GRID = 14
N_BLOCKS = 12
UP = 4

names = sorted(d.name for d in RESULTS.iterdir() if (d / "entropies.npz").exists())
print(f"images: {len(names)}")


# %% [markdown]
# ## 1. Per-image evolution.gif (input + H + ∇H + streamlines)

# %%
def upsample(arr2d: np.ndarray, factor: int = UP) -> np.ndarray:
    return scipy.ndimage.zoom(arr2d, factor, order=3)


def render_per_image(name: str) -> None:
    Hs = np.load(RESULTS / name / "entropies.npz")["H"]   # (12, 14, 14)
    img_pil = Image.open(IMAGES / f"{name}.jpg").convert("RGB")

    vmin, vmax = float(Hs.min()), float(Hs.max())
    yy, xx = np.mgrid[:GRID, :GRID]
    xs = np.linspace(0, GRID - 1, GRID * UP)
    ys = np.linspace(0, GRID - 1, GRID * UP)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(img_pil)
    axes[0].set_title(f"input: {name}", fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    def draw_layer(L: int):
        H = Hs[L]
        gy, gx = np.gradient(H)
        H_up = upsample(H)
        gy_u, gx_u = np.gradient(H_up)

        for ax in axes[1:]:
            ax.clear()
        # H heatmap
        axes[1].imshow(H, vmin=vmin, vmax=vmax, cmap="magma")
        axes[1].set_title(f"H @ L{L+1}", fontsize=11)
        # H + ∇H quiver
        axes[2].imshow(H, vmin=vmin, vmax=vmax, cmap="magma")
        axes[2].quiver(xx, yy, gx, -gy, color="cyan",
                       scale_units="xy", scale=0.8, width=0.012, alpha=0.85)
        axes[2].set_title(f"H + ∇H @ L{L+1}", fontsize=11)
        # streamlines on upsampled
        axes[3].imshow(H_up, vmin=vmin, vmax=vmax, cmap="magma",
                       extent=[0, GRID - 1, GRID - 1, 0])
        try:
            axes[3].streamplot(xs, ys, gx_u, -gy_u, color="cyan",
                               density=1.2, linewidth=0.8, arrowsize=0.7)
        except Exception:
            pass
        axes[3].set_title(f"streamlines ∇H @ L{L+1}", fontsize=11)
        for ax in axes[1:]:
            ax.set_xticks([]); ax.set_yticks([])

    draw_layer(0)
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, draw_layer, frames=N_BLOCKS, interval=500,
                                   blit=False)
    out = RESULTS / name / "evolution.gif"
    anim.save(out, writer="pillow", dpi=110)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


for n in names:
    render_per_image(n)

# %% [markdown]
# ## 2. cross_evolution.gif — all 8 H fields animating simultaneously
#
# Static thumbnail strip at the top; below, an 8-panel grid where each cell
# is the H field for that image at the current layer. Per-cell vmin/vmax
# fixed across layers so frame-to-frame changes are honest.

# %%
H_per_image = {n: np.load(RESULTS / n / "entropies.npz")["H"] for n in names}
vmin_per = {n: float(H.min()) for n, H in H_per_image.items()}
vmax_per = {n: float(H.max()) for n, H in H_per_image.items()}

ncols = len(names)
fig = plt.figure(figsize=(2.5 * ncols, 6.0))
gs = fig.add_gridspec(2, ncols, height_ratios=[0.9, 1.4], hspace=0.30, wspace=0.18)

# Top row: static thumbnails.
img_axes = []
for c, name in enumerate(names):
    ax = fig.add_subplot(gs[0, c])
    ax.imshow(Image.open(IMAGES / f"{name}.jpg").convert("RGB"))
    ax.set_title(name, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    img_axes.append(ax)

# Bottom row: H-field axes. We re-create the imshow each frame for cleanliness.
field_axes = [fig.add_subplot(gs[1, c]) for c in range(ncols)]
ims = []
for c, name in enumerate(names):
    ax = field_axes[c]
    im = ax.imshow(H_per_image[name][0], vmin=vmin_per[name], vmax=vmax_per[name],
                   cmap="magma")
    ax.set_xticks([]); ax.set_yticks([])
    ims.append(im)
suptitle = fig.suptitle("H field across depth   L = 1 / 12", y=1.0, fontsize=13)


def draw(L: int):
    for c, name in enumerate(names):
        ims[c].set_data(H_per_image[name][L])
        field_axes[c].set_xlabel(f"L{L+1}", fontsize=10)
    suptitle.set_text(f"H field across depth   L = {L+1} / {N_BLOCKS}")


fig.tight_layout()
anim = animation.FuncAnimation(fig, draw, frames=N_BLOCKS, interval=600, blit=False)
out = RESULTS / "cross_evolution.gif"
anim.save(out, writer="pillow", dpi=120)
plt.close(fig)
print(f"saved {out.relative_to(REPO)}")
