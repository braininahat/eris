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
# # video_volume — Step B2: volumetric driver
#
# For each (video, arch), treat `H` across frames at a fixed layer L as a
# 3-D scalar field on `(t, gy, gx)`. Compute:
#
# - 3-D gradient `(Ht, Hy, Hx)` and 3-D Laplacian via
#   `eris.fields.gradient_3d` / `laplacian_3d`.
# - Top 5% of `|∇H|` voxels → 3-D connected-component labelling
#   (`scipy.ndimage.label`). Largest CC ≡ "the tube".
# - On the synthetic blob: IoU vs ground-truth tube (binary mask placing
#   1s at the patch closest to each frame's blob centre, dilated by 1
#   patch).
# - Plotly Streamtube of `(Hx, Hy, Ht)` (cubic-upsampled spatially to
#   56×56 for prettier streamlines, numerics on the raw grid).
# - 2-D `outlier_tubes_L6.gif` — high-|∇H| binary mask overlaid on H
#   field, frame-by-frame.
#
# **Stop condition.** If on the synthetic blob the IoU < 0.3 at L=6, the
# volumetric framing isn't load-bearing. We surface that decision in the
# final report.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eris.extract import load_model
from eris.fields import gradient_3d, laplacian_3d
from eris.video import (
    blob_centres_to_patch,
    synthesise_translating_blob,
    synthetic_tube_mask,
)
from eris.volumetric import (
    extract_outlier_tubes,
    iou_3d,
    render_streamtubes_html,
)


try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:
    NB_DIR = Path.cwd()
REPO = NB_DIR if (NB_DIR / "pyproject.toml").exists() else NB_DIR.parent
RESULTS = REPO / "results" / "video"
IMAGES = REPO / "images"
print(f"REPO={REPO}")

SYNTH_NAME = "synth_translating_blob"
REAL_NAME = "real_clip"
VIDEO_NAMES = (SYNTH_NAME, REAL_NAME)
ARCHS = ("vit_b16", "dinov2_b")
N_FRAMES = 64
SIZE = 224
LAYERS_FOR_VOLUME = (6, 12)        # 1-indexed
LAYER_FOR_RENDER = 6               # the headline layer (1-indexed)
TOP_PCT = 5.0
DILATE_PATCHES = 1
SPATIAL_UPSAMPLE_RENDER = 4        # 14*4=56, 16*4=64 — meets ≥56 spec

# %% [markdown]
# ## 1. Load `H_video` from B1, recover synthetic ground-truth tube

# %%
def load_H_video(arch: str, video_name: str) -> np.ndarray:
    p = RESULTS / video_name / arch / "H_video.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}; run notebooks/video_frames.py first"
        )
    return np.load(p)["H"]


# Need ground-truth blob centres for the synthetic IoU.
_, synth_meta = synthesise_translating_blob(
    n_frames=N_FRAMES, size=SIZE, sigma=12.0, amplitude=100.0, seed=0,
)
print(f"synthetic centres recovered: {synth_meta.centres_xy.shape}")


def gt_tube_for_arch(arch: str) -> np.ndarray:
    """`(t, gy, gx)` boolean ground-truth tube for the synthetic stimulus."""
    _, _, _, spec = load_model(arch)
    grid = spec.grid_size
    return synthetic_tube_mask(
        synth_meta.centres_xy, image_size=SIZE, grid=grid,
        n_frames=N_FRAMES, dilate_patches=DILATE_PATCHES,
    )


# %% [markdown]
# ## 2. Volumetric pipeline per (video, arch, layer)
#
# For both layers in {6, 12}: gradient → top-5% threshold → connected-
# component label → IoU vs ground-truth (synth only). Render the Streamtube
# HTML and outlier-tubes GIF only at L=6 (the headline layer).

# %%
def render_outlier_tubes_gif(
    H_volume: np.ndarray,
    mask_volume: np.ndarray,
    gt_centres_patch: np.ndarray | None,
    out_path: Path,
    title: str,
) -> None:
    """2-D animation: H field + binary outlier mask overlay per frame."""
    n_frames, gy, gx = H_volume.shape
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    fig.suptitle(title, fontsize=10)
    vmin, vmax = float(H_volume.min()), float(H_volume.max())
    im = ax.imshow(H_volume[0], cmap="magma", vmin=vmin, vmax=vmax)
    # Use a separate red overlay imshow with alpha for the mask.
    mask_rgba = np.zeros((gy, gx, 4), dtype=float)
    mask_rgba[..., 0] = 1.0     # red
    overlay = ax.imshow(mask_rgba, alpha=0.0)
    gt_marker, = ax.plot([], [], "+", color="cyan", mew=2.0, ms=10.0,
                         label="GT centre")
    if gt_centres_patch is not None:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    txt = fig.text(0.5, 0.02, "frame 1", ha="center", fontsize=9)

    def draw(t: int):
        im.set_data(H_volume[t])
        rgba = mask_rgba.copy()
        rgba[..., 3] = mask_volume[t].astype(float) * 0.55
        overlay.set_data(rgba)
        if gt_centres_patch is not None:
            gt_marker.set_data([gt_centres_patch[t, 0]],
                               [gt_centres_patch[t, 1]])
        txt.set_text(f"frame {t + 1} / {n_frames}")
        return im, overlay, gt_marker, txt

    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    anim = animation.FuncAnimation(
        fig, draw, frames=n_frames, interval=100, blit=False,
    )
    anim.save(out_path, writer="pillow", dpi=110, fps=10)
    plt.close(fig)


# %%
metric_rows: list[dict] = []
streamtube_paths: list[Path] = []
outlier_gif_paths: list[Path] = []

for arch in ARCHS:
    for video_name in VIDEO_NAMES:
        out_dir = RESULTS / video_name / arch
        out_dir.mkdir(parents=True, exist_ok=True)
        H_video = load_H_video(arch, video_name)
        # H_video shape: (n_frames, n_layers, gy, gx)
        n_frames, n_layers, gy, gx = H_video.shape
        print(f"{video_name} / {arch}: H_video={H_video.shape}")

        gt = gt_tube_for_arch(arch) if video_name == SYNTH_NAME else None
        gt_centres_patch = (
            blob_centres_to_patch(synth_meta.centres_xy, SIZE, (gy, gx))
            if video_name == SYNTH_NAME else None
        )

        for layer_one_idx in LAYERS_FOR_VOLUME:
            layer_zero_idx = layer_one_idx - 1
            if layer_zero_idx >= n_layers:
                continue
            H_volume = H_video[:, layer_zero_idx]    # (t, gy, gx)
            Ht, Hy, Hx = gradient_3d(H_volume)
            grad_mag = np.sqrt(Ht ** 2 + Hy ** 2 + Hx ** 2)
            tube = extract_outlier_tubes(
                grad_mag, top_pct=TOP_PCT, connectivity=1,
            )
            iou = iou_3d(tube.largest_mask, gt) if gt is not None else float("nan")
            metric_rows.append({
                "video": video_name,
                "arch": arch,
                "layer": layer_one_idx,
                "n_voxels_top5pct": tube.n_voxels,
                "largest_cc_size": tube.largest_size,
                "n_components": tube.n_components,
                "iou_vs_gt": iou,
            })
            print(
                f"  L{layer_one_idx:>2}: "
                f"thr={tube.threshold:.4f} "
                f"top5%={tube.n_voxels:>5d} "
                f"cc={tube.n_components:>4d} "
                f"largest={tube.largest_size:>4d} "
                f"iou_vs_gt={iou:.3f}"
            )

            # Heavy renders only at L=6.
            if layer_one_idx == LAYER_FOR_RENDER:
                # Outlier-tubes 2-D animation: top-5% mask on H field.
                mask_volume = grad_mag >= tube.threshold
                gif_path = out_dir / f"outlier_tubes_L{layer_one_idx}.gif"
                render_outlier_tubes_gif(
                    H_volume=H_volume,
                    mask_volume=mask_volume,
                    gt_centres_patch=gt_centres_patch,
                    out_path=gif_path,
                    title=(
                        f"{video_name} · {arch} — top-{TOP_PCT:.0f}% |∇H| "
                        f"@ L{layer_one_idx}"
                    ),
                )
                outlier_gif_paths.append(gif_path)
                print(f"    wrote {gif_path.relative_to(REPO)}")

                # 3-D plotly streamtube. Cubic spatial upsample done inside
                # `render_streamtubes_html`; numerics above stayed on raw grid.
                html_path = out_dir / f"volume_streamlines_L{layer_one_idx}.html"
                render_streamtubes_html(
                    Hx=Hx, Hy=Hy, Ht=Ht,
                    out_path=html_path,
                    title=(
                        f"{video_name} · {arch} — streamlines (Hx, Hy, Ht) "
                        f"@ L{layer_one_idx}"
                    ),
                    starts=4,
                    upsample_xy=SPATIAL_UPSAMPLE_RENDER,
                    sizeref=0.5,
                )
                streamtube_paths.append(html_path)
                size_mb = html_path.stat().st_size / (1 << 20)
                print(f"    wrote {html_path.relative_to(REPO)}  "
                      f"({size_mb:.2f} MB)")

# %% [markdown]
# ## 3. Per-(video, arch) volume_metrics.csv
#
# Single tabular dump of every (video, arch, layer) row. NaN for
# `iou_vs_gt` on the real video (no ground-truth available).

# %%
metrics_df = pd.DataFrame(metric_rows).sort_values(
    ["video", "arch", "layer"]
).reset_index(drop=True)
print(metrics_df.to_string(index=False))

# Per-cell CSV (deliverable spec) — one row of metrics rows that match
# this (video, arch).
for arch in ARCHS:
    for video_name in VIDEO_NAMES:
        sub = metrics_df[
            (metrics_df["video"] == video_name) & (metrics_df["arch"] == arch)
        ]
        out_csv = RESULTS / video_name / arch / "volume_metrics.csv"
        sub.to_csv(out_csv, index=False)
        print(f"wrote {out_csv.relative_to(REPO)}  ({len(sub)} rows)")

# Top-level aggregated copy for downstream summary.
metrics_df.to_csv(RESULTS / "volume_metrics_all.csv", index=False)
print(f"wrote {(RESULTS / 'volume_metrics_all.csv').relative_to(REPO)}")

# %% [markdown]
# ## 4. Stop-condition check
#
# The plan says: stop the volumetric framing if on the synthetic blob
# IoU(largest CC vs ground-truth tube) < 0.3 at L=6. Print that
# verdict per arch.

# %%
synth_L6 = metrics_df[
    (metrics_df["video"] == SYNTH_NAME) & (metrics_df["layer"] == LAYER_FOR_RENDER)
].set_index("arch")
for arch in ARCHS:
    iou = float(synth_L6.loc[arch, "iou_vs_gt"])
    verdict = "PASS" if iou >= 0.3 else "FAIL (IoU < 0.3 → volumetric framing not load-bearing)"
    print(f"  synth blob @ L6 / {arch}: IoU = {iou:.3f}  → {verdict}")
