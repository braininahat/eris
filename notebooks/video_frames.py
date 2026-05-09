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
# # video_frames — Step B1: sequence-of-frames driver
#
# Treat each video frame as an independent still and run the entropy-stack
# pipeline. For each (video, arch) combination we save:
#
# - `H_video.npz` — `(n_frames, n_layers, gy, gx)`.
# - `depth_x_time.gif` — H @ L=6 animated across frames at ~10 fps, with
#   the input frame paired alongside.
# - `transition_per_frame.png` — argmax-ΔH-std layer per frame.
# - `layer_time_grid.png` — `(layer × frame)` heatmap of `H_std` per layer
#   per frame.
#
# Two videos × two architectures = 4 outputs each.
#
# **Stimuli.** The synthetic blob is regenerated here if missing; the real
# video is fetched on-demand and cached under `data/`.
#
# **Stop condition for Step B (paired with B2).** If on the synthetic
# blob the IoU of the largest-component vs ground-truth tube < 0.3 at L=6,
# the volumetric framing isn't load-bearing. (B2 owns the IoU; this
# notebook produces the per-frame view either way.)

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eris.extract import extract_entropy_stack, load_model
from eris.fields import per_layer_stats, transition_layer
from eris.video import (
    SyntheticBlobMeta,
    blob_centres_to_patch,
    frames_to_pil,
    load_real_video,
    synthesise_translating_blob,
    write_video,
)


try:
    NB_DIR = Path(__file__).resolve().parent
except NameError:
    NB_DIR = Path.cwd()
REPO = NB_DIR if (NB_DIR / "pyproject.toml").exists() else NB_DIR.parent
IMAGES = REPO / "images"
RESULTS = REPO / "results" / "video"
DATA = REPO / "data" / "video_cache"
RESULTS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
IMAGES.mkdir(parents=True, exist_ok=True)
print(f"REPO={REPO}")

# %% [markdown]
# ## 1. Stimuli — synthetic translating blob + a real public clip

# %%
SYNTH_NAME = "synth_translating_blob"
SYNTH_PATH = IMAGES / f"{SYNTH_NAME}.mp4"
SYNTH_GT_CSV = IMAGES / f"{SYNTH_NAME}_gt.csv"
N_FRAMES = 64
SIZE = 224

if not SYNTH_PATH.exists() or not SYNTH_GT_CSV.exists():
    print("synthesising translating-blob video …")
    synth_frames, synth_meta = synthesise_translating_blob(
        n_frames=N_FRAMES, size=SIZE, sigma=12.0, amplitude=100.0, seed=0,
    )
    write_video(SYNTH_PATH, synth_frames, fps=10)
    # Ground-truth centres in pixel + per-arch patch coordinates.
    centres = synth_meta.centres_xy
    rows: list[dict] = []
    grids = {"vit_b16": (14, 14), "dinov2_b": (16, 16)}
    cx_patches = {k: blob_centres_to_patch(centres, SIZE, g)[:, 0] for k, g in grids.items()}
    cy_patches = {k: blob_centres_to_patch(centres, SIZE, g)[:, 1] for k, g in grids.items()}
    for t in range(N_FRAMES):
        row = {
            "frame": t,
            "cx_px": float(centres[t, 0]),
            "cy_px": float(centres[t, 1]),
        }
        for k in grids:
            row[f"{k}_cx_patch"] = int(cx_patches[k][t])
            row[f"{k}_cy_patch"] = int(cy_patches[k][t])
        rows.append(row)
    pd.DataFrame(rows).to_csv(SYNTH_GT_CSV, index=False)
    print(f"  wrote {SYNTH_PATH.relative_to(REPO)} and {SYNTH_GT_CSV.relative_to(REPO)}")
else:
    print(f"reusing existing {SYNTH_PATH.name} + {SYNTH_GT_CSV.name}")
    synth_frames, synth_meta = synthesise_translating_blob(
        n_frames=N_FRAMES, size=SIZE, sigma=12.0, amplitude=100.0, seed=0,
    )

print(f"synthetic frames: {synth_frames.shape}, dtype={synth_frames.dtype}")

# %%
REAL_NAME = "real_clip"
REAL_DIR = RESULTS / REAL_NAME
REAL_DIR.mkdir(parents=True, exist_ok=True)
REAL_CACHE = DATA / "real_clip"
real_frames, real_meta = load_real_video(
    out_path=REAL_DIR / "fetched.mp4",
    cache_dir=REAL_CACHE,
    n_frames=N_FRAMES,
    size=SIZE,
    stride=2,
    skip_first=200,
)
print(f"real video: {real_frames.shape}, source={real_meta['source_name']}")

# Document the source — required deliverable.
(REAL_DIR / "SOURCE.md").write_text(
    f"# Real-video stimulus source\n\n"
    f"- Source name: `{real_meta['source_name']}`\n"
    f"- Source URL: {real_meta['source_url']}\n"
    f"- FPS (original): {real_meta['fps']}\n"
    f"- Total frames in source: {real_meta['n_total_frames']}\n"
    f"- Frames used: {real_meta['n_frames_used']} (stride={real_meta['stride']}, skip_first={real_meta['skip_first']})\n"
    f"- Pre-processing: centre-crop to square, resize to {SIZE}×{SIZE}, BILINEAR.\n"
    f"\nClip is public-domain / Creative-Commons (Wikimedia Commons).\n"
)
print(f"  wrote SOURCE.md → {(REAL_DIR / 'SOURCE.md').relative_to(REPO)}")

# Mirror the SOURCE.md under the synthetic stimulus dir for symmetry.
SYNTH_DIR = RESULTS / SYNTH_NAME
SYNTH_DIR.mkdir(parents=True, exist_ok=True)
(SYNTH_DIR / "SOURCE.md").write_text(
    "# Synthetic stimulus source\n\n"
    "Synthesised in-process by `eris.video.synthesise_translating_blob`:\n"
    f"- {N_FRAMES} frames, {SIZE}×{SIZE}, σ=12.0 px, amp=100/255, "
    "noise σ=12.0, seed=0.\n"
    "- Trajectory: constant horizontal velocity across inner 80% of the\n"
    "  frame, half-cycle vertical sinusoid (amplitude ≈ 22 px).\n"
    f"- Ground-truth centres in `images/{SYNTH_NAME}_gt.csv`.\n"
)

# %% [markdown]
# ## 2. Per-(video, arch) extraction
#
# 64 frames × 2 archs × 2 videos = 256 forwards. The big cost is keeping
# transformers loaded — we re-use them across the two videos.

# %%
VIDEOS = {
    SYNTH_NAME: (synth_frames, synth_meta),
    REAL_NAME: (real_frames, real_meta),
}
ARCHS = ("vit_b16", "dinov2_b")

H_videos: dict[tuple[str, str], np.ndarray] = {}
specs: dict[str, object] = {}

for arch in ARCHS:
    for video_name, (frames, _meta) in VIDEOS.items():
        out_dir = RESULTS / video_name / arch
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / "H_video.npz"
        if out_npz.exists():
            blob = np.load(out_npz)
            H_video = blob["H"]
            print(f"  reuse {arch} / {video_name}: H_video={H_video.shape}")
            H_videos[(arch, video_name)] = H_video
            if arch not in specs:
                # We still need the spec for figure conventions.
                _, _, _, spec = load_model(arch)
                specs[arch] = spec
            continue
        print(f"running {arch} / {video_name} …")
        # extract_entropy_stack iterates over images → list of PIL frames.
        pil_frames = list(frames_to_pil(frames))
        stack, spec = extract_entropy_stack(
            arch, pil_frames,
            progress=lambda i, n: (
                print(f"  frame {i}/{n}", end="\r")
                if i % 8 == 0 or i == n else None
            ),
        )
        print()
        # ``stack`` from extract_entropy_stack is shape
        # (n_frames, n_layers, gy, gx) when `images` is a list of frames.
        H_video = stack
        np.savez_compressed(out_npz, H=H_video)
        H_videos[(arch, video_name)] = H_video
        specs[arch] = spec
        print(f"  saved {out_npz.relative_to(REPO)}  shape={H_video.shape}")

# %% [markdown]
# ## 3. Per-(video, arch) figures
#
# Three figures per cell:
#
# - `depth_x_time.gif` — input frame ⊕ H @ L=6, animated across frames.
# - `transition_per_frame.png` — argmax ΔH-std layer per frame.
# - `layer_time_grid.png` — `(layer × frame)` heatmap of `H_std`.

# %%
def render_depth_x_time(
    frames: np.ndarray, H_video: np.ndarray, layer_idx: int, out_path: Path, title: str,
) -> None:
    """Animate the chosen layer's H field paired with the input frame."""
    n_frames = H_video.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4))
    fig.suptitle(title, fontsize=11)
    vmin = float(H_video[:, layer_idx].min())
    vmax = float(H_video[:, layer_idx].max())

    im_frame = axes[0].imshow(frames[0])
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title("input frame", fontsize=10)

    im_H = axes[1].imshow(
        H_video[0, layer_idx], cmap="magma", vmin=vmin, vmax=vmax,
    )
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title(f"H @ L{layer_idx + 1}", fontsize=10)
    cb = fig.colorbar(im_H, ax=axes[1], shrink=0.85)
    cb.ax.tick_params(labelsize=8)
    txt = fig.text(0.5, 0.02, "frame 1", ha="center", fontsize=9)

    def draw(t: int) -> tuple:
        im_frame.set_data(frames[t])
        im_H.set_data(H_video[t, layer_idx])
        txt.set_text(f"frame {t + 1} / {n_frames}")
        return im_frame, im_H, txt

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    anim = animation.FuncAnimation(
        fig, draw, frames=n_frames, interval=100, blit=False,
    )
    anim.save(out_path, writer="pillow", dpi=110, fps=10)
    plt.close(fig)


def render_transition_per_frame(
    H_video: np.ndarray, out_path: Path, title: str,
) -> np.ndarray:
    """Per-frame argmax-ΔH-std layer."""
    transitions = np.array(
        [transition_layer(H_video[t]) for t in range(H_video.shape[0])]
    )
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(np.arange(len(transitions)) + 1, transitions, marker="o", color="C3")
    ax.set_xlabel("frame")
    ax.set_ylabel("argmax ΔH-std layer")
    ax.set_ylim(0.5, H_video.shape[1] + 0.5)
    ax.axhline(int(np.median(transitions)), color="C0", ls="--", lw=0.8,
               label=f"median = {int(np.median(transitions))}")
    ax.legend(loc="best", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return transitions


def render_layer_time_grid(
    H_video: np.ndarray, out_path: Path, title: str,
) -> np.ndarray:
    """`(layer × frame)` heatmap of `H_std`."""
    n_frames, n_layers = H_video.shape[0], H_video.shape[1]
    H_std_grid = np.empty((n_layers, n_frames), dtype=np.float32)
    for t in range(n_frames):
        s = per_layer_stats(H_video[t])
        H_std_grid[:, t] = s["H_std"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    im = ax.imshow(H_std_grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("frame")
    ax.set_ylabel("layer (1-indexed bottom→top)")
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([str(L + 1) for L in range(n_layers)])
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("H_std")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return H_std_grid


# %%
LAYER_FOR_GIF = 5   # 0-indexed → layer 6 in 1-indexed display
all_transitions: dict[tuple[str, str], np.ndarray] = {}

for arch in ARCHS:
    for video_name, (frames, _meta) in VIDEOS.items():
        out_dir = RESULTS / video_name / arch
        H_video = H_videos[(arch, video_name)]
        title_prefix = f"{video_name} · {arch}"
        # 1) depth_x_time.gif — paired input+H @ L=6
        render_depth_x_time(
            frames, H_video, layer_idx=LAYER_FOR_GIF,
            out_path=out_dir / "depth_x_time.gif",
            title=f"{title_prefix} — H @ L{LAYER_FOR_GIF + 1}",
        )
        print(f"wrote {(out_dir / 'depth_x_time.gif').relative_to(REPO)}")
        # 2) transition_per_frame.png
        transitions = render_transition_per_frame(
            H_video, out_path=out_dir / "transition_per_frame.png",
            title=f"{title_prefix} — argmax ΔH-std per frame",
        )
        all_transitions[(arch, video_name)] = transitions
        print(f"wrote {(out_dir / 'transition_per_frame.png').relative_to(REPO)}  median={int(np.median(transitions))}")
        # 3) layer_time_grid.png
        render_layer_time_grid(
            H_video, out_path=out_dir / "layer_time_grid.png",
            title=f"{title_prefix} — H_std (layer × frame)",
        )
        print(f"wrote {(out_dir / 'layer_time_grid.png').relative_to(REPO)}")

# %% [markdown]
# ## 4. Quick numeric summary for the per-frame transition stability (P2)
#
# A useful single number per (video, arch): how stable is the
# transition layer across frames? We report `median ± stdev` and
# `frac(|trans - median| <= 1)`.

# %%
rows = []
for (arch, video_name), trans in all_transitions.items():
    med = float(np.median(trans))
    sd = float(trans.std())
    within_one = float(np.mean(np.abs(trans - med) <= 1))
    rows.append({
        "video": video_name, "arch": arch,
        "transition_median": med,
        "transition_std": sd,
        "frac_within_one_of_median": within_one,
    })
trans_df = pd.DataFrame(rows).sort_values(["video", "arch"]).reset_index(drop=True)
trans_df.to_csv(RESULTS / "transition_stability_summary.csv", index=False)
print(trans_df.to_string(index=False))
print(f"\nsaved {(RESULTS / 'transition_stability_summary.csv').relative_to(REPO)}")
