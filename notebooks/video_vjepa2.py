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
# # video_vjepa2 — entropy field on a native-3D video ViT
#
# Step E. ViT-B/16 and DINO-v2 (Step B) are still-image ViTs we apply
# frame-by-frame, then volumetrically. V-JEPA 2 is the natural
# control: a *native-3D* tubelet ViT trained on video, so its residual
# stream at every layer is a 3-D scalar field on the (tubelet-time ×
# spatial-y × spatial-x) grid by construction.
#
# Two questions:
# 1. Does V-JEPA 2 also exhibit a sharp depth-wise phase transition in
#    standardised per-token entropy, like the still-image ViTs?
# 2. On the synthetic translating-blob stimulus, does the volumetric
#    high-|∇H| component at the transition layer track the ground-truth
#    blob trajectory in (t, y, x)? In Step B (still-image ViTs +
#    per-frame fields, then stitched volumetrically) this metric was
#    dominated by register-token edge artefacts (Darcet et al. 2024).
#    A native-3D model has a chance to do better.
#
# Two stimuli, ViT-L (24 layers, 1024 hidden, 32×16×16 token grid for
# fpc=64 / 256² input):
# - synthetic blob (same trajectory as Step B, just 256² instead of 224²
#   for V-JEPA 2's preprocessor).
# - Big Buck Bunny clip (same source, same 64-frame slice, 256²).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eris.extract import extract_entropy_volume
from eris.fields import gradient_3d
from eris.video import (
    blob_centres_to_patch,
    load_real_video,
    synthesise_translating_blob,
)
from eris.volumetric import extract_outlier_tubes, iou_3d, render_streamtubes_html

REPO = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
RESULTS = REPO / "results" / "video_vjepa2"
RESULTS.mkdir(parents=True, exist_ok=True)

ARCH = "vjepa2_l"
N_FRAMES = 64
IMG_SIZE = 256

# %% [markdown]
# ## 1. Build / load the two stimuli at 256²

# %%
synth_frames, synth_meta = synthesise_translating_blob(
    n_frames=N_FRAMES, size=IMG_SIZE, sigma=14.0, amplitude=110.0,
    noise_std=12.0, seed=0,
)
print(f"synth: {synth_frames.shape}  blob crosses {synth_meta.centres_xy[0]} → "
      f"{synth_meta.centres_xy[-1]}")

real_frames, real_meta = load_real_video(
    out_path=RESULTS / "real_clip.mp4",
    cache_dir=REPO / "data" / "cache",
    n_frames=N_FRAMES, size=IMG_SIZE, stride=2, skip_first=900,
)
print(f"real: {real_frames.shape}  source={real_meta['source_name']}")

# %% [markdown]
# ### Constant-depth control
#
# Input volume with no temporal variation: 64 identical frames. If
# V-JEPA 2 actually uses the temporal axis in its residual stream, the
# resulting (gt, gy, gx) entropy volume should be ~constant along
# `t` (modulo numerical noise) — every tubelet sees the same content.
# If we observe structure along `t` anyway, that's an artefact of the
# temporal positional encoding rather than scene dynamics, and the
# volumetric framing on real videos needs caveating.
#
# Use the middle frame of the real clip (more naturalistic statistics
# than the synth blob) as the static content.

# %%
static_frame = real_frames[N_FRAMES // 2]                          # (256, 256, 3)
const_frames = np.broadcast_to(static_frame, (N_FRAMES, IMG_SIZE, IMG_SIZE, 3)).copy()
print(f"const: {const_frames.shape}   "
      f"all frames identical: {np.array_equal(const_frames[0], const_frames[-1])}")

# %% [markdown]
# ## 2. Run V-JEPA 2 → per-layer 3-D entropy volume per clip

# %%
def cached_volume(arch: str, name: str, frames: np.ndarray) -> np.ndarray:
    out_npz = RESULTS / f"{arch}_{name}_H_volume.npz"
    if out_npz.exists():
        return np.load(out_npz)["H"]
    vol, spec = extract_entropy_volume(
        arch, frames,
        progress=lambda L, n: print(f"  {name} L={L}/{n}", end="\r"),
    )
    np.savez_compressed(
        out_npz, H=vol,
        grid_size_3d=np.array(spec.grid_size_3d),
        n_layers=spec.n_layers,
    )
    print()
    return vol


H_synth = cached_volume(ARCH, "synth", synth_frames)
H_real = cached_volume(ARCH, "real", real_frames)
H_const = cached_volume(ARCH, "const", const_frames)
print(f"H_synth: {H_synth.shape}   H_real: {H_real.shape}   "
      f"H_const: {H_const.shape}")

# %% [markdown]
# ## 3. Phase curves vs depth (mean, std, |∇H|, ΔH-std)

# %%
def phase_curves(H_vol: np.ndarray) -> dict[str, np.ndarray]:
    """H_vol of shape (n_layers, gt, gy, gx) → per-layer summary statistics."""
    n_L = H_vol.shape[0]
    flat = H_vol.reshape(n_L, -1)
    H_mean = flat.mean(axis=1)
    H_std = flat.std(axis=1)
    grad_mag = np.empty(n_L)
    for L in range(n_L):
        Ht, Hy, Hx = gradient_3d(H_vol[L])
        grad_mag[L] = np.sqrt(Ht ** 2 + Hy ** 2 + Hx ** 2).mean()
    dH_std = np.r_[0.0, np.abs(np.diff(H_std))]
    return {"H_mean": H_mean, "H_std": H_std,
            "grad_mag": grad_mag, "dH_std": dH_std}


curves = {name: phase_curves(vol) for name, vol in
          [("synth", H_synth), ("real", H_real), ("const", H_const)]}
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True)
metric_titles = [("H_mean", "H mean"),
                 ("H_std", "H std (within-layer spread)"),
                 ("grad_mag", "mean |∇₃H| per layer"),
                 ("dH_std", "ΔH-std (cross-layer change)")]
for ax, (key, title) in zip(axes, metric_titles):
    for name, c in curves.items():
        n_L = len(c[key])
        ax.plot(np.arange(1, n_L + 1), c[key], "-o", label=name, ms=4)
    ax.set_title(title); ax.set_xlabel("layer")
    ax.grid(True, alpha=0.3)
axes[0].legend()
fig.suptitle(f"{ARCH} — per-layer entropy field summary, two video stimuli",
             y=1.01)
fig.tight_layout()
fig.savefig(RESULTS / "phase_curves.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {RESULTS / 'phase_curves.png'}")

# %% [markdown]
# ### Identify the V-JEPA 2 transition layer per stimulus

# %%
trans_summary = []
for name, c in curves.items():
    L_argmax = int(np.argmax(c["dH_std"][1:]) + 1) + 1   # 1-based, ignore L=1 padding
    peak = float(c["dH_std"].max())
    median = float(np.median(c["dH_std"][1:]))
    sharpness = peak / max(median, 1e-9)
    trans_summary.append({"video": name, "L_trans": L_argmax,
                          "peak_dHstd": peak, "median_dHstd": median,
                          "peak_over_median": sharpness})
    print(f"{name}: L_trans={L_argmax}  peak/median ΔH-std = {sharpness:.2f}")

# %% [markdown]
# ## 4. Per-tubelet transition stability
#
# At each tubelet timestep `t` (out of 32), compute the per-`t` slice
# `(n_layers, gy, gx)` and ask which layer maximises ΔH-std. If the
# transition is video-stationary the per-tubelet `L_trans` should equal
# the overall `L_trans` for ~all `t`, with std ≈ 0.

# %%
def per_tubelet_transition(H_vol: np.ndarray) -> np.ndarray:
    """Per tubelet t, transition layer = argmax_L |H_std(L,t) − H_std(L−1,t)|."""
    gt = H_vol.shape[1]
    out = np.empty(gt, dtype=int)
    for t in range(gt):
        slab = H_vol[:, t, :, :]
        std_per_L = slab.reshape(slab.shape[0], -1).std(axis=1)
        out[t] = int(np.argmax(np.abs(np.diff(std_per_L))) + 1) + 1   # 1-based
    return out


fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (name, vol) in zip(axes,
                           [("synth", H_synth), ("real", H_real),
                            ("const", H_const)]):
    ts = per_tubelet_transition(vol)
    ax.plot(np.arange(1, len(ts) + 1), ts, "o-", ms=4)
    ax.axhline(int(np.median(ts)), ls=":", c="gray",
               label=f"median = {int(np.median(ts))}")
    ax.set_xlabel("tubelet index t (1..32)")
    ax.set_ylabel("argmax ΔH-std layer")
    ax.set_title(f"{name}  (std across tubelets = {ts.std():.2f})")
    ax.grid(True, alpha=0.3); ax.legend()
fig.suptitle(f"{ARCH} — per-tubelet transition layer stability", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "transition_per_tubelet.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {RESULTS / 'transition_per_tubelet.png'}")

# %% [markdown]
# ## 5. Layer × tubelet H_std heatmap — global view of the depth × time field

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, (name, vol) in zip(axes,
                           [("synth", H_synth), ("real", H_real),
                            ("const", H_const)]):
    n_L, gt, gy, gx = vol.shape
    grid = vol.reshape(n_L, gt, -1).std(axis=2)            # (n_L, gt)
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   extent=(1, gt, 1, n_L), cmap="magma")
    ax.set_ylabel("layer")
    ax.set_title(f"{name} — H_std(L, t)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="H_std")
axes[-1].set_xlabel("tubelet index t")
fig.suptitle(f"{ARCH} — per-(layer, tubelet) within-spatial spread",
             y=1.0)
fig.tight_layout()
fig.savefig(RESULTS / "layer_tubelet_grid.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {RESULTS / 'layer_tubelet_grid.png'}")

# %% [markdown]
# ## 5b. Const-input diagnostic — does V-JEPA 2 invent temporal structure?
#
# H_const has identical input frames. Define
# `Δt-std(L) = std over tubelet t of  spatial mean of  H_const[L, t, :, :]`.
# If V-JEPA 2 strictly encoded content, Δt-std should be ~0 across all
# layers; any non-zero values are a temporal positional / register
# effect injected by the model. Compare against `Δt-std` for the synth
# blob (where genuine temporal motion is present) and for the real
# clip.

# %%
def t_std(H_vol: np.ndarray) -> np.ndarray:
    """For each layer, std over tubelet t of the spatial-mean H field."""
    return H_vol.mean(axis=(2, 3)).std(axis=1)


def xy_std(H_vol: np.ndarray) -> np.ndarray:
    """Per-layer mean spatial std (within each tubelet)."""
    return H_vol.std(axis=(2, 3)).mean(axis=1)


fig, ax = plt.subplots(figsize=(8, 4.5))
for name, vol in [("synth", H_synth), ("real", H_real), ("const", H_const)]:
    ax.plot(np.arange(1, vol.shape[0] + 1), t_std(vol), "-o",
            ms=4, label=f"{name}  (Δt-std)")
ax.set_xlabel("layer")
ax.set_ylabel("std over tubelet t  of spatial-mean H")
ax.set_title("Temporal variance per layer.\n"
             "const should be ~0 if V-JEPA 2 only encodes content")
ax.grid(True, alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "const_input_temporal_variance.png", dpi=140,
            bbox_inches="tight")
plt.close(fig)
print(f"saved {RESULTS / 'const_input_temporal_variance.png'}")

# Numeric ratio: const Δt-std as fraction of real Δt-std (per layer).
import pandas as pd
ratios = pd.DataFrame({
    "L": np.arange(1, H_synth.shape[0] + 1),
    "synth_t_std": t_std(H_synth),
    "real_t_std": t_std(H_real),
    "const_t_std": t_std(H_const),
})
ratios["const_over_real"] = ratios["const_t_std"] / np.maximum(
    ratios["real_t_std"], 1e-12)
ratios["const_over_synth"] = ratios["const_t_std"] / np.maximum(
    ratios["synth_t_std"], 1e-12)
ratios.to_csv(RESULTS / "const_input_temporal_variance.csv", index=False)
print(ratios.round(4).to_string(index=False))

# %% [markdown]
# ## 5c. Volumetric projections — see the field's shape in (t, y, x)
#
# For each clip + a few representative layers, project the H volume
# and the gradient-magnitude volume |∇H| along each axis (max-project
# for |∇H| to surface outlier tubes; mean-project for H to show the
# bulk distribution). For the synth blob, the diagonal (t, x)
# trajectory of the moving Gaussian should be visible in the
# y-projection of |∇H|. For the const clip, all three projections
# should show negligible temporal structure if V-JEPA 2 is content-
# faithful.

# %%
def project_volume(vol: np.ndarray, op: str = "mean") -> dict[str, np.ndarray]:
    """3-D `(gt, gy, gx)` volume → three 2-D projections.

    Returns ``{"t": (gy, gx), "y": (gt, gx), "x": (gt, gy)}``.
    `op` is "mean" or "max"; max is the more useful one for sparse
    fields like |∇H|.
    """
    fn = vol.mean if op == "mean" else vol.max
    return {
        "t": fn(axis=0),                           # collapse t → (gy, gx)
        "y": fn(axis=1),                           # collapse y → (gt, gx)
        "x": fn(axis=2),                           # collapse x → (gt, gy)
    }


def plot_projection_grid(
    vols_named: list[tuple[str, np.ndarray]], out_path: Path,
    layers_of_interest: list[int], title: str, op: str = "mean",
    cmap: str = "viridis",
) -> None:
    """One row per (clip, layer) pair; 3 columns = (t, y, x) projections."""
    n_clips = len(vols_named)
    n_L = len(layers_of_interest)
    fig, axes = plt.subplots(
        n_clips * n_L, 3, figsize=(11, 3.0 * n_clips * n_L),
        squeeze=False,
    )
    for ci, (name, vol) in enumerate(vols_named):
        for li, L in enumerate(layers_of_interest):
            row = ci * n_L + li
            slab = vol[L - 1]                       # (gt, gy, gx)
            projs = project_volume(slab, op=op)
            for ax, axis_label in zip(axes[row], ["t", "y", "x"]):
                im = ax.imshow(projs[axis_label], cmap=cmap, aspect="auto")
                ax.set_title(f"{name}  L={L}  {op}-proj along {axis_label}",
                             fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# H projections: mean over the collapsed axis.
# Pick layers spanning the depth: shallow / mid / late.
LAYERS_INTEREST = [4, 12, 24]
plot_projection_grid(
    [("synth", H_synth), ("real", H_real), ("const", H_const)],
    out_path=RESULTS / "H_projections.png",
    layers_of_interest=LAYERS_INTEREST, op="mean",
    title=f"{ARCH} — mean-projections of per-layer H volume "
          f"(rows = clip × layer, cols = (t, y, x) projections)",
    cmap="viridis",
)
print(f"saved {RESULTS / 'H_projections.png'}")

# |∇H| projections: max-project so sparse outlier tubes survive.
def gradmag_volume(vol: np.ndarray) -> np.ndarray:
    """Per-layer |∇₃H| volume of shape (n_layers, gt, gy, gx)."""
    n_L = vol.shape[0]
    out = np.empty_like(vol)
    for L in range(n_L):
        Ht, Hy, Hx = gradient_3d(vol[L])
        out[L] = np.sqrt(Ht ** 2 + Hy ** 2 + Hx ** 2)
    return out


grad_synth = gradmag_volume(H_synth)
grad_real = gradmag_volume(H_real)
grad_const = gradmag_volume(H_const)

plot_projection_grid(
    [("synth", grad_synth), ("real", grad_real), ("const", grad_const)],
    out_path=RESULTS / "gradH_projections.png",
    layers_of_interest=LAYERS_INTEREST, op="max",
    title=f"{ARCH} — max-projections of per-layer |∇₃H| volume "
          f"(rows = clip × layer, cols = (t, y, x); look for diagonals "
          f"in 'y-proj' for synth's translating blob)",
    cmap="inferno",
)
print(f"saved {RESULTS / 'gradH_projections.png'}")

# %% [markdown]
# ## 6. Volumetric outlier-tube tracking on the synthetic blob
#
# Step B's volumetric tracking on still-image ViTs failed (bundle IoU =
# 0 for ViT-B/16) due to register-token edge artefacts: 60–80 % of the
# top-5% |∇H| voxels were on frame edges. Native-3D V-JEPA 2 has no
# explicit register tokens; if the entropy field carries motion
# structure, the largest connected component should now intersect the
# ground-truth blob trajectory.
#
# Build the GT tubelet mask in (gt=32, gy=16, gx=16). Each tubelet
# covers 2 input frames; take the midpoint of those two frames'
# centres.

# %%
gt_synth, gy_synth, gx_synth = H_synth.shape[1:]
centres = synth_meta.centres_xy                 # (n_frames=64, 2) in (cx_px, cy_px)
# average each pair of frames → 32 tubelet centres
tubelet_centres = centres.reshape(gt_synth, 2, 2).mean(axis=1)
patches = blob_centres_to_patch(
    tubelet_centres, image_size=IMG_SIZE, grid=(gy_synth, gx_synth)
)                                                # (gt, 2) cx, cy
gt_mask = np.zeros((gt_synth, gy_synth, gx_synth), dtype=bool)
for t in range(gt_synth):
    cxp, cyp = patches[t]
    gt_mask[t, cyp, cxp] = True
# 1-patch dilate (y, x only) to allow some jitter
import scipy.ndimage as ndi
struct2d = ndi.generate_binary_structure(2, 1)
for t in range(gt_synth):
    gt_mask[t] = ndi.binary_dilation(gt_mask[t], structure=struct2d, iterations=1)
print(f"GT tube voxels: {gt_mask.sum()} / {gt_mask.size}")

# %% [markdown]
# ### Per-layer tube IoU on synth (largest connected component vs GT)

# %%
def tube_metrics_per_layer(H_vol: np.ndarray, gt_mask: np.ndarray,
                           top_pct: float = 5.0) -> np.ndarray:
    n_L = H_vol.shape[0]
    rows = []
    for L in range(n_L):
        Ht, Hy, Hx = gradient_3d(H_vol[L])
        gmag = np.sqrt(Ht ** 2 + Hy ** 2 + Hx ** 2)
        # diagnose register-token edge dominance
        edge = np.zeros_like(gmag, dtype=bool)
        edge[:, 0, :] = edge[:, -1, :] = True
        edge[:, :, 0] = edge[:, :, -1] = True
        thresh = float(np.percentile(gmag, 100.0 - top_pct))
        topmask = gmag >= thresh
        edge_frac = float((topmask & edge).sum()) / max(int(topmask.sum()), 1)
        # largest connected component IoU
        ext = extract_outlier_tubes(gmag, top_pct=top_pct, connectivity=2)
        rows.append({
            "L": L + 1,
            "top_thresh": thresh,
            "n_top_voxels": int(topmask.sum()),
            "edge_frac": edge_frac,
            "n_components": ext.n_components,
            "largest_size": ext.largest_size,
            "iou_largest_vs_gt": iou_3d(ext.largest_mask, gt_mask),
            "iou_topmask_vs_gt": iou_3d(topmask, gt_mask),
        })
    return rows


import pandas as pd
rows = tube_metrics_per_layer(H_synth, gt_mask, top_pct=5.0)
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "tube_metrics_synth.csv", index=False)
print(df.round(4).to_string(index=False))

# %% [markdown]
# ### Plot tube-tracking results

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(df["L"], df["iou_largest_vs_gt"], "o-",
             label="largest CC vs GT")
axes[0].plot(df["L"], df["iou_topmask_vs_gt"], "s--",
             label="raw top-5% mask vs GT", ms=4)
axes[0].set_xlabel("layer"); axes[0].set_ylabel("IoU")
axes[0].set_title("3-D tube IoU vs ground-truth (synth)")
axes[0].grid(True, alpha=0.3); axes[0].legend()
axes[1].plot(df["L"], df["edge_frac"], "o-", color="firebrick")
axes[1].axhline(4/16, ls="--", c="gray",
                label="chance edge frac = 4/16 = 0.25")
axes[1].set_xlabel("layer"); axes[1].set_ylabel("edge fraction")
axes[1].set_title("Top-5% |∇H| voxels on frame edge\n(register-token diagnostic)")
axes[1].grid(True, alpha=0.3); axes[1].legend()
axes[2].plot(df["L"], df["n_components"], "o-")
axes[2].set_xlabel("layer"); axes[2].set_ylabel("n connected components")
axes[2].set_title("Connected-component count")
axes[2].grid(True, alpha=0.3)
fig.suptitle(f"{ARCH} — volumetric tube tracking on synth blob "
             f"(GT trajectory in (t, y, x))", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "tube_metrics_summary.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved {RESULTS / 'tube_metrics_summary.png'}")

# %% [markdown]
# ## 7. Best-IoU layer — animate the largest CC + the GT mask

# %%
best_L = int(df.loc[df["iou_largest_vs_gt"].idxmax(), "L"])
print(f"best IoU layer for synth tube tracking: L={best_L}, "
      f"IoU={df['iou_largest_vs_gt'].max():.3f}")

L_idx = best_L - 1
Ht, Hy, Hx = gradient_3d(H_synth[L_idx])
gmag = np.sqrt(Ht ** 2 + Hy ** 2 + Hx ** 2)
ext = extract_outlier_tubes(gmag, top_pct=5.0, connectivity=2)

# Render side-by-side animation: GT mask | predicted mask | overlay
import matplotlib.animation as anim
gt_, gy_, gx_ = gt_mask.shape
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
ims = [axes[i].imshow(np.zeros((gy_, gx_), dtype=float),
                       cmap="gray" if i == 0 else
                       ("inferno" if i == 1 else "viridis"),
                       vmin=0, vmax=1) for i in range(3)]
titles = ["GT tubelet mask", f"largest CC at L={best_L}",
          "GT (cyan) + pred (red)"]
for ax, t in zip(axes, titles):
    ax.set_title(t, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

def init():
    return ims

def update(t):
    overlay = np.zeros((gy_, gx_, 3), dtype=float)
    overlay[..., 1] = gt_mask[t].astype(float)         # GT in green/cyan
    overlay[..., 2] = gt_mask[t].astype(float)
    overlay[..., 0] = ext.largest_mask[t].astype(float)
    ims[0].set_data(gt_mask[t].astype(float))
    ims[1].set_data(ext.largest_mask[t].astype(float))
    axes[2].clear()
    axes[2].imshow(overlay)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].set_title("GT (cyan) + pred (red)", fontsize=10)
    fig.suptitle(f"{ARCH} synth blob, L={best_L}, t={t+1}/{gt_}", y=1.0)
    return ims

ani = anim.FuncAnimation(fig, update, frames=gt_, init_func=init,
                         interval=100, blit=False)
out_gif = RESULTS / f"tube_compare_L{best_L}.gif"
ani.save(out_gif, writer="pillow", fps=8)
plt.close(fig)
print(f"saved {out_gif}")

# %% [markdown]
# ## 8. Per-frame H@best_L animation (visual on the input video)

# %%
def render_field_over_video(frames: np.ndarray, H_vol: np.ndarray, L_idx: int,
                            out_gif: Path, title: str) -> None:
    """Animate H_vol[L_idx, t, :, :] heatmap upsampled onto the input frame."""
    n_frames = frames.shape[0]
    gt_, gy_, gx_ = H_vol.shape[1:]
    H_layer = H_vol[L_idx]                 # (gt, gy, gx)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    img_ax, h_ax = axes
    ima = img_ax.imshow(frames[0])
    img_ax.set_xticks([]); img_ax.set_yticks([])
    img_ax.set_title("input frame", fontsize=10)
    imh = h_ax.imshow(H_layer[0], cmap="viridis",
                      vmin=H_layer.min(), vmax=H_layer.max())
    h_ax.set_xticks([]); h_ax.set_yticks([])
    h_ax.set_title(f"H @ L={L_idx+1} (per-tubelet)", fontsize=10)
    fig.colorbar(imh, ax=h_ax, fraction=0.04, pad=0.02, label="H (nats)")

    def update(frame_t):
        ima.set_data(frames[frame_t])
        # 2 frames per tubelet: int(frame_t / 2)
        tubelet = min(int(frame_t // 2), gt_ - 1)
        imh.set_data(H_layer[tubelet])
        fig.suptitle(f"{title}  frame {frame_t+1}/{n_frames}  "
                     f"tubelet {tubelet+1}/{gt_}", y=1.0)
        return ima, imh

    ani = anim.FuncAnimation(fig, update, frames=n_frames,
                             interval=80, blit=False)
    ani.save(out_gif, writer="pillow", fps=10)
    plt.close(fig)


render_field_over_video(synth_frames, H_synth, best_L - 1,
                        RESULTS / f"synth_H_L{best_L}.gif",
                        f"{ARCH} synth blob")
render_field_over_video(real_frames, H_real, best_L - 1,
                        RESULTS / f"real_H_L{best_L}.gif",
                        f"{ARCH} real clip")
print(f"saved depth_x_time GIFs at L={best_L}")

# %% [markdown]
# ## 9. 3-D streamtubes interactive HTML at L_trans

# %%
L_idx = trans_summary[0]["L_trans"] - 1   # synth's transition layer
Ht_s, Hy_s, Hx_s = gradient_3d(H_synth[L_idx])
out_html = render_streamtubes_html(
    Hx=Hx_s, Hy=Hy_s, Ht=Ht_s,
    out_path=RESULTS / f"synth_streamtubes_L{L_idx+1}.html",
    title=f"{ARCH} synth blob L={L_idx+1}",
    starts=4, upsample_xy=4, sizeref=0.4,
)
print(f"saved {out_html}")

# %% [markdown]
# ## 10. Persist transition + tracking summary
#
# CSV the headline numbers next to Step B's so the comparison is one
# step away.

# %%
import json
summary = {
    "arch": ARCH,
    "model": "facebook/vjepa2-vitl-fpc64-256",
    "n_layers": int(H_synth.shape[0]),
    "grid_size_3d": list(H_synth.shape[1:]),
    "transition_per_video": trans_summary,
    "best_synth_tube_iou": {
        "L": best_L,
        "iou_largest_vs_gt": float(df["iou_largest_vs_gt"].max()),
        "edge_frac_at_best": float(df.loc[df["L"] == best_L, "edge_frac"].iloc[0]),
        "iou_topmask_vs_gt_at_best":
            float(df.loc[df["L"] == best_L, "iou_topmask_vs_gt"].iloc[0]),
    },
    "step_b_reference": {
        "vit_b16_synth_best_iou_largest": 0.000,
        "dinov2_synth_best_iou_largest_at_L6": 0.299,
        "vit_b16_synth_edge_frac_top5pct": "0.60–0.80",
    },
}
with (RESULTS / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
