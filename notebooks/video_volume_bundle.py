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
# # video_volume_bundle — bundle-of-tubes & per-frame metrics
#
# Step B's first pass measured "IoU of the *largest single connected
# component* of high-|∇H| voxels vs the ground-truth blob tube" and
# got 0.0 across every config. That metric is wrong: outlier patches
# are sparse and distributed, so the volume fragments into hundreds of
# small CCs (387–427 of them) — there is no single coherent tube to
# match. The right things to ask are:
#
# 1. **Bundle IoU** — IoU between (union of top-K largest CCs) and the
#    GT tube, swept over K. If outlier patches form a *bundle* travelling
#    with the object, bundle IoU should rise sharply with K.
# 2. **Per-frame spatial overlap** — at each frame t, what fraction of
#    that frame's top-|∇H| patches sit inside the GT blob region?
#    Average across frames. Tests whether outliers are *spatially
#    locked* to image content even if they don't form temporally
#    connected tubes.
# 3. **Centroid trajectory** — per frame, centre-of-mass of top-|∇H|
#    voxels vs the GT blob centre. Pearson r in (cx, cy).
#
# Reuses Step B's H_video.npz tensors; does not re-run any forward.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage

REPO = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
RESULTS = REPO / "results" / "video"
GT_CSV = REPO / "images" / "synth_translating_blob_gt.csv"

GT = pd.read_csv(GT_CSV)
print(f"GT frames: {len(GT)}")
GT.head()


# %%
def gt_tube(arch: str, n_frames: int, gy: int, gx: int, dilate: int = 1) -> np.ndarray:
    """Binary GT tube of shape (n_frames, gy, gx). 1 at the GT patch and
    its `dilate`-neighbourhood per frame; 0 elsewhere."""
    cx_col = f"{arch}_cx_patch"
    cy_col = f"{arch}_cy_patch"
    if cx_col not in GT.columns:
        raise KeyError(f"GT csv missing {cx_col} (cols={list(GT.columns)})")
    tube = np.zeros((n_frames, gy, gx), dtype=bool)
    for t in range(min(n_frames, len(GT))):
        cx = int(GT.iloc[t][cx_col])
        cy = int(GT.iloc[t][cy_col])
        for dy in range(-dilate, dilate + 1):
            for dx in range(-dilate, dilate + 1):
                yy, xx = cy + dy, cx + dx
                if 0 <= yy < gy and 0 <= xx < gx:
                    tube[t, yy, xx] = True
    return tube


def grad_volume_magnitude(H_video: np.ndarray, layer: int) -> np.ndarray:
    """|∇H_L| per voxel of shape (t, gy, gx). H_video: (n_frames, L, gy, gx)."""
    H_L = H_video[:, layer, :, :]                      # (t, gy, gx)
    Ht = np.gradient(H_L, axis=0)
    Hy = np.gradient(H_L, axis=1)
    Hx = np.gradient(H_L, axis=2)
    return np.sqrt(Ht ** 2 + Hy ** 2 + Hx ** 2)


def topk_mask(volume: np.ndarray, k_frac: float) -> np.ndarray:
    """Top-(k_frac × N) voxels by magnitude → boolean mask same shape."""
    n_keep = max(1, int(volume.size * k_frac))
    thr = np.partition(volume.ravel(), -n_keep)[-n_keep]
    return volume >= thr


def bundle_iou(mask: np.ndarray, gt: np.ndarray, top_k_components: int) -> float:
    """IoU of (union of K largest CCs in `mask`) vs `gt`."""
    labels, n = scipy.ndimage.label(mask)
    if n == 0:
        return 0.0
    sizes = scipy.ndimage.sum_labels(mask, labels, index=range(1, n + 1))
    keep = np.argsort(sizes)[::-1][:top_k_components] + 1
    bundle = np.isin(labels, keep)
    inter = np.logical_and(bundle, gt).sum()
    union = np.logical_or(bundle, gt).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def per_frame_overlap(mask: np.ndarray, gt: np.ndarray) -> tuple[float, np.ndarray]:
    """Per-frame fraction of `mask` voxels that fall inside `gt`.

    Returns ``(mean_across_frames, per_frame_array)``.
    """
    n_frames = mask.shape[0]
    ratios = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        m = mask[t]; g = gt[t]
        denom = m.sum()
        ratios[t] = (np.logical_and(m, g).sum() / denom) if denom > 0 else 0.0
    return float(ratios.mean()), ratios


def centroid_trajectory(volume: np.ndarray, mask: np.ndarray):
    """Per-frame |grad|-weighted centre of mass of voxels inside `mask`.

    Returns array of shape ``(n_frames, 2)`` with (cy, cx) per frame
    (NaN for frames with empty mask).
    """
    n_frames, gy, gx = volume.shape
    out = np.full((n_frames, 2), np.nan, dtype=np.float64)
    yy, xx = np.mgrid[:gy, :gx]
    for t in range(n_frames):
        m = mask[t]
        if not m.any():
            continue
        w = volume[t][m]
        out[t, 0] = float((yy[m] * w).sum() / w.sum())
        out[t, 1] = float((xx[m] * w).sum() / w.sum())
    return out


# %%
configs = [
    # 0-indexed layer; H_stack shape is (12, gy, gx) for ViT-B/16 and
    # DINO-v2-base. Layer 5 = post-transition L=6 in smoke-test convention;
    # layer 11 = final block.
    ("synth_translating_blob", "vit_b16", 5),
    ("synth_translating_blob", "vit_b16", 11),
    ("synth_translating_blob", "dinov2_b", 5),
    ("synth_translating_blob", "dinov2_b", 11),
    ("real_clip", "vit_b16", 5),
    ("real_clip", "vit_b16", 11),
    ("real_clip", "dinov2_b", 5),
    ("real_clip", "dinov2_b", 11),
]

# Sweep top-K CCs and top-K-fraction thresholds.
TOPK_CCS = [1, 5, 20, 50, 100, 250, 1000]
KFRACS = [0.02, 0.05, 0.10, 0.20]   # top 2/5/10/20% voxels by |∇H|

rows = []
for video, arch, layer in configs:
    npz = RESULTS / video / arch / "H_video.npz"
    if not npz.exists():
        print(f"missing {npz}; skipping")
        continue
    H_video = np.load(npz)["H"]                        # (n_frames, L, gy, gx)
    n_frames, n_layers, gy, gx = H_video.shape
    grad_mag = grad_volume_magnitude(H_video, layer)

    # Real clip has no patch-grid GT; skip GT-relative metrics and only
    # report a frame-edge concentration diagnostic (Darcet 2024 register-
    # token signal) so we know if outlier mass is image-content vs edge-
    # artefact-dominated.
    if video == "real_clip":
        for kfrac in KFRACS:
            mask = topk_mask(grad_mag, kfrac)
            edge_mask = np.zeros((gy, gx), dtype=bool)
            edge_mask[0] = edge_mask[-1] = edge_mask[:, 0] = edge_mask[:, -1] = True
            edge_voxels = mask & edge_mask[None]
            edge_frac = float(edge_voxels.sum() / max(mask.sum(), 1))
            rows.append({
                "video": video, "arch": arch, "layer": layer,
                "kfrac": kfrac, "k_ccs": -1,
                "metric": "edge_frac", "value": edge_frac,
            })
        continue

    gt = gt_tube(arch, n_frames=n_frames, gy=gy, gx=gx, dilate=1)

    for kfrac in KFRACS:
        mask = topk_mask(grad_mag, kfrac)
        # Bundle IoU sweep
        for k_ccs in TOPK_CCS:
            iou = bundle_iou(mask, gt, top_k_components=k_ccs)
            rows.append({
                "video": video, "arch": arch, "layer": layer,
                "kfrac": kfrac, "k_ccs": k_ccs,
                "metric": "bundle_iou", "value": iou,
            })
        # Per-frame overlap (single value per kfrac)
        ovr_mean, _ = per_frame_overlap(mask, gt)
        rows.append({
            "video": video, "arch": arch, "layer": layer,
            "kfrac": kfrac, "k_ccs": -1,
            "metric": "per_frame_overlap", "value": ovr_mean,
        })

    # Centroid trajectory at the median kfrac (5%).
    mask = topk_mask(grad_mag, 0.05)
    centroids = centroid_trajectory(grad_mag, mask)
    cx_col = f"{arch}_cx_patch"; cy_col = f"{arch}_cy_patch"
    gt_cy = GT[cy_col].values[:n_frames]; gt_cx = GT[cx_col].values[:n_frames]
    valid = ~np.isnan(centroids[:, 0])
    if valid.sum() >= 4:
        from scipy.stats import pearsonr
        r_y, _ = pearsonr(centroids[valid, 0], gt_cy[valid])
        r_x, _ = pearsonr(centroids[valid, 1], gt_cx[valid])
    else:
        r_y = r_x = float("nan")
    rows.append({
        "video": video, "arch": arch, "layer": layer,
        "kfrac": 0.05, "k_ccs": -1,
        "metric": "centroid_pearson_r_x", "value": r_x,
    })
    rows.append({
        "video": video, "arch": arch, "layer": layer,
        "kfrac": 0.05, "k_ccs": -1,
        "metric": "centroid_pearson_r_y", "value": r_y,
    })

df = pd.DataFrame(rows)
out_csv = RESULTS / "volume_metrics_bundle.csv"
df.to_csv(out_csv, index=False)
print(f"saved {out_csv}  ({len(df)} rows)")

# %% [markdown]
# ## Headline tables

# %%
print("=== bundle IoU at top-20% voxels ===")
print(df[(df["metric"] == "bundle_iou") & (df["kfrac"] == 0.20)]
      .pivot_table(index=["arch", "layer"], columns="k_ccs", values="value")
      .round(3))

# %%
print("\n=== per-frame spatial overlap (fraction of top-K-frac voxels inside GT region) ===")
print(df[df["metric"] == "per_frame_overlap"]
      .pivot_table(index=["arch", "layer"], columns="kfrac", values="value")
      .round(3))

# %%
print("\n=== centroid Pearson r vs GT (kfrac=0.05) ===")
print(df[df["metric"].str.startswith("centroid_pearson")]
      .pivot_table(index=["arch", "layer"], columns="metric", values="value")
      .round(3))
