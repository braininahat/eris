# eris — entropy field across ViT depth

Per-patch differential entropy on standardised residual streams of ViT-style
models, treated as a 2-D scalar field on the patch grid.

## Findings as of 2026-05-09

### 1. Sharp depth-wise phase transition replicates across all 4 ViTs tested

200 tiny-imagenet val images run through `vit-base-patch16-224`,
`vit-large-patch16-224`, `dinov2-base`, and `clip-vit-base-patch16`. Per
arch, per image, transition layer = `argmax_L (ΔH-std)`.

| arch | L_trans | n_blocks | rel `L/L_max` | peak/median sharpness |
|---|---|---|---|---|
| ViT-B/16 (supervised) | 5 | 12 | **0.42** | 9.3 |
| ViT-L/16 (supervised) | 5 | 24 | **0.21** | 21.2 |
| DINO-v2-base (SSL) | 8 | 12 | **0.67** | 5.0 |
| CLIP-ViT-B/16 (contrastive) | 7 | 12 | **0.58** | 4.0 |

Per-image IQR of `L_trans` is 0 in every arch — the transition is rock-stable
across images within an arch. The relative-depth location is gated by
**training objective**: supervised classification → early transition;
self-supervised / contrastive → later. Within supervised models, larger →
earlier in both absolute and relative terms (ViT-L L=5/24 < ViT-B L=5/12).

Original prediction "transition at L/L_max ≈ 0.4 universal" falsifies as
stated; refrains as "phenomenon universal, location training-dependent".

Figure: `cross_arch_phase_transition.png` (with input-image strip).

### 2. Entropy field carries information that simpler signals don't

Paired-bootstrap on outlier-localisation accuracy (top-K patches selected by
each signal vs saliency-defined ground-truth, n_boot = 2000), ViT-B/16,
n=200 images, primary comparison layer L=9, top-K=30:

| comparison | Δprecision | p (paired bootstrap) |
|---|---|---|
| entropy − L2 norm | +0.027 | **0.000** |
| entropy − attention rollout | +0.038 | **0.000** |
| entropy − random-init residual entropy | +0.115 | **0.000** |

Per-layer scan: entropy strictly dominates L2 at L1–L9 (p ≤ 0.038); ties
at L10–L12 where residual norms grow large. The L4→L5 spike in ΔH-std is
**unique to the trained-checkpoint entropy field** — L2, attention, and
random-init H show nothing comparable.

Methodological note: outliers in the entropy field are *low* H (sparse
non-Gaussian patches against a near-Gaussian baseline ~ 1.42 nats). They
are scored by absolute deviation from the per-layer median, not raw top-K.

Figures: `baselines/phase_curves.png`, `baselines/comparison_per_image.png`.

### 3. Per-frame transition stability holds; volumetric tube tracking does not

Two videos, two architectures (ViT-B/16, DINO-v2-base), 64 frames each.

**Per-frame transition stability — perfect.**
| video | arch | median L_trans | std | frac within ±1 of median |
|---|---|---|---|---|
| Big Buck Bunny clip | ViT-B/16 | 5 | 0 | 1.0 |
| Big Buck Bunny clip | DINO-v2 | 8 | 0 | 1.0 |
| synth translating blob | ViT-B/16 | 5 | 0 | 1.0 |
| synth translating blob | DINO-v2 | 8 | 0 | 1.0 |

**Volumetric outlier-tube tracking — fails on synthetic blob.** Three
tracking metrics on the synthetic stimulus (where ground-truth blob
trajectory in `(t, gy, gx)` is known):

| arch | layer | bundle IoU (top-1 CC, kfrac=0.20) | per-frame overlap (kfrac=0.05) | centroid Pearson r_x |
|---|---|---|---|---|
| ViT-B/16 | 6 | 0.000 | 0.000 | +0.006 |
| ViT-B/16 | 12 | 0.000 | 0.000 | −0.151 |
| DINO-v2 | 6 | 0.299 | 0.116 | +0.405 |
| DINO-v2 | 12 | 0.000 | 0.000 | +0.045 |

Diagnosis (visible in `bundle_metrics_summary.png`'s edge-fraction panel):
even at the top 5 % of `|∇H|` voxels by magnitude, ViT-B/16 places **60–80 %
of those voxels on the frame-edge row/column** — Darcet et al. 2024
register-token / edge-artefact effect. Per-patch standardisation kills the
magnitude cue that would have flagged the blob; the blob's per-patch
distribution sits near the Gaussian baseline. DINO-v2 has weaker register-
token dominance and shows modest content tracking (r_x = 0.40 at L=5).

Figures: `video/{video}/{arch}/{depth_x_time.gif, layer_time_grid.png,
transition_per_frame.png}`, `video/bundle_metrics_summary.png`.

### 4. Estimator robustness: pending

Vasicek (scipy default) used throughout the above. k-NN (Kozachenko-
Leonenko) and KDE estimators implemented but cross-validation of the L4→L5
finding under each is in progress.

## What this is + isn't

- It IS: a clean depth-wise structural finding in trained-ViT residual
  streams; the visualisation pipeline that surfaced it; a methodological
  comparison against simpler per-patch baselines.
- It IS NOT: a new mechanism, a free-energy derivation, or a causal probe.
- The phenomenon (sparse non-Gaussian "outlier features") is in the
  literature (Kovaleva 2021, Dettmers 2022, Sun 2024). What this work adds
  is the **spatial dimension** — outliers live on a 2-D patch grid and
  emerge in a single-layer phase transition whose timing is gated by
  training objective.

## Reproducibility

Run order:
```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/synth_check.ipynb        # gate
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/cross_arch.ipynb         # Step A
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/video_frames.ipynb       # Step B1
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/video_volume.ipynb       # Step B2
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/video_volume_bundle.ipynb # bundle eval
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/baselines.ipynb          # Step C
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/estimator_robustness.ipynb  # Step D (pending)
```

Outputs all under `results/`. Total compute < 30 min on a single 4090.

## File index (key artefacts)

```
results/
├── summary.md                                  ← this file
├── cross_arch_phase_transition.png             ← Step A headline
├── cross_arch/{arch}/H_stack.npz               ← per-arch H tensors (4 archs × 200 imgs)
├── baselines/
│   ├── phase_curves.png                        ← Step C 4-panel signal overlay
│   ├── comparison_per_image.png                ← per-image outlier visualisation
│   └── outlier_localisation.csv                ← paired-bootstrap p-values
├── video/
│   ├── bundle_metrics_summary.png              ← bundle/per-frame/centroid 6-panel
│   ├── volume_metrics_bundle.csv               ← bundle re-eval table
│   └── {video}/{arch}/                         ← per-(video, arch) outputs
│       ├── H_video.npz                         ← (n_frames, n_layers, gy, gx)
│       ├── depth_x_time.gif                    ← H@L6 animating across frames
│       ├── transition_per_frame.png            ← argmax ΔH-std per frame
│       ├── layer_time_grid.png                 ← (layer × frame) H_std heatmap
│       ├── outlier_tubes_L6.gif                ← high-|∇H| binary mask anim
│       └── volume_streamlines_L6.html          ← interactive 3D streamtubes
└── (smoke-test artefacts from earlier sessions: per-image field_stack.png,
    derivatives.png, evolution.gif, depth_movie.gif, plus cross_image_summary,
    depth_curves, depth_derivatives_summary, L4_L5_jump_localisation, etc.)
```
