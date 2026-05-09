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

### 4. Estimator robustness: L4→L5 transition is not a Vasicek artefact

ViT-B/16, same 200 tiny-imagenet val images, three estimators (Vasicek
scipy default, k-NN Kozachenko-Leonenko k=3, KDE Gaussian kernel
Silverman bandwidth):

| estimator | median L_trans | IQR | within ±1 of Vasicek median |
|---|---|---|---|
| Vasicek | 5.0 | 0.0 | 100% |
| k-NN | 5.0 | 0.0 | 100% |
| KDE | 4.0 | 1.0 | 100% |

ΔH-std peak/median on the body of the curve: Vasicek 9.3, k-NN 4.6,
KDE 6.1 — all ≫ 2.0 sharpness threshold. Phase curves overlay near-
identically on z-scored axes; the L=5 spike is the dominant feature
under every estimator. Field-stack at L=6 on a single image: all three
flag the *same* outlier patch cluster.

Anomaly: KDE's median sits at L=4 vs L=5 for Vasicek/k-NN; still inside
the ±1 envelope. KDE is ~5× slower than the others (52 min vs 4 min
for k-NN on the same 200×12 forward set) — Gaussian KDE log-pdf is
O(n²) in 768 samples.

Figures: `estimator_robustness/{phase_curves.png,
field_stack_compare.png, transition_layer_agreement.png}`,
`estimator_robustness/agreement.csv`.

### 5. The L4→L5 transition has a geometric correlate in residual space

UMAP of per-patch raw residual vectors (768-D for ViT-B/16, 768-D for
DINO-v2), pooled across 200 tiny-imagenet val images, at four layers
per arch, two normalisations (raw / per-patch standardised). Coloured
by per-patch H.

ViT-B/16 — at L=1 the residual cloud is one cohesive blob with
near-uniform H. At L=5 (transition) a discrete low-H cluster splinters
from the bulk; the split persists through L=12. Quantitatively, low-H
"outlier" patches (bottom 1% of H) sit at mean-50-NN distance 8–20×
**tighter** to each other than the bulk under per-patch standardisation
(ratio 0.05–0.12 at L=5/6/12, vs 0.62 at L=1). Under raw residuals the
ratio inverts to >1 at deep layers because outlier-feature *magnitudes*
spread them in raw L2 space — the geometric clustering is a
distribution-shape effect that standardisation reveals.

DINO-v2 — no comparable bifurcation. NN-distance ratios stay 0.31–0.95
across all four layers, both normalisations. Low-H patches are present
but diffuse, not clustered.

Read: ViT-B/16's entropy phase transition has a clean geometric
correlate (sparse non-Gaussian patches form a residual-space attractor);
DINO-v2 doesn't. Consistent with the training-objective gating from
Finding 1 — supervised classification produces sharp outlier-feature
separation, SSL produces graded.

Figures: `umap_embeddings/{vit_b16,dinov2_b}_umap_residuals.png`,
`umap_embeddings/outlier_geometry.csv`.

(The earlier `umap/whole_stack_umap.png` etc. that UMAP'd the *derived*
1-D entropy field per patch shows at-chance class separation —
expected, since collapsing 768-D → 1-D loses the geometry. The
finding above lives in the raw residual stream, not its scalar
projection.)

### 6. V-JEPA 2 (native-3D video ViT) injects temporal positional structure that overwhelms motion content in mid-network residuals

`facebook/vjepa2-vitl-fpc64-256` — 24-block self-supervised video ViT
(326M params, 64 frames × 256², tubelet=2 → 32×16×16 token grid). Three
clips, all 64 frames at 256²:

- **synth**: translating Gaussian blob (same trajectory as Step B).
- **real**: Big Buck Bunny, same source clip as Step B.
- **const**: 64 identical copies of `real[32]` — a "constant-content"
  control. If V-JEPA 2 only encodes content, the resulting (gt, gy,
  gx) entropy volume should have ~zero variance along `t`.

**Phase signature differs qualitatively from supervised / SSL still-
image ViTs.** Per-layer H_mean rises monotonically (does not drop);
H_std and mean |∇₃H| form a *broad arch* peaking at L=11-17, not a
sharp single-layer spike. Per-clip transition layers (argmax ΔH-std)
all sit at L=24 — the final-layer-norm bump — with peak/median ratios
3.5–10.5×, lower than supervised ViTs' 4–21× spike sharpness. So
V-JEPA 2 has no L4→L5-style phase transition in the same sense.

**The const clip exposes V-JEPA 2's temporal positional encoding.**
Define `Δt-std(L) = std over tubelet t of spatial-mean H[L, t, :, :]`.
Predictions:
- if V-JEPA 2 is content-faithful, const should have Δt-std ≈ 0 for
  all layers;
- synth (slow translation) should have moderate Δt-std;
- real (rich scene dynamics) should have largest Δt-std.

Actual:
- `const` has the *highest* Δt-std at every mid-network layer, peaking
  at 0.10 around L=15; only matches the others' near-zero values at
  L=1-2 and L=24.
- `synth` has the *lowest* Δt-std for L=1-21.
- `real` sits between, near 0.03 across all layers.

V-JEPA 2 invents temporal structure on a static-content input and
that injected structure is *larger* than the content-driven motion
variance at every mid-network layer. The per-tubelet transition layer
on `const` is bimodal between L≈6 and L=24 with std=9.57 across t
(synth std=0, real std=4.7).

**Volumetric outlier-tube tracking on the synth blob:** unlike
ViT-B/16 (Step B — bundle IoU = 0, 60-80 % of top-5 % |∇H| voxels on
frame edges), V-JEPA 2 tracks the blob trajectory **at shallow layers
only**. Largest connected component vs ground-truth tubelet mask:

| layer | IoU largest CC | edge fraction (chance = 0.25) | n components |
|---|---|---|---|
| 1 | 0.156 | 0.108 | 27 |
| **2** | **0.236** | **0.115** | 22 |
| 3 | 0.197 | 0.094 | 19 |
| 4 | 0.066 | 0.330 | 98 |
| 7 | 0.000 | 0.679 | 174 |
| 15 | 0.000 | 0.918 | 159 |
| 24 | 0.000 | 0.937 | 105 |

L=2 IoU = 0.24 vs ViT-B/16's 0.00 and DINO-v2's 0.30 (at L=6). The
*content* signal in V-JEPA 2 lives in the early embedding layers,
before the temporal-PE pattern contaminates the gradient field. By
L=4 onwards, ≥0.5–0.9 of high-|∇H| voxels sit on the (t, y, x) volume
boundary — V-JEPA 2 develops its own register-token-like artefacts,
just at deeper layers than ViT-B/16.

**Read.** Native-3D pretraining doesn't escape the artefact; it shifts
*where* in depth the artefact dominates. The volumetric framing is
viable in a narrow window (L=1-3 for V-JEPA 2 ViT-L) — at the cost
of giving up the deep-layer outlier-feature mechanism that drives the
still-image phase transition.

Figures: `video_vjepa2/{phase_curves.png,
const_input_temporal_variance.png, layer_tubelet_grid.png,
transition_per_tubelet.png, H_projections.png, gradH_projections.png,
tube_metrics_summary.png}`,
`video_vjepa2/{tube_metrics_synth.csv,
const_input_temporal_variance.csv, summary.json}`,
`video_vjepa2/synth_streamtubes_L24.html`.

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
