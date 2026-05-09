# eris

Per-patch differential entropy on standardised residual streams of ViT-style
models, treated as a 2-D (or 3-D in video) scalar field on the patch grid.

**Headline finding so far.** Across `vit-base-patch16-224`,
`vit-large-patch16-224`, `dinov2-base`, and `clip-vit-base-patch16` (all
loaded ungated from HuggingFace, no fine-tuning), per-image transition
layer = `argmax_L (ΔH-std)` is rock-stable (IQR = 0 across 200 tiny-imagenet
val images per arch), the transition is sharp (peak/median 4×–21×), and its
location is gated by **training objective** rather than universal: supervised
classification ≪ self-supervised / contrastive in `L_trans / L_max`. Per-
patch entropy beats per-patch L2 norm, attention-rollout entropy, and
random-init residual entropy at p < 0.001 on outlier-localisation accuracy.
The phase transition is robust to entropy-estimator choice (Vasicek / k-NN
/ KDE all agree within ±1 layer on ViT-B/16) and has a clean geometric
correlate: under per-patch standardisation, low-H "outlier" patches sit
8-20× tighter to each other in residual space than the bulk, forming a
discrete attractor. On video the transition layer is frame-stationary (std
= 0 across all frames in both ViT-B/16 and DINO-v2), but volumetric
outlier-tube tracking on a synthetic translating blob fails for ViT-B/16
due to register-token edge dominance (Darcet et al. 2024). Native-3D
V-JEPA 2 escapes the spatial register-token artefact at shallow layers
(tube IoU = 0.24 at L=2 vs ViT-B/16's 0.00) but injects a temporal-PE
pattern that overwhelms content motion at mid-network — even on a
constant-content video, mid-network residuals exhibit larger temporal
variance than for a genuinely moving clip.

See `results/summary.md` for the full read-out and `results/` for figures
and tensors.

## Reproduce

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/synth_check.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/cross_arch.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/video_frames.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/video_volume.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/video_volume_bundle.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/baselines.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/estimator_robustness.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/umap_embeddings.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/video_vjepa2.ipynb
```

Total compute < 30 min on a single 4090; mostly forwards (no fine-tuning).

## Library

`src/eris/` exports:

```python
from eris.extract import (extract_entropy_stack, load_model,
                          extract_entropy_volume, load_video_model)
from eris.estimators import vasicek_entropy, knn_entropy, kde_entropy
from eris.fields import (gradient_2d, laplacian_2d, depth_derivatives,
                         gradient_3d, laplacian_3d,
                         transition_layer, per_layer_stats)
from eris.video import synthesise_translating_blob, load_real_video
from eris.volumetric import extract_outlier_tubes, render_streamtubes_html
from eris.baselines import (l2_norm_field, attention_rollout_field,
                            random_init_entropy_field)
```

`load_model` registry covers `vit_b16`, `vit_l16`, `dinov2_b`, `clip_b16`
out of the box for still images; arch-specific CLS / patch-grid / forward
conventions are hidden behind the unified `extract_entropy_stack` API.
For video, `load_video_model` covers `vjepa2_l` and `vjepa2_g`; use
`extract_entropy_volume(arch, frames)` to get a `(n_layers, gt, gy, gx)`
tensor per clip.

## What this is + isn't

- **It IS:** a clean depth-wise structural finding in trained-ViT residual
  streams; the visualisation pipeline that surfaced it; a methodological
  comparison against simpler per-patch baselines.
- **It IS NOT:** a new mechanism, a free-energy derivation, or a causal
  probe. The phenomenon (sparse non-Gaussian "outlier features") is in
  the literature (Kovaleva 2021, Dettmers 2022, Sun 2024). What this work
  adds is the **spatial dimension** — outliers live on a 2-D patch grid
  and emerge in a single-layer phase transition whose timing is gated by
  training objective.
