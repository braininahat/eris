"""eris — per-patch entropy field across ViT depth.

Public API mirrors the smoke-test notebooks but generalised over any
HuggingFace ViT-style model:

    from eris.extract import extract_entropy_stack, load_model
    from eris.estimators import vasicek_entropy, knn_entropy, kde_entropy
    from eris.fields import (
        gradient_2d, laplacian_2d, depth_derivatives,
        gradient_3d, laplacian_3d,
        transition_layer, per_layer_stats,
    )
    from eris.video import (
        synthesise_translating_blob, synthetic_tube_mask,
        load_real_video, write_video, frames_to_pil,
    )
    from eris.volumetric import (
        extract_outlier_tubes, iou_3d,
        upsample_volume_xy, render_streamtubes_html,
    )

See ``notebooks/cross_arch.py`` (Step A), ``notebooks/video_frames.py``
(Step B1), and ``notebooks/video_volume.py`` (Step B2) for reference
drivers.
"""
