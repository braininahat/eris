"""Differential-entropy estimators with a shared API.

All estimators take a 2-D array of shape ``(n_patches, n_samples_per_patch)``
and return a 1-D array of shape ``(n_patches,)`` — one entropy estimate
per row, in nats. Standardisation (zero-mean unit-var per row) is the
caller's responsibility; not done in here.

Why three? Per-patch ``n_samples = hidden_dim`` (e.g. 768) for ViT
sits below the conventional Vasicek stability floor (~1k); a single
estimator could conceivably impose its own depth-wise pattern as an
artefact. Step D of the plan checks the L4→L5 transition and outlier-
patch story under all three.
"""

from __future__ import annotations

import numpy as np
import scipy.stats
import scipy.special
from sklearn.neighbors import NearestNeighbors


def vasicek_entropy(samples: np.ndarray) -> np.ndarray:
    """Vasicek estimator (scipy default ``differential_entropy``).

    Args:
        samples: ``(n_patches, n_samples)``.
    Returns:
        ``(n_patches,)`` differential entropy in nats.
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be (n_patches, n_samples); got {samples.shape}")
    out = np.empty(samples.shape[0], dtype=np.float32)
    for i in range(samples.shape[0]):
        out[i] = scipy.stats.differential_entropy(samples[i])
    return out


def knn_entropy(samples: np.ndarray, k: int = 3) -> np.ndarray:
    """Kozachenko–Leonenko k-NN entropy estimator (1-D variant).

    For 1-D samples, the K-L estimator with k-th nearest neighbour is

        H ≈ ψ(n) − ψ(k) + log(2 ⟨r_k⟩)

    where ``r_k`` is the distance from each sample to its k-th nearest
    neighbour and ``⟨·⟩`` is the geometric mean (i.e. average of
    ``log r_k``). The leading constant in the log is ``2`` for the 1-D
    "ball" (interval) volume.
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be (n_patches, n_samples); got {samples.shape}")
    n = samples.shape[1]
    out = np.empty(samples.shape[0], dtype=np.float32)
    psi_n = scipy.special.digamma(n)
    psi_k = scipy.special.digamma(k)
    for i, row in enumerate(samples):
        x = np.sort(row.reshape(-1, 1), axis=0)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
        dists, _ = nn.kneighbors(x)
        # column k is the k-th neighbour (column 0 is self at distance 0)
        rk = dists[:, k]
        # avoid log(0) for pathological ties
        rk = np.maximum(rk, 1e-12)
        out[i] = float(psi_n - psi_k + np.mean(np.log(2.0 * rk)))
    return out


def kde_entropy(samples: np.ndarray, bandwidth: str | float = "silverman") -> np.ndarray:
    """KDE entropy estimator using ``scipy.stats.gaussian_kde``.

    Computes ``H = -⟨log p̂(x)⟩`` over the samples themselves, where
    ``p̂`` is a Gaussian KDE fit to those samples. Bandwidth defaults to
    Silverman's rule.
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be (n_patches, n_samples); got {samples.shape}")
    out = np.empty(samples.shape[0], dtype=np.float32)
    for i, row in enumerate(samples):
        kde = scipy.stats.gaussian_kde(row, bw_method=bandwidth)
        log_p = np.log(np.maximum(kde(row), 1e-300))
        out[i] = float(-np.mean(log_p))
    return out


ESTIMATORS = {
    "vasicek": vasicek_entropy,
    "knn": knn_entropy,
    "kde": kde_entropy,
}
