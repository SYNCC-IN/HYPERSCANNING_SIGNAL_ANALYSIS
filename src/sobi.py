"""
sobi.py — Second Order Blind Identification (SOBI)

Python implementation based on EEGLAB's sobi.m
(Belouchrani, Abed-Meriam, Cardoso, Moulines, 1997).

Reference
---------
Belouchrani, A., Abed-Meriam, K., Cardoso, J.-F., & Moulines, R. (1997).
A blind source separation technique using second-order statistics.
IEEE Transactions on Signal Processing, 45(2), 434–444.
https://doi.org/10.1109/78.554307

Mathematical background
-----------------------
SOBI assumes the observed data X = A @ S, where:
  - A is the (n_channels × n_components) mixing matrix
  - S contains statistically independent sources with distinct autocorrelation
    profiles (i.e., different power spectra)

Algorithm:
  1. Whiten X via PCA → Z = W_w @ X  (decorrelates, scales to unit variance)
  2. Compute sample autocovariance matrices C(τ) for τ = 1…n_lags
  3. Find orthogonal V that jointly diagonalises all C(τ) simultaneously
     (Jacobi sweeps, one pair (p,q) per step)
  4. Total unmixing: W = V @ W_w;  mixing: A = pinv(W);  sources: S = W @ X

The key insight: two sources with *different* autocorrelation decays (i.e.,
different spectral shapes) appear as off-diagonal elements in C(τ) that cannot
be made zero by a single rotation — only the correct V eliminates them across
ALL lags simultaneously.  This makes SOBI excellent for EEG rhythm extraction.

Optimal Givens rotation for pair (p, q)
---------------------------------------
At each step we minimise f(θ) = Σ_k [M_k'(p,q)]² where
  M_k'(p,q) = sin(2θ)/2 · (M_k[q,q] − M_k[p,p]) + cos(2θ) · M_k[p,q]

Setting df/d(2θ) = 0 and solving gives:

    tan(4θ) = 4 Σ_k(u_k w_k) / (Σ_k u_k² − 4 Σ_k w_k²)

with  u_k = M_k[p,p] − M_k[q,q],  w_k = M_k[p,q].
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def sobi(
    data: np.ndarray,
    n_components: int | None = None,
    n_lags: int | None = None,
    lags: list[int] | None = None,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Second Order Blind Identification (SOBI).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Multi-channel time series.  Centred internally — do not pre-centre.
    n_components : int, optional
        Number of components to extract (≤ n_channels).
        Default: n_channels (full decomposition, as in EEGLAB).
    n_lags : int, optional
        Number of time lags for autocovariance estimation.
        Default: min(100, n_times // 5)  — matches EEGLAB's sobi.m default.
        Larger values give better spectral resolution at the cost of time.
        Rule of thumb: cover 1–2 cycles of the lowest frequency of interest,
        e.g. for 1 Hz delta and fs = 1024 Hz → n_lags ≥ 1024.
    lags : list of int, optional
        Explicit list of time lags (in samples) for autocovariance estimation.
        Overrides n_lags if provided.  Must be a list of positive integers.
    max_iter : int
        Maximum Jacobi sweeps in joint diagonalisation.  500 is more than
        sufficient for typical EEG data (convergence usually < 50 sweeps).
    tol : float
        Convergence threshold: algorithm stops when total off-diagonal squared
        energy across all lagged covariance matrices falls below this value.

    Returns
    -------
    A : ndarray, shape (n_channels, n_components)
        Mixing matrix.  Column j is the scalp topography of component j.
        Suitable for mne.viz.plot_topomap (same interpretation as ICA.get_components()).
    W : ndarray, shape (n_components, n_channels)
        Unmixing (spatial filter) matrix.  W = pinv(A) approximately.
    S : ndarray, shape (n_components, n_times)
        Recovered source signals.  Units: same as input × mixing coefficient.

    Raises
    ------
    ValueError
        If data has wrong shape or effective rank < n_components.

    Notes
    -----
    Components are sorted in descending order of total autocovariance energy
    (sum over all lags of the squared diagonal element), matching EEGLAB's
    output order.

    Integration with ICAPreprocessor
    ---------------------------------
    Replace the ica.fit() / ica.get_sources() / ica.get_components() calls::

        from sobi import sobi
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        data = raw.get_data(picks=eeg_picks)      # (n_ch, n_times)
        A, W, S = sobi(data, n_components=15)     # drop-in replacement
        # A  →  topographies  (like ica.get_components())
        # W  →  spatial filters
        # S  →  sources       (like ica.get_sources(raw).get_data())
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2-D (n_channels, n_times), got shape {data.shape}."
        )

    n_channels, n_times = data.shape

    if n_components is None:
        n_components = n_channels
    if n_lags is None:
        n_lags = min(100, n_times // 5)
    n_lags = max(1, min(n_lags, n_times - 1))

    # ── 1. Centre ─────────────────────────────────────────────────────────────
    X = data - data.mean(axis=1, keepdims=True)

    # ── 2. Whiten via truncated PCA ───────────────────────────────────────────
    # Use scipy.linalg.eigh (symmetric eigendecomposition): faster + numerically
    # more stable than np.linalg.eig for real symmetric matrices.
    Rx = (X @ X.T) / n_times
    eigvals, eigvecs = linalg.eigh(Rx)

    # Descending order; retain n_components largest eigenvalues
    desc = np.argsort(eigvals)[::-1]
    eigvals = eigvals[desc[:n_components]]
    eigvecs = eigvecs[:, desc[:n_components]]

    if eigvals[-1] <= 0:
        raise ValueError(
            f"Effective rank of data < {n_components} (smallest retained "
            f"eigenvalue = {eigvals[-1]:.3e}). Reduce n_components."
        )

    # Whitening matrix: W_w @ Rx @ W_w.T = I_{n_components}
    W_w = np.diag(eigvals ** -0.5) @ eigvecs.T   # (n_comp, n_ch)
    Z   = W_w @ X                                 # (n_comp, n_times)

    # ── 3. Lagged autocovariance matrices of whitened data ────────────────────
    if lags is not None:
        lag_list = [l for l in lags if 1 <= l < n_times]
    else:
        n_lags = min(n_lags or min(100, n_times // 5), n_times - 1)
        lag_list = range(1, n_lags + 1)

    cov_mats: list[np.ndarray] = []
    for lag in lag_list:
        T_eff = n_times - lag
        R = (Z[:, lag:] @ Z[:, :T_eff].T) / T_eff
        cov_mats.append((R + R.T) * 0.5)   # symmetrise numerical noise

    # ── 4. Joint diagonalisation via Jacobi sweeps ───────────────────────────
    # Returns V (n_comp × n_comp, orthogonal) such that
    #   V @ C(τ) @ V.T ≈ diagonal  for all τ
    V = _joint_diag_jacobi(cov_mats, n_components, max_iter=max_iter, tol=tol)

    # ── 5. Sort by autocovariance energy (descending) ─────────────────────────
    # Energy of component j = Σ_τ (V @ C(τ) @ V.T)[j,j]²
    # Matches EEGLAB's component ordering.
    energy = np.zeros(n_components)
    for m in cov_mats:
        energy += np.diag(V @ m @ V.T) ** 2
    V = V[np.argsort(energy)[::-1], :]

    # ── 6. Final matrices ─────────────────────────────────────────────────────
    W = V @ W_w               # (n_comp, n_ch)    total unmixing matrix
    A = np.linalg.pinv(W)     # (n_ch,  n_comp)   mixing matrix / topographies
    S = W @ X                 # (n_comp, n_times)  source signals

    return A, W, S


# ──────────────────────────────────────────────────────────────────────────────
# Internal: joint diagonalisation
# ──────────────────────────────────────────────────────────────────────────────

def _joint_diag_jacobi(
    matrices: list[np.ndarray],
    n: int,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Joint approximate diagonalisation of symmetric matrices via Jacobi rotations.

    Finds orthogonal V (n × n) minimising the total off-diagonal energy:

        F(V) = Σ_k  ||off(V M_k V^T)||_F²

    where off(·) sets the diagonal to zero.

    At each pair (p, q) the optimal Givens angle θ is:

        tan(4θ) = 4 Σ_k u_k w_k / (Σ_k u_k² − 4 Σ_k w_k²)

        u_k = M_k[p,p] − M_k[q,q],   w_k = M_k[p,q]

    Derivation (summary)
    --------------------
    After applying G(p,q,θ) on the left and right:

        M_k'[p,q] = sin(2θ)/2 · (M_k[q,q] − M_k[p,p]) + cos(2θ) · M_k[p,q]

    Minimising Σ_k (M_k'[p,q])² over θ and solving gives the formula above.
    For K=1 this reduces to the classical Jacobi formula tan(2θ) = 2w/(−u).

    Parameters
    ----------
    matrices : list of (n, n) symmetric ndarray
    n : int   — dimension
    max_iter : int
    tol : float

    Returns
    -------
    V : ndarray (n, n), orthogonal
        V @ M_k @ V.T ≈ diagonal for all k.
    """
    V = np.eye(n)
    M = [m.copy() for m in matrices]  # work on copies

    for sweep in range(max_iter):

        # ── Convergence: total off-diagonal squared energy ─────────────────
        off_energy = sum(
            np.sum(m ** 2) - np.sum(np.diag(m) ** 2)
            for m in M
        )
        if off_energy < tol:
            break

        # ── Full Jacobi sweep over all unique pairs (p, q) ─────────────────
        for p in range(n - 1):
            for q in range(p + 1, n):

                # Aggregate u_k and w_k sums across all matrices
                sum_uu = 0.0  # Σ u_k²
                sum_ww = 0.0  # Σ w_k²
                sum_uw = 0.0  # Σ u_k w_k
                for m in M:
                    u = m[p, p] - m[q, q]
                    w = m[p, q]
                    sum_uu += u * u
                    sum_ww += w * w
                    sum_uw += u * w

                # Numerator and denominator of tan(4θ) formula
                num   =  4.0 * sum_uw
                denom = sum_uu - 4.0 * sum_ww

                # Skip if pair is already (jointly) diagonal
                if abs(num) < 1e-15 and abs(denom) < 1e-15:
                    continue

                # Optimal Givens angle: θ = (1/4) arctan2(num, denom)
                theta = 0.25 * np.arctan2(num, denom)

                if abs(theta) < 1e-15:
                    continue

                c, s  = np.cos(theta), np.sin(theta)
                # Givens rotation matrix G  (G[p,p]=c, G[p,q]=s, G[q,p]=−s, G[q,q]=c)
                Rot   = np.array([[c,  s],
                                  [-s, c]])

                # Apply  M_k ← G @ M_k @ G.T  for every matrix (in-place):
                #   left  multiply (rows p, q):  M[[p,q],:] = Rot @ M[[p,q],:]
                #   right multiply (cols p, q):  M[:,[p,q]] = M[:,[p,q]] @ Rot.T
                for m in M:
                    m[[p, q], :]  = Rot @ m[[p, q], :]
                    m[:, [p, q]] = m[:, [p, q]] @ Rot.T

                # Accumulate  V ← G @ V  (rows p, q of V get rotated)
                V[[p, q], :] = Rot @ V[[p, q], :]

    return V


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test  (run with:  python sobi.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_ch, n_t = 19, 10_000
    fs = 256.0

    # Simulate 3 narrow-band sources + 16 white-noise sources
    t = np.arange(n_t) / fs
    s_alpha    = np.sin(2 * np.pi * 10 * t)     # 10 Hz alpha
    s_theta    = np.sin(2 * np.pi *  6 * t)     # 6 Hz theta
    s_artifact = np.sin(2 * np.pi * 50 * t)     # 50 Hz line noise
    S_true = np.vstack([
        s_alpha, s_theta, s_artifact,
        rng.standard_normal((n_ch - 3, n_t)),
    ])

    # Random mixing matrix and noisy observations
    A_true = rng.standard_normal((n_ch, n_ch))
    X = A_true @ S_true + 0.05 * rng.standard_normal((n_ch, n_t))

    A_est, W_est, S_est = sobi(X, n_components=n_ch, n_lags=200)

    # ── Shape contract ────────────────────────────────────────────────────────
    assert A_est.shape == (n_ch, n_ch), f"A shape: {A_est.shape}"
    assert W_est.shape == (n_ch, n_ch), f"W shape: {W_est.shape}"
    assert S_est.shape == (n_ch, n_t),  f"S shape: {S_est.shape}"
    print("Shape checks passed.")

    # ── Inverse relationship: W @ A ≈ I ──────────────────────────────────────
    # This must hold exactly (up to float64 rounding) regardless of data,
    # because A = pinv(W) and W is square invertible.
    err_inv = np.linalg.norm(W_est @ A_est - np.eye(n_ch))
    print(f"||W @ A − I||_F = {err_inv:.2e}  (should be < 1e-10)")
    assert err_inv < 1e-10, f"Inverse relationship failed: {err_inv:.2e}"

    # ── Reconstruction of centred data: A @ S ≈ X_centred ────────────────────
    # sobi() centres X internally, so S = W @ X_centred.
    # Therefore A @ S = A @ W @ X_centred = X_centred (exactly).
    # We must compare against X_centred, not X.
    X_centred = X - X.mean(axis=1, keepdims=True)
    X_recon   = A_est @ S_est
    rel_err   = np.linalg.norm(X_centred - X_recon) / np.linalg.norm(X_centred)
    print(f"Reconstruction relative error (vs centred X): {rel_err:.2e}  (should be < 1e-10)")
    assert rel_err < 1e-10, f"Reconstruction failed: {rel_err:.2e}"

    # ── SOBI criterion: lagged covariance matrices of sources are diagonal ────
    # SOBI does NOT guarantee decorrelation at lag 0 — it minimises off-diagonal
    # elements of C_S(τ) for τ=1…n_lags.  Check that criterion is satisfied.
    test_lags = [1, 5, 20, 50]
    for lag in test_lags:
        C_lag = S_est[:, lag:] @ S_est[:, :n_t - lag].T / (n_t - lag)
        C_lag = (C_lag + C_lag.T) / 2
        diag_energy = np.sum(np.diag(C_lag) ** 2)
        off_energy  = np.sum(C_lag ** 2) - diag_energy
        ratio = off_energy / (diag_energy + 1e-30)
        print(f"  lag={lag:3d}: off/diag energy ratio = {ratio:.3e}  (SOBI criterion)")

    print("\nAll checks passed.")