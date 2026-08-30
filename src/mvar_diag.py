"""MVAR model-order selection and fit-quality diagnostics (Stage 3).

Stage 3's default estimation path is the **windowed-ACF-averaged** fit: a
design matrix is cut into short windows (`src.design.window_stack`,
`detrend_windows`) stacked on a trials axis, and `src.mtmvar.ar_coeff`'s
existing `count_corr`-based averaging (over that trials axis) turns them into
one MVAR estimate -- the Kaminski sDTF core. `fit_mvar_avg_acf` is a thin,
documented wrapper over that path; `residual_whiteness` and `ar_root_stability`
are the two diagnostics that ask whether the fitted VAR is trustworthy. No
ffDTF here (Stage 4); this module only asks "is the fit well-behaved", not
"what does it say about coupling".

`residual_whiteness`/`ar_root_stability` work identically on a 3-D windowed
stack or on a plain `(k, n_samples)` design reshaped to one window
(`design[:, :, np.newaxis]`) -- the same functions serve both the windowed fit
and a global single-window comparison fit, which is what the Stage 3 gate
uses to show the windowed method's improvement.

All functions preserve whatever channel order the caller passes in -- see
`scripts/stage03_mvar_order.py` for how the fixed `src.design.DESIGN_VARIABLES`
order is threaded through.
"""

import numpy as np
from statsmodels.tsa.stattools import acf

try:
    from .mtmvar import ar_coeff, mvar_criterion
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.mtmvar import ar_coeff, mvar_criterion


def fit_mvar_avg_acf(design_3d, p):
    """Fit one MVAR from the autocovariance averaged across windows.

    Thin wrapper over `src.mtmvar.ar_coeff`: the averaging across the trials
    axis (here, windows) happens inside `src.mtmvar.count_corr`, which
    `ar_coeff` already calls when given 3-D input -- this function only
    asserts the expected shape and documents the averaged-ACF semantics that
    make it correct to call this way (the Kaminski sDTF core).

    Parameters
    ----------
    design_3d : np.ndarray, shape (k, win_len, n_windows)
        Windowed, per-window-detrended design matrix
        (``detrend_windows(window_stack(design, win_len, step))``).
    p : int
        Model order.

    Returns
    -------
    ar_coeffs : np.ndarray, shape (k, k, p)
        ``ar_coeffs[:, :, m]`` is the lag-``(m + 1)`` coefficient matrix
        (row = target, column = source).
    variance : np.ndarray, shape (k, k)
        Residual covariance matrix, averaged across windows.
    """
    if design_3d.ndim != 3:
        raise ValueError(f"design_3d must be 3-D (k, win_len, n_windows), got shape {design_3d.shape}")
    return ar_coeff(design_3d, p)


def _windowed_residuals(design_3d, ar_coeffs):
    """One-step-ahead residuals computed independently within each window (no cross-window prediction)."""
    k, win_len, n_windows = design_3d.shape
    p = ar_coeffs.shape[2]

    residuals = np.zeros((k, win_len - p, n_windows))
    for window in range(n_windows):
        for t in range(p, win_len):
            predicted = np.zeros(k)
            for lag in range(1, p + 1):
                predicted += ar_coeffs[:, :, lag - 1] @ design_3d[:, t - lag, window]
            residuals[:, t - p, window] = design_3d[:, t, window] - predicted
    return residuals


def residual_whiteness(design_3d, ar_coeffs, max_lag):
    """Per-variable residual autocorrelation, averaged across windows.

    Reconstructs one-step-ahead residuals independently within each window
    (never predicting across a window boundary), computes each window's ACF
    up to `max_lag`, and averages the ACF curves across windows -- the same
    averaging philosophy as the fit itself, rather than concatenating windows
    into one pseudo-series (which would fabricate boundary autocorrelation).

    Parameters
    ----------
    design_3d : np.ndarray, shape (k, win_len, n_windows)
        The same (detrended, windowed) design matrix `ar_coeffs` was fit on
        (via `fit_mvar_avg_acf`) -- or a global design reshaped to one window
        (``design[:, :, np.newaxis]``) for a single-window comparison fit.
    ar_coeffs : np.ndarray, shape (k, k, p)
        AR coefficient tensor from `fit_mvar_avg_acf` (or `src.mtmvar.ar_coeff`).
    max_lag : int
        Number of lags to compute/average (must be well below
        ``win_len - p`` for the per-window ACF to be meaningful).

    Returns
    -------
    residual_acf : np.ndarray, shape (k, max_lag + 1)
        Per-variable ACF (lag 0 to `max_lag`), averaged across windows.
    whiteness_summary : dict
        ``{channel_index: fraction_of_lags_within_band}`` -- for each
        variable, the fraction of lags 1..max_lag whose averaged ACF falls
        within the ``+/- 1.96 / sqrt(n_eff)`` Bartlett white-noise band
        (``n_eff = win_len - p``). 1.0 = fully white by this summary.
    """
    residuals = _windowed_residuals(design_3d, ar_coeffs)
    n_channels, n_eff, n_windows = residuals.shape

    acf_per_window = np.stack([
        np.stack([acf(residuals[channel, :, window], nlags=max_lag, fft=False) for window in range(n_windows)])
        for channel in range(n_channels)
    ])  # (n_channels, n_windows, max_lag + 1)
    residual_acf = acf_per_window.mean(axis=1)

    bartlett_band = 1.96 / np.sqrt(n_eff)
    within_band = np.abs(residual_acf[:, 1:]) <= bartlett_band
    whiteness_summary = {channel: float(within_band[channel].mean()) for channel in range(n_channels)}

    return residual_acf, whiteness_summary


def ar_root_stability(ar_coeffs):
    """Companion-matrix eigenvalues and stability of a fitted AR coefficient tensor.

    Parameters
    ----------
    ar_coeffs : np.ndarray, shape (k, k, p)
        AR coefficient tensor from `fit_mvar_avg_acf` (or `src.mtmvar.ar_coeff`).

    Returns
    -------
    roots : np.ndarray, shape (k * p,), complex
        Eigenvalues of the ``(k*p, k*p)`` companion matrix (top block-row
        ``[A_1 ... A_p]``, sub-diagonal identity blocks) -- for plotting on
        the complex unit circle.
    max_abs_root : float
        The largest eigenvalue modulus (the headline stability number).
    is_stable : bool
        Whether every eigenvalue modulus is strictly below 1 (inside the unit
        circle).
    """
    k, _, p = ar_coeffs.shape
    top_row = np.concatenate([ar_coeffs[:, :, lag] for lag in range(p)], axis=1)

    if p > 1:
        sub_identity = np.eye(k * (p - 1))
        sub_zeros = np.zeros((k * (p - 1), k))
        bottom_rows = np.concatenate([sub_identity, sub_zeros], axis=1)
        companion = np.vstack([top_row, bottom_rows])
    else:
        companion = top_row

    roots = np.linalg.eigvals(companion)
    max_abs_root = float(np.abs(roots).max())
    return roots, max_abs_root, bool(max_abs_root < 1.0)


def select_order(system, max_model_order, crit_types):
    """Select the MVAR model order under each of several information criteria.

    Thin wrapper over `src.mtmvar.mvar_criterion`, run once per criterion.
    `mvar_criterion` only accepts a 2-D `(m, n_samples)` array, so this always
    runs on the global (non-windowed) design -- see the Stage 3 script's
    documented caveat on why order selection stays 2-D.

    Parameters
    ----------
    system : np.ndarray, shape (m, n_samples)
        Any 2-D design matrix -- the full multivariate system, or a sub-block
        (e.g. just the EEG or just the HRV rows).
    max_model_order : int
        Maximum model order to evaluate.
    crit_types : list of str
        Criteria to evaluate, each one of ``'AIC'``, ``'HQ'``, ``'SC'``.

    Returns
    -------
    optimal_orders : dict
        ``{crit_type: optimal_model_order}``.
    curves : dict
        ``{crit_type: criterion values array, length max_model_order}``.
    model_order_range : np.ndarray
        The evaluated order range (``1..max_model_order``), shared by every
        criterion's curve.
    """
    optimal_orders = {}
    curves = {}
    model_order_range = None
    for crit_type in crit_types:
        crit, model_order_range, optimal_model_order = mvar_criterion(
            system, max_model_order, crit_type=crit_type, plot=False,
        )
        optimal_orders[crit_type] = int(optimal_model_order)
        curves[crit_type] = crit
    return optimal_orders, curves, model_order_range
