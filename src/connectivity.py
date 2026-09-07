"""Granger_estimator connectivity estimation (Stage 4).

The Stage 4 estimator reuses Stage 3's windowed-ACF-averaged MVAR core
(`src.design.window_stack`/`detrend_windows` -> `src.mtmvar.full_freq_dtf`/
`multivariate_spectra` with an explicit `optimal_model_order`) rather than a
single global 2-D fit. Stage 3 established empirically that this averaged
estimator is more stable and whiter-residualed than a global fit, primarily
for the non-stationary HRV variables -- see
`DTF_analysis_notes/pipeline_plan.md` Stage 4 and `scripts/stage03_mvar_order.py`.

This is the default estimation path, not a deferred swap-in: `Granger_estimator`
is the stable interface Stage 4 (and later stages) call, so a future
Bayesian-MVAR core could replace the internals without changing callers.
"""

try:
    from .design import detrend_windows, window_stack
    from .mtmvar import dtf_estimator, multivariate_spectra
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.design import detrend_windows, window_stack
    from src.mtmvar import dtf_estimator, multivariate_spectra


def Granger_estimator(design, freqs, fs, p, win_len, step, detrend_type="linear", ESTIMATOR="dDTF", box_cox_lambda=-1):
    """Windowed-ACF-averaged Granger_estimator and multivariate spectra at a fixed order.

    Cuts the per-film design matrix into overlapping, per-window-detrended
    windows (the same geometry Stage 3 selected and recorded) and fits ONE
    MVAR from the block-autocovariance averaged across those windows -- the
    Kaminski sDTF core, which Stage 3 established as more stable and
    whiter-residualed than a single global fit, primarily on the
    non-stationary HRV variables. Granger_estimator and spectra are then read off that one
    averaged fit.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
        z-scored design matrix in `src.design.DESIGN_VARIABLES` order
        (from `src.design.assemble_design_matrix`).
    freqs : np.ndarray
        Frequency axis (Hz) for the Granger_estimator/spectra cubes.
    fs : float
        Sampling frequency (Hz) of `design`.
    p : int
        MVAR model order (Stage 3's `p_used`; never None on the 3-D stack).
    win_len, step : int
        Window length / step in samples (Stage 3's `win_len` / `step`).
    detrend_type : {'linear', 'constant'}, optional
        Per-window detrend type (Stage 3's `detrend_type`).
    ESTIMATOR : {'dDTF', 'ffDTF', 'GPDC'}, optional
        Estimator type for the directed transfer function calculation (default is "dDTF").
    box_cox_lambda : float, optional
        Box-Cox exponent `(x**box_cox_lambda - 1) / box_cox_lambda` applied to the
        Granger_estimator cube (see `src.mtmvar.box_cox_transform`). Default -1 = no
        transform, preserving prior behaviour; spectra are never transformed.

    Returns
    -------
    granger_estimator : np.ndarray, shape (k, k, n_freqs)
        Granger_estimator cube. `granger_estimator[target, source, f]` = flow source -> target.
    spectra : np.ndarray, shape (k, k, n_freqs), complex
        Multivariate spectra on the same averaged fit.
    """
    assert win_len > p, f"win_len={win_len} must exceed model order p={p}"
    stack = detrend_windows(window_stack(design, win_len, step), dtype=detrend_type)
    granger_estimator = dtf_estimator(stack, freqs, fs, optimal_model_order=p, ESTIMATOR=ESTIMATOR, box_cox_lambda=box_cox_lambda)
    spectra = multivariate_spectra(stack, freqs, fs, optimal_model_order=p)
    return granger_estimator, spectra
