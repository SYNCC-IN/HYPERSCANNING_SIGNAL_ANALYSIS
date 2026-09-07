"""Film-matched surrogate-dyad null construction (Stage 5).

A surrogate stitches one dyad's child variables with a *different* dyad's
caregiver variables (same film) into a design matrix in the canonical
`src.design.DESIGN_VARIABLES` order. Both partners watched the same film but
were never in the room together, so any coupling recovered from a surrogate
reflects the shared stimulus + generic physiology, not real-time interaction
-- the null that real dyads' ffDTF is compared against
(`delta_and_z`) to isolate genuine interpersonal coupling. See
`DTF_analysis_notes/pipeline_plan.md` Stage 5 and `scripts/stage05_surrogate.py`
for the locked/open design decisions this module implements.

`Delta`/`z` stay signed everywhere -- never `abs()` or clipped -- because the
sign of a surviving effect is scientifically meaningful (e.g. negative
interpersonal HRV synchrony can be adaptive).
"""

import numpy as np
import xarray as xr
from scipy.stats import median_abs_deviation

try:
    from .design import DESIGN_VARIABLES, assemble_design_matrix, detrend_windows, window_stack
    from .mvar_diag import ar_root_stability, fit_mvar_avg_acf
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.design import DESIGN_VARIABLES, assemble_design_matrix, detrend_windows, window_stack
    from src.mvar_diag import ar_root_stability, fit_mvar_avg_acf


def surrogate_pairs(dyad_ids, group_of=None):
    """All ordered (child_dyad, cg_dyad) pairs with child_dyad != cg_dyad.

    The film is fixed by the caller (surrogates are only ever built within one
    film), so this operates on the list of dyad ids present for that film.

    When ``group_of`` is given (a mapping ``dyad_id -> group label``), pairs are
    additionally restricted to partners from the *same* group -- the within-group
    null (Stage 5 D3 sensitivity, see `scripts/stage05_surrogate.py`). The
    default pooled null (L5) subtracts a *group-agnostic* per-film scalar, which
    is identical for TD and ASD and therefore cancels out of every between-group
    contrast (group main effect *and* film x group interaction): it centres the
    delta and protects a genuine group effect from being absorbed, but it removes
    nothing from the group comparisons and so cannot rule out a group-dependent
    stimulus response. Subtracting a same-group foreign-pair null instead removes
    each group's own shared-stimulus / generic-physiology baseline, so a
    surviving film x group interaction cannot be explained by the two groups
    responding to the same film differently. Foreign pairs never interacted in
    real time, so a same-group null carries no genuine coupling to absorb.
    ``group_of=None`` keeps the pooled behaviour unchanged.

    Parameters
    ----------
    dyad_ids : list of str
        Dyad ids present for one film.
    group_of : dict of {str: str}, optional
        Maps each dyad id to its group label. When provided, only ordered
        off-diagonal pairs whose two members share a group are returned. Default
        None (no group restriction).

    Returns
    -------
    list of tuple of str
        `(child_dyad_id, cg_dyad_id)`, ordered off-diagonal pairings. Length
        `N * (N - 1)` for `N` dyad ids when ``group_of`` is None; `sum_g
        n_g * (n_g - 1)` over groups when ``group_of`` is given.
    """
    pairs = [(child, cg) for child in dyad_ids for cg in dyad_ids if child != cg]
    if group_of is not None:
        pairs = [(child, cg) for child, cg in pairs if group_of[child] == group_of[cg]]
    return pairs


def assemble_surrogate_design(child_envelopes, cg_envelopes, zscore=True):
    """Stitch a foreign child with a foreign caregiver into one design matrix.

    Takes the child variables (`child:ROI`, `child:HRV`) from
    `child_envelopes` and the caregiver variables (`cg:ROI`, `cg:HRV`) from
    `cg_envelopes` (each a Stage 2 DataArray for the same film), truncates
    both to their common (shorter) time length -- the two segments are the
    same film but may differ by a sample or two -- and reassembles them in
    the canonical `DESIGN_VARIABLES` order with a fresh, shared time axis
    (the two source arrays' own time axes are dyad-specific and not
    meaningful once stitched). Z-scoring is delegated to
    `src.design.assemble_design_matrix`, so the surrogate design follows the
    identical per-channel z-score convention as a real one (single source of
    truth). No silent length-fixing beyond the documented truncate-to-min;
    a missing variable surfaces as the `.sel` `KeyError` it is.

    Parameters
    ----------
    child_envelopes : xarray.DataArray
        Stage 2 output for the dyad supplying the child variables, dims
        `("variable", "time")`.
    cg_envelopes : xarray.DataArray
        Stage 2 output for the dyad supplying the caregiver variables, same
        film, same sampling frequency.
    zscore : bool, optional
        Passed through to `src.design.assemble_design_matrix` (default True).

    Returns
    -------
    np.ndarray, shape (4, n_common)
        Rows in `DESIGN_VARIABLES` order, z-scored per row when `zscore` is
        True.
    """
    child_part = child_envelopes.sel(variable=["child:ROI", "child:HRV"])
    cg_part = cg_envelopes.sel(variable=["cg:ROI", "cg:HRV"])
    n_common = min(child_part.sizes["time"], cg_part.sizes["time"])

    child_values = child_part.isel(time=slice(0, n_common)).values
    cg_values = cg_part.isel(time=slice(0, n_common)).values
    fs = float(child_envelopes.attrs["fs"])

    data = np.stack([child_values[0], cg_values[0], child_values[1], cg_values[1]], axis=0)
    time = np.arange(n_common) / fs
    stitched = xr.DataArray(data, dims=("variable", "time"), coords={"variable": DESIGN_VARIABLES, "time": time})
    return assemble_design_matrix(stitched, zscore=zscore)


def windowed_ar_stability(design, win_len, step, p, detrend_type="linear"):
    """Companion-matrix stability of the windowed-ACF AR fit of one design matrix.

    Composition of existing diagnostics at a fixed order: `window_stack` ->
    `detrend_windows` -> `fit_mvar_avg_acf(p)` -> `ar_root_stability`. Used to
    gate surrogates and to flag real dyads at the Stage 5 common order.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
        z-scored design matrix (real or surrogate).
    win_len, step : int
        Window length / step in samples.
    p : int
        MVAR model order.
    detrend_type : {'linear', 'constant'}, optional
        Per-window detrend type (default 'linear').

    Returns
    -------
    max_abs_root : float
        Largest companion-matrix eigenvalue modulus.
    is_stable : bool
        Whether every eigenvalue modulus is strictly below 1.
    """
    stack = detrend_windows(window_stack(design, win_len, step), dtype=detrend_type)
    ar_coeffs, _ = fit_mvar_avg_acf(stack, p)
    _, max_abs_root, is_stable = ar_root_stability(ar_coeffs)
    return max_abs_root, is_stable


def band_average_cube(cube, freqs, band_hz):
    """Average a (k, k, n_freqs) cube over an inclusive frequency band -> (k, k).

    Identical maths to Stage 4's script-local `band_average`, lifted into
    `src/` so both stages share one definition.

    Parameters
    ----------
    cube : np.ndarray, shape (k, k, n_freqs)
        ffDTF (or similar) cube.
    freqs : np.ndarray
        Frequency axis (Hz) matching `cube`'s last axis.
    band_hz : tuple of float
        `(low, high)` band edges in Hz, inclusive.

    Returns
    -------
    np.ndarray, shape (k, k)
        Band-averaged matrix.
    """
    band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    return cube[:, :, band_mask].mean(axis=2)


def delta_and_z(real_value, null_values):
    """Signed delta and z of a real edge value against its surrogate null.

    `delta = real - median(null)`; `z = (real - median(null)) / MAD(null)`. Both the
    centre and the spread are robust statistics (median, median absolute deviation)
    rather than mean/std, so a handful of extreme surrogate draws cannot dominate
    either. `null_std` (kept under its historical name for schema compatibility with
    downstream tables) is actually the MAD scaled by `scipy.stats.median_abs_deviation`'s
    `scale="normal"` (`MAD * 1.4826`), which makes it a consistent estimator of the
    standard deviation under a normal null and keeps `z` on the same scale it had
    when computed from `std(ddof=1)`. Signed on purpose -- never `abs()` or clipped,
    since the sign of what survives null subtraction is scientifically meaningful. A
    degenerate near-zero null MAD is left to surface as a large/inf `z` and is visible
    via `null_std`/`n_null` in the caller's table -- not silently patched.

    Parameters
    ----------
    real_value : float
        Real dyad's band-averaged ffDTF for one edge.
    null_values : array-like
        Surrogate null draws for the same edge.

    Returns
    -------
    dict
        Keys: `delta`, `z`, `null_median`, `null_std`, `n_null`.
    """
    null_values = np.asarray(null_values, dtype=float)
    null_median = float(np.median(null_values))
    null_std = float(median_abs_deviation(null_values, scale="normal"))
    delta = float(real_value - null_median)
    z = delta / null_std
    return {"delta": delta, "z": z, "null_median": null_median, "null_std": null_std, "n_null": int(null_values.size)}
