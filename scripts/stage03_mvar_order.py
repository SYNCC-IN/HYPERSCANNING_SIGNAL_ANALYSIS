"""Stage 3 - design matrix, model order, and windowed-ACF-averaged MVAR fit.

Reads Stage 2's `02_envelopes/<dyad_id>_<film>.nc` design files and, for every
case:

1. Builds the z-scored `(4, n_samples)` design matrix
   (`src.design.assemble_design_matrix`) and selects the model order `p_used`
   from the **global 2-D signal** (`src.mvar_diag.select_order` ->
   `src.mtmvar.mvar_criterion`, 2-D only -- see the caveat below).
2. Cuts that design matrix into short, overlapping windows
   (`src.design.window_stack`), detrends each window independently
   (`detrend_windows`), and fits **one** MVAR from the autocovariance averaged
   across those windows (`src.mvar_diag.fit_mvar_avg_acf` -> the existing
   `count_corr`/`ar_coeff` averaging path -- the Kaminski sDTF core). This is
   the **default** estimation path, not a deferred swap-in.
3. Runs the same diagnostics (`residual_whiteness`, `ar_root_stability`) on
   both the windowed fit and a global single-window comparison fit, so the
   gate can show the windowed method's improvement directly -- most visibly
   for the two HRV variables, whose within-film non-stationarity (drifting
   RSA centre frequency and variance) biases a single global fit.

No Granger_estimator here -- that is Stage 4, which will reuse the identical detrended,
windowed 3-D representation with the explicit `p_used` this stage selects
(`full_freq_dtf(stack, freqs, fs, optimal_model_order=p_used)`; never `None`
on a 3-D stack -- see `DTF_analysis_notes/pipeline_plan.md` Stage 4).

Writes, per dyad x film, `03_mvar/<dyad_id>_<film>_order.json`; across all
cases, `03_mvar/stage03_manifest.csv`; a small window-choice sensitivity check
on 1-2 dyads (gate-only, not written to per-dyad outputs); and a QC gate
(`qc/*.png` figures + `mvar_order_gate.html`).

Each case also gets one Granger_estimator/spectra grid figure (`src.mtmvar.mvar_plot`:
off-diagonal panels = pairwise Granger_estimator, diagonal panels = auto power spectra),
fit at a fixed model order and window geometry (`GRID_MODEL_ORDER`,
`GRID_WIN_LEN_S`/`GRID_OVERLAP_FRAC` below) independent of the per-case
`p_used`/window selection above -- a fixed, comparable view across all cases.

Known caveat, documented and not fixed here (simplicity convention):
`src.mtmvar.mvar_criterion` unpacks `n_channels, n_samples = data.shape` and
therefore crashes on 3-D input; even patched, its `n_samples` penalty term
would need to reflect the summed sample budget across windows, not one
window's. Order selection therefore stays on the global 2-D signal.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.design import DESIGN_VARIABLES, assemble_design_matrix, detrend_windows, window_stack
from src.io_utils import ensure_dir
from src.mtmvar import ar_coeff, full_freq_dtf, multivariate_spectra, mvar_plot, direct_dtf, dtf_estimator
from src.mvar_diag import ar_root_stability, fit_mvar_avg_acf, residual_whiteness, select_order

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ANALYSIS_ROOT = PROJECT_ROOT / "Interbrain_ffDTF_analysis"
ENVELOPES_DIR = ANALYSIS_ROOT / "02_envelopes"
OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / "03_mvar")
QC_DIR = ensure_dir(OUTPUT_DIR / "qc")

FILMS = ["Peppa", "Incredibles", "Brave"]
TARGET_SFREQ = 2.5  # must match Stage 2's realized design-file rate

# Order selection: global 2-D signal only (mvar_criterion caveat, see module docstring).
CRIT_TYPES = ["AIC", "HQ", "SC"]
PRIMARY_CRIT = "SC"       # parsimonious default for n ~ 150 (see Stage 3 §5 in the plan)
MAX_MODEL_ORDER = 15      # short-segment guard, not a scientific claim
EEG_ROWS = [0, 1]         # child:ROI, cg:ROI -- diagnostic-only sub-block order
HRV_ROWS = [2, 3]         # child:HRV, cg:HRV -- diagnostic-only sub-block order

# Locked window geometry (pipeline_plan.md Stage 3): 1/WIN_LEN_S = 0.1 Hz stays
# below the ~0.2-1 Hz coupling band, each window comfortably exceeds a small
# expected p, and 50% overlap lifts a 60 s film from 6 to ~11 windows.
WIN_LEN_S = 10.0
OVERLAP_FRAC = 0.5
DETREND_TYPE = "linear"

RESIDUAL_ACF_MAX_LAG = 8   # must stay well below win_len - p_used
MIN_WHITE_FRACTION = 0.8   # quality flag: per-variable fraction of lags within the Bartlett band

# Per-case Granger_estimator/spectra grid figure (src.mtmvar.mvar_plot): fixed model order and
# window geometry, independent of the p_used/window selected above, for a
# comparable view across all cases.
ESTIMATOR = "dDTF"  # or "ffDTF" for full-frequency DTF, "GPDC" for generalized partial directed coherence
BOX_COX_LAMBDA = 0.25  # (x**lambda - 1) / lambda applied to the Granger_estimator cube; -1 = no transform (src.mtmvar.box_cox_transform)
GRID_MODEL_ORDER = 4
GRID_WIN_LEN_S = 15.0
GRID_OVERLAP_FRAC = 0.5
GRID_FREQS = np.linspace(0.02, TARGET_SFREQ / 2 - 0.02, 100)
GRID_SCALE = "linear"

# Window-choice sensitivity check (gate-only): validates the locked default
# (10 s/50%) against two alternatives, holding p_used fixed (it is selected on
# the window-independent global 2-D signal).
SENSITIVITY_DYADS = [("W_030", "Peppa"), ("W_000", "Peppa")]
SENSITIVITY_WINDOW_CONFIGS = [(10.0, 0.5), (15.0, 0.5), (10.0, 0.0)]
SENSITIVITY_FREQS = np.linspace(0.02, TARGET_SFREQ / 2 - 0.02, 100)
COUPLING_BAND_HZ = (0.15, 0.5)  # band-averaged Granger_estimator for the DV substrate
PRIMARY_EDGES = [("cg:ROI", "child:ROI"), ("child:ROI", "cg:ROI"), ("cg:HRV", "child:HRV"), ("child:HRV", "cg:HRV")]


def parse_case_filename(nc_path):
    """Recover ``(dyad_id, film)`` from a Stage 2 output filename.

    Parameters
    ----------
    nc_path : pathlib.Path
        A `02_envelopes/<dyad_id>_<film>.nc` file.

    Returns
    -------
    tuple of str
        ``(dyad_id, film)``. Raises `ValueError` if the stem does not end in
        one of `FILMS` -- an unexpected filename is a real error, not a case
        to silently skip.
    """
    stem = nc_path.stem
    for film in FILMS:
        suffix = f"_{film}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], film
    raise ValueError(f"Cannot parse dyad_id/film from {nc_path.name}")


def window_geometry(win_len_s, overlap_frac, target_sfreq):
    """Derive integer window length/step (samples) from a length/overlap spec.

    Parameters
    ----------
    win_len_s : float
        Window length in seconds.
    overlap_frac : float
        Fractional overlap between consecutive windows (0 = none, 0.5 = half).
    target_sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    win_len : int
        Window length in samples.
    step : int
        Step between window starts, in samples.
    """
    win_len = round(win_len_s * target_sfreq)
    step = round(win_len * (1 - overlap_frac))
    return win_len, step


def select_p_used(design, max_model_order, crit_types, primary_crit, eeg_rows, hrv_rows):
    """Select the shared model order for the joint system, plus diagnostic sub-block orders.

    `p_used` is fit on the joint 4-variable system (required for the
    exploratory cross brain-heart edges, which only exist in the joint model);
    `p_eeg`/`p_hrv` are reported only to expose an EEG/HRV order mismatch, not
    to justify splitting the fit.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
        Global (non-windowed) z-scored design matrix.
    max_model_order, crit_types, primary_crit : see `src.mvar_diag.select_order`.
    eeg_rows, hrv_rows : list of int
        Row indices for the EEG-only and HRV-only sub-blocks.

    Returns
    -------
    dict
        ``{p_used, p_eeg, p_hrv, orders_full, orders_eeg, orders_hrv,
        curves_full, curves_eeg, curves_hrv, order_range, order_at_cap}``.
    """
    orders_full, curves_full, order_range = select_order(design, max_model_order, crit_types)
    orders_eeg, curves_eeg, _ = select_order(design[eeg_rows], max_model_order, crit_types)
    orders_hrv, curves_hrv, _ = select_order(design[hrv_rows], max_model_order, crit_types)
    return {
        "p_used": orders_full[primary_crit], "p_eeg": orders_eeg[primary_crit], "p_hrv": orders_hrv[primary_crit],
        "orders_full": orders_full, "orders_eeg": orders_eeg, "orders_hrv": orders_hrv,
        "curves_full": curves_full, "curves_eeg": curves_eeg, "curves_hrv": curves_hrv,
        "order_range": order_range, "order_at_cap": any(o == max_model_order for o in orders_full.values()),
    }


def plot_order_curves(order_range, curves_by_block, orders_by_block, max_model_order, title):
    """Plot AIC/HQ/SC criterion curves for the full system and each sub-block."""
    figure, axes = plt.subplots(ncols=len(curves_by_block), figsize=(4 * len(curves_by_block), 4))
    for axis, block_label in zip(axes, curves_by_block):
        for crit_type, curve in curves_by_block[block_label].items():
            line, = axis.plot(order_range, curve, label=crit_type)
            axis.axvline(orders_by_block[block_label][crit_type], color=line.get_color(), linestyle=":", alpha=0.6)
        axis.axvline(max_model_order, color="black", linestyle="--", label="cap")
        axis.set_title(block_label)
        axis.set_xlabel("model order p")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("criterion value")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_roots_comparison(roots_global, roots_windowed, max_abs_root_global, max_abs_root_windowed, title):
    """Plot AR companion eigenvalues for the global vs windowed fit on one unit circle."""
    figure, axis = plt.subplots(figsize=(4.5, 4.5))
    theta = np.linspace(0, 2 * np.pi, 200)
    axis.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
    axis.scatter(roots_global.real, roots_global.imag, color="steelblue",
                 label=f"global (max={max_abs_root_global:.3f})", zorder=3)
    axis.scatter(roots_windowed.real, roots_windowed.imag, color="crimson", marker="x",
                 label=f"windowed (max={max_abs_root_windowed:.3f})", zorder=4)
    axis.set_xlabel("Re")
    axis.set_ylabel("Im")
    axis.set_aspect("equal")
    axis.legend(fontsize=8)
    axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_acf_comparison(acf_global, acf_windowed, band_global, band_windowed, variable_names, title):
    """Plot pooled residual ACF, global vs windowed fit, one panel per variable."""
    figure, axes = plt.subplots(ncols=len(variable_names), figsize=(3.2 * len(variable_names), 3), sharey=True)
    lags = np.arange(acf_global.shape[1])
    for channel, (axis, name) in enumerate(zip(axes, variable_names)):
        width = 0.35
        axis.bar(lags[1:] - width / 2, acf_global[channel, 1:], width=width, color="steelblue", label="global")
        axis.bar(lags[1:] + width / 2, acf_windowed[channel, 1:], width=width, color="crimson", label="windowed")
        axis.axhline(band_global, color="steelblue", linestyle="--", linewidth=0.8)
        axis.axhline(-band_global, color="steelblue", linestyle="--", linewidth=0.8)
        axis.axhline(band_windowed, color="crimson", linestyle="--", linewidth=0.8)
        axis.axhline(-band_windowed, color="crimson", linestyle="--", linewidth=0.8)
        axis.set_title(name, fontsize=9)
        axis.set_xlabel("lag")
    axes[0].set_ylabel("residual ACF")
    axes[0].legend(fontsize=7)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_detrend_example(stack, stack_detrended, variable_names, win_len_s, coupling_band_hz, title, n_examples=3):
    """Plot a few example windows, pre- vs post-detrend, one row per variable."""
    figure, axes = plt.subplots(nrows=len(variable_names), figsize=(8, 2.2 * len(variable_names)), sharex=True)
    for channel, (axis, name) in enumerate(zip(axes, variable_names)):
        for window in range(min(n_examples, stack.shape[2])):
            offset = window * stack.shape[1]
            time = offset + np.arange(stack.shape[1])
            axis.plot(time, stack[channel, :, window], color="steelblue", alpha=0.6,
                      label="raw" if window == 0 else None)
            axis.plot(time, stack_detrended[channel, :, window], color="crimson", alpha=0.8,
                      label="detrended" if window == 0 else None)
        axis.set_ylabel(name, fontsize=9)
    axes[0].legend(fontsize=7)
    axes[-1].set_xlabel("sample (example windows concatenated for display)")
    figure.suptitle(
        f"{title}\nlinear detrend attenuates below ~1/win_len = {1 / win_len_s:.2f} Hz "
        f"(coupling band {coupling_band_hz[0]}-{coupling_band_hz[1]} Hz)"
    )
    figure.tight_layout()
    return figure


def plot_mvar_grid(design, model_order, win_len_s, overlap_frac, freqs, variable_names, title, scale="linear", ESTIMATOR="dDTF", box_cox_lambda=-1):
    """Grid of pairwise Granger_estimator (off-diagonal) and auto power spectra (diagonal).

    Windows and detrends `design` at a fixed geometry and fits one
    windowed-ACF-averaged MVAR at a fixed model order -- independent of the
    `p_used`/window geometry selected elsewhere in this script, so every
    case's grid figure is directly comparable. Delegates the actual figure to
    `src.mtmvar.mvar_plot`, which creates its own figure rather than
    returning one -- `plt.gcf()` recovers it for saving.

    Parameters
    ----------
    design : np.ndarray, shape (k, n_samples)
        Global (non-windowed) z-scored design matrix for one dyad x film.
    model_order : int
        Fixed MVAR model order.
    win_len_s, overlap_frac : float
        Fixed window geometry (seconds, fractional overlap).
    freqs : np.ndarray
        Frequency axis (Hz) for the granger_estimator/spectra grid.
    variable_names : list of str
        Channel labels, in `design`'s row order (`src.design.DESIGN_VARIABLES`).
    title : str
        Figure title.
    scale : {'linear', 'sqrt', 'log'}, optional
        Amplitude scale passed to `mvar_plot`.
    ESTIMATOR : {'dDTF', 'ffDTF', 'GPDC'}, optional
        Estimator type for the directed transfer function calculation.
    box_cox_lambda : float, optional
        Box-Cox exponent applied to the granger_estimator cube (see
        `src.mtmvar.box_cox_transform`). Default -1 = no transform.

    Returns
    -------
    matplotlib.figure.Figure
    """
    win_len, step = window_geometry(win_len_s, overlap_frac, TARGET_SFREQ)
    assert win_len > model_order, f"win_len={win_len} must exceed model_order={model_order}"
    stack = detrend_windows(window_stack(design, win_len, step), dtype=DETREND_TYPE)
    spectra = multivariate_spectra(stack, freqs, TARGET_SFREQ, optimal_model_order=model_order)
    granger_estimator = dtf_estimator(stack, freqs, TARGET_SFREQ, optimal_model_order=model_order, ESTIMATOR=ESTIMATOR, box_cox_lambda=box_cox_lambda)
    mvar_plot(spectra, granger_estimator, freqs, x_label="from ", y_label="to ", chan_names=variable_names,
              top_title=title, scale=scale, fig_size=(9, 9), band_hz=COUPLING_BAND_HZ)
    return plt.gcf()


# ---------------------------------------------------------------------------
# 1. Per-case: order selection, windowed fit, global-comparison fit, diagnostics
# ---------------------------------------------------------------------------
nc_paths = sorted(ENVELOPES_DIR.glob("*.nc"))
print(f"Stage 3: {len(nc_paths)} dyad x film design files found in {ENVELOPES_DIR}")

manifest_rows = []
gate_entries = []

for nc_path in nc_paths:
    dyad_id, film = parse_case_filename(nc_path)
    envelopes = xr.load_dataarray(nc_path)
    design = assemble_design_matrix(envelopes, zscore=True)
    k, n_samples = design.shape

    order = select_p_used(design, MAX_MODEL_ORDER, CRIT_TYPES, PRIMARY_CRIT, EEG_ROWS, HRV_ROWS)
    p_used = order["p_used"]

    win_len, step = window_geometry(WIN_LEN_S, OVERLAP_FRAC, TARGET_SFREQ)
    assert win_len > p_used, f"win_len={win_len} must exceed p_used={p_used} for {dyad_id} {film}"

    stack = window_stack(design, win_len, step)
    stack_detrended = detrend_windows(stack, dtype=DETREND_TYPE)
    n_windows = stack_detrended.shape[2]

    # Windowed-ACF-averaged fit (the default estimator) + diagnostics.
    ar_coeffs, _ = fit_mvar_avg_acf(stack_detrended, p_used)
    roots, max_abs_root, stable = ar_root_stability(ar_coeffs)
    residual_acf, whiteness_summary = residual_whiteness(stack_detrended, ar_coeffs, RESIDUAL_ACF_MAX_LAG)
    n_eff_windowed = win_len - p_used
    band_windowed = 1.96 / np.sqrt(n_eff_windowed)

    # Global single-window comparison fit, same p_used, no per-window detrend.
    ar_coeffs_global, _ = ar_coeff(design, p_used)
    roots_global, max_abs_root_global, stable_global = ar_root_stability(ar_coeffs_global)
    design_1window = design[:, :, np.newaxis]
    residual_acf_global, whiteness_summary_global = residual_whiteness(design_1window, ar_coeffs_global, RESIDUAL_ACF_MAX_LAG)
    band_global = 1.96 / np.sqrt(n_samples - p_used)

    quality_reasons = []
    if not stable:
        quality_reasons.append("unstable")
    if any(fraction < MIN_WHITE_FRACTION for fraction in whiteness_summary.values()):
        quality_reasons.append("residual_autocorrelation")
    quality_ok = len(quality_reasons) == 0

    record = {
        "dyad_id": dyad_id, "film": film,
        "group": str(envelopes.attrs.get("group", "")),
        "age_months": float(envelopes.attrs["age_months"]),
        "n_samples": n_samples,
        "p_eeg": order["p_eeg"], "p_hrv": order["p_hrv"], "p_used": p_used,
        "primary_crit": PRIMARY_CRIT,
        "orders_full": order["orders_full"], "orders_eeg": order["orders_eeg"], "orders_hrv": order["orders_hrv"],
        "order_at_cap": order["order_at_cap"],
        "win_len": win_len, "step": step, "step_s": step / TARGET_SFREQ, "n_windows": n_windows,
        "detrend_type": DETREND_TYPE,
        "max_abs_root": max_abs_root, "stable": stable,
        "whiteness_summary": {DESIGN_VARIABLES[c]: v for c, v in whiteness_summary.items()},
        "max_abs_root_global": max_abs_root_global, "stable_global": stable_global,
        "whiteness_summary_global": {DESIGN_VARIABLES[c]: v for c, v in whiteness_summary_global.items()},
        "quality_ok": quality_ok, "quality_reasons": quality_reasons,
    }
    (OUTPUT_DIR / f"{dyad_id}_{film}_order.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    manifest_rows.append({
        "dyad_id": dyad_id, "film": film, "group": record["group"], "n_samples": n_samples,
        "p_used": p_used, "p_eeg": order["p_eeg"], "p_hrv": order["p_hrv"], "order_at_cap": order["order_at_cap"],
        "n_windows": n_windows, "max_abs_root": max_abs_root, "stable": stable,
        "max_abs_root_global": max_abs_root_global,
        "min_white_fraction": min(whiteness_summary.values()),
        "min_white_fraction_global": min(whiteness_summary_global.values()),
        "quality_ok": quality_ok, "quality_reasons": ";".join(quality_reasons),
    })

    # --- QC figures ---
    case_title = f"{dyad_id} {film}"
    fig = plot_order_curves(
        order["order_range"],
        {"full (4-var)": order["curves_full"], "EEG (2-var)": order["curves_eeg"], "HRV (2-var)": order["curves_hrv"]},
        {"full (4-var)": order["orders_full"], "EEG (2-var)": order["orders_eeg"], "HRV (2-var)": order["orders_hrv"]},
        MAX_MODEL_ORDER, f"{case_title}: model-order criteria (global 2-D signal)",
    )
    order_curves_path = QC_DIR / f"{dyad_id}_{film}_order_curves.png"
    fig.savefig(order_curves_path)
    plt.close(fig)

    fig = plot_roots_comparison(roots_global, roots, max_abs_root_global, max_abs_root,
                                 f"{case_title}: AR roots, global vs windowed (p={p_used})")
    roots_path = QC_DIR / f"{dyad_id}_{film}_roots_comparison.png"
    fig.savefig(roots_path)
    plt.close(fig)

    fig = plot_acf_comparison(residual_acf_global, residual_acf, band_global, band_windowed, DESIGN_VARIABLES,
                               f"{case_title}: residual ACF, global vs windowed (p={p_used})")
    acf_path = QC_DIR / f"{dyad_id}_{film}_acf_comparison.png"
    fig.savefig(acf_path)
    plt.close(fig)

    fig = plot_detrend_example(stack, stack_detrended, DESIGN_VARIABLES, WIN_LEN_S, COUPLING_BAND_HZ,
                                f"{case_title}: example windows, pre- vs post-detrend")
    detrend_path = QC_DIR / f"{dyad_id}_{film}_detrend_example.png"
    fig.savefig(detrend_path)
    plt.close(fig)

    fig = plot_mvar_grid(
        design, GRID_MODEL_ORDER, GRID_WIN_LEN_S, GRID_OVERLAP_FRAC, GRID_FREQS, DESIGN_VARIABLES,
        f"{case_title}: {ESTIMATOR} grid (p={GRID_MODEL_ORDER}, window={GRID_WIN_LEN_S:g}s/{int(GRID_OVERLAP_FRAC * 100)}%)",
        scale=GRID_SCALE, ESTIMATOR=ESTIMATOR, box_cox_lambda=BOX_COX_LAMBDA
    )
    mvar_grid_path = QC_DIR / f"{dyad_id}_{film}_mvar_grid.png"
    fig.savefig(mvar_grid_path)
    plt.close(fig)

    gate_entries.append({
        "dyad_id": dyad_id, "film": film, "group": record["group"],
        "p_used": p_used, "n_windows": n_windows, "max_abs_root": max_abs_root, "stable": stable,
        "min_white_fraction": min(whiteness_summary.values()), "quality_ok": quality_ok,
        "order_curves": order_curves_path.name, "roots": roots_path.name,
        "acf": acf_path.name, "detrend": detrend_path.name, "mvar_grid": mvar_grid_path.name,
    })

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(OUTPUT_DIR / "stage03_manifest.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Run summary
# ---------------------------------------------------------------------------
print(f"\n=== Stage 3 summary ({len(manifest_df)} cases, primary_crit={PRIMARY_CRIT}, "
      f"window={WIN_LEN_S}s/{int(OVERLAP_FRAC * 100)}% overlap) ===")
for group_label, group_df in manifest_df.groupby("group"):
    n = len(group_df)
    n_stable = int(group_df["stable"].sum())
    n_white = int((group_df["min_white_fraction"] >= MIN_WHITE_FRACTION).sum())
    n_quality_ok = int(group_df["quality_ok"].sum())
    n_stable_g = int((group_df["max_abs_root_global"] < 1.0).sum())
    n_white_g = int((group_df["min_white_fraction_global"] >= MIN_WHITE_FRACTION).sum())
    print(f"{group_label}: n={n}  windowed: stable={n_stable}/{n} white={n_white}/{n} quality_ok={n_quality_ok}/{n}  "
          f"|  global: stable={n_stable_g}/{n} white={n_white_g}/{n}")

n_at_cap = int(manifest_df["order_at_cap"].sum())
if n_at_cap:
    print(f"\n{n_at_cap} case(s) had a criterion pick land at MAX_MODEL_ORDER={MAX_MODEL_ORDER}:")
    for _, row in manifest_df[manifest_df["order_at_cap"]].iterrows():
        print(f"  {row['dyad_id']} {row['film']}")

print(f"\nWrote {len(manifest_df)} order files + manifest to {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# 3. Window-choice sensitivity check (gate-only)
# ---------------------------------------------------------------------------
sensitivity_results = []
for dyad_id, film in SENSITIVITY_DYADS:
    nc_path = ENVELOPES_DIR / f"{dyad_id}_{film}.nc"
    envelopes = xr.load_dataarray(nc_path)
    design = assemble_design_matrix(envelopes, zscore=True)

    order = select_p_used(design, MAX_MODEL_ORDER, CRIT_TYPES, PRIMARY_CRIT, EEG_ROWS, HRV_ROWS)
    p_used = order["p_used"]  # held fixed across configs: selected on the window-independent global signal

    spectra_by_config = {}
    for win_len_s, overlap_frac in SENSITIVITY_WINDOW_CONFIGS:
        win_len, step = window_geometry(win_len_s, overlap_frac, TARGET_SFREQ)
        assert win_len > p_used, f"win_len={win_len} must exceed p_used={p_used} for sensitivity config {win_len_s}s"
        stack = detrend_windows(window_stack(design, win_len, step), dtype=DETREND_TYPE)
        granger_estimator = dtf_estimator(stack, SENSITIVITY_FREQS, TARGET_SFREQ, optimal_model_order=p_used, ESTIMATOR=ESTIMATOR, box_cox_lambda=BOX_COX_LAMBDA)
        spectra_by_config[(win_len_s, overlap_frac)] = granger_estimator

    band_mask = (SENSITIVITY_FREQS >= COUPLING_BAND_HZ[0]) & (SENSITIVITY_FREQS <= COUPLING_BAND_HZ[1])

    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), sharex=True)
    band_table_rows = []
    for edge_idx, (source_name, target_name) in enumerate(PRIMARY_EDGES):
        source, target = DESIGN_VARIABLES.index(source_name), DESIGN_VARIABLES.index(target_name)
        axis = axes.flat[edge_idx]
        for config, granger_estimator in spectra_by_config.items():
            label = f"{config[0]:g}s/{int(config[1] * 100)}%"
            axis.plot(SENSITIVITY_FREQS, granger_estimator[target, source], label=label)
            band_table_rows.append({
                "edge": f"{source_name} -> {target_name}", "window_config": label,
                "band_avg_granger": float(granger_estimator[target, source][band_mask].mean()),
            })
        axis.axvspan(*COUPLING_BAND_HZ, color="grey", alpha=0.15)
        axis.set_title(f"{source_name} -> {target_name}", fontsize=9)
        axis.set_xlabel("Frequency (Hz)")
        axis.legend(fontsize=7)
    axes[0, 0].set_ylabel(f"{ESTIMATOR} (Granger causality)")
    axes[1, 0].set_ylabel(f"{ESTIMATOR} (Granger causality)")
    figure.suptitle(f"{dyad_id} {film}: window-choice sensitivity (p_used={p_used})")
    figure.tight_layout()
    sensitivity_plot_path = QC_DIR / f"{dyad_id}_{film}_sensitivity.png"
    figure.savefig(sensitivity_plot_path)
    plt.close(figure)

    band_table = pd.DataFrame(band_table_rows)
    sensitivity_results.append({
        "dyad_id": dyad_id, "film": film, "p_used": p_used,
        "plot": sensitivity_plot_path.name, "band_table": band_table,
    })

sensitivity_table_path = OUTPUT_DIR / "sensitivity_band_averages.csv"
pd.concat([r["band_table"].assign(dyad_id=r["dyad_id"], film=r["film"]) for r in sensitivity_results]).to_csv(
    sensitivity_table_path, index=False,
)
print(f"\nWrote window-choice sensitivity check ({len(sensitivity_results)} cases) to {sensitivity_table_path}")

# ---------------------------------------------------------------------------
# 4. Interactive HTML gate
# ---------------------------------------------------------------------------
gate_by_dyad = {}
for entry in gate_entries:
    gate_by_dyad.setdefault(entry["dyad_id"], []).append(entry)
gate_dyad_ids = sorted(gate_by_dyad.keys())

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 3 windowed-MVAR gate</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 1.5em; color: #1a1a1a; }
  h1 { font-size: 1.3em; }
  select { font-size: 1em; padding: 0.3em; margin-bottom: 1em; }
  .dyad-panel { display: none; }
  .dyad-panel.active { display: block; }
  .film-block { border-top: 1px solid #ccc; padding-top: 1em; margin-top: 1em; }
  .header-line { font-family: monospace; margin-bottom: 0.5em; }
  .badge { padding: 0.1em 0.5em; border-radius: 3px; color: white; font-size: 0.85em; }
  .badge-ok { background: #2e8b2e; }
  .badge-bad { background: #b03030; }
  .row { display: flex; flex-wrap: wrap; gap: 0.5em; }
  .row img { max-width: 500px; border: 1px solid #ccc; }
  .row img.mvar-grid { max-width: 700px; }
  h2, h3 { margin-bottom: 0.3em; }
  #summary, #sensitivity { margin-bottom: 1.5em; white-space: pre; font-family: monospace; }
</style>
</head>
<body>
<h1>Stage 3 windowed-MVAR gate</h1>
<p>estimator=<b>__ESTIMATOR__</b> (box_cox_lambda=<b>__BOX_COX_LAMBDA__</b>, -1 = no transform), primary_crit=<b>__PRIMARY_CRIT__</b>, window=<b>__WIN_LEN_S__ s / __OVERLAP_PCT__% overlap</b>,
   detrend=<b>__DETREND_TYPE__</b>, min_white_fraction=__MIN_WHITE_FRACTION__.</p>
<div id="summary">__SUMMARY__</div>
<h2>Window-choice sensitivity</h2>
<div id="sensitivity">__SENSITIVITY_TEXT__</div>
<div class="row">__SENSITIVITY_IMAGES__</div>
<h2>Per-case diagnostics</h2>
<label for="dyad-select">Dyad: </label>
<select id="dyad-select"></select>
<div id="panels">__PANELS__</div>
<script>
const dyadIds = __DYAD_IDS_JSON__;
const select = document.getElementById('dyad-select');
for (const id of dyadIds) {
  const opt = document.createElement('option');
  opt.value = id; opt.textContent = id;
  select.appendChild(opt);
}
function showDyad(id) {
  document.querySelectorAll('.dyad-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById('panel-' + id);
  if (panel) panel.classList.add('active');
}
select.onchange = () => showDyad(select.value);
if (dyadIds.length) showDyad(dyadIds[0]);
</script>
</body>
</html>
"""


def render_dyad_panel(dyad_id, entries):
    """Render one dyad's QC panel (one film-block per case) as an HTML fragment."""
    html = [f'<div class="dyad-panel" id="panel-{dyad_id}"><h2>{dyad_id}</h2>']
    for entry in entries:
        badge_class = "badge-ok" if entry["quality_ok"] else "badge-bad"
        badge_text = "quality_ok" if entry["quality_ok"] else "quality_fail"
        html.append(f'<div class="film-block"><h3>{entry["film"]} (group={entry["group"]})</h3>')
        html.append(
            f'<div class="header-line">n_windows={entry["n_windows"]}  p_used={entry["p_used"]}  '
            f'max_abs_root={entry["max_abs_root"]:.3f}  stable={entry["stable"]}  '
            f'min_white_fraction={entry["min_white_fraction"]:.2f}  '
            f'<span class="badge {badge_class}">{badge_text}</span></div>'
        )
        html.append('<div class="row">')
        html.append(f'<img src="qc/{entry["order_curves"]}" alt="order curves">')
        html.append(f'<img src="qc/{entry["roots"]}" alt="AR roots global vs windowed">')
        html.append(f'<img src="qc/{entry["acf"]}" alt="residual ACF global vs windowed">')
        html.append(f'<img src="qc/{entry["detrend"]}" alt="pre vs post detrend">')
        html.append(f'<img src="qc/{entry["mvar_grid"]}" alt="ffDTF grid" class="mvar-grid">')
        html.append('</div></div>')
    html.append('</div>')
    return "\n".join(html)


panels_html = "\n".join(render_dyad_panel(dyad_id, gate_by_dyad[dyad_id]) for dyad_id in gate_dyad_ids)

summary_lines = [f"{len(manifest_df)} cases total"]
for group_label, group_df in manifest_df.groupby("group"):
    n = len(group_df)
    summary_lines.append(
        f"{group_label}: n={n}  windowed: stable={int(group_df['stable'].sum())}/{n} "
        f"white={int((group_df['min_white_fraction'] >= MIN_WHITE_FRACTION).sum())}/{n} "
        f"quality_ok={int(group_df['quality_ok'].sum())}/{n}  |  "
        f"global: stable={int((group_df['max_abs_root_global'] < 1.0).sum())}/{n} "
        f"white={int((group_df['min_white_fraction_global'] >= MIN_WHITE_FRACTION).sum())}/{n}"
    )
summary_text = "\n".join(summary_lines)

sensitivity_text_lines = []
for result in sensitivity_results:
    sensitivity_text_lines.append(f"\n{result['dyad_id']} {result['film']} (p_used={result['p_used']}):")
    sensitivity_text_lines.append(result["band_table"].to_string(index=False))
sensitivity_text = "\n".join(sensitivity_text_lines)
sensitivity_images = "\n".join(f'<img src="qc/{r["plot"]}" alt="sensitivity {r["dyad_id"]} {r["film"]}">' for r in sensitivity_results)

html = HTML_TEMPLATE.replace("__PANELS__", panels_html)
html = html.replace("__SUMMARY__", summary_text)
html = html.replace("__SENSITIVITY_TEXT__", sensitivity_text)
html = html.replace("__SENSITIVITY_IMAGES__", sensitivity_images)
html = html.replace("__DYAD_IDS_JSON__", json.dumps(gate_dyad_ids))
html = html.replace("__ESTIMATOR__", ESTIMATOR)
html = html.replace("__BOX_COX_LAMBDA__", str(BOX_COX_LAMBDA))
html = html.replace("__PRIMARY_CRIT__", PRIMARY_CRIT).replace("__WIN_LEN_S__", str(WIN_LEN_S))
html = html.replace("__OVERLAP_PCT__", str(int(OVERLAP_FRAC * 100))).replace("__DETREND_TYPE__", DETREND_TYPE)
html = html.replace("__MIN_WHITE_FRACTION__", str(MIN_WHITE_FRACTION))
(OUTPUT_DIR / "mvar_order_gate.html").write_text(html, encoding="utf-8")
print(f"Wrote interactive gate to {OUTPUT_DIR / 'mvar_order_gate.html'}")
