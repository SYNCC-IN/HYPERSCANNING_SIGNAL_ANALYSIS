"""Stage 5 - surrogate dyads, delta_dtf and z_vs_surrogate.

Stage 4 is done and untouched: for every `02_envelopes/<dyad>_<film>.nc` it
wrote a per-case ffDTF/spectra cube at that case's Stage-3-selected order
`p_used`. Those cubes remain the descriptive per-case estimate.

Stage 5 builds the group-model dependent variable: for every real
dyad x film x edge, `delta_dtf` and `z_vs_surrogate` against a film-matched
surrogate null (a foreign child + foreign caregiver who watched the same
film but were never in the room together). Subtracting the null removes the
common-stimulus + generic-physiology component and isolates genuine
interpersonal coupling. `delta_dtf`/`z_vs_surrogate` stay SIGNED everywhere
-- never `abs()` or clipped -- since the sign of what survives (especially on
HRV/H4) is scientifically meaningful.

Locked decisions (see `DTF_analysis_notes/pipeline_plan.md` Stage 5 -- honour,
do not silently resolve differently):

- L1: common model order `p=4` for every fit in this stage, real AND
  surrogate. ffDTF depends on order; Stage 3/4's per-case `p_used` varies
  across dyads, which would put each dyad's Delta baseline at a different
  order -- a confound in a group DV. Consequence: the real ffDTF entering
  Delta here is RECOMPUTED at p=4, not read from Stage 4's `.npz` cubes.
- L2: a surrogate is a re-paired design matrix (foreign child rows + foreign
  caregiver rows, same film) re-estimated through the exact Stage 4 path
  (`src.connectivity.Granger_estimator`) -- ffDTF is a joint 4-variable
  quantity and cannot be composed from two real cubes.
- L3: window geometry stays locked at Stage 3/4's values (10 s / 50 %
  overlap, linear per-window detrend) for real and surrogate alike. Each
  case's `03_mvar/<dyad>_<film>_order.json` `win_len`/`step`/`detrend_type`
  is asserted equal to the config constants below -- a consistency guard,
  since only `p` is deliberately overridden.
- L4: DV substrate = band-averaged ffDTF over the coupling band 0.2-1.0 Hz,
  same frequency grid as Stage 4 (100 pts, 0.02 Hz -> Nyquist - 0.02).
- L5: one pooled null per film x edge, shared across all dyads of that film,
  pooled across both groups (TD and ASD) -- group-agnostic so it cannot
  absorb the group effect the model tests. The pool includes surrogates that
  share one member with a given real dyad; that is standard and fine.
- L6: full ordered mismatched-pair set, `N*(N-1)` surrogates per film,
  deterministic, no random subsampling by default.
- L7: instability filter, explicit and counted. Each surrogate's windowed-ACF
  AR (p=4) is checked for companion-matrix stability
  (`windowed_ar_stability`); unstable surrogates are excluded from the null
  and the excluded count is reported per film. Real dyads are never dropped;
  their p=4 stability is recorded as `real_stable` and surfaced.
- L8: fixed variable order `DESIGN_VARIABLES` and
  `ffdtf[target, source, f]` (flow source->target) throughout, matching
  Stages 3/4. All 12 directed edges are computed; the scientifically named
  subset is tagged via `EDGE_CLASS` (H2/H4 primary+reverse, exploratory
  cross brain-heart, else "other").

Open decisions, surfaced here rather than resolved silently (see plan doc):

- D1: real value recomputed at p=4 (default, per L1) vs reusing Stage 4's
  `p_used` cube (rejected -- would mix orders in Delta).
- D2: null scope = film-wide pooled (default reference, L5) vs
  leave-one-dyad-out per real dyad. Only pooled is implemented; a
  leave-one-dyad-out scope would give a different null per dyad and break
  the one-null-per-film output contract -- not built here.
- D3: null pooling across groups (reference, L5) vs within group. BOTH are
  now run, controlled by `NULL_POOL_SCOPES=("film", "within_group")`. The
  pooled "film" null is the locked reference (its artifacts and CSV schema
  are byte-for-byte unchanged and Stage 6 consumes it). "within_group" is a
  SENSITIVITY: a same-group foreign-pair null, one per (film x group),
  written to `_within_group`-suffixed companion files. It exists because the
  pooled null is a group-invariant per-film scalar and therefore cancels out
  of every TD-vs-ASD contrast (group main effect AND film x group
  interaction) -- so it cannot separate genuine interpersonal coupling from a
  group-dependent stimulus response, which is exactly what the surviving
  interaction risks being. The within-group null subtracts each group's own
  stimulus/physiology baseline, so an interaction that survives it is genuine
  coupling. Requires n_g >= 2 dyads per (film x group) (asserted loudly);
  small groups yield few draws and a noisy z (surfaced via n_null).

Note for Stage 6 (not a Stage 5 decision): the plan's group-model formula
uses `(1|dyad) + (1|child) + (1|caregiver)`, but the envelope/order metadata
carries only `dyad_id` -- child and caregiver ids are 1:1 with dyad in this
design. Whether/how to expand crossed random-effect terms from `dyad_id` is
a Stage 6 modelling decision; Stage 5 simply carries `dyad_id`, `group`,
`age_months` in the tidy table.

Writes (reference "film" scope, unchanged), per film,
`05_surrogate/<film>_null.npz` (pooled null draws per edge, pairing metadata,
exclusion counts) and QC figures; per real dyad x film,
`05_surrogate/<dyad>_<film>_delta.npz` (delta/z per edge); across all cases,
the tidy hand-off table `05_surrogate/stage05_delta_table.csv`; and an
interactive QC gate (`surrogate_gate.html`).

Additionally, for the "within_group" (D3) sensitivity, per (film x group),
`<film>_<group>_null_within_group.npz` and QC histograms; per real dyad x
film, `<dyad>_<film>_delta_within_group.npz`; and the companion hand-off
table `stage05_delta_table_within_group.csv` (IDENTICAL column schema to the
reference table -- the null scope is in the filename, not a column, so Stage 6
runs unchanged on either). The gate gains a within-group sensitivity section.
The reference `.npz` files gain two provenance keys (`null_scope`,
`null_group`); their delta/z values and the reference CSV schema are
unchanged.
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
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.connectivity import Granger_estimator
from src.design import DESIGN_VARIABLES, assemble_design_matrix
from src.io_utils import ensure_dir
from src.surrogate import assemble_surrogate_design, band_average_cube, delta_and_z, surrogate_pairs, windowed_ar_stability

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ANALYSIS_ROOT = PROJECT_ROOT / "Interbrain_ffDTF_analysis"
ENVELOPES_DIR = ANALYSIS_ROOT / "02_envelopes"
ORDER_DIR = ANALYSIS_ROOT / "03_mvar"
OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / "05_surrogate")
QC_DIR = ensure_dir(OUTPUT_DIR / "qc")

FILMS = ["Peppa", "Incredibles", "Brave"]
TARGET_SFREQ = 2.5  # must match Stage 2/3/4's realized design-file rate (asserted against each file's attrs below)

COMMON_MODEL_ORDER = 4  # L1: fixed order for every fit in this stage, real and surrogate

# Locked window geometry (L3), identical to Stage 3/4 -- asserted equal to
# each case's order.json below rather than recomputed independently.
WIN_LEN_S = 10.0
OVERLAP_FRAC = 0.5
DETREND_TYPE = "linear"

FREQS = np.linspace(0.02, TARGET_SFREQ / 2 - 0.02, 100)  # L4, identical to Stage 4's grid
COUPLING_BAND_HZ = (0.2, 1.0)
ESTIMATOR = "dDTF"  # or "ffDTF" for full-frequency DTF, "GPDC" for generalized partial directed coherence -- must match Stage 3/4's ESTIMATOR
BOX_COX_LAMBDA = 0.25  # (x**lambda - 1) / lambda applied to the Granger_estimator cube; -1 = no transform (src.mtmvar.box_cox_transform) -- must match Stage 3/4's BOX_COX_LAMBDA

ALL_EDGES = [(source, target) for source in DESIGN_VARIABLES for target in DESIGN_VARIABLES if source != target]
EDGE_CLASS = {
    ("cg:ROI", "child:ROI"): "H2_primary",
    ("child:ROI", "cg:ROI"): "H2_reverse",
    ("cg:HRV", "child:HRV"): "H4_primary",
    ("child:HRV", "cg:HRV"): "H4_reverse",
    ("cg:HRV", "child:ROI"): "exploratory",
    ("child:HRV", "cg:ROI"): "exploratory",
}  # remaining 6 directed edges default to "other" via edge_class_for()

FOUR_PRIMARY_EDGES = [("cg:ROI", "child:ROI"), ("child:ROI", "cg:ROI"), ("cg:HRV", "child:HRV"), ("child:HRV", "cg:HRV")]
SIX_EMPHASIS_EDGES = FOUR_PRIMARY_EDGES + [("cg:HRV", "child:ROI"), ("child:HRV", "cg:ROI")]

SURROGATE_STABILITY_MAX_ROOT = 1.0  # L7: exclude surrogate from the null if max_abs_root >= this

# L5/D3: null pooling scopes to run, in order. "film" is the LOCKED reference
# (one pooled null per film, shared across both groups, L5) -- it writes the
# canonical Stage 5 artifacts that Stage 6 consumes and its behaviour is
# byte-for-byte unchanged. "within_group" is the D3 SENSITIVITY: a same-group
# foreign-pair null, one per (film x group), written to suffixed companion
# files, never overwriting the reference. Rationale: the pooled null is a
# group-invariant per-film scalar, so it cancels out of every TD-vs-ASD
# contrast -- including the film x group interaction -- and therefore cannot
# rule out a group-dependent stimulus response; the within-group null subtracts
# each group's own stimulus/physiology baseline, so a surviving interaction is
# genuine interpersonal coupling, not shared-film driving. "leave_one_dyad_out"
# (D2) is still not implemented.
NULL_POOL_SCOPES = ("film", "within_group")
REFERENCE_SCOPE = "film"
SCOPE_SUFFIX = {"film": "", "within_group": "_within_group"}

MAX_SURROGATES_PER_FILM = None  # L6: None = full N*(N-1) set; set an int to cap
SURROGATE_SUBSAMPLE_SEED = 0    # only used if MAX_SURROGATES_PER_FILM is set

FFDTF_ROWSUM_TOL = 1e-6
GRID_SCALE = "linear"

assert REFERENCE_SCOPE == "film" and REFERENCE_SCOPE == NULL_POOL_SCOPES[0], \
    "REFERENCE_SCOPE must be the locked pooled 'film' null and run first (L5)"
assert set(NULL_POOL_SCOPES) <= {"film", "within_group"}, \
    f"unsupported scope(s) in {NULL_POOL_SCOPES} -- only 'film' (L5) and 'within_group' (D3) exist; leave_one_dyad_out (D2) not implemented"
assert set(SCOPE_SUFFIX) >= set(NULL_POOL_SCOPES), "SCOPE_SUFFIX must cover every scope in NULL_POOL_SCOPES"
# Subsampling is defined per film (whole-film pair pool); it has no per-group
# meaning, so the within_group sensitivity always uses the full same-group set.
assert not ("within_group" in NULL_POOL_SCOPES and MAX_SURROGATES_PER_FILM is not None), \
    "unset MAX_SURROGATES_PER_FILM to run the within_group sensitivity (subsampling is per film, not per group)"


def parse_case_filename(nc_path):
    """Recover ``(dyad_id, film)`` from a Stage 2 output filename.

    Mirrors `scripts/stage03_mvar_order.py`/`scripts/stage04_ffdtf.py`'s
    `parse_case_filename` exactly (duplicated rather than imported, since
    those scripts have no ``__main__`` guard and importing one would re-run
    its whole pipeline).

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

    Identical maths to Stage 3/4's script-local `window_geometry`; used here
    only to derive the locked config's expected `win_len`/`step` so each
    case's `order.json` can be asserted against it (L3).

    Parameters
    ----------
    win_len_s : float
        Window length in seconds.
    overlap_frac : float
        Fractional overlap between consecutive windows.
    target_sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    win_len : int
    step : int
    """
    win_len = round(win_len_s * target_sfreq)
    step = round(win_len * (1 - overlap_frac))
    return win_len, step


def edge_class_for(source_name, target_name):
    """Return this directed edge's H2/H4/exploratory/"other" tag (see `EDGE_CLASS`)."""
    return EDGE_CLASS.get((source_name, target_name), "other")


def real_edge_value(band_avg, source_name, target_name):
    """Read one directed edge's band-averaged ffDTF out of a (4, 4) matrix (row=target, col=source)."""
    source, target = DESIGN_VARIABLES.index(source_name), DESIGN_VARIABLES.index(target_name)
    return float(band_avg[target, source])


def plot_null_vs_real_violin(edges_to_plot, null_matrix, real_by_dyad, title, delta_space=False):
    """Split violin of the surrogate null vs real dyads (TD left / ASD right), per edge.

    Density-normalised (KDE), not a raw-count histogram: the surrogate null pool (hundreds
    to thousands of draws) vastly outnumbers real dyads (tens), so a count-based plot would
    make the null dwarf the real distributions regardless of effect size. Each of the three
    densities (null, TD, ASD) is independently normalised to unit area by `gaussian_kde`
    (area, not count), then all three share one width-scale constant -- so violin width
    reflects relative density, not sample size. The null (grey, both halves, should look
    roughly symmetric about its mean) sits in the background; TD occupies the left half and
    ASD the right half of the same y-scale, for an at-a-glance group-vs-group and
    group-vs-null read.

    Parameters
    ----------
    edges_to_plot : list of tuple(str, str)
        `(source_name, target_name)` edges to render, one panel each.
    null_matrix : np.ndarray, shape (n_pairs_kept, len(ALL_EDGES))
        Pooled surrogate null draws, columns in `ALL_EDGES` order.
    real_by_dyad : dict
        `{dyad_id: {"band_avg": (4,4) array, "group": str, ...}}` for this film.
    title : str
        Figure title.
    delta_space : bool, optional
        If False (default), plot raw band-averaged estimator values. If True, shift every
        value in a panel by that panel's own `-null_median` before plotting -- i.e. plot
        `delta_dtf` (`src.surrogate.delta_and_z`'s signed `real - median(null)`) instead of
        the raw value. The null violin is then centred on zero by construction; each real
        dyad's offset from zero IS its `delta_dtf`, read directly off the y-axis. Uses the
        median (matching `delta_and_z`), not the mean, so the delta shown here is exactly
        the `delta_dtf` value in the tidy table -- not a different, mean-centred quantity.

    Returns
    -------
    matplotlib.figure.Figure
    """
    group_colors = {"TD": "tab:blue", "ASD": "tab:orange"}
    group_sides = {"TD": -1, "ASD": 1}
    n_cols = 3
    n_rows = int(np.ceil(len(edges_to_plot) / n_cols))
    half_width = 0.4  # max half-violin width (x-axis units), shared by null/TD/ASD
    figure, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3.5 * n_cols, 4 * n_rows))
    axes_flat = list(np.atleast_1d(axes).flat)

    for panel_idx, (axis, (source_name, target_name)) in enumerate(zip(axes_flat, edges_to_plot)):
        edge_idx = ALL_EDGES.index((source_name, target_name))
        null_values = null_matrix[:, edge_idx]
        offset = np.median(null_values) if delta_space else 0.0
        null_values = null_values - offset
        values_by_group = {
            group_label: np.array([real_edge_value(info["band_avg"], source_name, target_name) - offset
                                    for info in real_by_dyad.values() if info["group"] == group_label])
            for group_label in group_sides
        }

        all_values = np.concatenate([null_values] + list(values_by_group.values()))
        y_grid = np.linspace(all_values.min(), all_values.max(), 200)

        null_density = gaussian_kde(null_values)(y_grid)
        scale = half_width / null_density.max()
        axis.fill_betweenx(y_grid, -null_density * scale, null_density * scale,
                            color="lightgrey", alpha=0.6, zorder=1, label="surrogate null")
        axis.axhline(np.median(null_values), color="black", linestyle="--", linewidth=1, zorder=2, label="null median")

        for group_label, side in group_sides.items():
            values = values_by_group[group_label]
            if values.size < 2:
                continue
            density = gaussian_kde(values)(y_grid) * scale
            color = group_colors[group_label]
            axis.fill_betweenx(y_grid, 0, side * density, color=color, alpha=0.7, zorder=3, label=group_label)
            tick_x = sorted([0, side * 0.6 * half_width])
            axis.hlines(np.median(values), tick_x[0], tick_x[1], color=color, linewidth=2, zorder=4)

        axis.set_title(f"{source_name} -> {target_name} ({edge_class_for(source_name, target_name)})", fontsize=9)
        axis.set_xlim(-half_width * 1.1, half_width * 1.1)
        axis.set_xticks([-half_width / 2, half_width / 2])
        axis.set_xticklabels(["TD", "ASD"])
        if panel_idx == 0:
            axis.legend(fontsize=6, loc="upper right")

    box_cox_suffix = "" if BOX_COX_LAMBDA == -1 else f", box_cox_lambda={BOX_COX_LAMBDA}"
    y_axis_label = (f"delta_dtf ({ESTIMATOR}, real - null_median{box_cox_suffix})" if delta_space
                     else f"band-avg {ESTIMATOR}{box_cox_suffix}")
    for axis in axes_flat[: n_rows * n_cols : n_cols]:
        axis.set_ylabel(y_axis_label)
    for axis in axes_flat[len(edges_to_plot):]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_delta_summary(delta_table_df, edges_to_plot, title):
    """Per-group mean +/- SEM of `delta_dtf` for the given edges.

    Parameters
    ----------
    delta_table_df : pd.DataFrame
        The tidy Stage 5 table (or a subset), with `group`, `source`,
        `target`, `delta_dtf` columns.
    edges_to_plot : list of tuple(str, str)
        `(source_name, target_name)` edges, in display order.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    edge_labels = [f"{s}->{t}" for s, t in edges_to_plot]
    x = np.arange(len(edge_labels))
    width = 0.35
    group_labels = sorted(delta_table_df["group"].unique())
    figure, axis = plt.subplots(figsize=(7, 4))
    for i, group_label in enumerate(group_labels):
        group_df = delta_table_df[delta_table_df["group"] == group_label]
        means, sems = [], []
        for source_name, target_name in edges_to_plot:
            edge_df = group_df[(group_df["source"] == source_name) & (group_df["target"] == target_name)]
            means.append(edge_df["delta_dtf"].mean())
            sems.append(edge_df["delta_dtf"].std(ddof=1) / np.sqrt(len(edge_df)))
        offset = (i - (len(group_labels) - 1) / 2) * width
        axis.bar(x + offset, means, width=width, yerr=sems, label=group_label, capsize=3)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(edge_labels, rotation=20, fontsize=8)
    axis.set_ylabel("mean delta_dtf (real - null), +/- SEM")
    axis.legend(title="group")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def compute_null(candidate_pairs, envelopes_by_dyad, win_len, step):
    """Estimate a surrogate null matrix for one set of candidate mismatched pairs.

    Scope-agnostic core shared by the pooled reference ("film") and the D3
    "within_group" sensitivity: the caller decides which pairs go in (all
    off-diagonal for the pooled null; same-group off-diagonal for a within-group
    null). Each pair is stitched (`assemble_surrogate_design`), stability-gated
    at p=4 (`windowed_ar_stability`, L7), estimated (`Granger_estimator`) and
    band-averaged; kept draws are stacked in `ALL_EDGES` column order.

    Parameters
    ----------
    candidate_pairs : list of tuple(str, str)
        Ordered `(child_dyad, cg_dyad)` pairs to attempt.
    envelopes_by_dyad : dict
        `{dyad_id: (envelopes DataArray, order_record)}` for this film.
    win_len, step : int
        Locked window geometry (L3).

    Returns
    -------
    dict
        Keys: `null_matrix` (n_kept, len(ALL_EDGES)), `kept_child_dyads`,
        `kept_cg_dyads`, `n_excluded_unstable`, `n_attempted`.
    """
    null_rows, kept_child_dyads, kept_cg_dyads = [], [], []
    n_excluded_unstable = 0
    for child_dyad, cg_dyad in candidate_pairs:
        assert child_dyad != cg_dyad
        child_envelopes, _ = envelopes_by_dyad[child_dyad]
        cg_envelopes, _ = envelopes_by_dyad[cg_dyad]
        design = assemble_surrogate_design(child_envelopes, cg_envelopes, zscore=True)

        max_abs_root, _ = windowed_ar_stability(design, win_len, step, COMMON_MODEL_ORDER, DETREND_TYPE)
        if max_abs_root >= SURROGATE_STABILITY_MAX_ROOT:
            n_excluded_unstable += 1
            continue

        ffdtf, _ = Granger_estimator(design, FREQS, TARGET_SFREQ, COMMON_MODEL_ORDER, win_len, step, DETREND_TYPE, ESTIMATOR=ESTIMATOR, box_cox_lambda=BOX_COX_LAMBDA)
        band_avg = band_average_cube(ffdtf, FREQS, COUPLING_BAND_HZ)
        null_rows.append([real_edge_value(band_avg, s, t) for s, t in ALL_EDGES])
        kept_child_dyads.append(child_dyad)
        kept_cg_dyads.append(cg_dyad)

    return {
        "null_matrix": np.array(null_rows),
        "kept_child_dyads": kept_child_dyads,
        "kept_cg_dyads": kept_cg_dyads,
        "n_excluded_unstable": n_excluded_unstable,
        "n_attempted": len(candidate_pairs),
    }


# ---------------------------------------------------------------------------
# 1. Per-film: real values at p=4, surrogate null(s), delta/z
#    Scope "film" = locked pooled reference (L5); "within_group" = D3 sensitivity.
# ---------------------------------------------------------------------------
nc_paths = sorted(ENVELOPES_DIR.glob("*.nc"))
print(f"Stage 5: {len(nc_paths)} dyad x film design files found in {ENVELOPES_DIR}")

cases_by_film = {film: [] for film in FILMS}
for nc_path in nc_paths:
    dyad_id, film = parse_case_filename(nc_path)
    cases_by_film[film].append(dyad_id)

locked_win_len, locked_step = window_geometry(WIN_LEN_S, OVERLAP_FRAC, TARGET_SFREQ)
assert locked_win_len > COMMON_MODEL_ORDER, f"win_len={locked_win_len} must exceed p={COMMON_MODEL_ORDER}"

delta_table_rows_by_scope = {scope: [] for scope in NULL_POOL_SCOPES}
film_summaries = []            # reference ("film") scope -- drives the interactive per-dyad gate
within_group_summaries = []    # D3 sensitivity -- drives the gate appendix
gate_by_dyad = {}              # reference scope only

for film in FILMS:
    dyad_ids = sorted(set(cases_by_film[film]))
    n_dyads = len(dyad_ids)
    print(f"\n--- {film}: {n_dyads} dyads ---")

    envelopes_by_dyad = {}
    for dyad_id in dyad_ids:
        envelopes = xr.load_dataarray(ENVELOPES_DIR / f"{dyad_id}_{film}.nc")
        assert envelopes.attrs["fs"] == TARGET_SFREQ, f"fs mismatch for {dyad_id} {film}: {envelopes.attrs['fs']}"
        order_record = json.loads((ORDER_DIR / f"{dyad_id}_{film}_order.json").read_text(encoding="utf-8"))
        assert order_record["win_len"] == locked_win_len, f"win_len mismatch for {dyad_id} {film}: {order_record['win_len']} != {locked_win_len}"
        assert order_record["step"] == locked_step, f"step mismatch for {dyad_id} {film}: {order_record['step']} != {locked_step}"
        assert order_record["detrend_type"] == DETREND_TYPE, f"detrend_type mismatch for {dyad_id} {film}: {order_record['detrend_type']}"
        envelopes_by_dyad[dyad_id] = (envelopes, order_record)

    # --- Real values, recomputed at the common order p=4 (L1/D1). Scope-independent:
    #     computed once and reused by every null-pooling scope below. ---
    real_by_dyad = {}
    for dyad_id in dyad_ids:
        envelopes, order_record = envelopes_by_dyad[dyad_id]
        fs = envelopes.attrs["fs"]
        design = assemble_design_matrix(envelopes, zscore=True)

        ffdtf, _ = Granger_estimator(design, FREQS, fs, COMMON_MODEL_ORDER, locked_win_len, locked_step, DETREND_TYPE, ESTIMATOR=ESTIMATOR, box_cox_lambda=BOX_COX_LAMBDA)
        # [0,1] boundedness and ffDTF's row(target)-sum-to-1 are properties of the RAW
        # (untransformed) cube -- ffDTF/GPDC are ratios bounded by construction and dDTF is
        # a product of two [0,1] quantities, but Box-Cox (x**lambda - 1)/lambda maps [0,1]
        # outside that range for lambda != 1, so both checks only apply when no transform
        # is active (BOX_COX_LAMBDA == -1, the sentinel for "no transform").
        if BOX_COX_LAMBDA == -1:
            assert -1e-9 <= ffdtf.min() and ffdtf.max() <= 1 + 1e-9, f"{ESTIMATOR} out of [0,1] for {dyad_id} {film}: [{ffdtf.min()}, {ffdtf.max()}]"
            if ESTIMATOR == "ffDTF":
                row_sums = ffdtf.sum(axis=(1, 2))
                assert np.max(np.abs(row_sums - 1.0)) < FFDTF_ROWSUM_TOL, f"ffDTF rows not normalised for {dyad_id} {film}"

        band_avg = band_average_cube(ffdtf, FREQS, COUPLING_BAND_HZ)
        max_abs_root, real_stable = windowed_ar_stability(design, locked_win_len, locked_step, COMMON_MODEL_ORDER, DETREND_TYPE)

        real_by_dyad[dyad_id] = {
            "band_avg": band_avg, "real_stable": real_stable, "max_abs_root": max_abs_root,
            "group": order_record["group"], "age_months": order_record["age_months"],
        }

    group_of = {dyad_id: real_by_dyad[dyad_id]["group"] for dyad_id in dyad_ids}
    group_counts = {g: sum(1 for d in dyad_ids if group_of[d] == g) for g in sorted(set(group_of.values()))}
    print(f"  dyads per group: {group_counts}")

    n_real_unstable = sum(1 for info in real_by_dyad.values() if not info["real_stable"])
    if n_real_unstable:
        unstable_dyads = [d for d, info in real_by_dyad.items() if not info["real_stable"]]
        print(f"  WARNING: {n_real_unstable} real dyad(s) unstable at p={COMMON_MODEL_ORDER} (kept, flagged real_stable=False): {unstable_dyads}")

    for scope in NULL_POOL_SCOPES:
        suffix = SCOPE_SUFFIX[scope]

        # --- Build the null pool(s) for this scope. `null_by_group` maps a group
        #     key -> compute_null(...) result: pooled "film" keeps a single shared
        #     pool under key None (used for every dyad, L5); "within_group" (D3)
        #     builds one pool per group, and each dyad uses its own group's pool. ---
        if scope == "film":
            candidate_pairs = surrogate_pairs(dyad_ids)  # full ordered off-diagonal set (L2/L6)
            expected_n_pairs = n_dyads * (n_dyads - 1)
            assert len(candidate_pairs) == expected_n_pairs, f"{film}: expected {expected_n_pairs} pairs, got {len(candidate_pairs)}"
            if MAX_SURROGATES_PER_FILM is not None:
                rng = np.random.default_rng(SURROGATE_SUBSAMPLE_SEED)
                chosen_idx = sorted(rng.choice(len(candidate_pairs), size=min(MAX_SURROGATES_PER_FILM, len(candidate_pairs)), replace=False))
                candidate_pairs = [candidate_pairs[i] for i in chosen_idx]
            null_by_group = {None: compute_null(candidate_pairs, envelopes_by_dyad, locked_win_len, locked_step)}
            null_group_keys = [None]

            def null_pool_for(group_label, _pools=null_by_group):
                return _pools[None]
        else:  # within_group (D3 sensitivity)
            wg_pairs = surrogate_pairs(dyad_ids, group_of=group_of)  # same-group ordered off-diagonal only
            expected_wg = sum(c * (c - 1) for c in group_counts.values())
            assert len(wg_pairs) == expected_wg, f"{film}: expected {expected_wg} within-group pairs, got {len(wg_pairs)}"
            pairs_by_group = {g: [] for g in group_counts}
            for child_dyad, cg_dyad in wg_pairs:
                pairs_by_group[group_of[child_dyad]].append((child_dyad, cg_dyad))
            null_by_group = {}
            for g in sorted(group_counts):
                assert group_counts[g] >= 2, (
                    f"{film}/{g}: only {group_counts[g]} dyad(s) -- within_group null (D3) needs >=2 per "
                    f"(film x group). This sensitivity cell is infeasible; drop 'within_group' from "
                    f"NULL_POOL_SCOPES or exclude this cell before re-running.")
                null_by_group[g] = compute_null(pairs_by_group[g], envelopes_by_dyad, locked_win_len, locked_step)
            null_group_keys = sorted(group_counts)

            def null_pool_for(group_label, _pools=null_by_group):
                return _pools[group_label]

        # --- Persist null pool(s) + QC histogram(s) for this scope ---
        for gkey in null_group_keys:
            pool = null_by_group[gkey]
            gtag = "" if gkey is None else f"_{gkey}"
            assert len(pool["kept_child_dyads"]) + pool["n_excluded_unstable"] == pool["n_attempted"]
            np.savez(
                OUTPUT_DIR / f"{film}{gtag}_null{suffix}.npz",
                null_matrix=pool["null_matrix"], edge_labels=np.array([f"{s}->{t}" for s, t in ALL_EDGES]),
                pair_child_dyads=np.array(pool["kept_child_dyads"]), pair_cg_dyads=np.array(pool["kept_cg_dyads"]),
                n_excluded_unstable=pool["n_excluded_unstable"], n_pairs_kept=len(pool["kept_child_dyads"]),
                n_pairs_attempted=pool["n_attempted"], n_dyads=n_dyads,
                null_scope=scope, null_group=("pooled" if gkey is None else gkey),
                p=COMMON_MODEL_ORDER, win_len=locked_win_len, step=locked_step, detrend_type=DETREND_TYPE,
                coupling_band=np.array(COUPLING_BAND_HZ), freqs=FREQS, variable_order=np.array(DESIGN_VARIABLES),
            )

            reals_for_hist = (real_by_dyad if gkey is None
                              else {d: info for d, info in real_by_dyad.items() if info["group"] == gkey})
            hist_title = (f"{film}{'' if gkey is None else ' ' + gkey}: surrogate null vs real "
                          f"({scope}, p={COMMON_MODEL_ORDER})")
            fig = plot_null_vs_real_violin(SIX_EMPHASIS_EDGES, pool["null_matrix"], reals_for_hist, hist_title)
            fig.savefig(QC_DIR / f"{film}{gtag}_null_hist{suffix}.png")
            plt.close(fig)

            delta_title = (f"{film}{'' if gkey is None else ' ' + gkey}: delta_dtf, surrogate null vs real "
                           f"({scope}, p={COMMON_MODEL_ORDER})")
            delta_fig = plot_null_vs_real_violin(SIX_EMPHASIS_EDGES, pool["null_matrix"], reals_for_hist, delta_title, delta_space=True)
            delta_fig.savefig(QC_DIR / f"{film}{gtag}_delta_violin{suffix}.png")
            plt.close(delta_fig)

        # --- Delta / z per real dyad x edge, against this scope's null ---
        for dyad_id in dyad_ids:
            info = real_by_dyad[dyad_id]
            null_matrix = null_pool_for(info["group"])["null_matrix"]
            deltas, zs, null_medians, null_stds, n_nulls, reals = [], [], [], [], [], []
            film_gate_rows = []
            for edge_idx, (source_name, target_name) in enumerate(ALL_EDGES):
                real_value = real_edge_value(info["band_avg"], source_name, target_name)
                result = delta_and_z(real_value, null_matrix[:, edge_idx])
                deltas.append(result["delta"]); zs.append(result["z"])
                null_medians.append(result["null_median"]); null_stds.append(result["null_std"]); n_nulls.append(result["n_null"])
                reals.append(real_value)

                edge_class = edge_class_for(source_name, target_name)
                delta_table_rows_by_scope[scope].append({
                    "dyad_id": dyad_id, "film": film, "source": source_name, "target": target_name,
                    "edge": f"{source_name}->{target_name}", "edge_class": edge_class,
                    "group": info["group"], "age_months": info["age_months"],
                    "real_ffdtf": real_value, "null_median": result["null_median"], "null_std": result["null_std"],
                    "n_null": result["n_null"], "delta_dtf": result["delta"], "z_vs_surrogate": result["z"],
                    "real_stable": info["real_stable"],
                })
                film_gate_rows.append({
                    "edge": f"{source_name}->{target_name}", "edge_class": edge_class,
                    "real": real_value, "null_median": result["null_median"], "null_std": result["null_std"],
                    "delta": result["delta"], "z": result["z"],
                })

            np.savez(
                OUTPUT_DIR / f"{dyad_id}_{film}_delta{suffix}.npz",
                delta=np.array(deltas), z=np.array(zs), real=np.array(reals),
                null_median=np.array(null_medians), null_std=np.array(null_stds), n_null=np.array(n_nulls),
                edge_labels=np.array([f"{s}->{t}" for s, t in ALL_EDGES]),
                edge_class=np.array([edge_class_for(s, t) for s, t in ALL_EDGES]),
                group=info["group"], age_months=info["age_months"], real_stable=info["real_stable"],
                null_scope=scope, p=COMMON_MODEL_ORDER, coupling_band=np.array(COUPLING_BAND_HZ),
            )

            if scope == REFERENCE_SCOPE:
                gate_by_dyad.setdefault(dyad_id, []).append({
                    "film": film, "group": info["group"], "real_stable": info["real_stable"],
                    "max_abs_root": info["max_abs_root"], "rows": film_gate_rows,
                })

        # --- Per-scope film summary ---
        if scope == REFERENCE_SCOPE:
            pool = null_by_group[None]
            film_summaries.append({
                "film": film, "n_dyads": n_dyads, "n_attempted": pool["n_attempted"],
                "n_pairs_kept": len(pool["kept_child_dyads"]), "n_excluded_unstable": pool["n_excluded_unstable"],
                "n_real_unstable": n_real_unstable, "null_hist_image": f"{film}_null_hist.png",
                "delta_violin_image": f"{film}_delta_violin.png",
                "null_matrix": pool["null_matrix"],
            })
            print(f"  [{scope}] surrogates: attempted={pool['n_attempted']} "
                  f"kept={len(pool['kept_child_dyads'])} excluded_unstable={pool['n_excluded_unstable']}")
        else:
            per_group = []
            for g in sorted(group_counts):
                pool = null_by_group[g]
                per_group.append({
                    "group": g, "n_dyads": group_counts[g], "n_attempted": pool["n_attempted"],
                    "n_pairs_kept": len(pool["kept_child_dyads"]), "n_excluded_unstable": pool["n_excluded_unstable"],
                    "null_hist_image": f"{film}_{g}_null_hist{suffix}.png",
                    "delta_violin_image": f"{film}_{g}_delta_violin{suffix}.png",
                })
                print(f"  [{scope}/{g}] surrogates: attempted={pool['n_attempted']} "
                      f"kept={len(pool['kept_child_dyads'])} excluded_unstable={pool['n_excluded_unstable']}")
            within_group_summaries.append({"film": film, "per_group": per_group})

# --- Tidy hand-off tables: one CSV per scope. IDENTICAL schema in every scope
#     (the null scope is encoded in the filename, not a column), so Stage 6 runs
#     unchanged on either -- point it at the reference table for the primary
#     result and at the *_within_group table for the D3 sensitivity. ---
delta_table_df = pd.DataFrame(delta_table_rows_by_scope[REFERENCE_SCOPE])  # reference, drives the summary + gate below
delta_table_df.to_csv(OUTPUT_DIR / "stage05_delta_table.csv", index=False)
for scope in NULL_POOL_SCOPES:
    if scope == REFERENCE_SCOPE:
        continue
    scope_df = pd.DataFrame(delta_table_rows_by_scope[scope])
    scope_csv = OUTPUT_DIR / f"stage05_delta_table{SCOPE_SUFFIX[scope]}.csv"
    scope_df.to_csv(scope_csv, index=False)
    print(f"Wrote {len(scope_df)}-row {scope} sensitivity table to {scope_csv}")

# ---------------------------------------------------------------------------
# 2. Run summary
# ---------------------------------------------------------------------------
print(f"\n=== Stage 5 summary ({len(delta_table_df)} rows, {delta_table_df['dyad_id'].nunique()} dyads x "
      f"{delta_table_df['film'].nunique()} films x {len(ALL_EDGES)} edges) ===")
summary_lines = []
for film_summary in film_summaries:
    film = film_summary["film"]
    line = (f"{film}: n_dyads={film_summary['n_dyads']} attempted={film_summary['n_attempted']} "
            f"kept={film_summary['n_pairs_kept']} excluded_unstable={film_summary['n_excluded_unstable']} "
            f"real_unstable={film_summary['n_real_unstable']}")
    print(line)
    summary_lines.append(line)
    film_df = delta_table_df[delta_table_df["film"] == film]
    for source_name, target_name in SIX_EMPHASIS_EDGES:
        edge_df = film_df[(film_df["source"] == source_name) & (film_df["target"] == target_name)]
        edge_line = (f"    {source_name}->{target_name} ({edge_df['edge_class'].iloc[0]}): "
                     f"null={edge_df['null_median'].iloc[0]:.4f}+/-{edge_df['null_std'].iloc[0]:.4f}  "
                     f"mean real/delta/z by group: " +
                     "  ".join(f"{g}: real={gdf['real_ffdtf'].mean():.4f} delta={gdf['delta_dtf'].mean():.4f} z={gdf['z_vs_surrogate'].mean():.2f}"
                               for g, gdf in edge_df.groupby("group")))
        print(edge_line)
        summary_lines.append(edge_line)

print(f"\nWrote {len(delta_table_df)}-row tidy table + per-film nulls + per-dyad deltas to {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# 3. Delta/z group summary figure (four H2/H4 edges, pooled across films)
# ---------------------------------------------------------------------------
delta_summary_fig = plot_delta_summary(delta_table_df, FOUR_PRIMARY_EDGES, "delta_dtf by group, H2/H4 edges (pooled across films)")
delta_summary_path = QC_DIR / "delta_summary.png"
delta_summary_fig.savefig(delta_summary_path)
plt.close(delta_summary_fig)
print(f"Wrote delta/z group summary figure to {delta_summary_path}")

# ---------------------------------------------------------------------------
# 3b. Within-group (D3) sensitivity companions: same-shape delta figure +
#     per-emphasis-edge delta/z-by-group summary, on the within_group null.
#     Empty if within_group was not in NULL_POOL_SCOPES.
# ---------------------------------------------------------------------------
within_group_summary_lines = []
within_group_delta_image = None
if "within_group" in NULL_POOL_SCOPES:
    wg_suffix = SCOPE_SUFFIX["within_group"]
    wg_df = pd.DataFrame(delta_table_rows_by_scope["within_group"])

    wg_delta_fig = plot_delta_summary(
        wg_df, FOUR_PRIMARY_EDGES,
        "delta_dtf by group, H2/H4 edges (within-group null, pooled across films)")
    within_group_delta_image = f"delta_summary{wg_suffix}.png"
    wg_delta_fig.savefig(QC_DIR / within_group_delta_image)
    plt.close(wg_delta_fig)

    print("\n=== Within-group (D3) sensitivity: delta/z by group on the same-group null ===")
    print("  (compare against the reference pooled-null summary above; the film x group "
          "interaction is what Stage 6 re-tests on stage05_delta_table_within_group.csv)")
    for wg_film_summary in within_group_summaries:
        film = wg_film_summary["film"]
        counts = "  ".join(f"{pg['group']}: n_dyads={pg['n_dyads']} kept={pg['n_pairs_kept']} "
                           f"excluded_unstable={pg['n_excluded_unstable']}" for pg in wg_film_summary["per_group"])
        head = f"{film}: {counts}"
        print(head)
        within_group_summary_lines.append(head)
        film_df = wg_df[wg_df["film"] == film]
        for source_name, target_name in SIX_EMPHASIS_EDGES:
            edge_df = film_df[(film_df["source"] == source_name) & (film_df["target"] == target_name)]
            edge_line = (f"    {source_name}->{target_name} ({edge_df['edge_class'].iloc[0]}): "
                         "real/delta/z by group: " +
                         "  ".join(f"{g}: real={gdf['real_ffdtf'].mean():.4f} delta={gdf['delta_dtf'].mean():.4f} "
                                   f"z={gdf['z_vs_surrogate'].mean():.2f} (null n={int(gdf['n_null'].iloc[0])})"
                                   for g, gdf in edge_df.groupby("group")))
            print(edge_line)
            within_group_summary_lines.append(edge_line)

# ---------------------------------------------------------------------------
# 4. Interactive HTML gate
# ---------------------------------------------------------------------------
gate_dyad_ids = sorted(gate_by_dyad.keys())

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 5 surrogate gate</title>
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
  .row img { max-width: 900px; border: 1px solid #ccc; }
  h2, h3 { margin-bottom: 0.3em; }
  #summary, #pairing, #within-group-summary { margin-bottom: 1.5em; white-space: pre; font-family: monospace; }
  code { font-family: monospace; background: #f0f0f0; padding: 0.05em 0.3em; border-radius: 3px; }
  table.edges { border-collapse: collapse; font-size: 0.85em; margin-bottom: 0.5em; }
  table.edges th, table.edges td { border: 1px solid #ccc; padding: 0.2em 0.5em; text-align: right; }
  table.edges th:first-child, table.edges td:first-child { text-align: left; }
  tr.emphasis { background: #eef6ff; font-weight: bold; }
  tr.other { color: #888; font-weight: normal; }
</style>
</head>
<body>
<h1>Stage 5 surrogate gate</h1>
<p>estimator=<b>__ESTIMATOR__</b> (must match Stage 3/4's ESTIMATOR), box_cox_lambda=<b>__BOX_COX_LAMBDA__</b>
   (-1 = no transform; must match Stage 3/4's BOX_COX_LAMBDA),
   common model order=<b>p=__COMMON_MODEL_ORDER__</b> (real AND surrogate, L1/D1 -- NOT Stage 4's per-case p_used),
   window=<b>__WIN_LEN_S__ s / __OVERLAP_PCT__% overlap</b>, detrend=<b>__DETREND_TYPE__</b>,
   coupling_band=<b>__COUPLING_BAND__ Hz</b>, reference null_pool_scope=<b>__REFERENCE_SCOPE__</b> (L5: one pooled
   null per film, across TD+ASD; D2 leave-one-dyad-out not built), surrogate_stability_max_root=__STABILITY_MAX_ROOT__.
   delta_dtf and z_vs_surrogate are SIGNED everywhere -- never abs() or clipped.
   A <b>within-group null (D3) sensitivity</b> is reported in its own section below; the pooled null is a
   group-invariant per-film constant, so it cancels out of every TD-vs-ASD contrast (including the film x group
   interaction) -- only the within-group null can separate genuine interpersonal coupling from a group-dependent
   stimulus response.</p>
<h2>Pairing sanity</h2>
<div id="pairing">__PAIRING__</div>
<h2>Surrogate null vs real, per film</h2>
<div class="row">__NULL_HIST_IMAGES__</div>
<h2>delta_dtf: surrogate null (centred at zero) vs real, per film</h2>
<p>Same violins as above, shifted by each edge's own <code>-null_median</code> -- the null is centred on zero by
   construction and every real dyad's offset from zero IS its <code>delta_dtf</code> (signed, never abs()/clipped).</p>
<div class="row">__DELTA_VIOLIN_IMAGES__</div>
<h2>Run summary</h2>
<div id="summary">__SUMMARY__</div>
<div class="row"><img src="qc/delta_summary.png" alt="delta_dtf group summary"></div>
<h2>Sensitivity: within-group null (D3)</h2>
<p>Same-group foreign-pair null, one per (film x group). Subtracts each group's own shared-stimulus /
   generic-physiology baseline, so a film x group interaction that survives here is genuine interpersonal
   coupling, not the two groups responding to the same film differently. Watch the per-group kept counts:
   small groups give few surrogate draws and a noisy z. Re-run Stage 6 on
   <code>stage05_delta_table_within_group.csv</code> for the sensitivity result.</p>
<div id="within-group-summary">__WITHIN_GROUP_SUMMARY__</div>
<div class="row">__WITHIN_GROUP_DELTA_IMAGE__</div>
<div class="row">__WITHIN_GROUP_NULL_HIST_IMAGES__</div>
<div class="row">__WITHIN_GROUP_DELTA_VIOLIN_IMAGES__</div>
<h2>Per-dyad diagnostics</h2>
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


def render_edge_table(rows):
    """Render one film's 12-edge table (real / null / delta / z) as an HTML fragment."""
    html = ['<table class="edges"><tr><th>edge</th><th>class</th><th>real</th><th>null_median</th>'
            '<th>null_std</th><th>delta_dtf</th><th>z_vs_surrogate</th></tr>']
    for row in rows:
        row_class = "other" if row["edge_class"] == "other" else "emphasis" if row["edge_class"] in ("H2_primary", "H2_reverse", "H4_primary", "H4_reverse") else ""
        html.append(
            f'<tr class="{row_class}"><td>{row["edge"]}</td><td>{row["edge_class"]}</td>'
            f'<td>{row["real"]:.4f}</td><td>{row["null_median"]:.4f}</td><td>{row["null_std"]:.4f}</td>'
            f'<td>{row["delta"]:+.4f}</td><td>{row["z"]:+.2f}</td></tr>'
        )
    html.append("</table>")
    return "\n".join(html)


def render_dyad_panel(dyad_id, entries):
    """Render one dyad's QC panel (one film-block per case) as an HTML fragment."""
    html = [f'<div class="dyad-panel" id="panel-{dyad_id}"><h2>{dyad_id}</h2>']
    for entry in entries:
        badge_class = "badge-ok" if entry["real_stable"] else "badge-bad"
        badge_text = "real_stable" if entry["real_stable"] else "real_UNSTABLE (p=4)"
        html.append(f'<div class="film-block"><h3>{entry["film"]} (group={entry["group"]})</h3>')
        html.append(
            f'<div class="header-line">max_abs_root={entry["max_abs_root"]:.3f}  '
            f'<span class="badge {badge_class}">{badge_text}</span></div>'
        )
        html.append(render_edge_table(entry["rows"]))
        html.append('</div>')
    html.append('</div>')
    return "\n".join(html)


panels_html = "\n".join(render_dyad_panel(dyad_id, gate_by_dyad[dyad_id]) for dyad_id in gate_dyad_ids)

pairing_lines = []
for film_summary in film_summaries:
    pairing_lines.append(
        f"{film_summary['film']}: n_dyads={film_summary['n_dyads']}  "
        f"attempted={film_summary['n_attempted']} (expected N*(N-1)={film_summary['n_dyads'] * (film_summary['n_dyads'] - 1)})  "
        f"kept={film_summary['n_pairs_kept']}  excluded_unstable={film_summary['n_excluded_unstable']}  "
        f"real_unstable={film_summary['n_real_unstable']}  "
        f"no surrogate is a real dyad; every pair shares its film ✓"
    )
pairing_text = "\n".join(pairing_lines)
null_hist_images = "\n".join(f'<img src="qc/{fs["null_hist_image"]}" alt="{fs["film"]} null distribution">' for fs in film_summaries)
delta_violin_images = "\n".join(f'<img src="qc/{fs["delta_violin_image"]}" alt="{fs["film"]} delta_dtf vs null">' for fs in film_summaries)

# --- Within-group (D3) appendix fragments (empty strings if it was not run) ---
if within_group_summary_lines:
    within_group_summary_html = "\n".join(within_group_summary_lines)
else:
    within_group_summary_html = "within_group not in NULL_POOL_SCOPES -- sensitivity not run."
within_group_delta_image_html = (
    f'<img src="qc/{within_group_delta_image}" alt="within-group delta_dtf group summary">'
    if within_group_delta_image else "")
within_group_null_hist_images = "\n".join(
    f'<img src="qc/{pg["null_hist_image"]}" alt="{wg["film"]} {pg["group"]} within-group null distribution">'
    for wg in within_group_summaries for pg in wg["per_group"])
within_group_delta_violin_images = "\n".join(
    f'<img src="qc/{pg["delta_violin_image"]}" alt="{wg["film"]} {pg["group"]} within-group delta_dtf vs null">'
    for wg in within_group_summaries for pg in wg["per_group"])

html = HTML_TEMPLATE.replace("__PANELS__", panels_html)
html = html.replace("__PAIRING__", pairing_text)
html = html.replace("__NULL_HIST_IMAGES__", null_hist_images)
html = html.replace("__DELTA_VIOLIN_IMAGES__", delta_violin_images)
html = html.replace("__SUMMARY__", "\n".join(summary_lines))
html = html.replace("__WITHIN_GROUP_SUMMARY__", within_group_summary_html)
html = html.replace("__WITHIN_GROUP_DELTA_IMAGE__", within_group_delta_image_html)
html = html.replace("__WITHIN_GROUP_NULL_HIST_IMAGES__", within_group_null_hist_images)
html = html.replace("__WITHIN_GROUP_DELTA_VIOLIN_IMAGES__", within_group_delta_violin_images)
html = html.replace("__DYAD_IDS_JSON__", json.dumps(gate_dyad_ids))
html = html.replace("__ESTIMATOR__", ESTIMATOR)
html = html.replace("__BOX_COX_LAMBDA__", str(BOX_COX_LAMBDA))
html = html.replace("__COMMON_MODEL_ORDER__", str(COMMON_MODEL_ORDER))
html = html.replace("__WIN_LEN_S__", str(WIN_LEN_S)).replace("__OVERLAP_PCT__", str(int(OVERLAP_FRAC * 100)))
html = html.replace("__DETREND_TYPE__", DETREND_TYPE)
html = html.replace("__COUPLING_BAND__", f"{COUPLING_BAND_HZ[0]}-{COUPLING_BAND_HZ[1]}")
html = html.replace("__REFERENCE_SCOPE__", REFERENCE_SCOPE)
html = html.replace("__STABILITY_MAX_ROOT__", str(SURROGATE_STABILITY_MAX_ROOT))
(OUTPUT_DIR / "surrogate_gate.html").write_text(html, encoding="utf-8")
print(f"Wrote interactive gate to {OUTPUT_DIR / 'surrogate_gate.html'}")
