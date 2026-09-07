"""Stage 6 - group model: Delta ffDTF vs surrogate (H2/H4).

Stage 5 is done and untouched: for every real dyad x film x edge it wrote a
signed `delta_dtf` and `z_vs_surrogate` against a film-matched surrogate null
(`Interbrain_ffDTF_analysis/05_surrogate/stage05_delta_table.csv`, 1512 rows =
42 dyads x 3 films x 12 edges). Stage 6 fits the group-level Bayesian model(s)
on that tidy table and answers H2 (caregiver->child TPJ coupling, reduced in
ASD?) and H4 (caregiver->child HRV co-regulation, altered in ASD -- sign not
assumed). `delta_dtf`/`z_vs_surrogate` stay SIGNED everywhere in every output
of this stage too -- never `abs()` or clipped.

Locked decisions (L1-L9) -- implemented as stated, not silently resolved
differently:

- L1: DV = `delta_dtf` (standardized per edge, D3), Student-t family for
  robustness to outliers. `z_vs_surrogate` is modelled too, but ONLY for the
  asymmetry track (L6) -- never for the plain per-edge models, and never
  `abs()`-ed either DV.
- L2: fixed effects `film * group`, sum-to-zero contrasts on both (`C(.,
  Sum)`) so `group` reads as a main effect and the interaction stays clean.
  `film` is a 3-level valence factor (Peppa/Incredibles/Brave), NOT an
  ordered scale -- never averaged across films as if it were one.
- L3: planned contrast `Incredibles vs (Peppa+Brave)/2`, reported two-sided
  (posterior mean, 95% HDI, `P(>0)` and `P(<0)`), overall and within each
  group. Bambi has no `brms::hypothesis()`/`emmeans` equivalent, so this
  script computes it the estimated-marginal-means way: predict the six
  film x group cell means at the population level
  (`include_group_specific=False`) and take linear combinations of those
  posterior draws -- algebraically identical to a contrast on the fixed
  effects, and it generalises cleanly to the interaction test.
- L4: inference = posterior mean, 95% HDI, and directional posterior
  probability `P(effect>0)`/`P(effect<0)` for every reported contrast. No
  p-values as the primary object (see L7 for the one supplementary
  exception).
- L5: confirmatory edge set = the 6 Stage-5 emphasis edges (H2_primary,
  H2_reverse, H4_primary, H4_reverse, and the 2 exploratory cross
  brain-heart edges). The 6 "other" edges are not modelled by default
  (`FIT_OTHER_EDGES` toggle below).
- L6: asymmetry track. For H2 and H4, `asym = value(primary) -
  value(reverse)` per dyad x film, same `film*group + (1|dyad_id)`
  structure. Primary DV = `z_vs_surrogate` (scale-free, robust to the
  child/adult maturation artefact); `delta_dtf` (standardized) is a
  sensitivity model. Intercept (== grand mean over the 6 film x group cells,
  read off the same posterior-prediction machinery as L3) `> 0` means
  caregiver-leading.
- L7: mild BH-FDR across the PRIMARY family only = {H2_primary group effect,
  H4_primary group effect} -- kept deliberately small and named
  (`PRIMARY_FAMILY` below). Feeds `2*min(P(>0),P(<0))`, an analog p-value
  documented as such, into `src.group_model.bh_fdr`. Reported as a
  supplementary column beside the Bayesian summary, never as the primary
  inference. Exploratory edges get no correction.
- L8: drop `real_stable == False` rows before fitting; report the drop count
  per model even when it is zero (currently: zero for every edge -- a
  visible no-op, not a silent one).
- L9: 4 chains, `DRAWS` post-warmup draws/chain (>= ~2000), `target_accept`
  starting at 0.95, fixed seed. Pass criteria: max Rhat < 1.01, min bulk/tail
  ESS > 400, zero divergences (else the count is reported, not hidden), a
  sane `pp_check`, LOO Pareto-k mostly < 0.7. No silent retry ladder: a
  model that fails L9 is reported as failing in the diagnostics table and the
  gate, which is the trigger to consider the 2-variable fallback -- not
  something this script patches on its own.

Open decisions (D0-D5) -- default implemented, alternative documented:

- D0: engine. The plan's literal contract names `stage06_group_model.R` +
  brms `hypothesis()`. Checked before writing any code: `brms`/`rstan`/
  `cmdstanr` are NOT installed in this environment's R, while `bambi` (0.17.2)
  and `arviz` (0.23.4) already are in `.venv`. Per this prompt's own
  instruction ("if the chosen engine is not installed, stop and say so"),
  this was surfaced to the project owner, who chose **Bambi + ArviZ**. This
  single Python script therefore does both the fit and the gate (keeping one
  toolchain, per the D0 Python-engine branch), reusing `src/group_model.py`
  for the pieces that don't involve MCMC. The output contract (tidy CSVs +
  PNGs + gate) is identical to what the brms branch would have produced.
- D1 (CORRECTNESS FIX): the plan/note formula is `(1|dyad_id) + (1|child_id)
  + (1|caregiver_id)`, but in this design child and caregiver are 1:1 with
  dyad (no member appears in more than one dyad) -- the three grouping
  factors are the same partition of the rows, so the three variance
  components are not jointly identifiable. Collapsed to `(1|dyad_id)` only
  (42 groups x 3 films per edge, a clean repeated-measures term). Surfaced
  in the run summary and the gate header for Jarek to ratify or override.
- D2: per-edge models (default, confirmatory) vs one pooled cross-edge model
  with `+ (1|edge)` partial pooling across the 6 emphasis edges
  (`FIT_POOLED_MODEL`, default True) as a shrinkage companion/cross-check --
  it does not replace the per-edge results.
- D3: `delta_dtf` is tiny (~1e-4 to 1e-3); standardized per edge (z-score
  across that edge's kept rows) before fitting, with explicit
  weakly-informative priors on the standardized scale (Intercept, fixed
  effects ~ Normal(0,1); group/edge SD, sigma ~ half-Student-t(3,0,1)).
  Contrasts are reported in standardized units and back-transformed to raw
  Delta-units (`unit` column). `z_vs_surrogate` (already ~unit scale) is
  modelled on its native scale with the same prior set, `unit="native"`.
- D4: baseline model (no covariate) is primary. `ADD_AGE_COVARIATE` /
  `ADD_IAF_COVARIATE` toggle sensitivity models adding mean-centred
  `age_months` and/or the dyadic `iaf_distance` column of
  `Exploratory_spectral_analysis/04_band_assignment/iaf_metrics.csv` (path
  and column name confirmed against the actual file before writing this
  script -- it does NOT live under `Interbrain_ffDTF_analysis/`). A missing
  file/column when a toggle is on errors loudly; no silent skip.
- D5: no expected sign is encoded for the H4 group effect --
  `P(ASD>TD)`/`P(ASD<TD)` are always reported symmetrically.

Hand-off note (not a Stage 6 decision, just a flag): `pipeline_plan.md` Stage
6 and `notatka_projekt_DTF_HRV_H2_H4.md` still describe the group model with
the stale three-term RE formula this script corrects via D1 -- worth fixing
in those docs, not done here.

Writes `Interbrain_ffDTF_analysis/06_group/` per the module-level OUTPUT_DIR:
fitted `idata` per model (`models/*.nc`), `stage06_contrasts.csv`,
`stage06_diagnostics.csv`, `stage06_primary_summary.csv`, QC figures, and the
interactive `group_model_gate.html`.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az
import bambi as bmb
from bambi.terms.group_specific import GroupSpecificTerm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.group_model import asymmetry_dv, bh_fdr, edge_subset, load_delta_table, standardize_within_edge
from src.io_utils import ensure_dir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ANALYSIS_ROOT = PROJECT_ROOT / "Interbrain_ffDTF_analysis"
INPUT_CSV = ANALYSIS_ROOT / "05_surrogate" / "stage05_delta_table_within_group.csv"  #"stage05_delta_table.csv"
OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / "06_group")
MODELS_DIR = ensure_dir(OUTPUT_DIR / "models")
QC_DIR = ensure_dir(OUTPUT_DIR / "qc")

ENGINE = "bambi"  # D0, ratified by the project owner (brms/cmdstanr not installed; see module docstring)

EMPHASIS_EDGES = [
    ("cg:ROI->child:ROI", "H2_primary"),
    ("child:ROI->cg:ROI", "H2_reverse"),
    ("cg:HRV->child:HRV", "H4_primary"),
    ("child:HRV->cg:HRV", "H4_reverse"),
    ("cg:HRV->child:ROI", "exploratory"),
    ("child:HRV->cg:ROI", "exploratory"),
]  # L5
FIT_OTHER_EDGES = True  # L5: the 6 non-emphasis edges are not modelled by default
FIT_POOLED_MODEL = True  # D2: shrinkage cross-check over the 6 emphasis edges, companion not replacement

ADD_AGE_COVARIATE = False  # D4
ADD_IAF_COVARIATE = False  # D4
IAF_METRICS_CSV = PROJECT_ROOT / "Exploratory_spectral_analysis" / "04_band_assignment" / "iaf_metrics.csv"
IAF_DISTANCE_COLUMN = "iaf_distance"  # confirmed against the actual file's header before writing this script

# L9 MCMC config
CHAINS = 4
DRAWS = 2000
TUNE = 2000            # L-conv-1: was 1000 -- longer warmup for mass-matrix adaptation at the (1|edge) funnel neck
TARGET_ACCEPT = 0.99   # L-conv-1: was 0.95 -- smaller step size to clear the funnel at 6 edge-groups
SEED = 0
HDI_PROB = 0.95
CORES = 1  # macOS fix: PyMC's default "fork" worker start crashes (SIGSEGV, child dies pre-exec) once
           # Accelerate/vecLib BLAS threads are live in the parent. "spawn"/"forkserver" avoid that
           # crash but both re-import this module as __main__ during worker bootstrap, and this
           # top-level pipeline script has no `if __name__ == "__main__":` guard -- that re-import
           # re-runs the whole pipeline recursively and fails the same way every time. cores=1 sidesteps
           # all three failure modes by never creating a worker subprocess: the CHAINS chains are still
           # drawn (for rhat/ESS across chains), just sequentially in this one process. Slower, not
           # parallel, but the only option that doesn't require restructuring the whole script.

# Convergence-fix toggles (Stage 6 addendum). L-conv-2: SD_PRIOR_EDGE scopes
# ONLY the (1|edge) intercept SD hyperprior -- 1|dyad_id and every M1/M2
# varying-slope SD keep their existing HalfStudentT(3,1) untouched.
SD_PRIOR_EDGE = "halfstudentt"  # D-conv-B: "halfstudentt" (nu=3,sigma=1, current) | "halfnormal" (sigma=1, tighter)
FIT_EDGE_AS_FIXED = True  # D-conv-A: M3 = film*group*edge fixed (sum contrasts), no (1|edge); side-by-side with M0/M1/M2, cannot funnel by construction

# L9 pass criteria
RHAT_MAX = 1.01
ESS_MIN = 400
PARETO_K_MAX = 0.7

# L7: the one named primary family for BH-FDR (kept deliberately small)
PRIMARY_FAMILY = [
    ("cg:ROI->child:ROI", "group_effect", "H2_primary"),
    ("cg:HRV->child:HRV", "group_effect", "H4_primary"),
]

DV_MAIN = "z_vs_surrogate"  # "delta_dtf"
FILMS = ["Peppa", "Incredibles", "Brave"]
GROUPS = ["TD", "ASD"]

# ROI info (display-only, for the gate header) -- must match Stage 2's
# ROI_LABEL/ROI_CHANNELS (scripts/stage02_envelopes.py), since the
# `child:ROI`/`cg:ROI` edges in INPUT_CSV are envelopes computed over this
# electrode set.
ROI_LABEL = "temporo-parietal"
ROI_CHANNELS = ["P7"]

# Estimator info (display-only, for the gate header) -- must match Stage 5's
# ESTIMATOR (scripts/stage05_surrogate.py), since INPUT_CSV's `real_ffdtf`/
# `delta_dtf`/`z_vs_surrogate` columns are computed from that estimator's
# cube (Stage 5's CSV carries no ESTIMATOR column of its own; this is a
# manually-synced label, not read from the data).
ESTIMATOR = "dDTF"  # or "ffDTF"/"GPDC" -- must match Stage 5's ESTIMATOR

# Box-Cox info (display-only, for the gate header) -- must match Stage 5's
# BOX_COX_LAMBDA (scripts/stage05_surrogate.py); same manually-synced,
# not-read-from-data caveat as ESTIMATOR above.
BOX_COX_LAMBDA = 0.25  # (x**lambda - 1) / lambda; -1 = no transform -- must match Stage 5's BOX_COX_LAMBDA


def add_covariates(df):
    """Add mean-centred covariate columns to `df` per the D4 toggles.

    A no-op copy when both toggles are off (the default/primary path). When a
    toggle is on and its source file/column is missing, `pd.read_csv`/column
    lookup raises naturally -- no silent skip.

    Parameters
    ----------
    df : pd.DataFrame
        Stage 5 delta table (or a subset).

    Returns
    -------
    pd.DataFrame
        Copy of `df`, with `age_months_c` and/or `iaf_distance_c` added.
    """
    out = df.copy()
    if ADD_AGE_COVARIATE:
        out["age_months_c"] = out["age_months"] - out["age_months"].mean()
    if ADD_IAF_COVARIATE:
        iaf = pd.read_csv(IAF_METRICS_CSV)
        dyad_iaf = iaf.drop_duplicates("dyad_id").set_index("dyad_id")[IAF_DISTANCE_COLUMN]
        out["iaf_distance_c"] = out["dyad_id"].map(dyad_iaf) - dyad_iaf.mean()
    return out


EXTRA_TERMS = (["age_months_c"] if ADD_AGE_COVARIATE else []) + (["iaf_distance_c"] if ADD_IAF_COVARIATE else [])


def build_formula(dv_col, extra_grouping=None):
    """Stage 6 model formula: `film * group` (sum contrasts) + dyad random intercept (D1).

    Parameters
    ----------
    dv_col : str
        Dependent variable column name.
    extra_grouping : str or None, optional
        Extra `(1|term)` grouping factor to append (used for the D2 pooled
        model's `(1|edge)`).

    Returns
    -------
    str
        A formula string bambi understands.
    """
    extra = "".join(f" + {t}" for t in EXTRA_TERMS)
    grouping = " + (1|dyad_id)" + (f" + (1|{extra_grouping})" if extra_grouping else "")
    return f"{dv_col} ~ C(film, Sum) * C(group, Sum){extra}{grouping}"


def build_priors():
    """D3 weakly-informative priors on the standardized (or ~unit, for `z_vs_surrogate`) scale.

    Unused keys (e.g. `1|edge` when the formula has no such term) are
    silently ignored by bambi -- not a masked error, just how bambi resolves
    a prior dict against a formula.

    Returns
    -------
    dict
        Bambi `Prior` objects keyed by term name.
    """
    group_sd_prior = bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfStudentT", nu=3, sigma=1))
    return {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
        "C(film, Sum)": bmb.Prior("Normal", mu=0, sigma=1),
        "C(group, Sum)": bmb.Prior("Normal", mu=0, sigma=1),
        "C(film, Sum):C(group, Sum)": bmb.Prior("Normal", mu=0, sigma=1),
        "1|dyad_id": group_sd_prior,
        "1|edge": bmb.Prior("Normal", mu=0, sigma=edge_sd_hyperprior()),
        "sigma": bmb.Prior("HalfStudentT", nu=3, sigma=1),
    }


def edge_sd_hyperprior():
    """The (1|edge) SD hyperprior, selected by SD_PRIOR_EDGE (D-conv-B).

    Scopes ONLY the `1|edge` intercept SD (L-conv-2) -- `1|dyad_id` and every
    M1/M2 varying-slope SD keep their own `HalfStudentT(3,1)` regardless of
    this toggle. Loud on an unrecognized value rather than a silent default.

    Returns
    -------
    bambi.priors.Prior
    """
    if SD_PRIOR_EDGE == "halfstudentt":
        return bmb.Prior("HalfStudentT", nu=3, sigma=1)
    if SD_PRIOR_EDGE == "halfnormal":
        return bmb.Prior("HalfNormal", sigma=1)
    raise ValueError(f"unknown SD_PRIOR_EDGE={SD_PRIOR_EDGE!r} (expected 'halfstudentt' or 'halfnormal')")


def fit_model(formula, data, family="t"):
    """Fit one bambi model at the Stage 6 L9 MCMC configuration.

    Parameters
    ----------
    formula : str
        Model formula (see `build_formula`).
    data : pd.DataFrame
        Rows to fit on.
    family : str, optional
        Bambi family name (default `"t"`, L1).

    Returns
    -------
    bambi.Model, arviz.InferenceData
        The fitted model and its posterior (with `log_likelihood` for LOO).
    """
    model = bmb.Model(formula, data, family=family, priors=build_priors())
    idata = model.fit(
        draws=DRAWS, tune=TUNE, chains=CHAINS, target_accept=TARGET_ACCEPT,
        random_seed=SEED, progressbar=False, idata_kwargs={"log_likelihood": True}, cores=CORES,
    )
    return model, idata


def reference_grid():
    """The 6 film x group cells (TD rows 0-2, ASD rows 3-5) used for every contrast.

    Any covariate in `EXTRA_TERMS` is held at its centred reference (0, since
    covariates are mean-centred) so contrasts read at that reference value.

    Returns
    -------
    pd.DataFrame
        6 rows: `film` (categorical, ordered as `FILMS`), `group`
        (categorical, ordered as `GROUPS`), plus any covariate columns at 0.
    """
    rows = [{"film": f, "group": g} for g in GROUPS for f in FILMS]
    grid = pd.DataFrame(rows)
    grid["film"] = pd.Categorical(grid["film"], categories=FILMS)
    grid["group"] = pd.Categorical(grid["group"], categories=GROUPS)
    for term in EXTRA_TERMS:
        grid[term] = 0.0
    return grid


def predict_cell_means(model, idata, grid):
    """Posterior draws of the population-level mean at each `reference_grid` row.

    Uses `include_group_specific=False` so the six cells are marginal
    (population-level) means, not tied to any specific dyad -- this is what
    makes the film/group contrasts below well-defined regardless of the sum
    contrast coding used to fit the model.

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    grid : pd.DataFrame
        From `reference_grid()`.

    Returns
    -------
    xarray.DataArray
        Dims `(chain, draw, __obs__)`, `__obs__` indexing `grid`'s rows.
    """
    preds = model.predict(idata, data=grid, kind="response_params", inplace=False, include_group_specific=False)
    return preds.posterior["mu"]


def cell_indices(grid, film=None, group=None):
    """Row positions of `grid` matching the given `film`/`group` (either may be None = any).

    Parameters
    ----------
    grid : pd.DataFrame
        From `reference_grid()`.
    film, group : str or None
        Value to match, or None to match all.

    Returns
    -------
    list of int
        Positional indices into `grid`.
    """
    mask = pd.Series(True, index=grid.index)
    if film is not None:
        mask &= grid["film"] == film
    if group is not None:
        mask &= grid["group"] == group
    return list(grid.index[mask])


def summarize_draws(draws):
    """Posterior mean, 95% HDI, and directional probabilities for a draws array (L4).

    Parameters
    ----------
    draws : xarray.DataArray or np.ndarray
        Posterior draws of one scalar contrast.

    Returns
    -------
    dict
        `estimate`, `hdi_low`, `hdi_high`, `p_gt0`, `p_lt0`.
    """
    flat = np.asarray(draws).flatten()
    hdi = az.hdi(flat, hdi_prob=HDI_PROB)
    return {
        "estimate": float(flat.mean()),
        "hdi_low": float(hdi[0]),
        "hdi_high": float(hdi[1]),
        "p_gt0": float((flat > 0).mean()),
        "p_lt0": float((flat < 0).mean()),
    }


def compute_contrasts(model, idata, grid):
    """The standard L3 contrast set read off one model's posterior (estimated marginal means).

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    grid : pd.DataFrame
        From `reference_grid()`.

    Returns
    -------
    dict of dict
        Keys `group_effect`, `film_contrast_overall`, `film_contrast_TD`,
        `film_contrast_ASD`, `interaction_film_group`, `grand_mean` -- each a
        `summarize_draws` dict of signed posterior draws.
    """
    mu = predict_cell_means(model, idata, grid)

    def cell(film=None, group=None):
        return mu.isel(__obs__=cell_indices(grid, film, group)).mean("__obs__")

    def film_contrast(group=None):
        return cell("Incredibles", group) - (cell("Peppa", group) + cell("Brave", group)) / 2

    group_effect = (cell(group="ASD") - cell(group="TD")).values.flatten()
    film_overall = film_contrast(group=None).values.flatten()
    film_td = film_contrast(group="TD").values.flatten()
    film_asd = film_contrast(group="ASD").values.flatten()
    interaction = film_asd - film_td
    grand_mean = mu.isel(__obs__=list(grid.index)).mean("__obs__").values.flatten()

    return {
        "group_effect": summarize_draws(group_effect),
        "film_contrast_overall": summarize_draws(film_overall),
        "film_contrast_TD": summarize_draws(film_td),
        "film_contrast_ASD": summarize_draws(film_asd),
        "interaction_film_group": summarize_draws(interaction),
        "grand_mean": summarize_draws(grand_mean),
    }


def back_transform(summary, sd):
    """Rescale a standardized contrast summary to raw Delta-units (D3).

    Valid for pure differences (all contrasts in `compute_contrasts` except
    `grand_mean`, which is an absolute level and needs `+ mean` too --
    handled by the caller, not here).

    Parameters
    ----------
    summary : dict
        A `summarize_draws` output.
    sd : float
        The edge's raw-scale standard deviation used to standardize.

    Returns
    -------
    dict
        Same keys, `estimate`/`hdi_low`/`hdi_high` scaled by `sd`;
        `p_gt0`/`p_lt0` unchanged (scaling by a positive number preserves sign).
    """
    return {
        "estimate": summary["estimate"] * sd,
        "hdi_low": summary["hdi_low"] * sd,
        "hdi_high": summary["hdi_high"] * sd,
        "p_gt0": summary["p_gt0"],
        "p_lt0": summary["p_lt0"],
    }


def convergence_row(model_label, idata, n_rows, n_dropped_unstable):
    """One `stage06_diagnostics.csv` row: Rhat/ESS/divergences/LOO for one fitted model (L9).

    Parameters
    ----------
    model_label : str
    idata : arviz.InferenceData
    n_rows : int
        Rows the model was fit on (after the L8 `real_stable` filter).
    n_dropped_unstable : int
        Rows dropped by the L8 filter (0 in the current dataset).

    Returns
    -------
    dict
        Row for `stage06_diagnostics.csv`, including a `pass_l9` boolean.
    """
    summary = az.summary(idata)
    n_divergent = int(idata.sample_stats["diverging"].values.sum())
    loo = az.loo(idata, pointwise=True)
    max_rhat = float(summary["r_hat"].max())
    min_bulk_ess = float(summary["ess_bulk"].min())
    min_tail_ess = float(summary["ess_tail"].min())
    max_pareto_k = float(np.max(loo.pareto_k.values))
    pass_l9 = (max_rhat < RHAT_MAX) and (min_bulk_ess > ESS_MIN) and (min_tail_ess > ESS_MIN) \
        and (n_divergent == 0) and (max_pareto_k < PARETO_K_MAX)
    return {
        "model": model_label,
        "max_rhat": max_rhat,
        "min_bulk_ess": min_bulk_ess,
        "min_tail_ess": min_tail_ess,
        "n_divergent": n_divergent,
        "loo_elpd": float(loo.elpd_loo),
        "loo_se": float(loo.se),
        "max_pareto_k": max_pareto_k,
        "n_rows": n_rows,
        "n_dropped_unstable": n_dropped_unstable,
        "pass_l9": bool(pass_l9),
    }


def safe_label(edge):
    """Filesystem-safe stem for an `"a->b"` edge string."""
    return edge.replace(":", "").replace("->", "_to_")


def plot_forest(rows, title):
    """Horizontal forest plot (posterior mean + 95% HDI) for a list of named contrasts.

    Parameters
    ----------
    rows : list of dict
        Each with `label`, `estimate`, `hdi_low`, `hdi_high`.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, axis = plt.subplots(figsize=(6.5, 0.6 * len(rows) + 1.2))
    ys = np.arange(len(rows))
    estimates = [r["estimate"] for r in rows]
    lo_err = [r["estimate"] - r["hdi_low"] for r in rows]
    hi_err = [r["hdi_high"] - r["estimate"] for r in rows]
    axis.errorbar(estimates, ys, xerr=[lo_err, hi_err], fmt="o", color="black", capsize=3)
    axis.axvline(0, color="red", linestyle="--", linewidth=1)
    axis.set_yticks(ys)
    axis.set_yticklabels([r["label"] for r in rows])
    axis.invert_yaxis()
    axis.set_xlabel("estimate (95% HDI)")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_ppc_figure(model, idata, title):
    """Posterior-predictive density overlay (L9 `pp_check`).

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    pps = model.predict(idata, kind="response", inplace=False)
    az.plot_ppc(pps, num_pp_samples=100)
    figure = plt.gcf()
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_pareto_k_figure(loo_result, title):
    """Pareto-k diagnostic scatter with the 0.7 warning line (L9).

    Parameters
    ----------
    loo_result : arviz.stats.ELPDData
        From `az.loo(idata, pointwise=True)`.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, axis = plt.subplots(figsize=(5, 3.5))
    pareto_k = loo_result.pareto_k.values
    axis.scatter(np.arange(len(pareto_k)), pareto_k, s=12)
    axis.axhline(PARETO_K_MAX, color="red", linestyle="--")
    axis.set_xlabel("observation index")
    axis.set_ylabel("pareto k")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_edge_funnel(idata, tag):
    """Save a plot_pair of each (edge offset, edge SD) pair with divergences flagged.

    Diagnoses the (1|edge)/(...|edge) funnel: divergences bunched at small SD
    are the classic neck, confirming the geometry is the cause. Loud if the
    model carries no edge SD variable (a wiring error by the caller).

    Parameters
    ----------
    idata : arviz.InferenceData
        A fitted D6 model's posterior (must include `sample_stats.diverging`).
    tag : str
        Slug for the output filename (e.g. "pooled_D2", "M1").

    Returns
    -------
    list of Path
        Saved figure paths, one per edge `*_sigma` variable found.
    """
    sigma_vars = [v for v in idata.posterior.data_vars if v.endswith("_sigma") and "edge" in v]
    assert sigma_vars, f"[{tag}] no edge '*_sigma' variable in posterior -- model has no (...|edge) term?"
    saved_paths = []
    for sigma_var in sigma_vars:
        offset_var = sigma_var[: -len("_sigma")]
        if offset_var not in idata.posterior.data_vars:
            continue  # a sigma with no matching offset vector (rare); skip this one, keep the others
        ax = az.plot_pair(
            idata, var_names=[sigma_var, offset_var], divergences=True,
            marginals=False, kind="scatter", scatter_kwargs={"alpha": 0.15},
        )
        fig = ax.ravel()[0].figure if hasattr(ax, "ravel") else ax.figure
        fig.suptitle(f"{tag}: {sigma_var} funnel (divergences in orange)", fontsize=9)
        fig.tight_layout()
        safe = sigma_var.replace("|", "_").replace(":", "-").replace("(", "").replace(")", "").replace(", ", "_").replace(" ", "")
        out_path = QC_DIR / f"funnel_{tag}_{safe}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        saved_paths.append(out_path)
    return saved_paths


# ---------------------------------------------------------------------------
# 1. Load and confirm schema (per this prompt's own instruction: confirm the
#    real file before trusting the spec's quoted schema)
# ---------------------------------------------------------------------------
delta_table = load_delta_table(INPUT_CSV)
required_columns = {
    "dyad_id", "film", "source", "target", "edge", "edge_class", "group", "age_months",
    "real_ffdtf", "null_median", "null_std", "n_null", "delta_dtf", "z_vs_surrogate", "real_stable",
}
assert required_columns.issubset(delta_table.columns), f"missing columns: {required_columns - set(delta_table.columns)}"
print(f"Loaded {INPUT_CSV} : {delta_table.shape}, engine={ENGINE} (D0)")

other_edges = sorted(set(delta_table["edge"].unique()) - {edge for edge, _ in EMPHASIS_EDGES})
edges_to_fit = list(EMPHASIS_EDGES) + ([(edge, "other") for edge in other_edges] if FIT_OTHER_EDGES else [])

contrast_rows = []
diagnostics_rows = []
forest_rows_by_edge = {}
edge_sd_lookup = {}

# ---------------------------------------------------------------------------
# 2. Per-edge models (confirmatory, D2 default)
# ---------------------------------------------------------------------------
for edge, edge_class in edges_to_fit:
    subset = edge_subset(delta_table, [edge])
    n_before = len(subset)
    subset = subset[subset["real_stable"]]
    n_dropped = n_before - len(subset)  # L8, reported even when zero
    subset = add_covariates(subset)
    subset, edge_stats = standardize_within_edge(subset, DV_MAIN)
    sd = float(edge_stats.loc[edge_stats["edge"] == edge, "sd"].iloc[0])
    edge_sd_lookup[edge] = sd

    formula = build_formula(f"{DV_MAIN}_z")
    model, idata = fit_model(formula, subset)
    az.to_netcdf(idata, MODELS_DIR / f"{safe_label(edge)}__baseline.nc")

    grid = reference_grid()
    contrasts = compute_contrasts(model, idata, grid)
    diagnostics_rows.append(convergence_row(f"{edge} ({edge_class})", idata, len(subset), n_dropped))

    forest_rows = []
    for contrast_name in ("group_effect", "film_contrast_overall", "film_contrast_TD", "film_contrast_ASD", "interaction_film_group"):
        summary_std = contrasts[contrast_name]
        summary_raw = back_transform(summary_std, sd)
        for unit, summary in (("std", summary_std), ("raw", summary_raw)):
            contrast_rows.append({
                "model": edge, "edge": edge, "edge_class": edge_class, "dv": DV_MAIN,
                "contrast": contrast_name, "unit": unit, **summary,
            })
        forest_rows.append({"label": f"{contrast_name} (raw)", **summary_raw})
    forest_rows_by_edge[edge] = forest_rows

    forest_fig = plot_forest(forest_rows, f"{edge} ({edge_class}) -- contrasts, raw Delta-units")
    forest_fig.savefig(QC_DIR / f"{safe_label(edge)}_forest.png")
    plt.close(forest_fig)

    ppc_fig = plot_ppc_figure(model, idata, f"{edge} pp_check")
    ppc_fig.savefig(QC_DIR / f"{safe_label(edge)}_ppcheck.png")
    plt.close(ppc_fig)

    loo_result = az.loo(idata, pointwise=True)
    loo_fig = plot_pareto_k_figure(loo_result, f"{edge} Pareto-k")
    loo_fig.savefig(QC_DIR / f"{safe_label(edge)}_loo.png")
    plt.close(loo_fig)

    print(f"Fit {edge} ({edge_class}): n={len(subset)} dropped_unstable={n_dropped} "
          f"max_rhat={diagnostics_rows[-1]['max_rhat']:.3f} n_divergent={diagnostics_rows[-1]['n_divergent']} "
          f"max_pareto_k={diagnostics_rows[-1]['max_pareto_k']:.2f} pass_l9={diagnostics_rows[-1]['pass_l9']}")

# ---------------------------------------------------------------------------
# 3. Asymmetry track (L6): H2 and H4, primary DV z_vs_surrogate + sensitivity delta_dtf
# ---------------------------------------------------------------------------
ASYMMETRY_SPECS = [
    ("H2", "cg:ROI->child:ROI", "child:ROI->cg:ROI"),
    ("H4", "cg:HRV->child:HRV", "child:HRV->cg:HRV"),
]
ASYMMETRY_DV_TRACKS = [("z_vs_surrogate", "native", "z"), (DV_MAIN, "std", "delta")]  # primary, sensitivity (L6)
asymmetry_summary_rows = []  # for run summary / gate primary section

for hypothesis, forward_edge, reverse_edge in ASYMMETRY_SPECS:
    for dv_col, unit_label, tag in ASYMMETRY_DV_TRACKS:
        asym = asymmetry_dv(delta_table, forward_edge, reverse_edge, dv_col)
        n_before = len(asym)
        asym = asym[asym["real_stable"]]
        n_dropped = n_before - len(asym)
        asym = add_covariates(asym)

        sd = None
        if unit_label == "std":
            asym = asym.assign(edge=f"{hypothesis}_asym_{tag}")
            asym, asym_stats = standardize_within_edge(asym, "asym")
            sd = float(asym_stats["sd"].iloc[0])
            fit_col = "asym_z"
        else:
            fit_col = "asym"

        formula = build_formula(fit_col)
        model, idata = fit_model(formula, asym)
        model_label = f"asym_{hypothesis}__{tag}"
        az.to_netcdf(idata, MODELS_DIR / f"{model_label}.nc")

        grid = reference_grid()
        contrasts = compute_contrasts(model, idata, grid)
        diagnostics_rows.append(convergence_row(f"{model_label} ({hypothesis} caregiver-leading, {dv_col})", idata, len(asym), n_dropped))

        forest_rows = []
        for contrast_name, summary_std in list(contrasts.items()):
            summary_report = back_transform(summary_std, sd) if sd is not None else summary_std
            contrast_rows.append({
                "model": model_label, "edge": f"{forward_edge} vs {reverse_edge}", "edge_class": f"{hypothesis}_asymmetry",
                "dv": dv_col, "contrast": contrast_name, "unit": ("raw" if sd is not None else unit_label), **summary_report,
            })
            label = "intercept_caregiver_leading" if contrast_name == "grand_mean" else contrast_name
            forest_rows.append({"label": f"{label} ({dv_col})", **summary_report})
            if contrast_name == "grand_mean":
                asymmetry_summary_rows.append({"hypothesis": hypothesis, "dv": dv_col, "tag": tag, **summary_report})

        forest_fig = plot_forest(forest_rows, f"{hypothesis} asymmetry ({dv_col}) -- caregiver-leading if intercept > 0")
        forest_fig.savefig(QC_DIR / f"{model_label}_forest.png")
        plt.close(forest_fig)

        ppc_fig = plot_ppc_figure(model, idata, f"{model_label} pp_check")
        ppc_fig.savefig(QC_DIR / f"{model_label}_ppcheck.png")
        plt.close(ppc_fig)

        print(f"Fit {model_label}: n={len(asym)} dropped_unstable={n_dropped} "
              f"max_rhat={diagnostics_rows[-1]['max_rhat']:.3f} pass_l9={diagnostics_rows[-1]['pass_l9']}")

# ---------------------------------------------------------------------------
# 4. Pooled cross-edge model (D2 companion, shrinkage cross-check)
# ---------------------------------------------------------------------------
pooled_diagnostics = None
pooled_contrasts = None
if FIT_POOLED_MODEL:
    pooled_frames = []
    for edge, edge_class in EMPHASIS_EDGES:
        subset = edge_subset(delta_table, [edge])
        subset = subset[subset["real_stable"]]
        subset = add_covariates(subset)
        subset, _ = standardize_within_edge(subset, DV_MAIN)
        pooled_frames.append(subset)
    pooled_data = pd.concat(pooled_frames, ignore_index=True)
    pooled_data["edge"] = pd.Categorical(pooled_data["edge"])

    pooled_formula = build_formula(f"{DV_MAIN}_z", extra_grouping="edge")
    pooled_model, pooled_idata = fit_model(pooled_formula, pooled_data)
    az.to_netcdf(pooled_idata, MODELS_DIR / "pooled_emphasis.nc")

    pooled_grid = reference_grid()
    pooled_contrasts = compute_contrasts(pooled_model, pooled_idata, pooled_grid)
    pooled_diagnostics = convergence_row("pooled_emphasis (6 edges, (1|edge))", pooled_idata, len(pooled_data), 0)
    diagnostics_rows.append(pooled_diagnostics)
    pooled_funnel_paths = plot_edge_funnel(pooled_idata, "pooled_D2")

    pooled_forest_rows = [
        {"label": name, **pooled_contrasts[name]}
        for name in ("group_effect", "film_contrast_overall", "film_contrast_TD", "film_contrast_ASD", "interaction_film_group")
    ]
    for row in pooled_forest_rows:
        contrast_rows.append({
            "model": "pooled_emphasis", "edge": "pooled(6 emphasis edges)", "edge_class": "pooled",
            "dv": DV_MAIN, "contrast": row["label"], "unit": "std",
            "estimate": row["estimate"], "hdi_low": row["hdi_low"], "hdi_high": row["hdi_high"],
            "p_gt0": row["p_gt0"], "p_lt0": row["p_lt0"],
        })
    pooled_forest_fig = plot_forest(pooled_forest_rows, "Pooled emphasis-edge model (standardized, shrinkage cross-check)")
    pooled_forest_fig.savefig(QC_DIR / "pooled_forest.png")
    plt.close(pooled_forest_fig)

    pooled_ppc_fig = plot_ppc_figure(pooled_model, pooled_idata, "pooled_emphasis pp_check")
    pooled_ppc_fig.savefig(QC_DIR / "pooled_ppcheck.png")
    plt.close(pooled_ppc_fig)

    print(f"Fit pooled_emphasis: n={len(pooled_data)} max_rhat={pooled_diagnostics['max_rhat']:.3f} "
          f"pass_l9={pooled_diagnostics['pass_l9']}")

# ---------------------------------------------------------------------------
# 5. Tidy CSV outputs
# ---------------------------------------------------------------------------
contrasts_df = pd.DataFrame(contrast_rows)
contrasts_df.to_csv(OUTPUT_DIR / "stage06_contrasts.csv", index=False)

diagnostics_df = pd.DataFrame(diagnostics_rows)
diagnostics_df.to_csv(OUTPUT_DIR / "stage06_diagnostics.csv", index=False)

primary_rows = []
p_like_values = []
for edge, contrast_name, tag in PRIMARY_FAMILY:
    row = contrasts_df[(contrasts_df["edge"] == edge) & (contrasts_df["contrast"] == contrast_name) & (contrasts_df["unit"] == "raw")].iloc[0]
    p_like = 2 * min(row["p_gt0"], row["p_lt0"])
    p_like_values.append(p_like)
    primary_rows.append({"family": "primary", "label": f"{tag} group_effect", **row.to_dict(), "p_like_2sided": p_like})
primary_summary_df = pd.DataFrame(primary_rows)
primary_summary_df["bh_fdr"] = bh_fdr(p_like_values)  # L7: BH applied within this named 2-member family only

for entry in asymmetry_summary_rows:
    label = f"{entry['hypothesis']}_asym_intercept_{entry['tag']}"
    primary_summary_df = pd.concat([primary_summary_df, pd.DataFrame([{
        "family": "asymmetry", "label": label, "model": f"asym_{entry['hypothesis']}__{entry['tag']}",
        "edge": None, "edge_class": f"{entry['hypothesis']}_asymmetry", "dv": entry["dv"], "contrast": "grand_mean",
        "unit": "raw" if entry["tag"] == "delta" else "native", "estimate": entry["estimate"],
        "hdi_low": entry["hdi_low"], "hdi_high": entry["hdi_high"], "p_gt0": entry["p_gt0"], "p_lt0": entry["p_lt0"],
        "p_like_2sided": np.nan, "bh_fdr": np.nan,
    }])], ignore_index=True)
primary_summary_df.to_csv(OUTPUT_DIR / "stage06_primary_summary.csv", index=False)

# ---------------------------------------------------------------------------
# 6. Primary-contrasts and asymmetry summary figures
# ---------------------------------------------------------------------------
primary_forest_rows = [
    {"label": row["label"], "estimate": row["estimate"], "hdi_low": row["hdi_low"], "hdi_high": row["hdi_high"]}
    for row in primary_rows
]
primary_fig = plot_forest(primary_forest_rows, "H2/H4 primary group effects (raw Delta-units)")
primary_fig.savefig(QC_DIR / "primary_contrasts.png")
plt.close(primary_fig)

asymmetry_forest_rows = [
    {"label": f"{entry['hypothesis']} caregiver-leading ({entry['dv']})", "estimate": entry["estimate"],
     "hdi_low": entry["hdi_low"], "hdi_high": entry["hdi_high"]}
    for entry in asymmetry_summary_rows
]
asymmetry_fig = plot_forest(asymmetry_forest_rows, "H2/H4 asymmetry: intercept > 0 = caregiver-leading")
asymmetry_fig.savefig(QC_DIR / "asymmetry.png")
plt.close(asymmetry_fig)

# ---------------------------------------------------------------------------
# 7. Run summary
# ---------------------------------------------------------------------------
summary_lines = [
    f"engine (D0) = {ENGINE}  |  RE structure (D1) = (1|dyad_id) only -- (1|child)+(1|caregiver) dropped, not identifiable (members 1:1 with dyad)",
    f"H4 group-effect sign is reported two-sided; TD>ASD is NOT assumed (D5).",
    f"MCMC config (Stage 6 addendum, L-conv-1): TARGET_ACCEPT={TARGET_ACCEPT} (was 0.95), TUNE={TUNE} (was 1000) "
    f"-- applied to every model in this run, per-edge through pooled/M1/M2/M3.",
    f"Sampler cores: cores={CORES} (sequential, no worker subprocesses) -- macOS fix: fork() of a process "
    f"with live Accelerate/vecLib BLAS threads segfaults the child pre-exec, and spawn/forkserver both "
    f"re-import this unguarded top-level script as __main__ during worker bootstrap, recursing into the "
    f"whole pipeline; applied to every model.fit() call in this run.",
    "",
    "Per-edge convergence (L9: rhat<{:.2f}, ess>{}, 0 divergences, pareto_k<{:.1f}):".format(RHAT_MAX, ESS_MIN, PARETO_K_MAX),
]
for row in diagnostics_rows:
    status = "PASS" if row["pass_l9"] else "FAIL"
    summary_lines.append(
        f"  [{status}] {row['model']}: n={row['n_rows']} dropped_unstable={row['n_dropped_unstable']} "
        f"max_rhat={row['max_rhat']:.3f} min_ess_bulk={row['min_bulk_ess']:.0f} min_ess_tail={row['min_tail_ess']:.0f} "
        f"n_divergent={row['n_divergent']} max_pareto_k={row['max_pareto_k']:.2f} loo_elpd={row['loo_elpd']:.1f}"
    )

summary_lines.append("")
summary_lines.append("H2/H4 primary group effects (raw Delta-units), BH-FDR over this 2-member family only:")
for _, row in primary_summary_df[primary_summary_df["family"] == "primary"].iterrows():
    summary_lines.append(
        f"  {row['label']}: estimate={row['estimate']:+.5f} HDI95=[{row['hdi_low']:+.5f}, {row['hdi_high']:+.5f}] "
        f"P(>0)={row['p_gt0']:.3f} P(<0)={row['p_lt0']:.3f} bh_fdr={row['bh_fdr']:.3f}"
    )

summary_lines.append("")
summary_lines.append("H2/H4 asymmetry (caregiver-leading if intercept > 0):")
for _, row in primary_summary_df[primary_summary_df["family"] == "asymmetry"].iterrows():
    summary_lines.append(
        f"  {row['label']}: estimate={row['estimate']:+.5f} HDI95=[{row['hdi_low']:+.5f}, {row['hdi_high']:+.5f}] "
        f"P(>0)={row['p_gt0']:.3f} P(<0)={row['p_lt0']:.3f}"
    )

failed_models = [row["model"] for row in diagnostics_rows if not row["pass_l9"]]
summary_lines.append("")
if failed_models:
    summary_lines.append(f"L9 FAILURES ({len(failed_models)}): {', '.join(failed_models)} -- consider the 2-variable fallback (project note S11).")
else:
    summary_lines.append("L9: all models pass convergence/PPC/LOO criteria.")

print("\n" + "\n".join(summary_lines))
(OUTPUT_DIR / "stage06_run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

# ---------------------------------------------------------------------------
# 8. Interactive HTML gate (mirrors Stage 5's surrogate_gate.html style)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 6 group model gate</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 1.5em; color: #1a1a1a; }
  h1 { font-size: 1.3em; }
  select { font-size: 1em; padding: 0.3em; margin-bottom: 1em; }
  .edge-panel { display: none; }
  .edge-panel.active { display: block; }
  .header-line { font-family: monospace; margin-bottom: 0.5em; }
  .badge { padding: 0.1em 0.5em; border-radius: 3px; color: white; font-size: 0.85em; }
  .badge-ok { background: #2e8b2e; }
  .badge-bad { background: #b03030; }
  .row { display: flex; flex-wrap: wrap; gap: 0.5em; }
  .row img { max-width: 900px; border: 1px solid #ccc; }
  h2, h3 { margin-bottom: 0.3em; }
  #summary { margin-bottom: 1.5em; white-space: pre; font-family: monospace; }
  table.contrasts, table.diag { border-collapse: collapse; font-size: 0.85em; margin-bottom: 1em; }
  table.contrasts th, table.contrasts td, table.diag th, table.diag td { border: 1px solid #ccc; padding: 0.2em 0.5em; text-align: right; }
  table.contrasts th:first-child, table.contrasts td:first-child,
  table.diag th:first-child, table.diag td:first-child { text-align: left; }
  .callout { background: #fff6e0; border: 1px solid #e0c060; padding: 0.6em 1em; margin-bottom: 1em; }
</style>
</head>
<body>
<h1>Stage 6 group model gate</h1>
<p class="header-line">
   engine (D0) = <b>__ENGINE__</b> (brms/cmdstanr not installed; bambi+arviz already were -- ratified by project owner).
   DV = standardized <b>delta_dtf</b> (per-edge z-score, D3), family=<b>Student-t</b>,
   formula=<b>film * group (sum contrasts) + (1|dyad_id)</b> (D1), planned film contrast =
   <b>Incredibles vs (Peppa+Brave)/2</b> (two-sided), FDR family (L7) = {H2_primary, H4_primary} group effects only.<br>
   input = <b>__INPUT_CSV__</b>. ROI (edges named <code>*:ROI</code>) = <b>__ROI_LABEL__</b>,
   electrodes <b>__ROI_CHANNELS__</b> (must match Stage 2's ROI_LABEL/ROI_CHANNELS).
   estimator = <b>__ESTIMATOR__</b> (must match Stage 5's ESTIMATOR -- not read from INPUT_CSV, which carries no
   ESTIMATOR column). box_cox_lambda = <b>__BOX_COX_LAMBDA__</b> (-1 = no transform; must match Stage 5's
   BOX_COX_LAMBDA, same not-read-from-data caveat).
</p>
<div class="callout">
  <b>D1 (correctness fix):</b> the plan's <code>(1|dyad_id) + (1|child_id) + (1|caregiver_id)</code> was dropped to
  <code>(1|dyad_id)</code> only -- in this design child and caregiver are 1:1 with dyad, so the three grouping factors
  are the same partition of the rows and are not jointly identifiable. Flagging for Jarek to ratify or override.<br>
  <b>H4 sign:</b> reported two-sided everywhere -- <code>P(ASD&gt;TD)</code> and <code>P(ASD&lt;TD)</code> both shown.
  TD &gt; ASD is NOT assumed (D5); negative interpersonal HRV synchrony can be adaptive.
</div>
<h2>Convergence (L9 gate)</h2>
<div id="diagnostics">__DIAGNOSTICS_TABLE__</div>
<h2>Primary contrasts</h2>
<div class="row">
  <img src="qc/primary_contrasts.png" alt="H2/H4 primary group effects">
  <img src="qc/asymmetry.png" alt="H2/H4 asymmetry">
</div>
<div id="primary">__PRIMARY_TABLE__</div>
<h2>Run summary</h2>
<div id="summary">__SUMMARY__</div>
<h2>Per-edge diagnostics</h2>
<label for="edge-select">Edge: </label>
<select id="edge-select"></select>
<div id="panels">__PANELS__</div>
__POOLED_SECTION__
<script>
const edgeIds = __EDGE_IDS_JSON__;
const select = document.getElementById('edge-select');
for (const id of edgeIds) {
  const opt = document.createElement('option');
  opt.value = id; opt.textContent = id;
  select.appendChild(opt);
}
function showEdge(id) {
  document.querySelectorAll('.edge-panel').forEach(p => p.classList.remove('active'));
  const panel = document.getElementById('panel-' + id);
  if (panel) panel.classList.add('active');
}
select.onchange = () => showEdge(select.value);
if (edgeIds.length) showEdge(edgeIds[0]);
</script>
</body>
</html>
"""


def render_diagnostics_table(rows):
    """Render `stage06_diagnostics.csv` rows as an HTML table with L9 pass/fail badges."""
    html = ['<table class="diag"><tr><th>model</th><th>n</th><th>dropped</th><th>max_rhat</th>'
            '<th>min_ess_bulk</th><th>min_ess_tail</th><th>n_divergent</th><th>max_pareto_k</th>'
            '<th>loo_elpd</th><th>status</th></tr>']
    for row in rows:
        badge_class = "badge-ok" if row["pass_l9"] else "badge-bad"
        badge_text = "PASS" if row["pass_l9"] else "FAIL"
        html.append(
            f'<tr><td>{row["model"]}</td><td>{row["n_rows"]}</td><td>{row["n_dropped_unstable"]}</td>'
            f'<td>{row["max_rhat"]:.3f}</td><td>{row["min_bulk_ess"]:.0f}</td><td>{row["min_tail_ess"]:.0f}</td>'
            f'<td>{row["n_divergent"]}</td><td>{row["max_pareto_k"]:.2f}</td><td>{row["loo_elpd"]:.1f}</td>'
            f'<td><span class="badge {badge_class}">{badge_text}</span></td></tr>'
        )
    html.append("</table>")
    return "\n".join(html)


def render_contrast_table(rows_df):
    """Render a set of `stage06_contrasts.csv` rows (one edge, raw units) as an HTML table."""
    html = ['<table class="contrasts"><tr><th>contrast</th><th>estimate</th><th>HDI95 low</th>'
            '<th>HDI95 high</th><th>P(&gt;0)</th><th>P(&lt;0)</th></tr>']
    for _, row in rows_df.iterrows():
        html.append(
            f'<tr><td>{row["contrast"]}</td><td>{row["estimate"]:+.5f}</td><td>{row["hdi_low"]:+.5f}</td>'
            f'<td>{row["hdi_high"]:+.5f}</td><td>{row["p_gt0"]:.3f}</td><td>{row["p_lt0"]:.3f}</td></tr>'
        )
    html.append("</table>")
    return "\n".join(html)


def render_primary_table(df):
    """Render `stage06_primary_summary.csv` (both families) as an HTML table, with BH-FDR shown for the primary family."""
    html = ['<table class="contrasts"><tr><th>label</th><th>dv</th><th>unit</th><th>estimate</th>'
            '<th>HDI95 low</th><th>HDI95 high</th><th>P(&gt;0)</th><th>P(&lt;0)</th><th>bh_fdr</th></tr>']
    for _, row in df.iterrows():
        bh_text = f'{row["bh_fdr"]:.3f}' if pd.notna(row["bh_fdr"]) else "&mdash;"
        html.append(
            f'<tr><td>{row["label"]}</td><td>{row["dv"]}</td><td>{row["unit"]}</td>'
            f'<td>{row["estimate"]:+.5f}</td><td>{row["hdi_low"]:+.5f}</td><td>{row["hdi_high"]:+.5f}</td>'
            f'<td>{row["p_gt0"]:.3f}</td><td>{row["p_lt0"]:.3f}</td><td>{bh_text}</td></tr>'
        )
    html.append("</table>")
    return "\n".join(html)


def render_edge_panel(edge, edge_class):
    """Render one emphasis edge's QC panel: forest/ppcheck/loo images + its contrast table."""
    label = safe_label(edge)
    rows_df = contrasts_df[(contrasts_df["edge"] == edge) & (contrasts_df["unit"] == "raw")]
    tag = "hypothesis-generating -- no correction" if edge_class in ("exploratory",) else edge_class
    html = [f'<div class="edge-panel" id="panel-{edge}"><h2>{edge} <small>({tag})</small></h2>']
    html.append('<div class="row">'
                f'<img src="qc/{label}_forest.png" alt="{edge} forest">'
                f'<img src="qc/{label}_ppcheck.png" alt="{edge} pp_check">'
                f'<img src="qc/{label}_loo.png" alt="{edge} pareto-k">'
                '</div>')
    html.append(render_contrast_table(rows_df))
    html.append("</div>")
    return "\n".join(html)


panels_html = "\n".join(render_edge_panel(edge, edge_class) for edge, edge_class in edges_to_fit)
edge_ids = [edge for edge, _ in edges_to_fit]

pooled_section = ""
if FIT_POOLED_MODEL:
    pooled_rows_df = contrasts_df[contrasts_df["model"] == "pooled_emphasis"]
    pooled_section = (
        '<h2>Pooled cross-edge model (D2 shrinkage cross-check)</h2>'
        '<p>Standardized units only (edges have different raw scales); a companion to the per-edge results above, not a replacement.</p>'
        '<div class="row"><img src="qc/pooled_forest.png" alt="pooled forest">'
        '<img src="qc/pooled_ppcheck.png" alt="pooled pp_check"></div>'
        + render_contrast_table(pooled_rows_df)
    )

html = HTML_TEMPLATE.replace("__DIAGNOSTICS_TABLE__", render_diagnostics_table(diagnostics_rows))
html = html.replace("__PRIMARY_TABLE__", render_primary_table(primary_summary_df))
html = html.replace("__SUMMARY__", "\n".join(summary_lines))
html = html.replace("__PANELS__", panels_html)
html = html.replace("__POOLED_SECTION__", pooled_section)
html = html.replace("__EDGE_IDS_JSON__", json.dumps(edge_ids))
html = html.replace("__ENGINE__", ENGINE)
html = html.replace("__INPUT_CSV__", str(INPUT_CSV.relative_to(PROJECT_ROOT)))
html = html.replace("__ROI_LABEL__", ROI_LABEL)
html = html.replace("__ROI_CHANNELS__", "/".join(ROI_CHANNELS))
html = html.replace("__ESTIMATOR__", ESTIMATOR)
html = html.replace("__BOX_COX_LAMBDA__", str(BOX_COX_LAMBDA))
(OUTPUT_DIR / "group_model_gate.html").write_text(html, encoding="utf-8")
print(f"\nWrote gate to {OUTPUT_DIR / 'group_model_gate.html'}")

# ---------------------------------------------------------------------------
# D6 (open, exploratory) -- cross-edge localization of the film x group
# interaction. Appended after the Stage 6 gate above; nothing above this line
# is touched.
#
# The pooled model (Section 4) found a credible film x group interaction in
# standardized units, while the per-edge H2_primary model shows none -- so
# the pooled interaction is a cross-edge signal that does not live on the
# primary H2 edge. This is NOT a two-stage omnibus-then-post-hoc test: the
# pooled fixed effect and any per-edge re-test would be separate estimators
# with no shared multiplicity protection (the classic forking-paths failure
# mode). Instead this fits one richer hierarchy that lets the film x group
# interaction vary by edge (M1, primary) and, as a heavier sensitivity check,
# lets the whole film*group surface vary by edge (M2, gated by
# `FIT_M2_FULL_VARYING`). Both are compared to the interaction-fixed pooled
# model (M0 = `pooled_model`/`pooled_idata` from Section 4, reused unrefit)
# via `az.compare` (LOO); per-edge localization is read off M1's shrunk
# group-specific deviations, warranted only if letting the interaction vary
# actually improves predictive fit.
#
# Caveats (surfaced again in the gate section below):
#  - Exploratory, uncorrected. The interaction was never in the L7 FDR family
#    (PRIMARY_FAMILY = the two H2/H4 GROUP effects only). Every quantity here
#    is hypothesis-generating, not confirmatory.
#  - Six edges is very little for a variance component -- the edge-level SD
#    is estimated from 6 groups, so per-edge deviations are heavily shrunk
#    and the `az.compare` ranking is low-powered. A "win" for the varying
#    model is weak evidence.
#  - `(...|edge)` treats the 6 emphasis edges as exchangeable, which is
#    scientifically questionable (interbrain-EEG, interbrain-HRV, and cross
#    brain-heart edges are qualitatively different objects). A typed
#    grouping could be more defensible -- flagged as an open modelling
#    choice, not resolved here.
#  - Standardized units only (`unit="std"`), same as the pooled model -- no
#    raw back-transform across edges with different raw scales.
#  - Signed everywhere. Never abs() or clip any interaction/contrast.
# ---------------------------------------------------------------------------
if not FIT_POOLED_MODEL:
    raise RuntimeError("D6 requires the D2 pooled model -- set FIT_POOLED_MODEL=True.")

FIT_M2_FULL_VARYING = True  # D6 sensitivity: full film*group | edge (heavy, 6-edge-limited)

edge_class_lookup = dict(EMPHASIS_EDGES)


def fit_model_with_priors(formula, data, priors, required_group_terms, family="t"):
    """Fit one bambi model at the Stage 6 L9 MCMC config with an explicit priors dict.

    Mirrors `fit_model`, but takes `priors` directly instead of
    `build_priors()` -- the D6 varying-slope models below need extra
    group-specific prior keys `build_priors()` does not define. Deliberate
    small duplication of `fit_model`'s body (explicit over silently patching
    a shared function mid-file). Asserts that `required_group_terms` are
    exactly among the group-specific term names bambi assigned: a mismatch
    would mean a prior silently fell back to bambi's default, which must be
    caught loudly rather than fit anyway.

    Parameters
    ----------
    formula : str
    data : pd.DataFrame
    priors : dict
        Bambi `Prior` objects keyed by term name.
    required_group_terms : iterable of str
        Group-specific term names that must appear in the built model (e.g.
        `["C(film, Sum):C(group, Sum)|edge"]`).
    family : str, optional
        Bambi family name (default `"t"`, L1).

    Returns
    -------
    bambi.Model, arviz.InferenceData
    """
    model = bmb.Model(formula, data, family=family, priors=priors)
    term_names = set(model.distributional_components["mu"].terms.keys())
    missing = set(required_group_terms) - term_names
    assert not missing, f"prior keys not found among model terms {sorted(term_names)}: {missing}"
    idata = model.fit(
        draws=DRAWS, tune=TUNE, chains=CHAINS, target_accept=TARGET_ACCEPT,
        random_seed=SEED, progressbar=False, idata_kwargs={"log_likelihood": True}, cores=CORES,
    )
    return model, idata


def per_edge_contrasts(model, idata, edges, grid_categories):
    """Per-edge film/interaction contrasts from an edge-varying model (standardized units, signed).

    Predicts the 6 film x group cell means for every edge at one shared,
    in-sample `dyad_id` (so the dyad random intercept is identical within
    each edge and cancels exactly in every within-edge contrast) with
    `include_group_specific=True` (so the per-edge group-specific deviations
    enter), then reads `Incredibles - (Peppa+Brave)/2` within each group and
    their difference, per edge -- the same linear combinations as
    `compute_contrasts`, edge-scoped. Model-agnostic: reused unchanged for
    both M1 and M2.

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    edges : list of str
        Edge strings to score (the 6 emphasis edges).
    grid_categories : list of str
        `edge` categories exactly as used at fit time (category order
        matters for bambi's internal indexing).

    Returns
    -------
    dict of dict
        `edge -> {"interaction_film_group", "film_contrast_TD",
        "film_contrast_ASD"} -> summarize_draws() dict`.
    """
    ref_dyad = pooled_data["dyad_id"].iloc[0]
    rows = [{"film": f, "group": g, "edge": e, "dyad_id": ref_dyad}
            for e in edges for g in GROUPS for f in FILMS]
    grid = pd.DataFrame(rows)
    grid["film"] = pd.Categorical(grid["film"], categories=FILMS)
    grid["group"] = pd.Categorical(grid["group"], categories=GROUPS)
    grid["edge"] = pd.Categorical(grid["edge"], categories=grid_categories)
    grid["dyad_id"] = grid["dyad_id"].astype(str)
    for term in EXTRA_TERMS:  # D4 covariates held at centred reference
        grid[term] = 0.0

    preds = model.predict(idata, data=grid, kind="response_params", inplace=False, include_group_specific=True)
    mu = preds.posterior["mu"]  # dims (chain, draw, __obs__)

    def cell(edge, film=None, group=None):
        mask = grid["edge"] == edge
        if film is not None:
            mask &= grid["film"] == film
        if group is not None:
            mask &= grid["group"] == group
        return mu.isel(__obs__=list(grid.index[mask])).mean("__obs__")

    def film_contrast(edge, group):
        return cell(edge, "Incredibles", group) - (cell(edge, "Peppa", group) + cell(edge, "Brave", group)) / 2

    out = {}
    for e in edges:
        td = film_contrast(e, "TD").values.flatten()
        asd = film_contrast(e, "ASD").values.flatten()
        out[e] = {
            "film_contrast_TD": summarize_draws(td),
            "film_contrast_ASD": summarize_draws(asd),
            "interaction_film_group": summarize_draws(asd - td),  # signed, no abs()
        }
    return out


edge_categories = list(pooled_data["edge"].cat.categories)
localization_diagnostics_rows = []
group_sd_prior = bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfStudentT", nu=3, sigma=1))

# --- M1: interaction varies by edge (primary D6 model) ---------------------
m1_formula = build_formula(f"{DV_MAIN}_z", extra_grouping="edge") + " + (0 + C(film, Sum):C(group, Sum) | edge)"
m1_priors = {**build_priors(), "C(film, Sum):C(group, Sum)|edge": group_sd_prior}
m1_model, m1_idata = fit_model_with_priors(
    m1_formula, pooled_data, m1_priors, required_group_terms=["C(film, Sum):C(group, Sum)|edge"],
)
az.to_netcdf(m1_idata, MODELS_DIR / "pooled_emphasis_varying_interaction.nc")
localization_diagnostics_rows.append(convergence_row("pooled_varying_interaction (D6, M1)", m1_idata, len(pooled_data), 0))
m1_funnel_paths = plot_edge_funnel(m1_idata, "M1")
print(f"Fit D6 M1 (interaction varies by edge): max_rhat={localization_diagnostics_rows[-1]['max_rhat']:.3f} "
      f"n_divergent={localization_diagnostics_rows[-1]['n_divergent']} pass_l9={localization_diagnostics_rows[-1]['pass_l9']}")

# --- M2: full film*group varies by edge (sensitivity, gated) ---------------
m2_model, m2_idata = None, None
if FIT_M2_FULL_VARYING:
    m2_formula = (
        build_formula(f"{DV_MAIN}_z", extra_grouping="edge")
        + " + (0 + C(film, Sum) + C(group, Sum) + C(film, Sum):C(group, Sum) | edge)"
    )
    m2_priors = {
        **build_priors(),
        "C(film, Sum)|edge": group_sd_prior,
        "C(group, Sum)|edge": group_sd_prior,
        "C(film, Sum):C(group, Sum)|edge": group_sd_prior,
    }
    m2_model, m2_idata = fit_model_with_priors(
        m2_formula, pooled_data, m2_priors,
        required_group_terms=["C(film, Sum)|edge", "C(group, Sum)|edge", "C(film, Sum):C(group, Sum)|edge"],
    )
    az.to_netcdf(m2_idata, MODELS_DIR / "pooled_emphasis_varying_full.nc")
    localization_diagnostics_rows.append(convergence_row("pooled_varying_full (D6, M2)", m2_idata, len(pooled_data), 0))
    print(f"Fit D6 M2 (full film*group varies by edge): max_rhat={localization_diagnostics_rows[-1]['max_rhat']:.3f} "
          f"n_divergent={localization_diagnostics_rows[-1]['n_divergent']} pass_l9={localization_diagnostics_rows[-1]['pass_l9']}")

# --- M3: edge as a FIXED factor (no pooling) -- D-conv-A --------------------
# Un-shrunk counterpart to M0's full pooling (M1 reads between the two). By
# construction there is no (1|edge)/(...|edge) hyperprior, so M3 cannot
# funnel; plot_edge_funnel is deliberately NOT called on it (its assert would
# correctly fire on a model with no edge SD term).
m3_model, m3_idata = None, None
if FIT_EDGE_AS_FIXED:
    extra = "".join(f" + {t}" for t in EXTRA_TERMS)
    m3_formula = f"{DV_MAIN}_z ~ C(film, Sum) * C(group, Sum) * C(edge, Sum){extra} + (1|dyad_id)"
    # Fixed three-way crossing -> enumerate the common terms bambi actually
    # builds and set each to Normal(0,1); catch any silently-defaulted term
    # loudly rather than fitting with an unreviewed default prior.
    m3_probe = bmb.Model(m3_formula, pooled_data, family="t")
    common_terms = [
        name for name, term in m3_probe.distributional_components["mu"].terms.items()
        if not isinstance(term, GroupSpecificTerm) and name != "Intercept"
    ]
    m3_priors = {
        "Intercept": bmb.Prior("Normal", mu=0, sigma=1),
        "1|dyad_id": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfStudentT", nu=3, sigma=1)),
        "sigma": bmb.Prior("HalfStudentT", nu=3, sigma=1),
        **{name: bmb.Prior("Normal", mu=0, sigma=1) for name in common_terms},
    }
    m3_model, m3_idata = fit_model_with_priors(
        m3_formula, pooled_data, m3_priors, required_group_terms=[],
    )
    az.to_netcdf(m3_idata, MODELS_DIR / "pooled_emphasis_edge_fixed.nc")
    localization_diagnostics_rows.append(
        convergence_row("edge_fixed (D6, M3, no pooling)", m3_idata, len(pooled_data), 0))
    print(f"Fit D6 M3 (edge as fixed factor): max_rhat={localization_diagnostics_rows[-1]['max_rhat']:.3f} "
          f"n_divergent={localization_diagnostics_rows[-1]['n_divergent']} pass_l9={localization_diagnostics_rows[-1]['pass_l9']}")

# --- loo_compare: the localization warrant ----------------------------------
models_for_loo = {"interaction_fixed": pooled_idata, "interaction_varying": m1_idata}
if FIT_M2_FULL_VARYING:
    models_for_loo["full_varying"] = m2_idata
if FIT_EDGE_AS_FIXED:
    models_for_loo["edge_fixed"] = m3_idata  # D-conv-C: reference point only, never drives the heterogeneity verdict
loo_compare = az.compare(models_for_loo, ic="loo")
compare_reset = loo_compare.reset_index().rename(columns={"index": "model"})
compare_reset.to_csv(OUTPUT_DIR / "stage06_localization_compare.csv", index=False)

compare_fig, compare_ax = plt.subplots(figsize=(6, 2.5))
az.plot_compare(loo_compare, ax=compare_ax)
compare_fig.tight_layout()
compare_fig.savefig(QC_DIR / "localization_loo_compare.png")
plt.close(compare_fig)

best_model = loo_compare.index[0]
second_row = loo_compare.iloc[1]
elpd_diff = float(second_row["elpd_diff"])
dse = float(second_row["dse"])

if best_model == "interaction_varying" and elpd_diff > 2 * dse:
    verdict = "cross-edge heterogeneity supported; per-edge localization is interpretable (weakly, n=6 edges)."
elif best_model == "interaction_fixed":
    verdict = "no support for heterogeneity; the pooled interaction is best read as a single shared effect, not localized."
elif best_model == "full_varying":
    m2_pass = localization_diagnostics_rows[-1]["pass_l9"]
    if m2_pass:
        verdict = ("the richer surface (M2) fits best, but at 6 edges M2 is over-parameterized; treat M1's "
                   "per-edge interaction as the estimate and M2 as agreement/robustness only, not the reference.")
    else:
        verdict = ("M2 ranks best by ELPD but FAILED the L9 convergence gate -- this ranking is not trustworthy; "
                   "fall back to M1 as the localization model of record.")
elif best_model == "edge_fixed":
    verdict = ("the no-pooling M3 (edge fixed) ranks first, but it is a different model class "
               "(un-shrunk edge estimates) and is NOT evidence of shrinkage-model heterogeneity; "
               "read the M0/M1/M2 comparison for the heterogeneity verdict, with M3 as an un-shrunk reference only.")
else:
    verdict = "inconclusive; models indistinguishable in predictive fit."

if bool(loo_compare["warning"].any()):
    verdict += " NOTE: az.compare raised a Pareto-k warning for at least one model -- treat this comparison itself as unreliable."

print(f"D6 loo_compare verdict: {verdict}")

def population_interaction_marginal_over_edge(model, idata, edges, grid_categories):
    """Film x group interaction marginalized (equal-weight average) over a FIXED edge factor.

    Needed for M3 only: `edge` there is a common (fixed), Sum-coded term, not
    a `(...|edge)` group-specific one, so `reference_grid()` (no `edge`
    column at all) cannot be predicted from M3's formula and
    `include_group_specific=False` has nothing to switch off. Equal-weight
    averaging over Sum-coded levels exactly cancels the edge main effect and
    every edge interaction term (that is what Sum contrasts are for), so this
    reproduces the same population quantity `compute_contrasts` reads off
    directly for M1/M2 -- just via explicit marginalization instead of
    dropping a group-specific term.

    Parameters
    ----------
    model : bambi.Model
    idata : arviz.InferenceData
    edges : list of str
        The edge categories to average over (the 6 emphasis edges).
    grid_categories : list of str
        `edge` categories exactly as used at fit time.

    Returns
    -------
    dict
        `summarize_draws()` output for the marginal `interaction_film_group`.
    """
    ref_dyad = pooled_data["dyad_id"].iloc[0]
    rows = [{"film": f, "group": g, "edge": e, "dyad_id": ref_dyad}
            for g in GROUPS for f in FILMS for e in edges]
    grid = pd.DataFrame(rows)
    grid["film"] = pd.Categorical(grid["film"], categories=FILMS)
    grid["group"] = pd.Categorical(grid["group"], categories=GROUPS)
    grid["edge"] = pd.Categorical(grid["edge"], categories=grid_categories)
    grid["dyad_id"] = grid["dyad_id"].astype(str)
    for term in EXTRA_TERMS:
        grid[term] = 0.0

    preds = model.predict(idata, data=grid, kind="response_params", inplace=False, include_group_specific=True)
    mu = preds.posterior["mu"]

    def cell(film=None, group=None):
        mask = pd.Series(True, index=grid.index)
        if film is not None:
            mask &= grid["film"] == film
        if group is not None:
            mask &= grid["group"] == group
        return mu.isel(__obs__=list(grid.index[mask])).mean("__obs__")  # equal-weight average, incl. over edges

    def film_contrast(group):
        return cell(film="Incredibles", group=group) - (cell(film="Peppa", group=group) + cell(film="Brave", group=group)) / 2

    interaction = (film_contrast("ASD") - film_contrast("TD")).values.flatten()
    return summarize_draws(interaction)


# --- per-edge localization table (from one posterior each, M1 and M2) ------
localization_rows = []


def add_localization_rows(model_tag, model, idata):
    """Append per-edge + population interaction/film rows for one D6 model to `localization_rows`."""
    per_edge = per_edge_contrasts(model, idata, [e for e, _ in EMPHASIS_EDGES], edge_categories)
    for edge, contrasts in per_edge.items():
        for contrast_name, summary in contrasts.items():
            localization_rows.append({
                "model": model_tag, "edge": edge, "edge_class": edge_class_lookup[edge],
                "contrast": contrast_name, "unit": "std", **summary,
            })
    edge_is_fixed_term = "C(edge, Sum)" in model.distributional_components["mu"].terms
    if edge_is_fixed_term:
        population = population_interaction_marginal_over_edge(
            model, idata, [e for e, _ in EMPHASIS_EDGES], edge_categories)
    else:
        population = compute_contrasts(model, idata, reference_grid())["interaction_film_group"]
    localization_rows.append({
        "model": model_tag, "edge": "population", "edge_class": "population",
        "contrast": "interaction_film_group", "unit": "std", **population,
    })


add_localization_rows("M1_interaction_varying", m1_model, m1_idata)
if FIT_M2_FULL_VARYING:
    add_localization_rows("M2_full_varying", m2_model, m2_idata)
if FIT_EDGE_AS_FIXED:
    add_localization_rows("M3_edge_fixed", m3_model, m3_idata)

localization_df = pd.DataFrame(localization_rows)
localization_df.to_csv(OUTPUT_DIR / "stage06_localization.csv", index=False)

interaction_rows_m1 = localization_df[
    (localization_df["model"] == "M1_interaction_varying") & (localization_df["contrast"] == "interaction_film_group")
]
localized_edges = [
    row["edge"] for _, row in interaction_rows_m1.iterrows()
    if row["edge"] != "population" and (row["hdi_low"] > 0 or row["hdi_high"] < 0)
]

localization_forest_rows = [
    {"label": f"{row['edge']} ({row['edge_class']})" if row["edge"] != "population" else "population (M1)",
     "estimate": row["estimate"], "hdi_low": row["hdi_low"], "hdi_high": row["hdi_high"]}
    for _, row in interaction_rows_m1.iterrows()
]

interaction_rows_m2 = None
m2_localized_edges = []
if FIT_M2_FULL_VARYING:
    interaction_rows_m2 = localization_df[
        (localization_df["model"] == "M2_full_varying") & (localization_df["contrast"] == "interaction_film_group")
    ]
    m2_localized_edges = [
        row["edge"] for _, row in interaction_rows_m2.iterrows()
        if row["edge"] != "population" and (row["hdi_low"] > 0 or row["hdi_high"] < 0)
    ]
    localization_forest_rows += [
        {"label": f"{row['edge']} -- M2" if row["edge"] != "population" else "population (M2)",
         "estimate": row["estimate"], "hdi_low": row["hdi_low"], "hdi_high": row["hdi_high"]}
        for _, row in interaction_rows_m2.iterrows()
    ]

interaction_rows_m3 = None
m3_localized_edges = []
if FIT_EDGE_AS_FIXED:
    interaction_rows_m3 = localization_df[
        (localization_df["model"] == "M3_edge_fixed") & (localization_df["contrast"] == "interaction_film_group")
    ]
    m3_localized_edges = [
        row["edge"] for _, row in interaction_rows_m3.iterrows()
        if row["edge"] != "population" and (row["hdi_low"] > 0 or row["hdi_high"] < 0)
    ]
    localization_forest_rows += [
        {"label": f"{row['edge']} -- M3" if row["edge"] != "population" else "population (M3)",
         "estimate": row["estimate"], "hdi_low": row["hdi_low"], "hdi_high": row["hdi_high"]}
        for _, row in interaction_rows_m3.iterrows()
    ]

localization_forest_title = "D6: per-edge film x group interaction (standardized) -- M1 primary" + (
    ", M2 sensitivity below" if FIT_M2_FULL_VARYING else ""
) + (", M3 (un-shrunk, no pooling) below" if FIT_EDGE_AS_FIXED else "")
localization_forest_fig = plot_forest(localization_forest_rows, localization_forest_title)
localization_forest_fig.savefig(QC_DIR / "localization_interaction_forest.png")
plt.close(localization_forest_fig)

# --- D6 summary text ---------------------------------------------------------
localization_summary_lines = [
    "D6 (open, exploratory) -- cross-edge localization of the film x group interaction.",
    "Not in the L7 FDR family; hypothesis-generating only. Standardized units (unit=std) throughout, signed.",
    f"Sampler/prior settings (Stage 6 addendum): TARGET_ACCEPT={TARGET_ACCEPT}, TUNE={TUNE}, "
    f"SD_PRIOR_EDGE={SD_PRIOR_EDGE!r} (scopes 1|edge only), FIT_EDGE_AS_FIXED={FIT_EDGE_AS_FIXED}.",
    "",
    "loo_compare:",
]
for _, row in compare_reset.iterrows():
    localization_summary_lines.append(
        f"  {row['model']}: rank={int(row['rank'])} elpd_loo={row['elpd_loo']:.2f} p_loo={row['p_loo']:.2f} "
        f"elpd_diff={row['elpd_diff']:.2f} dse={row['dse']:.2f} weight={row['weight']:.3f} warning={row['warning']}"
    )
localization_summary_lines.append(f"  verdict: {verdict}")
localization_summary_lines.append("")
localization_summary_lines.append("M1/M2 convergence:")
for row in localization_diagnostics_rows:
    status = "PASS" if row["pass_l9"] else "FAIL"
    localization_summary_lines.append(
        f"  [{status}] {row['model']}: max_rhat={row['max_rhat']:.3f} min_ess_bulk={row['min_bulk_ess']:.0f} "
        f"n_divergent={row['n_divergent']} max_pareto_k={row['max_pareto_k']:.2f}"
    )
localization_summary_lines.append("")
localization_summary_lines.append("M1 per-edge film x group interaction (standardized):")
for _, row in interaction_rows_m1.iterrows():
    localization_summary_lines.append(
        f"  {row['edge']} ({row['edge_class']}): estimate={row['estimate']:+.4f} "
        f"HDI95=[{row['hdi_low']:+.4f}, {row['hdi_high']:+.4f}] P(>0)={row['p_gt0']:.3f} P(<0)={row['p_lt0']:.3f}"
    )
if FIT_M2_FULL_VARYING:
    localization_summary_lines.append("")
    localization_summary_lines.append("M2 per-edge film x group interaction (standardized, sensitivity):")
    for _, row in interaction_rows_m2.iterrows():
        localization_summary_lines.append(
            f"  {row['edge']} ({row['edge_class']}): estimate={row['estimate']:+.4f} "
            f"HDI95=[{row['hdi_low']:+.4f}, {row['hdi_high']:+.4f}] P(>0)={row['p_gt0']:.3f} P(<0)={row['p_lt0']:.3f}"
        )
if FIT_EDGE_AS_FIXED:
    localization_summary_lines.append("")
    localization_summary_lines.append("M3 per-edge film x group interaction (standardized, un-shrunk, no pooling):")
    for _, row in interaction_rows_m3.iterrows():
        localization_summary_lines.append(
            f"  {row['edge']} ({row['edge_class']}): estimate={row['estimate']:+.4f} "
            f"HDI95=[{row['hdi_low']:+.4f}, {row['hdi_high']:+.4f}] P(>0)={row['p_gt0']:.3f} P(<0)={row['p_lt0']:.3f}"
        )
localization_summary_lines.append("")
localization_summary_lines.append(
    f"M1 edges with interaction HDI excluding zero (which edges carry the effect): "
    f"{', '.join(localized_edges) if localized_edges else 'none'}"
)
if FIT_M2_FULL_VARYING:
    localization_summary_lines.append(
        f"M2 agreement on that set: {', '.join(m2_localized_edges) if m2_localized_edges else 'none'}"
    )
if FIT_EDGE_AS_FIXED:
    localization_summary_lines.append(
        f"M3 (un-shrunk) agreement on that set: {', '.join(m3_localized_edges) if m3_localized_edges else 'none'}"
    )

print("\n" + "\n".join(localization_summary_lines))
(OUTPUT_DIR / "stage06_localization_summary.txt").write_text("\n".join(localization_summary_lines), encoding="utf-8")

# --- D6 gate section (appended into the already-written gate HTML) ---------


def render_localization_table(df):
    """Render a set of D6 per-edge/population interaction rows (adds edge/edge_class to render_contrast_table's columns)."""
    rows_html = ['<table class="contrasts"><tr><th>edge</th><th>edge_class</th><th>contrast</th><th>estimate</th>'
                 '<th>HDI95 low</th><th>HDI95 high</th><th>P(&gt;0)</th><th>P(&lt;0)</th></tr>']
    for _, row in df.iterrows():
        rows_html.append(
            f'<tr><td>{row["edge"]}</td><td>{row["edge_class"]}</td><td>{row["contrast"]}</td>'
            f'<td>{row["estimate"]:+.5f}</td><td>{row["hdi_low"]:+.5f}</td><td>{row["hdi_high"]:+.5f}</td>'
            f'<td>{row["p_gt0"]:.3f}</td><td>{row["p_lt0"]:.3f}</td></tr>'
        )
    rows_html.append("</table>")
    return "\n".join(rows_html)


m1_rows_gate = localization_df[localization_df["model"] == "M1_interaction_varying"]
if FIT_M2_FULL_VARYING:
    m2_rows_gate = localization_df[localization_df["model"] == "M2_full_varying"]
    m2_gate_section = (
        '<h4>M2 sensitivity (full film*group varies by edge -- heavier, agreement check only)</h4>'
        + render_localization_table(m2_rows_gate)
    )
else:
    m2_gate_section = "<h4>M2 sensitivity</h4><p>not fit (FIT_M2_FULL_VARYING=False)</p>"

if FIT_EDGE_AS_FIXED:
    m3_rows_gate = localization_df[localization_df["model"] == "M3_edge_fixed"]
    m3_gate_section = (
        '<h4>M3 (edge as a fixed factor -- un-shrunk, no pooling; D-conv-A)</h4>'
        '<p>The un-shrunk counterpart to M0\'s full pooling, with M1 read between them. '
        'By construction M3 has no <code>(...|edge)</code> hyperprior and cannot funnel; it is shown as an ELPD '
        'reference point only (D-conv-C) and never drives the heterogeneity verdict above.</p>'
        + render_localization_table(m3_rows_gate)
    )
else:
    m3_gate_section = "<h4>M3 (edge fixed)</h4><p>not fit (FIT_EDGE_AS_FIXED=False)</p>"

funnel_row_items = "".join(
    f'<img src="qc/{p.name}" alt="{p.stem}">' for p in (pooled_funnel_paths + m1_funnel_paths)
)

d6_section = f"""
<h2>Cross-edge localization of the film&times;group interaction (D6 &mdash; exploratory, not confirmatory)</h2>
<div class="callout">
  <b>Exploratory, uncorrected:</b> the film&times;group interaction was never in the L7 FDR family (only the two
  H2/H4 group effects are). Every quantity in this section is hypothesis-generating.<br>
  <b>Low power:</b> the edge-level SD is estimated from only 6 edges; per-edge deviations are heavily shrunk and
  the loo_compare ranking below is low-powered.<br>
  <b>Exchangeability:</b> <code>(...|edge)</code> treats the 6 emphasis edges as exchangeable, which is
  scientifically questionable (interbrain-EEG, interbrain-HRV, and cross brain-heart edges are qualitatively
  different) -- a typed grouping could be more defensible; not resolved here.<br>
  <b>Units:</b> standardized only (<code>unit=std</code>); no raw back-transform across edges.<br>
  <b>Model status:</b> <b>M1</b> (interaction varies by edge) is primary. <b>M2</b> (full film*group varies by
  edge) is a heavier, 6-edge-limited sensitivity check shown for agreement only. <b>M3</b> (edge as a fixed
  factor, un-shrunk) sits alongside M0/M1/M2 as the no-pooling end of the shrinkage spectrum
  (M0 fully pooled &rarr; M1 partially pooled &rarr; M3 un-shrunk) -- never promoted to the reference answer on
  an ELPD win alone.<br>
  <b>Convergence-fix settings (Stage 6 addendum):</b> TARGET_ACCEPT=<code>{TARGET_ACCEPT}</code>,
  TUNE=<code>{TUNE}</code>, SD_PRIOR_EDGE=<code>{SD_PRIOR_EDGE}</code> (scopes the <code>1|edge</code> SD only).
</div>
<p><b>loo_compare verdict:</b> {verdict}</p>
<div class="row"><img src="qc/localization_loo_compare.png" alt="D6 loo_compare"></div>
<h3>M1/M2/M3 convergence</h3>
{render_diagnostics_table(localization_diagnostics_rows)}
<h3>(1|edge) funnel diagnostic</h3>
<p>Divergences clustered at small SD = confirmed funnel, treatable by the SD_PRIOR_EDGE / target_accept levers;
divergences elsewhere = look beyond the edge geometry. M3 has no edge SD term and is intentionally not plotted.</p>
<div class="row">{funnel_row_items}</div>
<h3>Per-edge interaction (standardized)</h3>
<div class="row"><img src="qc/localization_interaction_forest.png" alt="D6 per-edge interaction forest"></div>
<h4>M1 (primary)</h4>
{render_localization_table(m1_rows_gate)}
{m2_gate_section}
{m3_gate_section}
"""

gate_path = OUTPUT_DIR / "group_model_gate.html"
gate_html = gate_path.read_text(encoding="utf-8")
gate_html = gate_html.replace("</body>", d6_section + "\n</body>")
gate_path.write_text(gate_html, encoding="utf-8")
print(f"\nAppended D6 localization section to {gate_path}")
