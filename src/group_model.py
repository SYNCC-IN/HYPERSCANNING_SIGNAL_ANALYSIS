"""Reusable helpers for the Stage 6 group model (Delta ffDTF vs surrogate).

Small, literal-free pieces shared between the model-fitting script
(`scripts/stage06_group_model.py`) and anything downstream that needs to read
the same tidy table the same way: category ordering, edge selection,
within-edge standardization (so standardized contrasts can be back-transformed
to raw Delta-units), the forward-minus-reverse asymmetry DV, and a small
Benjamini-Hochberg helper for the one named primary family. Model fitting,
priors, and orchestration stay in the script -- this module has no MCMC code.
"""

import numpy as np
import pandas as pd


def load_delta_table(csv_path):
    """Load Stage 5's tidy delta table with `film`/`group` as ordered categoricals.

    Parameters
    ----------
    csv_path : str or Path
        Path to `stage05_delta_table.csv`.

    Returns
    -------
    pd.DataFrame
        Same columns as the CSV. `film` is a categorical ordered
        `[Peppa, Incredibles, Brave]` (a fixed factor, not a scale -- ordering
        only fixes label order in tables/plots). `group` is categorical
        `[TD, ASD]`. `dyad_id` and `edge_class` are plain categoricals.
    """
    df = pd.read_csv(csv_path)
    df["film"] = pd.Categorical(df["film"], categories=["Peppa", "Incredibles", "Brave"])
    df["group"] = pd.Categorical(df["group"], categories=["TD", "ASD"])
    df["dyad_id"] = pd.Categorical(df["dyad_id"])
    df["edge_class"] = pd.Categorical(df["edge_class"])
    return df


def edge_subset(df, edges):
    """Rows of `df` whose `edge` matches any entry in `edges`.

    Parameters
    ----------
    df : pd.DataFrame
        Stage 5 delta table (or a subset of it), with an `edge` column
        formatted as `"{source}->{target}"`.
    edges : list of str or list of tuple
        Each item is either an `"source->target"` string or a
        `(source, target)` tuple.

    Returns
    -------
    pd.DataFrame
        Matching rows (copy).
    """
    edge_strings = [e if isinstance(e, str) else f"{e[0]}->{e[1]}" for e in edges]
    return df[df["edge"].isin(edge_strings)].copy()


def standardize_within_edge(df, value_col):
    """Z-score `value_col` within each `edge` group.

    Standardizing per edge (rather than globally) is what makes the model's
    weakly-informative priors, set on a standardized scale, actually
    weakly-informative for every edge regardless of that edge's raw magnitude
    (see Stage 6 decision D3). The returned per-edge mean/sd let a caller
    back-transform standardized contrasts to raw Delta-units.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an `edge` column and `value_col`.
    value_col : str
        Column to standardize.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with an added `{value_col}_z` column.
    pd.DataFrame
        Per-edge `mean`/`sd` used, columns `edge`, `mean`, `sd`.
    """
    stats = df.groupby("edge", observed=True)[value_col].agg(["mean", "std"]).rename(columns={"std": "sd"})
    stats_reset = stats.reset_index()
    out = df.merge(stats, on="edge", how="left")
    out[f"{value_col}_z"] = (out[value_col] - out["mean"]) / out["sd"]
    out = out.drop(columns=["mean", "sd"])
    return out, stats_reset


def asymmetry_dv(df, forward_edge, reverse_edge, value_col):
    """Per dyad x film forward-minus-reverse asymmetry of `value_col`.

    `asym = value(forward_edge) - value(reverse_edge)`, signed: a positive
    value means the forward direction (caregiver->child for the H2/H4 primary
    edges) dominates -- "caregiver-leading" (Stage 6 L6).

    Parameters
    ----------
    df : pd.DataFrame
        Stage 5 delta table (all edges).
    forward_edge, reverse_edge : str or tuple
        Single-edge identifiers, as accepted by `edge_subset`.
    value_col : str
        Column to difference (e.g. `z_vs_surrogate` or `delta_dtf`).

    Returns
    -------
    pd.DataFrame
        One row per `dyad_id` x `film`: `dyad_id`, `film`, `group`,
        `age_months`, `asym`, `real_stable` (True only if both directions are
        stable for that dyad x film).
    """
    forward = edge_subset(df, [forward_edge]).set_index(["dyad_id", "film"])
    reverse = edge_subset(df, [reverse_edge]).set_index(["dyad_id", "film"])
    merged = forward[[value_col, "group", "age_months", "real_stable"]].join(
        reverse[[value_col, "real_stable"]], lsuffix="_fwd", rsuffix="_rev"
    )
    merged["asym"] = merged[f"{value_col}_fwd"] - merged[f"{value_col}_rev"]
    merged["real_stable"] = merged["real_stable_fwd"] & merged["real_stable_rev"]
    return merged.reset_index()[["dyad_id", "film", "group", "age_months", "asym", "real_stable"]]


def bh_fdr(pvalue_like):
    """Benjamini-Hochberg adjusted values for a small named family.

    Intended for a directional-probability-derived score such as
    `2 * min(P(effect > 0), P(effect < 0))` -- an analog p-value, not a
    frequentist one. The caller is responsible for documenting that; this
    function applies the standard BH step-up procedure exactly as it would to
    real p-values.

    Parameters
    ----------
    pvalue_like : array-like
        p-value-like scores, one per hypothesis in the family.

    Returns
    -------
    np.ndarray
        BH-adjusted values, same order as the input.
    """
    values = np.asarray(pvalue_like, dtype=float)
    n = values.size
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out
