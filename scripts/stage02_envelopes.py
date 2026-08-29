"""Stage 2 - individual-band envelopes for the interbrain ffDTF + HRV pipeline.

Reads Stage 1's outputs (`Interbrain_ffDTF_analysis/01_coverage/`:
`dyad_selection.json`, `coverage.csv`) and, for every included dyad x film,
builds the four MVAR design variables -- `child:ROI`, `cg:ROI`, `child:HRV`,
`cg:HRV`:

- EEG variable (`*:ROI`): ROI-reduced fast-band amplitude envelope, using each
  participant's individual band from `band_assignments.csv`.
- HRV variable (`*:HRV`): the raw (interpolated) IBI signal, downsampled only
  -- no individualized band-pass, no Hilbert. This reverses the project note's
  original HF-envelope choice: on inspection of the real signals, the EEG
  rhythm envelopes fluctuate in a band that overlaps the raw IBI (RSA,
  ~0.2-1 Hz), whereas the HF-IBI envelope is a second-order, much slower
  signal that no longer sits in that band. Feeding the raw IBI keeps both
  modalities in a comparable band for the shared low-rate MVAR. The
  consequence (accepted explicitly, see `docs/pipeline_plan.md` Stage 2): the
  EEG side is a second-order quantity (amplitude envelope of a fast rhythm)
  while the HRV side is a first-order oscillation (the IBI itself) --
  internally consistent within each modality, but relevant to interpreting the
  exploratory cross brain-heart edges.

After downsampling, both continuous variables additionally pass through one
*shared* band-pass (`DESIGN_HIGHPASS_HZ`-`DESIGN_LOWPASS_HZ`, 2nd-order
Butterworth, `filtfilt`) -- the same filter for both, so any group delay
matches -- confirmed on inspection of the real PSDs: no interesting HRV
activity above ~0.8 Hz, plus visible VLF drift below ~0.05 Hz.

Both variables are computed on each role's whole continuous `passive_movies`
chunk (EEG: individual-band filter -> Hilbert -> downsample -> shared
band-pass; HRV: downsample -> shared band-pass), *then* segmented to a film
window taken from Stage 1's already-QC'd `film_start_s`/`film_end_s` -- so all
filter/Hilbert edge transients fall in the discarded pre/post margins and
inter-film gaps, not inside the retained window. See
`DTF_analysis_notes/pipeline_plan.md` Stage 2 and `src/design.py` for the
underlying functions.

Writes one file per dyad x film to `Interbrain_ffDTF_analysis/02_envelopes/`:
`<dyad_id>_<film>.nc` (xarray.DataArray, dims (variable, time), physical
amplitude, not z-scored -- z-scoring is a Stage 3 concern), a
`stage02_manifest.csv` (one row per included dyad x film, written or
skipped-with-reason), and a QC gate (`qc/*.png` figures + `envelopes_gate.html`
index). QC plots z-score every variable first (plotting only, never persisted)
so the EEG envelope and raw IBI -- which differ by orders of magnitude in
physical units -- are visually comparable.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assemble import ROLE_CODE_OF, assemble_dyad
from src.design import roi_band_envelope, segment_signal, stack_design_variables
from src.envelopes import (
    average_channels,
    bandpass_filter,
    downsample,
    filter_individual_band,
    hilbert_envelope,
    plot_eeg_hrv_envelopes,
    plot_signal_filtered_envelope,
)
from src.io_utils import ensure_dir, get_participant_files
from src.psd import compute_psd_multitaper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DRIVE_ROOT = Path(
    "/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/"
    "Mój dysk/SYNCC-IN/WP4          - Joint study/UniWAW Data collection"
)
EEG_CLEANED_ROOT = DRIVE_ROOT / "UNIWAW_EEG_exported_BY_TASKS" / "ICA_output" / "EEG_ICA_CLEANED"
IBI_ROOT = DRIVE_ROOT / "UNIWAW_EEG_exported_BY_TASKS" / "IBI"

ANALYSIS_ROOT = PROJECT_ROOT / "Interbrain_ffDTF_analysis"
COVERAGE_CSV = ANALYSIS_ROOT / "01_coverage" / "coverage.csv"
DYAD_SELECTION_JSON = ANALYSIS_ROOT / "01_coverage" / "dyad_selection.json"
BAND_ASSIGNMENTS_PATH = PROJECT_ROOT / "Exploratory_spectral_analysis" / "04_band_assignment" / "band_assignments.csv"

OUTPUT_DIR = ensure_dir(ANALYSIS_ROOT / "02_envelopes")
QC_DIR = ensure_dir(OUTPUT_DIR / "qc")

# ROI as config -- must match Stage 1's ROI_LABEL/ROI_CHANNELS, since
# dyad_selection.json's roi_ok gate was computed for this ROI.
ROI_LABEL = "temporo-parietal"
ROI_CHANNELS = ["P7", "P8"]
BAND_ROI_LABEL = ROI_LABEL  # row to read from band_assignments.csv

FILMS = ["Peppa", "Incredibles", "Brave"]
ROLES = ["child", "caregiver"]

BAND = "fast"
EEG_FILTER_ORDER = 4

# Set by the raw IBI (HRV_SIGNAL below), not by the EEG envelopes: the raw IBI
# carries RSA up to the top of the child HF-reference band (~1.04 Hz), so
# Nyquist must clear that -- 2.5 Hz gives Nyquist 1.25 Hz. resample_poly
# anti-aliases both signal types onto this shared rate.
TARGET_SFREQ = 2.5

# Shared post-downsample band-pass, applied identically to both the ROI envelope
# and the raw IBI on the continuous signal (before per-film segmentation), so
# both variables get exactly the same filter (and thus the same group delay, if
# any -- filtfilt is zero-phase, but the point is using one shared filter shape
# rather than two different ones). Confirmed on inspection of the real PSDs: no
# interesting HRV activity above ~0.8 Hz, plus visible VLF drift below ~0.05 Hz.
DESIGN_HIGHPASS_HZ = 0.05
DESIGN_LOWPASS_HZ = 1.0
DESIGN_FILTER_ORDER = 2

# Multitaper smoothing bandwidth (Hz) for the QC PSD comparison plot -- a plain
# periodogram on a ~60 s / ~150-sample segment is too noisy to read.
DESIGN_PSD_BANDWIDTH_HZ = 0.2

# specparam's reported bandwidth is 2-sided (2*std); band_assignments.csv's
# *_bw is stored as a half-width already inflated to match that 2-sided
# value (see src/bands.py _cluster_stats). filter_individual_band's
# `bandwidth` argument is itself a half-width (cf +/- bandwidth), so passing
# fast_bw/2 makes the filter passband equal specparam's 2-sided bandwidth.
BW_CONVENTION = "specparam_2sided__bandwidth=fast_bw/2"

# HRV variable = the raw (interpolated) IBI, downsampled only -- no band-pass,
# no Hilbert (reverses the project note's HF-envelope choice, see module
# docstring). HRV_HF_REFERENCE_* is recorded as metadata only, describing the
# age-adjusted HF band the raw IBI's RSA content is expected to occupy -- it
# is never used to filter anything.
HRV_SIGNAL = "raw_ibi"
HRV_HF_REFERENCE = {"child": (0.24, 1.04), "caregiver": (0.15, 0.40)}

# "average_envelopes": filter+Hilbert each ROI channel, then average envelopes (plan default).
# "average_raw": average raw ROI channels first, then filter+Hilbert once (what demo_envelopes.py does).
ROI_REDUCTION = "average_envelopes"

# QC plots/PSDs z-score every variable first (plotting only, never persisted to
# the .nc) so the EEG envelope (uV-scale) and raw IBI (hundreds of ms) are
# visually comparable on one axis -- see module docstring.
PLOT_ZSCORE = True

# ---------------------------------------------------------------------------
# 1. Load Stage 1 outputs
# ---------------------------------------------------------------------------
dyad_selection = json.loads(DYAD_SELECTION_JSON.read_text(encoding="utf-8"))
INCLUDED_DYADS = dyad_selection["INCLUDED_DYADS"]
coverage_df = pd.read_csv(COVERAGE_CSV)
band_assignments = pd.read_csv(BAND_ASSIGNMENTS_PATH)

participant_files = get_participant_files(EEG_CLEANED_ROOT)
print(f"Stage 2: {len(INCLUDED_DYADS)} included dyads from {DYAD_SELECTION_JSON}")


def film_window(dyad_id, film):
    """Look up a film's QC'd (start_s, end_s) window from Stage 1's coverage table.

    Parameters
    ----------
    dyad_id : str
    film : str

    Returns
    -------
    tuple of float
        ``(film_start_s, film_end_s)``, identical across role/modality rows
        for a given (dyad_id, film) since Stage 1 wrote them from one shared
        `film_windows` dict.
    """
    row = coverage_df.loc[(coverage_df["dyad_id"] == dyad_id) & (coverage_df["film"] == film)].iloc[0]
    return float(row["film_start_s"]), float(row["film_end_s"])


def band_lookup(dyad_id, role):
    """Look up one participant's individualized fast-band center/width at the ROI.

    Parameters
    ----------
    dyad_id : str
    role : str
        ``'child'`` or ``'caregiver'``.

    Returns
    -------
    tuple
        ``(fast_cf, fast_bw)`` in Hz, or ``(None, None)`` if no row exists
        for this participant/ROI or the fast peak is missing (NaN) -- e.g. no
        fast rhythm was detected at this ROI for this participant.
    """
    participant_id = f"{dyad_id}_{ROLE_CODE_OF[role]}"
    matches = band_assignments.loc[
        (band_assignments["participant_id"] == participant_id) & (band_assignments["roi"] == BAND_ROI_LABEL)
    ]
    if matches.empty:
        return None, None
    row = matches.iloc[0]
    fast_cf, fast_bw = row[f"{BAND}_cf"], row[f"{BAND}_bw"]
    if pd.isna(fast_cf) or pd.isna(fast_bw):
        return None, None
    return float(fast_cf), float(fast_bw)


def plot_continuous_psd_band(raw_avg, sfreq, fast_cf, fast_bw, title):
    """Plot a continuous ROI signal's PSD with the individualized band shaded.

    Parameters
    ----------
    raw_avg : np.ndarray, shape (n_times,)
        ROI-averaged raw signal, whole continuous chunk.
    sfreq : float
        Sampling frequency in Hz.
    fast_cf : float
        Individualized fast-rhythm center frequency in Hz.
    fast_bw : float
        Half-width of the individualized passband in Hz (``fast_bw / 2``
        already applied by the caller, matching the filter's own convention).
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    freqs, psd = compute_psd_multitaper(raw_avg[np.newaxis, :], sfreq, fmin=1.0, fmax=20.0, bandwidth=1.0)
    figure, axis = plt.subplots()
    axis.plot(freqs, psd[0])
    axis.axvspan(fast_cf - fast_bw, fast_cf + fast_bw, color="orange", alpha=0.3, label="individual fast band")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_continuous_overlay(role_continuous, films_windows, title):
    """Plot the continuous downsampled ROI envelope and raw IBI with film windows shaded.

    One row per design variable (`child:ROI`, `cg:ROI`, `child:HRV`, `cg:HRV`),
    so filter/anti-alias edge effects can be checked for all four signals that
    feed the design matrix, not just the EEG envelopes.

    Parameters
    ----------
    role_continuous : dict
        ``{'child': {...}, 'caregiver': {...}}`` entries from the main loop,
        each with ``roi_env``/``roi_env_sfreq``/``roi_t0`` and
        ``hrv_signal``/``hrv_signal_sfreq``/``hrv_t0``.
    films_windows : list of tuple
        ``(film_name, start_s, end_s)`` for every film, to shade as the
        retained (post-segmentation) regions.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    rows = [
        ("child", "roi_env", "roi_env_sfreq", "roi_t0", "child:ROI"),
        ("caregiver", "roi_env", "roi_env_sfreq", "roi_t0", "cg:ROI"),
        ("child", "hrv_signal", "hrv_signal_sfreq", "hrv_t0", "child:HRV"),
        ("caregiver", "hrv_signal", "hrv_signal_sfreq", "hrv_t0", "cg:HRV"),
    ]
    figure, axes = plt.subplots(nrows=len(rows), sharex=True, figsize=(10, 9))
    for axis, (role, signal_key, sfreq_key, t0_key, label) in zip(axes, rows):
        rc = role_continuous[role]
        time = rc[t0_key] + np.arange(rc[signal_key].size) / rc[sfreq_key]
        axis.plot(time, rc[signal_key])
        for film_name, start_s, end_s in films_windows:
            axis.axvspan(start_s, end_s, color="green", alpha=0.2)
            axis.text(start_s, axis.get_ylim()[1], film_name, fontsize=8, va="top")
        axis.set_ylabel(label)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def plot_design_variable_psd(segments, fs, title, plot_zscore, psd_bandwidth):
    """Plot the multitaper PSD of each downsampled design variable, to check for aliasing.

    Each variable is z-scored first (plotting only) so the EEG envelope and
    raw IBI -- which differ by orders of magnitude in physical units -- can be
    compared on one axis; this is what makes it possible to confirm they
    occupy a comparable frequency band (see module docstring). Multitaper
    (`compute_psd_multitaper`) is used instead of a plain periodogram, which is
    too noisy on a ~60 s segment to read.

    Parameters
    ----------
    segments : dict
        ``{'child': {'roi': array, 'hrv': array}, 'caregiver': {...}}``,
        already segmented to one film window.
    fs : float
        Sampling frequency in Hz (Nyquist is ``fs / 2``).
    title : str
    plot_zscore : bool
        Whether to z-score each variable before computing its PSD.
    psd_bandwidth : float
        Multitaper frequency smoothing bandwidth in Hz.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, axis = plt.subplots()
    for role in ["child", "caregiver"]:
        for variable in ["roi", "hrv"]:
            signal = segments[role][variable]
            if plot_zscore:
                signal = zscore(signal)
            freqs, psd = compute_psd_multitaper(signal[np.newaxis, :], fs, fmin=0.0, fmax=fs / 2, bandwidth=psd_bandwidth)
            axis.plot(freqs, psd[0], label=f"{role}:{variable}")
    axis.axvline(fs / 2, color="black", linestyle="--", label="Nyquist")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("PSD (z-scored input)" if plot_zscore else "PSD")
    axis.set_title(title)
    axis.legend(fontsize=7)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# 2. Per-dyad continuous envelopes, then per-film segmentation + write
# ---------------------------------------------------------------------------
manifest_rows = []
gate_entries = []
included_dyad_meta = []

for dyad_id in INCLUDED_DYADS:
    eeg_files = participant_files[participant_files["dyad_id"] == dyad_id]
    dyad = assemble_dyad(dyad_id, eeg_files, IBI_ROOT, ROI_CHANNELS)

    role_continuous = {}
    for role in ROLES:
        fast_cf, fast_bw = band_lookup(dyad_id, role)
        if fast_cf is None:
            role_continuous[role] = {"skip_reason": f"no fast band at {BAND_ROI_LABEL} for {dyad_id} {role}"}
            continue

        eeg_entry = dyad["eeg"][role]
        ibi_entry = dyad["ibi"][role]

        roi_env, roi_env_sfreq = roi_band_envelope(
            eeg_entry["data"], eeg_entry["sfreq"], fast_cf, fast_bw / 2, EEG_FILTER_ORDER, TARGET_SFREQ, ROI_REDUCTION,
        )
        hf_low, hf_high = HRV_HF_REFERENCE[role]
        # HRV variable is the raw IBI, downsampled only -- no band-pass, no Hilbert (see module docstring).
        hrv_signal, hrv_signal_sfreq = downsample(ibi_entry["data"], ibi_entry["sfreq"], TARGET_SFREQ)

        # Shared post-downsample band-pass on the continuous signal, before segmentation
        # (see DESIGN_HIGHPASS_HZ/DESIGN_LOWPASS_HZ config): identical filter for both
        # variables so any group delay matches, no interesting HRV content above ~0.8 Hz,
        # and VLF drift below ~0.05 Hz is removed.
        roi_env = bandpass_filter(roi_env, roi_env_sfreq, DESIGN_HIGHPASS_HZ, DESIGN_LOWPASS_HZ, DESIGN_FILTER_ORDER)
        hrv_signal = bandpass_filter(
            hrv_signal, hrv_signal_sfreq, DESIGN_HIGHPASS_HZ, DESIGN_LOWPASS_HZ, DESIGN_FILTER_ORDER,
        )

        # Full-rate raw/filtered/envelope trace for the QC gate only (a single
        # average-raw-then-filter trace regardless of ROI_REDUCTION, since the
        # gate's purpose is a visual sanity check, not the production signal).
        raw_avg = average_channels(eeg_entry["data"])
        filtered_avg = filter_individual_band(raw_avg, eeg_entry["sfreq"], fast_cf, fast_bw / 2, EEG_FILTER_ORDER)
        envelope_avg_full = hilbert_envelope(filtered_avg)

        role_continuous[role] = {
            "skip_reason": None,
            "fast_cf": fast_cf,
            "fast_bw": fast_bw,
            "hf_low": hf_low,
            "hf_high": hf_high,
            "roi_env": roi_env,
            "roi_env_sfreq": roi_env_sfreq,
            "roi_t0": float(eeg_entry["time"][0]),
            "hrv_signal": hrv_signal,
            "hrv_signal_sfreq": hrv_signal_sfreq,
            "hrv_t0": float(ibi_entry["time"][0]),
            "eeg_entry": eeg_entry,
            "ibi_entry": ibi_entry,
            "raw_avg": raw_avg,
            "filtered_avg": filtered_avg,
            "envelope_avg_full": envelope_avg_full,
        }

    dyad_skip_reason = next(
        (role_continuous[role]["skip_reason"] for role in ROLES if role_continuous[role]["skip_reason"]), None
    )

    dyad_qc = {}
    if dyad_skip_reason is None:
        included_dyad_meta.append({"dyad_id": dyad_id, "group": dyad["group"], **dyad["meta"]})
        films_windows = [(film, *film_window(dyad_id, film)) for film in FILMS]

        dyad_qc["psd_band"] = {}
        for role in ROLES:
            rc = role_continuous[role]
            fig = plot_continuous_psd_band(
                rc["raw_avg"], rc["eeg_entry"]["sfreq"], rc["fast_cf"], rc["fast_bw"] / 2,
                f"{dyad_id} {role} {ROI_LABEL} continuous PSD",
            )
            path = QC_DIR / f"{dyad_id}_{role}_continuous_psd_band.png"
            fig.savefig(path)
            plt.close(fig)
            dyad_qc["psd_band"][role] = path.name

        fig = plot_continuous_overlay(role_continuous, films_windows, f"{dyad_id} continuous ROI envelope + raw IBI + film windows")
        path = QC_DIR / f"{dyad_id}_continuous_overlay.png"
        fig.savefig(path)
        plt.close(fig)
        dyad_qc["overlay"] = path.name

    for film in FILMS:
        if dyad_skip_reason is not None:
            manifest_rows.append({
                "dyad_id": dyad_id, "film": film, "status": "skipped", "reason": dyad_skip_reason,
            })
            continue

        film_start_s, film_end_s = film_window(dyad_id, film)

        segments = {}
        for role in ROLES:
            rc = role_continuous[role]
            roi_seg, _ = segment_signal(rc["roi_env"], rc["roi_env_sfreq"], rc["roi_t0"], film_start_s, film_end_s)
            hrv_seg, _ = segment_signal(rc["hrv_signal"], rc["hrv_signal_sfreq"], rc["hrv_t0"], film_start_s, film_end_s)
            segments[role] = {"roi": roi_seg, "hrv": hrv_seg}

        common_len = min(
            segments["child"]["roi"].size, segments["caregiver"]["roi"].size,
            segments["child"]["hrv"].size, segments["caregiver"]["hrv"].size,
        )
        fs = role_continuous["child"]["roi_env_sfreq"]

        attrs = {
            "fs": fs,
            "roi_label": ROI_LABEL,
            "roi_channels": "|".join(ROI_CHANNELS),
            "child_roi_cf": role_continuous["child"]["fast_cf"],
            "child_roi_bw_half": role_continuous["child"]["fast_bw"] / 2,
            "cg_roi_cf": role_continuous["caregiver"]["fast_cf"],
            "cg_roi_bw_half": role_continuous["caregiver"]["fast_bw"] / 2,
            "eeg_filter_order": EEG_FILTER_ORDER,
            "hrv_signal": HRV_SIGNAL,
            "child_hf_reference_low": role_continuous["child"]["hf_low"],
            "child_hf_reference_high": role_continuous["child"]["hf_high"],
            "cg_hf_reference_low": role_continuous["caregiver"]["hf_low"],
            "cg_hf_reference_high": role_continuous["caregiver"]["hf_high"],
            "film": film,
            "dyad_id": dyad_id,
            "group": dyad["group"] or "",
            "age_months": dyad["meta"]["age_months"] or np.nan,
            "target_sfreq": TARGET_SFREQ,
            "zscored": 0,
            "roi_reduction": ROI_REDUCTION,
            "bw_convention": BW_CONVENTION,
            "design_highpass_hz": DESIGN_HIGHPASS_HZ,
            "design_lowpass_hz": DESIGN_LOWPASS_HZ,
            "design_filter_order": DESIGN_FILTER_ORDER,
        }

        segments_trimmed = {
            role: {variable: segments[role][variable][:common_len] for variable in ["roi", "hrv"]}
            for role in ROLES
        }

        design = stack_design_variables(
            segments_trimmed["child"]["roi"], segments_trimmed["caregiver"]["roi"],
            segments_trimmed["child"]["hrv"], segments_trimmed["caregiver"]["hrv"],
            fs, attrs,
        )
        out_path = OUTPUT_DIR / f"{dyad_id}_{film}.nc"
        design.to_netcdf(out_path)

        manifest_rows.append({
            "dyad_id": dyad_id, "film": film, "status": "written", "reason": "",
            "fs": fs, "n_samples": common_len,
            "child_fast_cf": role_continuous["child"]["fast_cf"], "child_fast_bw": role_continuous["child"]["fast_bw"],
            "cg_fast_cf": role_continuous["caregiver"]["fast_cf"], "cg_fast_bw": role_continuous["caregiver"]["fast_bw"],
        })

        # --- Film-level QC ---
        film_qc = {"psd_band": dyad_qc["psd_band"], "overlay": dyad_qc["overlay"], "filter_envelope": {}, "eeg_hrv": {}}
        for role in ROLES:
            rc = role_continuous[role]
            eeg_time = rc["eeg_entry"]["time"]
            mask = (eeg_time >= film_start_s) & (eeg_time <= film_end_s)
            fig = plot_signal_filtered_envelope(
                rc["raw_avg"][mask], rc["filtered_avg"][mask], rc["envelope_avg_full"][mask],
                rc["eeg_entry"]["sfreq"], f"{dyad_id} {role} {film} raw/filtered/envelope (retained window)",
            )
            path = QC_DIR / f"{dyad_id}_{film}_{role}_filter_envelope.png"
            fig.savefig(path)
            plt.close(fig)
            film_qc["filter_envelope"][role] = path.name

            roi_for_plot = zscore(segments_trimmed[role]["roi"]) if PLOT_ZSCORE else segments_trimmed[role]["roi"]
            hrv_for_plot = zscore(segments_trimmed[role]["hrv"]) if PLOT_ZSCORE else segments_trimmed[role]["hrv"]
            fig = plot_eeg_hrv_envelopes(
                roi_for_plot, fs, hrv_for_plot, fs,
                f"{dyad_id} {role} {film}: EEG {BAND} envelope and raw IBI"
                + (" (z-scored)" if PLOT_ZSCORE else ""),
            )
            path = QC_DIR / f"{dyad_id}_{film}_{role}_eeg_hrv.png"
            fig.savefig(path)
            plt.close(fig)
            film_qc["eeg_hrv"][role] = path.name

        fig = plot_design_variable_psd(
            segments_trimmed, fs, f"{dyad_id} {film} downsampled design variable PSD (aliasing check)",
            PLOT_ZSCORE, DESIGN_PSD_BANDWIDTH_HZ,
        )
        path = QC_DIR / f"{dyad_id}_{film}_design_psd.png"
        fig.savefig(path)
        plt.close(fig)
        film_qc["design_psd"] = path.name

        gate_entries.append({"dyad_id": dyad_id, "film": film, "status": "written", "qc": film_qc})

    if dyad_skip_reason is not None:
        for film in FILMS:
            gate_entries.append({"dyad_id": dyad_id, "film": film, "status": "skipped", "reason": dyad_skip_reason})

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(OUTPUT_DIR / "stage02_manifest.csv", index=False)

n_written = (manifest_df["status"] == "written").sum()
n_skipped = (manifest_df["status"] == "skipped").sum()
print(f"\n=== Stage 2 summary ===")
print(f"dyad x film cells: {len(manifest_df)} ({n_written} written, {n_skipped} skipped)")
if n_skipped:
    print("\nSkipped cells:")
    for _, r in manifest_df[manifest_df["status"] == "skipped"].drop_duplicates(["dyad_id", "reason"]).iterrows():
        print(f"  {r['dyad_id']} {r['film']:12s} {r['reason']}")
print(f"\nWrote {n_written} design files + manifest to {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# 3. Interactive HTML gate
# ---------------------------------------------------------------------------
gate_by_dyad = {}
for entry in gate_entries:
    gate_by_dyad.setdefault(entry["dyad_id"], []).append(entry)
gate_dyad_ids = sorted(gate_by_dyad.keys())

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Stage 2 envelopes gate</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 1.5em; color: #1a1a1a; }
  h1 { font-size: 1.3em; }
  select { font-size: 1em; padding: 0.3em; margin-bottom: 1em; }
  .dyad-panel { display: none; }
  .dyad-panel.active { display: block; }
  .film-block { border-top: 1px solid #ccc; padding-top: 1em; margin-top: 1em; }
  .skipped { color: #a33; font-weight: 600; }
  .row { display: flex; flex-wrap: wrap; gap: 0.5em; }
  .row img { max-width: 420px; border: 1px solid #ccc; }
  h2, h3 { margin-bottom: 0.3em; }
</style>
</head>
<body>
<h1>Stage 2 envelopes gate</h1>
<p>ROI: <b>__ROI_LABEL__</b> (__ROI_CHANNELS__), band: __BAND__, reduction: __ROI_REDUCTION__.</p>
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
    """Render one dyad's QC panel (continuous figures + per-film sections) as an HTML fragment.

    Parameters
    ----------
    dyad_id : str
    entries : list of dict
        This dyad's `gate_entries` rows, one per film.

    Returns
    -------
    str
        HTML fragment for the dyad's panel div.
    """
    html = [f'<div class="dyad-panel" id="panel-{dyad_id}"><h2>{dyad_id}</h2>']
    written = [e for e in entries if e["status"] == "written"]
    if written:
        qc = written[0]["qc"]
        html.append('<div class="row">')
        for role in ROLES:
            html.append(f'<img src="qc/{qc["psd_band"][role]}" alt="{role} continuous PSD">')
        html.append(f'<img src="qc/{qc["overlay"]}" alt="continuous overlay">')
        html.append('</div>')

    for entry in entries:
        html.append(f'<div class="film-block"><h3>{entry["film"]}</h3>')
        if entry["status"] == "skipped":
            html.append(f'<p class="skipped">Skipped: {entry["reason"]}</p>')
        else:
            qc = entry["qc"]
            html.append('<div class="row">')
            for role in ROLES:
                html.append(f'<img src="qc/{qc["filter_envelope"][role]}" alt="{role} filter/envelope">')
            for role in ROLES:
                html.append(f'<img src="qc/{qc["eeg_hrv"][role]}" alt="{role} EEG/HRV envelopes">')
            html.append(f'<img src="qc/{qc["design_psd"]}" alt="design variable PSD aliasing check">')
            html.append('</div>')
        html.append('</div>')
    html.append('</div>')
    return "\n".join(html)


panels_html = "\n".join(render_dyad_panel(dyad_id, gate_by_dyad[dyad_id]) for dyad_id in gate_dyad_ids)

html = HTML_TEMPLATE.replace("__PANELS__", panels_html)
html = html.replace("__DYAD_IDS_JSON__", json.dumps(gate_dyad_ids))
html = html.replace("__ROI_LABEL__", ROI_LABEL).replace("__ROI_CHANNELS__", "|".join(ROI_CHANNELS))
html = html.replace("__BAND__", BAND).replace("__ROI_REDUCTION__", ROI_REDUCTION)
(OUTPUT_DIR / "envelopes_gate.html").write_text(html, encoding="utf-8")
print(f"Wrote interactive gate to {OUTPUT_DIR / 'envelopes_gate.html'}")

# ---------------------------------------------------------------------------
# 4. Sample statistics for dyads included in the written design files
# ---------------------------------------------------------------------------
included_meta_df = pd.DataFrame(included_dyad_meta)

print(f"\n=== Dyads with written design files ({len(included_dyad_meta)}) sample statistics, by group ===")
sample_stats = {}
for group_label, group_meta in included_meta_df.groupby("group"):
    age = group_meta["age_months"].dropna()
    sex_counts = group_meta["sex"].value_counts()
    n_sexed = sex_counts.sum()
    sex_str = ", ".join(
        f"{sex_code}={count} ({100 * count / n_sexed:.1f}%)" for sex_code, count in sex_counts.items()
    )
    print(f"\n{group_label} (n={len(group_meta)}):")
    print(f"  age (months): mean={age.mean():.1f} +/- {age.std():.1f}, "
          f"range=[{age.min():.0f}, {age.max():.0f}] (n={len(age)})")
    print(f"  sex: {sex_str}")

    sample_stats[group_label] = {
        "n": len(group_meta),
        "age_months_mean": float(age.mean()),
        "age_months_std": float(age.std()),
        "age_months_min": float(age.min()),
        "age_months_max": float(age.max()),
        "sex_counts": {str(k): int(v) for k, v in sex_counts.items()},
        "sex_proportions": {str(k): float(v / n_sexed) for k, v in sex_counts.items()},
    }

sample_stats_path = OUTPUT_DIR / "stage02_sample_stats.json"
sample_stats_path.write_text(json.dumps(sample_stats, indent=2), encoding="utf-8")
print(f"\nWrote sample statistics to {sample_stats_path}")
