"""Run EEG rhythm-envelope and raw-IBI extraction for one dyad and film.

Consistent with ``DTF_analysis_notes/pipeline_plan.md`` (Stage 2), with two
modelling choices confirmed on inspection of the real signals:

1. **EEG variable = amplitude envelope of the individual rhythm.** Filter around
   the individual band, Hilbert, downsample.
2. **HRV variable = the original (interpolated) IBI from the ncdf files, NOT its
   HF envelope.** The EEG rhythm envelopes fluctuate in a frequency range that
   overlaps the *raw* IBI (RSA, ~0.2-1 Hz), whereas the envelope of HF-IBI is a
   second-order, much slower signal. Feeding the raw IBI keeps the brain and
   heart variables in a comparable band for a shared low-rate MVAR. (This
   reverses the earlier "HRV as HF envelope" note; see the project note.)

Ordering (plan section 3): filter/Hilbert/downsample (EEG) and downsample (IBI)
run on the **continuous** ``passive_movies`` chunk; the per-film fragment is cut
only afterwards, so filter/anti-alias transients fall in the discarded
margins/gaps.

Individual-band half-width (plan section 4.3, specparam-consistent): specparam
reports peak bandwidth as ``2*std`` (2-sided), so the passband whose total width
equals that bandwidth is ``cf +/- bw/2``; ``filter_individual_band`` takes a
half-width, so we pass ``fast_bw / 2``.

Stored/analysed signals are NOT z-scored (that is deferred to Stage 3). For the
QC plots and PSDs the signals ARE z-scored, so EEG-envelope and IBI can be
compared on one scale (raw they differ by orders of magnitude).
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envelopes import (
    average_channels,
    downsample,
    eeg_band_envelope,
    filter_individual_band,
    hilbert_envelope,
    plot_dyad_envelopes,
    plot_eeg_hrv_envelopes,
    plot_signal_filtered_envelope,
)
from src.ncdf import load_xarray_from_netcdf
from src.roi import define_rois_theory


# --- configuration (edit here) ---
DYAD_ID = "W_030"
ROI = "parietal"                # band_assignments 'roi' label to read cf/bw from
FILM = "Peppa"
BAND = "fast"
FILTER_ORDER = 4
# Nyquist must exceed the raw IBI's top RSA frequency (child HF up to ~1.04 Hz),
# so the target rate is now set by the raw IBI, not by the (slower) envelopes.
TARGET_SFREQ = 2.5             # Hz (Nyquist 1.25); 3.0 is a safer margin
OUT_DIR = Path("envelope_demo")

# These paths work for JZygierwicz. Anyone using this script has to set his/her own paths.
home_path = Path(
    "/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/"
    ".shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/"
    "WP4          - Joint study/UniWAW Data collection/UNIWAW_EEG_exported_BY_TASKS/"
)
DATA_ROOT = home_path
BAND_ASSIGNMENTS_PATH = Path(
    "../Exploratory_spectral_analysis/04_band_assignment/band_assignments.csv"
)
TASK = "passive_movies"
CHILD_CODE = "ch"
CAREGIVER_CODE = "cg"
CHILD_FOLDER = "child"
CAREGIVER_FOLDER = "caregiver"
# ---------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)


def zscore(signal):
    """Z-score a 1-D signal in time (for QC/PSD comparability only)."""
    return (signal - signal.mean()) / signal.std()


def continuous_roi_envelope(eeg_xr, roi_channels, center_freq, half_width, order, target_sfreq):
    """Average-of-envelopes ROI reduction on the continuous recording.

    Filter + Hilbert + downsample each ROI channel of the whole ``passive_movies``
    chunk, then average the per-channel envelopes (plan section 7.1 default).
    Returns the continuous ROI envelope and its realized sampling frequency.
    """
    sfreq = float(eeg_xr.attrs["sampling_freq"])
    channel_signals = eeg_xr.sel(channel=roi_channels).values  # (n_roi, n_times)
    channel_envelopes = []
    env_sfreq = None
    for channel_signal in channel_signals:
        envelope, env_sfreq = eeg_band_envelope(
            channel_signal, sfreq, center_freq, half_width, order, target_sfreq
        )
        channel_envelopes.append(envelope)
    return average_channels(np.vstack(channel_envelopes)), env_sfreq


def segment_to_film(signal, sfreq, t0, film_start, film_stop):
    """Cut a per-film fragment from a continuous downsampled signal.

    The downsampled time axis is reconstructed as ``t0 + arange(n)/sfreq`` and
    masked to the film window. Segmenting after downsampling keeps filter/
    anti-alias edges in the discarded margins/gaps (plan section 3).
    """
    time = t0 + np.arange(signal.size) / sfreq
    mask = (time >= film_start) & (time <= film_stop)
    return signal[mask]


# --- load continuous signals (NOT pre-cut) ---
child_eeg = load_xarray_from_netcdf(
    DATA_ROOT / "ICA_output/EEG_ICA_CLEANED" / DYAD_ID
    / f"{DYAD_ID}_EEG_{CHILD_CODE}_{TASK}_cleaned.nc"
)
caregiver_eeg = load_xarray_from_netcdf(
    DATA_ROOT / "ICA_output/EEG_ICA_CLEANED" / DYAD_ID
    / f"{DYAD_ID}_EEG_{CAREGIVER_CODE}_{TASK}_cleaned.nc"
)
child_ibi = load_xarray_from_netcdf(
    DATA_ROOT / "IBI" / DYAD_ID / CHILD_FOLDER / f"{DYAD_ID}_IBI_{CHILD_CODE}_{TASK}.nc"
)
caregiver_ibi = load_xarray_from_netcdf(
    DATA_ROOT / "IBI" / DYAD_ID / CAREGIVER_FOLDER / f"{DYAD_ID}_IBI_{CAREGIVER_CODE}_{TASK}.nc"
)

# --- film window (from the task-event structure), applied only after downsampling ---
film_event = next(
    event for event in child_eeg.attrs["task_events_structure"] if event["name"] == FILM
)
film_start = film_event["start_rel_s"]
film_stop = film_start + film_event["duration_s"]

# --- individual bands: passband half-width = fast_bw / 2 (specparam-consistent) ---
assignments = pd.read_csv(BAND_ASSIGNMENTS_PATH)
child_assignment = assignments.loc[
    (assignments["participant_id"] == f"{DYAD_ID}_{CHILD_CODE}")
    & (assignments["roi"] == ROI)
].iloc[0]
caregiver_assignment = assignments.loc[
    (assignments["participant_id"] == f"{DYAD_ID}_{CAREGIVER_CODE}")
    & (assignments["roi"] == ROI)
].iloc[0]

child_center_freq = child_assignment[f"{BAND}_cf"]
child_half_width = child_assignment[f"{BAND}_bw"] / 2.0
caregiver_center_freq = caregiver_assignment[f"{BAND}_cf"]
caregiver_half_width = caregiver_assignment[f"{BAND}_bw"] / 2.0

rois = define_rois_theory()
roi_channels = rois[ROI]

# --- continuous EEG envelopes (filter -> Hilbert -> downsample on the whole chunk) ---
child_eeg_env_cont, child_env_sfreq = continuous_roi_envelope(
    child_eeg, roi_channels, child_center_freq, child_half_width, FILTER_ORDER, TARGET_SFREQ
)
caregiver_eeg_env_cont, caregiver_env_sfreq = continuous_roi_envelope(
    caregiver_eeg, roi_channels, caregiver_center_freq, caregiver_half_width, FILTER_ORDER, TARGET_SFREQ
)

# --- continuous raw IBI, only downsampled (NO band-pass, NO Hilbert) ---
child_ibi_signal = child_ibi.values.squeeze()
caregiver_ibi_signal = caregiver_ibi.values.squeeze()
child_ibi_sfreq = float(child_ibi.attrs["sampling_freq"])
caregiver_ibi_sfreq = float(caregiver_ibi.attrs["sampling_freq"])

child_ibi_ds_cont, child_ibi_ds_sfreq = downsample(child_ibi_signal, child_ibi_sfreq, TARGET_SFREQ)
caregiver_ibi_ds_cont, caregiver_ibi_ds_sfreq = downsample(caregiver_ibi_signal, caregiver_ibi_sfreq, TARGET_SFREQ)

# --- segment to the film AFTER downsampling ---
child_eeg_t0 = float(child_eeg.coords["time"].values[0])
caregiver_eeg_t0 = float(caregiver_eeg.coords["time"].values[0])
child_ibi_t0 = float(child_ibi.coords["time"].values[0])
caregiver_ibi_t0 = float(caregiver_ibi.coords["time"].values[0])

child_eeg_env = segment_to_film(child_eeg_env_cont, child_env_sfreq, child_eeg_t0, film_start, film_stop)
caregiver_eeg_env = segment_to_film(caregiver_eeg_env_cont, caregiver_env_sfreq, caregiver_eeg_t0, film_start, film_stop)
child_ibi_seg = segment_to_film(child_ibi_ds_cont, child_ibi_ds_sfreq, child_ibi_t0, film_start, film_stop)
caregiver_ibi_seg = segment_to_film(caregiver_ibi_ds_cont, caregiver_ibi_ds_sfreq, caregiver_ibi_t0, film_start, film_stop)

# --- QC: band placement / edge handling on the CONTINUOUS EEG signal ---
# One representative ROI channel at full resolution, so filter transients are
# visible living in the margins (they are discarded by segmentation above).
child_repr_channel = child_eeg.sel(channel=roi_channels[0]).values
child_sfreq = float(child_eeg.attrs["sampling_freq"])
child_filtered = filter_individual_band(
    child_repr_channel, child_sfreq, child_center_freq, child_half_width, FILTER_ORDER
)
child_full_envelope = hilbert_envelope(child_filtered)

child_sanity_figure = plot_signal_filtered_envelope(
    child_repr_channel,
    child_filtered,
    child_full_envelope,
    child_sfreq,
    f"{DYAD_ID} child {roi_channels[0]} {BAND} rhythm (continuous)",
)
child_sanity_figure.savefig(OUT_DIR / "child_eeg_filter_envelope.png")
plt.close(child_sanity_figure)

# --- QC: segmented traces, z-scored for comparability ---
dyad_figure = plot_dyad_envelopes(
    zscore(child_eeg_env),
    zscore(caregiver_eeg_env),
    child_env_sfreq,
    f"{DYAD_ID} {FILM} {ROI} {BAND} EEG envelopes (z-scored)",
    ("Child", "Caregiver"),
)
dyad_figure.savefig(OUT_DIR / "dyad_eeg_envelopes.png")
plt.close(dyad_figure)

child_eeg_hrv_figure = plot_eeg_hrv_envelopes(
    zscore(child_eeg_env),
    child_env_sfreq,
    zscore(child_ibi_seg),
    child_ibi_ds_sfreq,
    f"{DYAD_ID} child {FILM}: EEG {BAND} envelope and raw IBI (z-scored)",
)
child_eeg_hrv_figure.savefig(OUT_DIR / "child_eeg_ibi.png")
plt.close(child_eeg_hrv_figure)

# --- QC: PSD comparison on z-scored downsampled signals ---
# With z-scoring every signal has unit variance, so the PSDs are directly
# comparable and it is clear whether the EEG envelope and the raw IBI share a
# frequency range (the motivation for using raw IBI rather than its HF envelope).
psd_signals = {
    "child EEG env": (zscore(child_eeg_env), child_env_sfreq),
    "caregiver EEG env": (zscore(caregiver_eeg_env), caregiver_env_sfreq),
    "child IBI": (zscore(child_ibi_seg), child_ibi_ds_sfreq),
    "caregiver IBI": (zscore(caregiver_ibi_seg), caregiver_ibi_ds_sfreq),
}
psd_figure, psd_axis = plt.subplots()
for label, (sig, fs) in psd_signals.items():
    nperseg = min(sig.size, 128)
    freqs, power = welch(sig, fs=fs, nperseg=nperseg)
    psd_axis.semilogy(freqs, power, label=label)
psd_axis.set_xlabel("Frequency (Hz)")
psd_axis.set_ylabel("PSD (z-scored signal)")
psd_axis.set_title(f"{DYAD_ID} {FILM}: downsampled-signal PSDs (z-scored)")
psd_axis.legend()
psd_figure.tight_layout()
psd_figure.savefig(OUT_DIR / "downsampled_psd_comparison.png")
plt.close(psd_figure)

print(f"Child EEG envelope sampling frequency: {child_env_sfreq} Hz")
print(f"Caregiver EEG envelope sampling frequency: {caregiver_env_sfreq} Hz")
print(f"Child IBI (downsampled) sampling frequency: {child_ibi_ds_sfreq} Hz")
print(f"Caregiver IBI (downsampled) sampling frequency: {caregiver_ibi_ds_sfreq} Hz")
print(f"Film window: [{film_start:.1f}, {film_stop:.1f}] s")
print(f"Segmented child EEG env samples: {child_eeg_env.size}")
print(f"Segmented child IBI samples: {child_ibi_seg.size}")