"""Run instantaneous EEG and HRV envelope extraction for one dyad and film."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envelopes import (
    average_channels,
    eeg_band_envelope,
    filter_individual_band,
    hilbert_envelope,
    hrv_hf_envelope,
    plot_dyad_envelopes,
    plot_eeg_hrv_envelopes,
    plot_signal_filtered_envelope,
)
from src.ncdf import load_xarray_from_netcdf
from src.roi import define_rois_theory


# --- configuration (edit here) ---
DYAD_ID = "W_030"
ROI = "parietal"
FILM = "Peppa"
BAND = "fast"
FILTER_ORDER = 4
TARGET_SFREQ = 2.5
HRV_HF_LOW_CHILD = 0.24
HRV_HF_HIGH_CHILD = 1.04
HRV_HF_LOW_CG = 0.15
HRV_HF_HIGH_CG = 0.40
HRV_ORDER = 4
OUT_DIR = Path("envelope_demo")

# These paths work for JZygierwicz. Anyone using this script has to set his/her own paths.
FUW_path = Path('/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/'
                '.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/'
                'WP4          - Joint study/UniWAW Data collection/'
                'UNIWAW_EEG_exported_BY_TASKS/ICA_output/EEG_ICA_CLEANED') #FUW path
home_path=Path(
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

child_eeg = load_xarray_from_netcdf(
    DATA_ROOT
    / "ICA_output/EEG_ICA_CLEANED"
    / DYAD_ID
    / f"{DYAD_ID}_EEG_{CHILD_CODE}_{TASK}_cleaned.nc"
)
caregiver_eeg = load_xarray_from_netcdf(
    DATA_ROOT
    / "ICA_output/EEG_ICA_CLEANED"
    / DYAD_ID
    / f"{DYAD_ID}_EEG_{CAREGIVER_CODE}_{TASK}_cleaned.nc"
)
child_ibi = load_xarray_from_netcdf(
    DATA_ROOT
    / "IBI"
    / DYAD_ID
    / CHILD_FOLDER
    / f"{DYAD_ID}_IBI_{CHILD_CODE}_{TASK}.nc"
)
caregiver_ibi = load_xarray_from_netcdf(
    DATA_ROOT
    / "IBI"
    / DYAD_ID
    / CAREGIVER_FOLDER
    / f"{DYAD_ID}_IBI_{CAREGIVER_CODE}_{TASK}.nc"
)

film_event = next(
    event
    for event in child_eeg.attrs["task_events_structure"]
    if event["name"] == FILM
)
film_start = film_event["start_rel_s"]
film_stop = film_start + film_event["duration_s"]

child_eeg_film = child_eeg.sel(time=slice(film_start, film_stop))
caregiver_eeg_film = caregiver_eeg.sel(time=slice(film_start, film_stop))
child_ibi_film = child_ibi.sel(time=slice(film_start, film_stop))
caregiver_ibi_film = caregiver_ibi.sel(time=slice(film_start, film_stop))

rois = define_rois_theory()
child_roi_signal = average_channels(
    child_eeg_film.sel(channel=rois[ROI]).values.T
)
caregiver_roi_signal = average_channels(
    caregiver_eeg_film.sel(channel=rois[ROI]).values.T
)
child_ibi_signal = child_ibi_film.values.squeeze()
caregiver_ibi_signal = caregiver_ibi_film.values.squeeze()

assignments = pd.read_csv(BAND_ASSIGNMENTS_PATH)
child_assignment = assignments.loc[
    (assignments["participant_id"] == f"{DYAD_ID}_{CHILD_CODE}")
    & (assignments["roi"] == ROI)
].iloc[0]
caregiver_assignment = assignments.loc[
    (assignments["participant_id"] == f"{DYAD_ID}_{CAREGIVER_CODE}")
    & (assignments["roi"] == ROI)
].iloc[0]

child_sfreq = float(child_eeg.attrs["sampling_freq"])
caregiver_sfreq = float(caregiver_eeg.attrs["sampling_freq"])
child_ibi_sfreq = float(child_ibi.attrs["sampling_freq"])
caregiver_ibi_sfreq = float(caregiver_ibi.attrs["sampling_freq"])
child_center_freq = child_assignment[f"{BAND}_cf"]
child_bandwidth = child_assignment[f"{BAND}_bw"]
caregiver_center_freq = caregiver_assignment[f"{BAND}_cf"]
caregiver_bandwidth = caregiver_assignment[f"{BAND}_bw"]

child_eeg_envelope, child_env_sfreq = eeg_band_envelope(
    child_roi_signal,
    child_sfreq,
    child_center_freq,
    child_bandwidth,
    FILTER_ORDER,
    TARGET_SFREQ,
)
caregiver_eeg_envelope, caregiver_env_sfreq = eeg_band_envelope(
    caregiver_roi_signal,
    caregiver_sfreq,
    caregiver_center_freq,
    caregiver_bandwidth,
    FILTER_ORDER,
    TARGET_SFREQ,
)
child_hrv_envelope, child_hrv_env_sfreq = hrv_hf_envelope(
    child_ibi_signal,
    child_ibi_sfreq,
    HRV_HF_LOW_CHILD,
    HRV_HF_HIGH_CHILD,
    HRV_ORDER,
    TARGET_SFREQ,
)
caregiver_hrv_envelope, caregiver_hrv_env_sfreq = hrv_hf_envelope(
    caregiver_ibi_signal,
    caregiver_ibi_sfreq,
    HRV_HF_LOW_CG,
    HRV_HF_HIGH_CG,
    HRV_ORDER,
    TARGET_SFREQ,
)

child_filtered = filter_individual_band(
    child_roi_signal,
    child_sfreq,
    child_center_freq,
    child_bandwidth,
    FILTER_ORDER,
)
child_full_envelope = hilbert_envelope(child_filtered)

child_sanity_figure = plot_signal_filtered_envelope(
    child_roi_signal,
    child_filtered,
    child_full_envelope,
    child_sfreq,
    f"{DYAD_ID} child {FILM} {ROI} {BAND} rhythm",
)
child_sanity_figure.savefig(OUT_DIR / "child_eeg_filter_envelope.png")
plt.close(child_sanity_figure)

dyad_figure = plot_dyad_envelopes(
    child_eeg_envelope,
    caregiver_eeg_envelope,
    child_env_sfreq,
    f"{DYAD_ID} {FILM} {ROI} {BAND} envelopes",
    ("Child", "Caregiver"),
)
dyad_figure.savefig(OUT_DIR / "dyad_eeg_envelopes.png")
plt.close(dyad_figure)

child_eeg_hrv_figure = plot_eeg_hrv_envelopes(
    child_eeg_envelope,
    child_env_sfreq,
    child_hrv_envelope,
    child_hrv_env_sfreq,
    f"{DYAD_ID} child {FILM}: EEG {BAND} and HRV HF envelopes",
)
child_eeg_hrv_figure.savefig(OUT_DIR / "child_eeg_hrv_envelopes.png")
plt.close(child_eeg_hrv_figure)

print(f"Child EEG envelope sampling frequency: {child_env_sfreq} Hz")
print(f"Caregiver EEG envelope sampling frequency: {caregiver_env_sfreq} Hz")
print(f"Child HRV envelope sampling frequency: {child_hrv_env_sfreq} Hz")
print(f"Caregiver HRV envelope sampling frequency: {caregiver_hrv_env_sfreq} Hz")
