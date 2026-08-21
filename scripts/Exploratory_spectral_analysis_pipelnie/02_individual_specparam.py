"""Standalone - Compute PSD and run specparam for a single dyad member.

Processes one participant (dyad member) end-to-end, without depending on the
PSDs saved by Step 1: loads the ICA-cleaned EEG NetCDF file, computes the
multitaper PSD per movie (same parameters as ``01_compute_psd.py``), averages
across movies, fits specparam per channel (same parameters as
``02_run_specparam.py``), and generates a fit gallery plot equivalent to
Artifact 2a for that one participant.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import get_participant_files, load_eeg_nc, trim_to_event_window, ensure_dir
from src.psd import compute_psd_multitaper, average_psd_across_conditions
from src.specparam_utils import fit_specparam, extract_fit_quality, extract_peak_params

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Participant to process: <dyad_id>, and role_code 'ch' (child) or 'cg' (caregiver).
DYAD_ID = 'W_017'
ROLE_CODE = 'cg'

# These paths work for JZygierwicz. Anyone using this script has to set his/her own paths.
home_path = Path(
    "/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/"
    ".shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/"
    "WP4          - Joint study/UniWAW Data collection/UNIWAW_EEG_exported_BY_TASKS/"
    "ICA_output/EEG_ICA_CLEANED"
)

DATA_DIR = home_path
OUTPUT_GALLERY = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '02_individual_specparam')

# PSD parameters, matching 01_compute_psd.py
FMIN, FMAX, BANDWIDTH = 1.0, 30.0, 2.0

# specparam parameters, matching 02_run_specparam.py
FREQ_RANGE = (3, 15)
PEAK_WIDTH_LIMITS = (1, 3)
MAX_N_PEAKS = 4
MIN_PEAK_HEIGHT = 0.05
APERIODIC_MODE = 'fixed'

GALLERY_CHANNELS = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']

# ---------------------------------------------------------------------------
# 1. Locate and load the participant's EEG file
# ---------------------------------------------------------------------------
files_df = get_participant_files(DATA_DIR)
match = files_df[(files_df['dyad_id'] == DYAD_ID) & (files_df['role_code'] == ROLE_CODE)]
if match.empty:
    raise FileNotFoundError(
        f'No cleaned EEG file found for dyad_id={DYAD_ID!r}, role_code={ROLE_CODE!r} under {DATA_DIR}'
    )
rec = load_eeg_nc(match.iloc[0]['filepath'])
participant_id = f"{rec['dyad_id']}_{rec['role_code']}"
print(f'Processing {participant_id} ({rec["role"]}, {rec["group"]})')

# ---------------------------------------------------------------------------
# 2. Compute PSD per movie, average across movies
# ---------------------------------------------------------------------------
freqs = None
channel_names = rec['channel_names']
psd_by_movie = {}

for movie in rec['movies']:
    data, time = trim_to_event_window(rec['data'], rec['time'], movie['duration_s'], start=movie['start_s'])
    movie_freqs, psd = compute_psd_multitaper(data, rec['sfreq'], FMIN, FMAX, BANDWIDTH)
    if freqs is None:
        freqs = movie_freqs
    else:
        psd = np.vstack([np.interp(freqs, movie_freqs, psd[ch, :]) for ch in range(psd.shape[0])])
    psd_by_movie[movie['name']] = psd

psd_avg = average_psd_across_conditions(psd_by_movie)
print(f'Computed PSD averaged across {len(psd_by_movie)} movies: {list(psd_by_movie)}')

# ---------------------------------------------------------------------------
# 3. Fit specparam per channel and plot the fit gallery (Artifact 2a equivalent)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(9, 7))
for ch_name, ax in zip(GALLERY_CHANNELS, axes.flat):
    ch_idx = channel_names.index(ch_name)
    sm = fit_specparam(freqs, psd_avg[ch_idx, :], FREQ_RANGE, PEAK_WIDTH_LIMITS,
                        MAX_N_PEAKS, MIN_PEAK_HEIGHT, APERIODIC_MODE)
    quality = extract_fit_quality(sm)
    n_peaks = len(extract_peak_params(sm))
    sm.plot(ax=ax, add_legend=(ch_name == GALLERY_CHANNELS[0]))
    ax.set_title(
        f"{ch_name}  r2={quality['r_squared']:.2f}  n_peaks={n_peaks}",
        fontsize=9,
    )
fig.suptitle(f"{participant_id}  ({rec['role']}, {rec['group']})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTPUT_GALLERY / f'{participant_id}_specparam_fits.png', dpi=300)
plt.close(fig)

print(f'Saved specparam fit gallery for {participant_id} to {OUTPUT_GALLERY}')
