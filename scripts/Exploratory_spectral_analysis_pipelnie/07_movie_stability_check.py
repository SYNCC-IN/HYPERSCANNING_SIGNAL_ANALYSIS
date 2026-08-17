"""Step 7 - Per-movie stability check.

Checks whether ROI-level dominant peak parameters are stable across the three
movies, validating the use of movie-averaged band definitions. Generates
artifact 7a: Bland-Altman plots per movie pair, and a stability summary table
(ICC across movies, mean absolute shift, % of participants shifting > 2 Hz).
"""

import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_dir
from src.roi import average_psd_within_roi
from src.specparam_utils import fit_specparam, extract_peak_params

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DERIVED_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / 'derived_data'
ROI_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '05_roi_definition'
OUTPUT_DIR = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '07_movie_stability')

FREQ_RANGE = (3, 25)
PEAK_WIDTH_LIMITS = (1, 6)
MAX_N_PEAKS = 4
MIN_PEAK_HEIGHT = 0.1
APERIODIC_MODE = 'fixed'
SHIFT_FLAG_HZ = 2.0

# ---------------------------------------------------------------------------
# 1. Load per-movie PSDs and final ROI definitions
# ---------------------------------------------------------------------------
with open(DERIVED_DIR / 'psd_data.pkl', 'rb') as f:
    psd_data = pickle.load(f)
freqs = psd_data['freqs']
channel_names = psd_data['channel_names']
psd_by_movie = psd_data['psd_by_movie']

with open(ROI_DIR / 'final_rois.json') as f:
    final_rois = json.load(f)

participant_meta = pd.read_csv(DERIVED_DIR / 'participant_metadata.csv').set_index('participant_id')
movies = ['Peppa', 'Incredibles', 'Brave']

# ---------------------------------------------------------------------------
# 2. Fit specparam per participant x ROI x movie, keep the dominant peak
# ---------------------------------------------------------------------------
rows = []
for pid, movie_psds in psd_by_movie.items():
    meta = participant_meta.loc[pid]
    for roi_name, roi_channels in final_rois.items():
        roi_channels_avail = [ch for ch in roi_channels if ch in channel_names]
        for movie in movies:
            if movie not in movie_psds:
                continue
            roi_psd = average_psd_within_roi(movie_psds[movie], channel_names, roi_channels_avail)
            model = fit_specparam(freqs, roi_psd, FREQ_RANGE, PEAK_WIDTH_LIMITS,
                                   MAX_N_PEAKS, MIN_PEAK_HEIGHT, APERIODIC_MODE)
            peaks = extract_peak_params(model)
            if not peaks:
                continue
            dominant = max(peaks, key=lambda p: p['power'])
            rows.append({
                'participant_id': pid, 'role': meta['role'], 'group': meta['group'],
                'roi': roi_name, 'movie': movie,
                'center_freq': dominant['center_freq'], 'power': dominant['power'],
            })

movie_peaks_df = pd.DataFrame(rows)
movie_peaks_df.to_csv(OUTPUT_DIR / 'roi_movie_peaks.csv', index=False)
print(f'Fitted dominant peaks for {len(movie_peaks_df)} participant x ROI x movie combinations.')

# ---------------------------------------------------------------------------
# 3. Stability metrics: ICC across movies, max absolute shift
# ---------------------------------------------------------------------------
def _icc_one_way(wide):
    """One-way random-effects ICC(1) (Shrout & Fleiss) for a subjects x raters matrix."""
    x = wide.dropna().values
    n, k = x.shape
    if n < 2:
        return np.nan
    grand_mean = x.mean()
    subject_means = x.mean(axis=1)
    ssb = k * np.sum((subject_means - grand_mean) ** 2)
    ssw = np.sum((x - subject_means[:, None]) ** 2)
    msb = ssb / (n - 1)
    msw = ssw / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


stability_rows = []
for roi_name in final_rois:
    for role in ['child', 'caregiver']:
        subset = movie_peaks_df[(movie_peaks_df['roi'] == roi_name) & (movie_peaks_df['role'] == role)]
        wide = subset.pivot_table(index='participant_id', columns='movie', values='center_freq')
        wide = wide.reindex(columns=movies)

        max_shift = wide.max(axis=1) - wide.min(axis=1)
        max_shift = max_shift.dropna()

        stability_rows.append({
            'roi': roi_name, 'role': role,
            'icc_center_freq': round(_icc_one_way(wide), 3),
            'mean_abs_shift_hz': round(max_shift.mean(), 3) if len(max_shift) else np.nan,
            'pct_shift_gt_2hz': round(100 * (max_shift > SHIFT_FLAG_HZ).mean(), 1) if len(max_shift) else np.nan,
            'n_complete': int(wide.dropna().shape[0]),
        })

stability_df = pd.DataFrame(stability_rows)
stability_df.to_csv(OUTPUT_DIR / 'movie_stability_summary.csv', index=False)
print(stability_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Artifact 7a (figure 1) - Bland-Altman plots per movie pair, children | caregivers
# ---------------------------------------------------------------------------
movie_pairs = list(combinations(movies, 2))
fig, axes = plt.subplots(2, len(movie_pairs), figsize=(5 * len(movie_pairs), 8), sharex=True, sharey=True)
for row, role in enumerate(['child', 'caregiver']):
    role_df = movie_peaks_df[movie_peaks_df['role'] == role]
    wide_all = role_df.pivot_table(index=['participant_id', 'roi'], columns='movie', values='center_freq')
    for col, (m1, m2) in enumerate(movie_pairs):
        ax = axes[row, col]
        pair = wide_all[[m1, m2]].dropna()
        mean_val = pair.mean(axis=1)
        diff_val = pair[m1] - pair[m2]
        ax.scatter(mean_val, diff_val, alpha=0.5, s=18)
        ax.axhline(0, color='black', lw=0.8)
        for line_val, style in [(1, ':'), (-1, ':'), (2, '--'), (-2, '--')]:
            ax.axhline(line_val, color='gray', linestyle=style, lw=0.8)
        ax.set_title(f'{m1} vs {m2}', fontsize=9)
        if row == 1:
            ax.set_xlabel('Mean center freq (Hz)')
        if col == 0:
            ax.set_ylabel(f'{role}\nDifference (Hz)')
fig.suptitle('Bland-Altman: ROI dominant peak center frequency across movies (dotted=1Hz, dashed=2Hz)')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTPUT_DIR / 'bland_altman_movie_pairs.png', dpi=300)
plt.close(fig)
print(f'Saved Bland-Altman figure to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 7a (figure 2) - Stability summary table rendered as a figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 0.5 * len(stability_df) + 1))
ax.axis('off')
table = ax.table(cellText=stability_df.values, colLabels=stability_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
ax.set_title('Peak stability across movies, by ROI and role', fontsize=12, pad=20)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'movie_stability_summary_table.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved artifact 7a to {OUTPUT_DIR}')
