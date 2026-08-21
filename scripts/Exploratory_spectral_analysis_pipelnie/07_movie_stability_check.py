"""Step 7 - Per-movie stability check, per band.

Checks whether the slow and fast rhythm peaks are stable across the three
movies, separately, since a stable fast rhythm doesn't guarantee a stable
slow rhythm (or vice versa). Generates artifact 7a (Bland-Altman plots per
ROI x band), 7b (stability summary table and heatmap), and 7c (per-participant
detection consistency heatmap for the primary ROI).
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
from src.specparam_utils import (
    extract_peak_params, find_peak_in_individual_window, find_peak_in_window, fit_specparam,
)
from src.viz import COLORS, plot_detection_consistency_heatmap, plot_stability_heatmap

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DERIVED_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / 'derived_data'
ROI_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '05_roi_definition'
BAND_ASSIGNMENT_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '04_band_assignment'
OUTPUT_DIR = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '07_movie_stability')

STABILITY_ROI_NAMES = ['sensorimotor', 'parietal', 'frontal-midline', 'central-midline', 'lateral-temporal', 'temporo-parietal']
# Include single-channel ROIs here — stability is about per-movie consistency,
# not ROI averaging, so single channels are relevant (unlike Step 6).
PRIMARY_ROI = 'parietal'  # used for artifact 7c, which shows one ROI at a time

SPECPARAM_SETTINGS = {
    'freq_range': (3, 14),
    'peak_width_limits': (1, 3),
    'max_n_peaks': 4,
    'min_peak_height': 0.1,
    'aperiodic_mode': 'fixed',
}
SLOW_FREQ_WINDOW = (3, 7)
FAST_FREQ_WINDOW = (7, 14)
MAX_ACCEPTABLE_SHIFT = 2.0  # Hz
MOVIES = ['Peppa', 'Incredibles', 'Brave']

# ---------------------------------------------------------------------------
# 1. Load per-movie PSDs, ROI definitions (Step 5), and band assignments (Step 4)
# ---------------------------------------------------------------------------
with open(DERIVED_DIR / 'psd_data.pkl', 'rb') as f:
    psd_data = pickle.load(f)
freqs = psd_data['freqs']
channel_names = psd_data['channel_names']
psd_by_movie = psd_data['psd_by_movie']

with open(ROI_DIR / 'roi_definitions.json') as f:
    roi_definitions = json.load(f)
STABILITY_ROIS = {name: roi_definitions[name]['channels'] for name in STABILITY_ROI_NAMES}

band_assignments_df = pd.read_csv(BAND_ASSIGNMENT_DIR / 'band_assignments.csv')
participant_meta = pd.read_csv(DERIVED_DIR / 'participant_metadata.csv').set_index('participant_id')

for roi_name in band_assignments_df['roi'].unique():
    subset = band_assignments_df[band_assignments_df['roi'] == roi_name]
    two_rhythm_rate = (subset['assignment_note'] == 'two_rhythms').mean()
    print(f'[step 4 cross-check] {roi_name}: {two_rhythm_rate:.0%} of participant x role rows assigned two_rhythms')

def _windows_overlap(a_low, a_high, b_low, b_high):
    return a_low <= b_high and b_low <= a_high


# ---------------------------------------------------------------------------
# 2. Fit specparam per participant x ROI x movie, find slow/fast peaks
#    (both fixed-window and individualized-window)
# ---------------------------------------------------------------------------
movie_band_rows = []
movie_band_rows_individual = []
for pid, movie_psds in psd_by_movie.items():
    meta = participant_meta.loc[pid]
    for roi_name, roi_channels in STABILITY_ROIS.items():
        roi_channels_avail = [ch for ch in roi_channels if ch in channel_names]

        # Individualized band boundaries for this participant x ROI, from the
        # movie-averaged assignment in Step 4 — looked up per-ROI, not
        # borrowed from a primary ROI.
        assignment_rows = band_assignments_df[
            (band_assignments_df['participant_id'] == pid) & (band_assignments_df['roi'] == roi_name)
        ]
        individual_bands = []
        if len(assignment_rows) > 0:
            assignment = assignment_rows.iloc[0]
            note = assignment['assignment_note']
            if note == 'two_rhythms':
                individual_bands = [
                    ('slow', assignment['slow_cf'], assignment['slow_bw']),
                    ('fast', assignment['fast_cf'], assignment['fast_bw']),
                ]
            elif note == 'single_dominant':
                dom_low = assignment['dominant_cf'] - assignment['dominant_bw']
                dom_high = assignment['dominant_cf'] + assignment['dominant_bw']
                if _windows_overlap(dom_low, dom_high, *SLOW_FREQ_WINDOW):
                    individual_bands.append(('slow', assignment['dominant_cf'], assignment['dominant_bw']))
                if _windows_overlap(dom_low, dom_high, *FAST_FREQ_WINDOW):
                    individual_bands.append(('fast', assignment['dominant_cf'], assignment['dominant_bw']))
            # 'no_peaks' (or no assignment row at all): individual_bands stays
            # empty, so this participant x ROI is excluded from the
            # individualized stability calc entirely.

        for movie in MOVIES:
            if movie not in movie_psds:
                continue
            roi_psd = average_psd_within_roi(movie_psds[movie], channel_names, roi_channels_avail)
            model = fit_specparam(freqs, roi_psd, **SPECPARAM_SETTINGS)
            peaks = extract_peak_params(model)

            for band_name, band_window in [('slow', SLOW_FREQ_WINDOW), ('fast', FAST_FREQ_WINDOW)]:
                peak = find_peak_in_window(peaks, band_window)
                movie_band_rows.append({
                    'participant_id': pid, 'role': meta['role'], 'group': meta['group'],
                    'roi': roi_name, 'band': band_name, 'movie': movie,
                    'center_freq': peak['center_freq'] if peak else np.nan,
                    'power': peak['power'] if peak else np.nan,
                })

            for band_name, cf, bw in individual_bands:
                peak = find_peak_in_individual_window(peaks, cf, bw)
                movie_band_rows_individual.append({
                    'participant_id': pid, 'role': meta['role'], 'group': meta['group'],
                    'roi': roi_name, 'band': band_name, 'movie': movie,
                    'center_freq': peak['center_freq'] if peak else np.nan,
                    'power': peak['power'] if peak else np.nan,
                })

movie_band_peaks_df = pd.DataFrame(movie_band_rows)
movie_band_peaks_df.to_csv(OUTPUT_DIR / 'movie_band_peaks.csv', index=False)
print(f'Fitted specparam for {len(psd_by_movie)} participants x {len(STABILITY_ROIS)} ROIs x '
      f'{len(MOVIES)} movies x 2 bands.')

movie_band_peaks_individual_df = pd.DataFrame(movie_band_rows_individual)
movie_band_peaks_individual_df.to_csv(OUTPUT_DIR / 'movie_band_peaks_individual.csv', index=False)
print(f'Found individualized-band peaks for {len(movie_band_peaks_individual_df)} '
      f'participant x roi x band x movie rows.')

# ---------------------------------------------------------------------------
# 3. Per participant x ROI x band stability metrics across the 3 movies
# ---------------------------------------------------------------------------
stability_rows = []
group_cols = ['participant_id', 'role', 'group', 'roi', 'band']
for keys, group_df in movie_band_peaks_df.groupby(group_cols):
    pid, role, group, roi_name, band_name = keys
    cf_by_movie = group_df.set_index('movie')['center_freq'].reindex(MOVIES)
    detected = cf_by_movie.dropna()
    n_movies_detected = len(detected)
    max_shift = float(detected.max() - detected.min()) if n_movies_detected >= 2 else np.nan
    mean_cf = float(detected.mean()) if n_movies_detected >= 1 else np.nan
    cf_range = float(detected.max() - detected.min()) if n_movies_detected >= 2 else np.nan
    stability_rows.append({
        'participant_id': pid, 'role': role, 'group': group, 'roi': roi_name, 'band': band_name,
        **{f'cf_{movie}': cf_by_movie.get(movie, np.nan) for movie in MOVIES},
        'n_movies_detected': n_movies_detected,
        'max_shift': max_shift,
        'mean_cf': mean_cf,
        'cf_range': cf_range,
    })

movie_stability_df = pd.DataFrame(stability_rows)
movie_stability_df.to_csv(OUTPUT_DIR / 'movie_stability.csv', index=False)
print(f'Saved per-participant stability metrics for {len(movie_stability_df)} participant x roi x band rows.')

# ---------------------------------------------------------------------------
# 3b. Same stability metrics, individualized band windows
# ---------------------------------------------------------------------------
stability_rows_individual = []
for keys, group_df in movie_band_peaks_individual_df.groupby(group_cols):
    pid, role, group, roi_name, band_name = keys
    cf_by_movie = group_df.set_index('movie')['center_freq'].reindex(MOVIES)
    detected = cf_by_movie.dropna()
    n_movies_detected = len(detected)
    max_shift = float(detected.max() - detected.min()) if n_movies_detected >= 2 else np.nan
    mean_cf = float(detected.mean()) if n_movies_detected >= 1 else np.nan
    cf_range = float(detected.max() - detected.min()) if n_movies_detected >= 2 else np.nan
    stability_rows_individual.append({
        'participant_id': pid, 'role': role, 'group': group, 'roi': roi_name, 'band': band_name,
        **{f'cf_{movie}': cf_by_movie.get(movie, np.nan) for movie in MOVIES},
        'n_movies_detected': n_movies_detected,
        'max_shift': max_shift,
        'mean_cf': mean_cf,
        'cf_range': cf_range,
    })

movie_stability_individual_df = pd.DataFrame(stability_rows_individual)
movie_stability_individual_df.to_csv(OUTPUT_DIR / 'movie_stability_individual.csv', index=False)
print(f'Saved individualized-band stability metrics for {len(movie_stability_individual_df)} '
      f'participant x roi x band rows.')

# ---------------------------------------------------------------------------
# Artifact 7a - Bland-Altman plots, per ROI x band
# ---------------------------------------------------------------------------
movie_pairs = list(combinations(MOVIES, 2))
for roi_name in STABILITY_ROIS:
    for band_name, _ in [('slow', SLOW_FREQ_WINDOW), ('fast', FAST_FREQ_WINDOW)]:
        subset = movie_band_peaks_df[(movie_band_peaks_df['roi'] == roi_name) & (movie_band_peaks_df['band'] == band_name)]

        fig, axes = plt.subplots(2, len(movie_pairs), figsize=(5 * len(movie_pairs), 8), sharex=True, sharey=True)
        for row, role in enumerate(['child', 'caregiver']):
            role_df = subset[subset['role'] == role]
            wide = role_df.pivot_table(index=['participant_id', 'group'], columns='movie', values='center_freq')
            for col, (m1, m2) in enumerate(movie_pairs):
                ax = axes[row, col]
                pair = wide[[m1, m2]].dropna().reset_index()
                mean_val = pair[[m1, m2]].mean(axis=1)
                diff_val = pair[m1] - pair[m2]
                for grp in ['TD', 'ASD']:
                    grp_mask = pair['group'] == grp
                    ax.scatter(mean_val[grp_mask], diff_val[grp_mask], color=COLORS.get(grp, 'gray'),
                               alpha=0.6, s=18, label=grp)
                ax.axhline(0, color='black', lw=0.8)
                for line_val, style in [(1, ':'), (-1, ':'), (MAX_ACCEPTABLE_SHIFT, '--'), (-MAX_ACCEPTABLE_SHIFT, '--')]:
                    ax.axhline(line_val, color='gray', linestyle=style, lw=0.8)
                ax.set_title(f'{m1} vs {m2}', fontsize=9)
                if row == 1:
                    ax.set_xlabel('Mean center freq (Hz)')
                if col == 0:
                    ax.set_ylabel(f'{role}\nDifference (Hz)')
                    ax.legend(fontsize=7)
        fig.suptitle(f'Bland-Altman: {roi_name} {band_name} band, center freq across movies '
                     f'(dotted=1Hz, dashed={MAX_ACCEPTABLE_SHIFT:.0f}Hz)')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(OUTPUT_DIR / f'bland_altman_{roi_name}_{band_name}.png', dpi=300)
        plt.close(fig)

print(f'Saved artifact 7a to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 7b - Stability summary table and heatmap
# ---------------------------------------------------------------------------
summary_rows = []
for roi_name in STABILITY_ROIS:
    for band_name in ['slow', 'fast']:
        for role in ['child', 'caregiver']:
            for group in ['TD', 'ASD']:
                subset = movie_stability_df[
                    (movie_stability_df['roi'] == roi_name) & (movie_stability_df['band'] == band_name)
                    & (movie_stability_df['role'] == role) & (movie_stability_df['group'] == group)
                ]
                n_participants = subset['participant_id'].nunique()
                detected_all_3 = subset[subset['n_movies_detected'] == 3]
                n_detected_all_3 = len(detected_all_3)
                pct_detected_all_3 = 100 * n_detected_all_3 / n_participants if n_participants > 0 else np.nan
                mean_max_shift = detected_all_3['max_shift'].mean() if len(detected_all_3) else np.nan
                pct_shift_over_2hz = (
                    100 * (detected_all_3['max_shift'] > MAX_ACCEPTABLE_SHIFT).mean() if len(detected_all_3) else np.nan
                )
                mean_cf_range = detected_all_3['cf_range'].mean() if len(detected_all_3) else np.nan
                summary_rows.append({
                    'roi': roi_name, 'band': band_name, 'role': role, 'group': group,
                    'n_participants': n_participants,
                    'n_detected_all_3': n_detected_all_3,
                    'pct_detected_all_3': round(pct_detected_all_3, 1) if not np.isnan(pct_detected_all_3) else np.nan,
                    'mean_max_shift': round(mean_max_shift, 3) if not np.isnan(mean_max_shift) else np.nan,
                    'pct_shift_over_2Hz': round(pct_shift_over_2hz, 1) if not np.isnan(pct_shift_over_2hz) else np.nan,
                    'mean_cf_range': round(mean_cf_range, 3) if not np.isnan(mean_cf_range) else np.nan,
                })

movie_stability_summary_df = pd.DataFrame(summary_rows)
movie_stability_summary_df.to_csv(OUTPUT_DIR / 'movie_stability_summary.csv', index=False)
print(movie_stability_summary_df.to_string(index=False))

fig = plot_stability_heatmap(movie_stability_summary_df, threshold=60.0)
fig.savefig(OUTPUT_DIR / 'movie_stability_heatmap.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 7b to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 7c - Detection consistency heatmap (primary ROI)
# ---------------------------------------------------------------------------
primary_roi_peaks_df = movie_band_peaks_df[movie_band_peaks_df['roi'] == PRIMARY_ROI]
fig = plot_detection_consistency_heatmap(
    primary_roi_peaks_df, movies=MOVIES, bands=('slow', 'fast'),
    title=f'Peak detection consistency: movie x band, per participant ({PRIMARY_ROI} ROI)',
)
fig.savefig(OUTPUT_DIR / 'detection_consistency_heatmap.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 7c to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 7d - Stability summary table and heatmap, individualized bands
# ---------------------------------------------------------------------------
summary_rows_individual = []
for roi_name in STABILITY_ROIS:
    for band_name in ['slow', 'fast']:
        for role in ['child', 'caregiver']:
            for group in ['TD', 'ASD']:
                subset = movie_stability_individual_df[
                    (movie_stability_individual_df['roi'] == roi_name)
                    & (movie_stability_individual_df['band'] == band_name)
                    & (movie_stability_individual_df['role'] == role)
                    & (movie_stability_individual_df['group'] == group)
                ]
                n_participants = subset['participant_id'].nunique()
                detected_all_3 = subset[subset['n_movies_detected'] == 3]
                n_detected_all_3 = len(detected_all_3)
                pct_detected_all_3 = 100 * n_detected_all_3 / n_participants if n_participants > 0 else np.nan
                mean_max_shift = detected_all_3['max_shift'].mean() if len(detected_all_3) else np.nan
                pct_shift_over_2hz = (
                    100 * (detected_all_3['max_shift'] > MAX_ACCEPTABLE_SHIFT).mean() if len(detected_all_3) else np.nan
                )
                mean_cf_range = detected_all_3['cf_range'].mean() if len(detected_all_3) else np.nan
                summary_rows_individual.append({
                    'roi': roi_name, 'band': band_name, 'role': role, 'group': group,
                    'n_participants': n_participants,
                    'n_detected_all_3': n_detected_all_3,
                    'pct_detected_all_3': round(pct_detected_all_3, 1) if not np.isnan(pct_detected_all_3) else np.nan,
                    'mean_max_shift': round(mean_max_shift, 3) if not np.isnan(mean_max_shift) else np.nan,
                    'pct_shift_over_2Hz': round(pct_shift_over_2hz, 1) if not np.isnan(pct_shift_over_2hz) else np.nan,
                    'mean_cf_range': round(mean_cf_range, 3) if not np.isnan(mean_cf_range) else np.nan,
                })

movie_stability_summary_individual_df = pd.DataFrame(summary_rows_individual)
movie_stability_summary_individual_df.to_csv(OUTPUT_DIR / 'movie_stability_summary_individual.csv', index=False)
print(movie_stability_summary_individual_df.to_string(index=False))

fig = plot_stability_heatmap(movie_stability_summary_individual_df, threshold=60.0, title_suffix=' (individualized bands)')
fig.savefig(OUTPUT_DIR / 'movie_stability_heatmap_individual.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 7d to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 7e - Detection consistency heatmap, individualized bands (primary ROI)
# ---------------------------------------------------------------------------
primary_roi_peaks_individual_df = movie_band_peaks_individual_df[
    movie_band_peaks_individual_df['roi'] == PRIMARY_ROI
]
fig = plot_detection_consistency_heatmap(
    primary_roi_peaks_individual_df, movies=MOVIES, bands=('slow', 'fast'),
    title=f'Peak detection consistency: movie x band, per participant, individualized bands ({PRIMARY_ROI} ROI)',
)
fig.savefig(OUTPUT_DIR / 'detection_consistency_heatmap_individual.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 7e to {OUTPUT_DIR}')
