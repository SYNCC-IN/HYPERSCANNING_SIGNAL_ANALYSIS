"""Step 3 - Peak inventory: distribution of detected peaks across scalp and frequency.

Loads the peak table from Step 2, computes prevalence by role x group x frequency
band, and generates artifacts 3a (frequency histograms), 3b (prevalence
topomaps), 3c (individual peak-frequency topomaps), and 3d (peak frequency vs. age).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_dir
from src.peaks import classify_channel_cluster, compute_peak_prevalence
from src.viz import (
    make_montage_info, plot_peak_freq_histogram, plot_peak_freq_topomap_individual,
    plot_peak_freq_vs_age,
)

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QUALITY_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '02_specparam_quality'
OUT_HIST = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '03_peak_freq_histograms')
OUT_PREVALENCE = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '03_peak_prevalence_topomaps')
OUT_INDIVIDUAL = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '03_peak_freq_topomaps_individual')
OUT_VS_AGE = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '03_peak_freq_vs_age')

FREQ_BINS = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13)]
CLUSTERS_3A = ['frontal', 'central', 'parietal', 'occipital']
POSTERIOR_CHANNELS = ['P3', 'Pz', 'P4', 'O1', 'O2']
N_EXEMPLARS = 3
RANDOM_SEED = 42

# Must match the specparam freq_range used in scripts/02_run_specparam.py, so
# artifact 3a's histogram window doesn't silently disagree with the fit range.
SPECPARAM_FREQ_RANGE = (3, 25)

# Peaks outside this range are excluded from artifacts 3c and 3d.
PEAK_FREQ_FILTER_RANGE = (3, 15)

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2',
]

# ---------------------------------------------------------------------------
# 1. Load peaks and metadata
# ---------------------------------------------------------------------------
all_peaks_df = pd.read_csv(QUALITY_DIR / 'all_peaks.csv')
fit_quality_df = pd.read_csv(QUALITY_DIR / 'fit_quality.csv')
all_peaks_df['channel_cluster'] = all_peaks_df['channel'].apply(classify_channel_cluster)

# ---------------------------------------------------------------------------
# 2. Prevalence tables per role x group x frequency bin
# ---------------------------------------------------------------------------
# fit_quality_df has one row per channel for every participant regardless of
# whether a peak was detected, so it gives the true cohort size (unlike
# all_peaks_df, which omits participants with zero peaks entirely).
def _n_participants(role, group):
    subset = fit_quality_df[fit_quality_df['role'] == role]
    if group is not None:
        subset = subset[subset['group'] == group]
    return subset['participant_id'].nunique()


prevalence_tables = {
    ('child', 'TD'): compute_peak_prevalence(all_peaks_df, CHANNEL_NAMES, FREQ_BINS, 'child', 'TD', n_participants=_n_participants('child', 'TD')),
    ('child', 'ASD'): compute_peak_prevalence(all_peaks_df, CHANNEL_NAMES, FREQ_BINS, 'child', 'ASD', n_participants=_n_participants('child', 'ASD')),
    ('caregiver', 'TD'): compute_peak_prevalence(all_peaks_df, CHANNEL_NAMES, FREQ_BINS, 'caregiver', 'TD', n_participants=_n_participants('caregiver', 'TD')),
    ('caregiver', 'ASD'): compute_peak_prevalence(all_peaks_df, CHANNEL_NAMES, FREQ_BINS, 'caregiver', 'ASD', n_participants=_n_participants('caregiver', 'ASD')),
}
for (role, group), table in prevalence_tables.items():
    table.to_csv(OUT_PREVALENCE / f'prevalence_{role}_{group}.csv')

# ---------------------------------------------------------------------------
# Artifact 3a - Peak frequency histograms (clusters x role, TD/ASD overlay)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, len(CLUSTERS_3A), figsize=(4 * len(CLUSTERS_3A), 6), sharex=True, sharey=True)
for row, role in enumerate(['child', 'caregiver']):
    for col, cluster in enumerate(CLUSTERS_3A):
        plot_peak_freq_histogram(all_peaks_df, cluster, role, group=None, bin_width=0.5,
                                  freq_range=SPECPARAM_FREQ_RANGE, ax=axes[row, col])
fig.suptitle('Peak center-frequency distributions by cluster, role, and group')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_HIST / 'peak_freq_histograms_grid.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 3a to {OUT_HIST}')

# ---------------------------------------------------------------------------
# Artifact 3b - Peak prevalence topomaps (freq bins x role/group), shared colorbar
# ---------------------------------------------------------------------------
info = make_montage_info(CHANNEL_NAMES)
row_labels = ['child-TD', 'child-ASD', 'caregiver-TD', 'caregiver-ASD']
row_keys = [('child', 'TD'), ('child', 'ASD'), ('caregiver', 'TD'), ('caregiver', 'ASD')]
bin_labels = [f'{lo}-{hi}' for lo, hi in FREQ_BINS]

fig, axes = plt.subplots(4, len(FREQ_BINS), figsize=(2.2 * len(FREQ_BINS), 9))
last_im = None
for row, (row_label, key) in enumerate(zip(row_labels, row_keys)):
    table = prevalence_tables[key]
    for col, bin_label in enumerate(bin_labels):
        ax = axes[row, col]
        last_im, _ = mne.viz.plot_topomap(
            table[bin_label].values, info, axes=ax, show=False, vlim=(0, 1), cmap='viridis', contours=0,
        )
        if row == 0:
            ax.set_title(f'{bin_label} Hz', fontsize=9)
        if col == 0:
            ax.text(-0.15, 0.5, row_label, fontsize=9, rotation=90, va='center', ha='center',
                    transform=ax.transAxes)
fig.suptitle('Peak prevalence by frequency bin, role, and group')
fig.subplots_adjust(right=0.9, wspace=0.3, hspace=0.2)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(last_im, cax=cbar_ax, label='Prevalence')
fig.savefig(OUT_PREVALENCE / 'peak_prevalence_topomaps_grid.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 3b to {OUT_PREVALENCE}')

peaks_in_range = all_peaks_df[
    (all_peaks_df['center_freq'] >= PEAK_FREQ_FILTER_RANGE[0]) &
    (all_peaks_df['center_freq'] <= PEAK_FREQ_FILTER_RANGE[1])
]
range_suffix = f'(peaks filtered to {PEAK_FREQ_FILTER_RANGE[0]}-{PEAK_FREQ_FILTER_RANGE[1]} Hz)'

# ---------------------------------------------------------------------------
# Artifact 3c - Individual peak frequency topomaps (median-quality exemplars)
# ---------------------------------------------------------------------------
mean_r2 = fit_quality_df.groupby(['participant_id', 'role'])['r_squared'].mean().reset_index()

for role in ['child', 'caregiver']:
    role_r2 = mean_r2[mean_r2['role'] == role].copy()
    median_val = role_r2['r_squared'].median()
    role_r2['dist_to_median'] = (role_r2['r_squared'] - median_val).abs()
    exemplars = role_r2.sort_values('dist_to_median').head(N_EXEMPLARS)['participant_id'].tolist()

    for pid in exemplars:
        participant_peaks = peaks_in_range[peaks_in_range['participant_id'] == pid]
        fig = plot_peak_freq_topomap_individual(participant_peaks, CHANNEL_NAMES, pid, f_range=PEAK_FREQ_FILTER_RANGE, ax=None)
        fig.axes[0].set_title(f'{fig.axes[0].get_title()}\n{range_suffix}', fontsize=9)
        fig.savefig(OUT_INDIVIDUAL / f'{pid}_peak_freq_topomap.png', dpi=300)
        plt.close(fig)

print(f'Saved artifact 3c to {OUT_INDIVIDUAL}')

# ---------------------------------------------------------------------------
# Artifact 3d - Peak frequency vs. age (posterior channels, children only)
# ---------------------------------------------------------------------------
fig = plot_peak_freq_vs_age(peaks_in_range, POSTERIOR_CHANNELS, role='child')
fig.axes[0].set_title(f'{fig.axes[0].get_title()}\n{range_suffix}', fontsize=9)
fig.savefig(OUT_VS_AGE / 'peak_freq_vs_age_posterior.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 3d to {OUT_VS_AGE}')
