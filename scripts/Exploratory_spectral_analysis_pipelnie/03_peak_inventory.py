"""Step 3 - Peak inventory: distribution of detected peaks across scalp and frequency.

Loads the peak table from Step 2, computes prevalence by role x group x frequency
band, and generates artifacts 3a (frequency histograms), 3b (prevalence
topomaps), 3c (individual peak-frequency topomaps), 3d (peak frequency vs. age),
3e (per-participant peak cluster topomaps), and 3f (peak cluster summary).
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
from src.peaks import (
    classify_channel_cluster, cluster_peaks_across_channels, cluster_peaks_all_rois,
    compute_peak_prevalence, count_peaks_per_roi,
)
from src.viz import (
    COLORS, make_montage_info, plot_individual_peak_profiles, plot_peak_cluster_topomaps,
    plot_peak_count_histogram, plot_peak_freq_histogram, plot_peak_freq_topomap_individual,
    plot_peak_freq_vs_age, plot_roi_cluster_scatter, plot_two_peak_scatter,
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
OUT_CLUSTERS = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '03_peak_clusters')
OUT_ROI_CLUSTERS = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '03_roi_peak_clusters')
OUT_BIMODALITY = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '03_bimodality_diagnostic')

FREQ_BINS = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13)]

FREQ_TOLERANCE = 1.0    # Hz — max distance to join a cluster
MIN_CHANNELS = 3        # minimum channels for a "robust" cluster
WITHIN_ROI_FREQ_TOLERANCE = 1.0  # Hz — max distance to join a within-ROI cluster
CLUSTERS_3A = ['frontal-midline', 'frontal-lateral', 'central-midline', 'sensorimotor', 'parietal', 'occipital', 'lateral-temporal', 'temporo-parietal']
POSTERIOR_CHANNELS = ['P3', 'Pz', 'P4', 'O1', 'O2']
N_EXEMPLARS = 3
RANDOM_SEED = 42

# Column order for artifact 3g (within-ROI cluster scatter): the 7 ROIs from
# Step 5's ROIS definition (excludes frontal-lateral, which only exists in the
# broader CHANNEL_CLUSTERS grouping below, not in the ROI set).
ROI_ORDER_3G = ['frontal-midline', 'sensorimotor', 'central-midline', 'parietal', 'occipital', 'lateral-temporal', 'temporo-parietal']

# Must match the specparam freq_range used in scripts/02_run_specparam.py, so
# artifact 3a's histogram window doesn't silently disagree with the fit range.
SPECPARAM_FREQ_RANGE = (3, 15)

# Peaks outside this range are excluded from artifacts 3c and 3d.
PEAK_FREQ_FILTER_RANGE = (3, 15)

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2',
]

CHANNEL_CLUSTERS = {
    'frontal-midline':  ['Fz'],
    'frontal-lateral':  ['F3', 'F4'],
    'central-midline':  ['Cz'],
    'sensorimotor':     ['C3', 'C4'],
    'parietal':         ['P3', 'Pz', 'P4'],
    'occipital':        ['O1', 'O2'],
    'lateral-temporal': ['T7', 'T8'],
    'temporo-parietal': ['P7', 'P8'],
}

# ROIs examined for within- vs between-subject bimodality of the dominant rhythm.
DIAGNOSTIC_ROIS = {
    'sensorimotor':     ['C3', 'C4'],
    'parietal':         ['P3', 'Pz', 'P4'],
    'lateral-temporal': ['T7', 'T8'],
    'temporo-parietal': ['P7', 'P8'],
}

# ---------------------------------------------------------------------------
# 1. Load peaks and metadata
# ---------------------------------------------------------------------------
all_peaks_df = pd.read_csv(QUALITY_DIR / 'all_peaks.csv')
fit_quality_df = pd.read_csv(QUALITY_DIR / 'fit_quality.csv')
all_peaks_df['channel_cluster'] = all_peaks_df['channel'].apply(classify_channel_cluster, channel_clusters=CHANNEL_CLUSTERS)

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

# ---------------------------------------------------------------------------
# Cross-channel peak clustering
# ---------------------------------------------------------------------------
participant_info = all_peaks_df[['participant_id', 'role', 'group']].drop_duplicates()

cluster_tables = []
for _, prow in participant_info.iterrows():
    pid = prow['participant_id']
    participant_peaks = all_peaks_df[all_peaks_df['participant_id'] == pid]
    clusters = cluster_peaks_across_channels(participant_peaks, freq_tolerance=FREQ_TOLERANCE, min_channels=MIN_CHANNELS)
    clusters['participant_id'] = pid
    clusters['role'] = prow['role']
    clusters['group'] = prow['group']
    cluster_tables.append(clusters)

peak_clusters_df = pd.concat(cluster_tables, ignore_index=True)
peak_clusters_df.to_csv(OUT_CLUSTERS / 'peak_clusters.csv', index=False)
print(f'Saved {len(peak_clusters_df)} peak clusters to {OUT_CLUSTERS}')

# ---------------------------------------------------------------------------
# Artifact 3e - Peak cluster topomaps (one figure per participant)
# ---------------------------------------------------------------------------
for _, prow in participant_info.iterrows():
    pid = prow['participant_id']
    participant_peaks = all_peaks_df[all_peaks_df['participant_id'] == pid]
    participant_clusters = peak_clusters_df[peak_clusters_df['participant_id'] == pid]
    fig = plot_peak_cluster_topomaps(participant_clusters, participant_peaks, CHANNEL_NAMES, pid, prow['role'], prow['group'])
    fig.savefig(OUT_CLUSTERS / f'{pid}_peak_clusters.png', dpi=300)
    plt.close(fig)

print(f'Saved artifact 3e to {OUT_CLUSTERS}')

# ---------------------------------------------------------------------------
# Artifact 3f - Peak cluster summary (frequency x n_channels, by role/group)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, role in zip(axes, ['child', 'caregiver']):
    role_clusters = peak_clusters_df[peak_clusters_df['role'] == role]
    for grp in ['TD', 'ASD']:
        grp_clusters = role_clusters[role_clusters['group'] == grp]
        ax.scatter(
            grp_clusters['mean_center_freq'], grp_clusters['n_channels'],
            s=grp_clusters['total_power'] * 40 + 10,
            color=COLORS.get(grp, 'gray'), label=grp, alpha=0.5, edgecolor='none',
        )
    ax.axhline(MIN_CHANNELS, color='black', linestyle='--', lw=1, label=f'min_channels = {MIN_CHANNELS}')
    ax.set_xlabel('Mean center frequency (Hz)')
    ax.set_title(role, fontsize=10)
    ax.legend(fontsize=7)
axes[0].set_ylabel('Number of channels')
fig.suptitle('Peak cluster topographic spread by frequency, role, and group')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_CLUSTERS / 'cluster_summary.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 3f to {OUT_CLUSTERS}')

# ---------------------------------------------------------------------------
# Within-ROI peak clustering
# ---------------------------------------------------------------------------
roi_cluster_tables = []
for _, prow in participant_info.iterrows():
    pid = prow['participant_id']
    participant_peaks = all_peaks_df[all_peaks_df['participant_id'] == pid]
    roi_clusters = cluster_peaks_all_rois(participant_peaks, CHANNEL_CLUSTERS, freq_tolerance=WITHIN_ROI_FREQ_TOLERANCE)
    roi_clusters['participant_id'] = pid
    roi_clusters['role'] = prow['role']
    roi_clusters['group'] = prow['group']
    roi_cluster_tables.append(roi_clusters)

roi_peak_clusters_df = pd.concat(roi_cluster_tables, ignore_index=True)
roi_peak_clusters_df.to_csv(OUT_ROI_CLUSTERS / 'roi_peak_clusters.csv', index=False)
print(f'Saved {len(roi_peak_clusters_df)} within-ROI peak clusters to {OUT_ROI_CLUSTERS}')

# ---------------------------------------------------------------------------
# Artifact 3g - Within-ROI cluster scatter (frequency vs. summed power)
# ---------------------------------------------------------------------------
fig = plot_roi_cluster_scatter(roi_peak_clusters_df, group_colors={'TD': COLORS['TD'], 'ASD': COLORS['ASD']}, roi_order=ROI_ORDER_3G)
fig.savefig(OUT_ROI_CLUSTERS / 'roi_cluster_scatter.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 3g to {OUT_ROI_CLUSTERS}')

# ---------------------------------------------------------------------------
# Within- vs between-subject bimodality diagnostic
# ---------------------------------------------------------------------------
# Use the within-ROI clustered peaks (one row per distinct oscillation) rather
# than raw all_peaks_df, so a single rhythm seen at multiple ROI electrodes
# isn't double-counted as separate peaks.
def _clusters_to_peaks_frame(cluster_subset, extra_cols=()):
    df = cluster_subset.rename(columns={
        'weighted_center_freq': 'center_freq', 'summed_power': 'power', 'mean_bandwidth': 'bandwidth',
    }).copy()
    df['channel'] = df['channel_list'].str.split(',').str[0]
    return df[['channel', 'center_freq', 'power', 'bandwidth', *extra_cols]]


peak_count_rows = []
for _, prow in participant_info.iterrows():
    pid = prow['participant_id']
    for roi_label, roi_channels in DIAGNOSTIC_ROIS.items():
        cluster_subset = roi_peak_clusters_df[
            (roi_peak_clusters_df['participant_id'] == pid) & (roi_peak_clusters_df['roi'] == roi_label)
        ]
        roi_peaks = _clusters_to_peaks_frame(cluster_subset)
        counts = count_peaks_per_roi(roi_peaks, roi_channels)
        peak_count_rows.append({
            'participant_id': pid,
            'role': prow['role'],
            'group': prow['group'],
            'roi': roi_label,
            'n_peaks': counts['n_peaks'],
            'peak_freqs': counts['peak_freqs'],
            'peak_powers': counts['peak_powers'],
        })

peak_counts_df = pd.DataFrame(peak_count_rows)
peak_counts_df.to_csv(OUT_BIMODALITY / 'peak_counts_per_roi.csv', index=False)
print(f'Saved {len(peak_counts_df)} peak-count rows to {OUT_BIMODALITY}')

two_peak_rows = []
for _, row in peak_counts_df[peak_counts_df['n_peaks'] >= 2].iterrows():
    freqs, powers = row['peak_freqs'], row['peak_powers']
    if len(freqs) > 2:
        top2 = np.argsort(powers)[::-1][:2]
        freqs, powers = np.array(freqs)[top2], np.array(powers)[top2]
    (f1, p1), (f2, p2) = sorted(zip(freqs, powers))
    two_peak_rows.append({
        'participant_id': row['participant_id'],
        'role': row['role'],
        'group': row['group'],
        'roi': row['roi'],
        'peak1_freq': f1,
        'peak2_freq': f2,
        'peak1_power': p1,
        'peak2_power': p2,
        'freq_gap': f2 - f1,
    })

two_peak_df = pd.DataFrame(two_peak_rows)
two_peak_df.to_csv(OUT_BIMODALITY / 'two_peak_participants.csv', index=False)
print(f'Saved {len(two_peak_df)} two-peak participants to {OUT_BIMODALITY}')

# ---------------------------------------------------------------------------
# Artifact 3h - Peak count histogram
# ---------------------------------------------------------------------------
fig = plot_peak_count_histogram(peak_counts_df, rois=list(DIAGNOSTIC_ROIS.keys()))
fig.savefig(OUT_BIMODALITY / 'peak_count_histogram.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 3h to {OUT_BIMODALITY}')

# ---------------------------------------------------------------------------
# Artifact 3i - Two-peak scatter
# ---------------------------------------------------------------------------
fig = plot_two_peak_scatter(two_peak_df)
fig.savefig(OUT_BIMODALITY / 'two_peak_scatter.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 3i to {OUT_BIMODALITY}')

# ---------------------------------------------------------------------------
# Artifact 3j - Individual peak profile strip plots (children only)
# ---------------------------------------------------------------------------
individual_filenames = {
    'sensorimotor': 'individual_peaks_sensorimotor.png',
    'parietal': 'individual_peaks_parietal.png',
    'lateral-temporal': 'individual_peaks_lateral-temporal.png',
    'temporo-parietal': 'individual_peaks_temporo-parietal.png',
}
for roi_label, roi_channels in DIAGNOSTIC_ROIS.items():
    cluster_subset = roi_peak_clusters_df[
        (roi_peak_clusters_df['roi'] == roi_label) & (roi_peak_clusters_df['role'] == 'child')
    ]
    roi_peaks = _clusters_to_peaks_frame(cluster_subset, extra_cols=['participant_id', 'group'])
    fig = plot_individual_peak_profiles(roi_peaks, roi_channels, roi_label, role='child')
    fig.savefig(OUT_BIMODALITY / individual_filenames[roi_label], dpi=300)
    plt.close(fig)
print(f'Saved artifact 3j to {OUT_BIMODALITY}')
