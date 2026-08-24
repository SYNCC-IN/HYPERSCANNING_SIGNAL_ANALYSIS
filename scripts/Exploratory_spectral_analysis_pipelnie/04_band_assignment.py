"""Step 4 - Two-band (slow/fast rhythm) assignment from raw ROI peaks.

Loads the raw per-channel specparam peaks from Step 2, pools them per ROI,
and assigns slow and fast rhythm bands per participant x ROI via seeded
power-weighted k-means (falling back to a single rhythm where the two
final centers aren't far enough apart). This replaces the greedy
within-ROI clustering + power-based top-2 selection, which could discard a
true oscillation split across adjacent peaks by spectral noise. Also
computes IAF and dyadic distance metrics, and generates artifacts 4a
(assignment strip plot), 4b (assignment summary table), 4c (IAF distance
by group), 4d (gap distribution), 4e (dyadic gap scatter), and 4f (merged
band center-frequency histograms).
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_dir
from src.bands import assign_bands_all_rois, compute_iaf_metrics
from src.viz import (
    PARTICIPANT_BAND_COLORS, plot_band_assignment_strips, plot_band_cf_histograms,
    plot_dyad_gap_scatter, plot_gap_distribution, plot_iaf_distance_by_group,
)

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QUALITY_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '02_specparam_quality'
DERIVED_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / 'derived_data'
OUTPUT_DIR = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '04_band_assignment')

ROI_CHANNELS = {
    'frontal-midline':  ['Fz'],
    'sensorimotor':     ['C3', 'C4'],
    'central-midline':  ['Cz'],
    'parietal':         ['P3', 'Pz', 'P4'],
    'occipital':        ['O1', 'O2'],
    'lateral-temporal': ['T7', 'T8'],
    'temporo-parietal': ['P7', 'P8'],
}  # ROIs to assign bands for
ASSIGNMENT_ROIS = list(ROI_CHANNELS)
MIN_GAP = 1.5  # Hz — minimum gap to split into slow/fast
SLOW_CF_RANGE = (3.0, 7.5)   # Hz — seeding range for a slow rhythm center
FAST_CF_RANGE = (7.5, 13.0)  # Hz — seeding range for a fast rhythm center
MAX_ITER = 50  # k-means iteration cap
PRIMARY_ROI = 'parietal'  # for IAF extraction
FALLBACK_ROI = 'sensorimotor'  # if parietal has no peak

# Shared with the per-participant overview (artifact 6c) for a consistent palette.
SLOW_COLOR = PARTICIPANT_BAND_COLORS['slow']
FAST_COLOR = PARTICIPANT_BAND_COLORS['fast']

# ---------------------------------------------------------------------------
# 1. Load raw specparam peaks and participant metadata
# ---------------------------------------------------------------------------
all_peaks_df = pd.read_csv(QUALITY_DIR / 'all_peaks.csv')
participant_meta = pd.read_csv(DERIVED_DIR / 'participant_metadata.csv').set_index('participant_id')

# ---------------------------------------------------------------------------
# 2. Assign slow/fast bands per participant x ROI
# ---------------------------------------------------------------------------
assignment_tables = []
for pid, meta in participant_meta.iterrows():
    participant_peaks = all_peaks_df[all_peaks_df['participant_id'] == pid]
    assignments = assign_bands_all_rois(
        participant_peaks, pid, meta['role'], ROI_CHANNELS, min_gap=MIN_GAP,
        slow_cf_range=SLOW_CF_RANGE, fast_cf_range=FAST_CF_RANGE, max_iter=MAX_ITER,
    )
    assignments['group'] = meta['group']
    assignments['dyad_id'] = meta['dyad_id']
    assignments['age_months'] = meta['age_months']
    assignment_tables.append(assignments)

band_assignments_df = pd.concat(assignment_tables, ignore_index=True)
band_assignments_df.to_csv(OUTPUT_DIR / 'band_assignments.csv', index=False)
print(f'Saved {len(band_assignments_df)} band assignments to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# 3. IAF and dyadic distance metrics
# ---------------------------------------------------------------------------
iaf_metrics_df = compute_iaf_metrics(band_assignments_df)
iaf_metrics_df.to_csv(OUTPUT_DIR / 'iaf_metrics.csv', index=False)
print(f'Saved IAF metrics for {len(iaf_metrics_df)} participants to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 4a - Band assignment strip plot
# ---------------------------------------------------------------------------
fig = plot_band_assignment_strips(band_assignments_df, rois=ASSIGNMENT_ROIS)
fig.savefig(OUTPUT_DIR / 'band_assignment_strips.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 4a to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 4b - Assignment summary statistics
# ---------------------------------------------------------------------------
summary_rows = []
for role in ['child', 'caregiver']:
    for group in ['TD', 'ASD']:
        for roi in ASSIGNMENT_ROIS:
            subset = band_assignments_df[
                (band_assignments_df['role'] == role)
                & (band_assignments_df['group'] == group)
                & (band_assignments_df['roi'] == roi)
            ]
            note_counts = subset['assignment_note'].value_counts()
            two_rhythms = subset[subset['assignment_note'] == 'two_rhythms']
            summary_rows.append({
                'role': role,
                'group': group,
                'roi': roi,
                'n_two_rhythms': note_counts.get('two_rhythms', 0),
                'n_single_slow': note_counts.get('single_slow', 0),
                'n_single_fast': note_counts.get('single_fast', 0),
                'n_no_peaks': note_counts.get('no_peaks', 0),
                'slow_cf_mean': two_rhythms['slow_cf'].mean(),
                'slow_cf_sd': two_rhythms['slow_cf'].std(),
                'fast_cf_mean': two_rhythms['fast_cf'].mean(),
                'fast_cf_sd': two_rhythms['fast_cf'].std(),
                'freq_gap_mean': two_rhythms['freq_gap'].mean(),
                'freq_gap_sd': two_rhythms['freq_gap'].std(),
            })

assignment_summary_df = pd.DataFrame(summary_rows)
print(assignment_summary_df.to_string(index=False))
assignment_summary_df.to_csv(OUTPUT_DIR / 'assignment_summary.csv', index=False)
print(f'Saved artifact 4b to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 4c - IAF distance by group
# ---------------------------------------------------------------------------
fig = plot_iaf_distance_by_group(iaf_metrics_df)
fig.savefig(OUTPUT_DIR / 'iaf_distance_by_group.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 4c to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 4d - Gap distribution
# ---------------------------------------------------------------------------
fig = plot_gap_distribution(band_assignments_df, min_gap=MIN_GAP)
fig.savefig(OUTPUT_DIR / 'gap_distribution.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 4d to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 4e - Dyadic gap scatter (caregiver vs. child)
# ---------------------------------------------------------------------------
fig = plot_dyad_gap_scatter(iaf_metrics_df)
fig.savefig(OUTPUT_DIR / 'dyad_gap_scatter.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 4e to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 4f - Merged band center-frequency histograms (post k-means)
# ---------------------------------------------------------------------------
fig = plot_band_cf_histograms(
    band_assignments_df, rois=ASSIGNMENT_ROIS, bin_width=0.5,
    slow_color=SLOW_COLOR, fast_color=FAST_COLOR,
)
fig.savefig(OUTPUT_DIR / 'band_cf_histograms.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 4f to {OUTPUT_DIR}')
