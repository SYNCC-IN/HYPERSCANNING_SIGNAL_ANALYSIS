"""Step 4 - Two-band (slow/fast rhythm) assignment from within-ROI peak clusters.

Loads the within-ROI peak clusters from Step 3, assigns slow and fast rhythm
bands per participant x ROI (falling back to a single dominant rhythm where
the two strongest clusters aren't far enough apart), computes IAF and dyadic
distance metrics, and generates artifacts 4a (assignment strip plot), 4b
(assignment summary table), 4c (IAF distance by group), and 4d (gap
distribution).
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
    plot_band_assignment_strips, plot_dyad_gap_scatter, plot_gap_distribution,
    plot_iaf_distance_by_group,
)

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROI_CLUSTERS_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '03_roi_peak_clusters'
DERIVED_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / 'derived_data'
OUTPUT_DIR = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '04_band_assignment')

ASSIGNMENT_ROIS = ['parietal', 'sensorimotor']  # ROIs to assign bands for
MIN_GAP = 1.5  # Hz — minimum gap to split into slow/fast
PRIMARY_ROI = 'parietal'  # for IAF extraction
FALLBACK_ROI = 'sensorimotor'  # if parietal has no peak

# ---------------------------------------------------------------------------
# 1. Load within-ROI peak clusters and participant metadata
# ---------------------------------------------------------------------------
roi_peak_clusters_df = pd.read_csv(ROI_CLUSTERS_DIR / 'roi_peak_clusters.csv')
participant_meta = pd.read_csv(DERIVED_DIR / 'participant_metadata.csv').set_index('participant_id')

# ---------------------------------------------------------------------------
# 2. Assign slow/fast bands per participant x ROI
# ---------------------------------------------------------------------------
assignment_tables = []
for pid, meta in participant_meta.iterrows():
    participant_clusters = roi_peak_clusters_df[roi_peak_clusters_df['participant_id'] == pid]
    assignments = assign_bands_all_rois(participant_clusters, pid, meta['role'], ASSIGNMENT_ROIS, min_gap=MIN_GAP)
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
                'n_single_dominant': note_counts.get('single_dominant', 0),
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
