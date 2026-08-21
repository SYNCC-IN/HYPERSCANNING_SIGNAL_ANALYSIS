"""Step 5 - ROI definition: peak-prevalence validation per slow/fast band.

Loads channel-level peaks from Step 2, computes peak prevalence within the
slow and fast rhythm frequency windows established in Step 4, validates each
ROI's viability separately per band and per role x group subgroup, and saves
the final ROI definitions (with per-band viability and a skip_rerun flag for
single-channel ROIs) plus artifact 5a.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_dir
from src.peaks import compute_peak_prevalence
from src.roi import validate_roi_two_bands, validate_roi_individual_bands
from src.viz import plot_roi_validation_heatmap, plot_roi_validation_heatmap_individual

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QUALITY_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '02_specparam_quality'
BAND_ASSIGNMENT_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '04_band_assignment'
OUTPUT_DIR = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '05_roi_definition')

ROIS = {
    'frontal-midline':  ['Fz'],
    'sensorimotor':     ['C3', 'C4'],
    'central-midline':  ['Cz'],
    'parietal':         ['P3', 'Pz', 'P4'],
    'occipital':        ['O1', 'O2'],
    'lateral-temporal': ['T7', 'T8'],
    'temporo-parietal': ['P7', 'P8'],
}

# Frequency windows for prevalence validation, derived from step 4 distributions
SLOW_FREQ_WINDOW = (3, 7)    # Hz — covers slow rhythm range across roles
FAST_FREQ_WINDOW = (7, 14)   # Hz — covers fast rhythm range across roles
MIN_PREVALENCE = 0.50         # minimum fraction of participants with a detected peak

ROLE_LABELS = {'child': 'child', 'cg': 'caregiver'}  # column-label -> all_peaks_df role value
GROUPS = ['TD', 'ASD']

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2',
]

# ---------------------------------------------------------------------------
# 1. Load peaks (Step 2/3 substrate) and Step 4 band assignments
# ---------------------------------------------------------------------------
all_peaks_df = pd.read_csv(QUALITY_DIR / 'all_peaks.csv')
fit_quality_df = pd.read_csv(QUALITY_DIR / 'fit_quality.csv')
band_assignments_df = pd.read_csv(BAND_ASSIGNMENT_DIR / 'band_assignments.csv')

for roi_name in band_assignments_df['roi'].unique():
    subset = band_assignments_df[band_assignments_df['roi'] == roi_name]
    two_rhythm_rate = (subset['assignment_note'] == 'two_rhythms').mean()
    print(f'[step 4 cross-check] {roi_name}: {two_rhythm_rate:.0%} of participant x role rows assigned two_rhythms')

# ---------------------------------------------------------------------------
# 2. Validate each ROI's peak prevalence, per band x role x group
# ---------------------------------------------------------------------------
rows = []
for roi_name, roi_channels in ROIS.items():
    row = {'ROI': roi_name, 'channels': ','.join(roi_channels)}
    validations = {}
    for role_label, role_actual in ROLE_LABELS.items():
        for group in GROUPS:
            n_participants = fit_quality_df[
                (fit_quality_df['role'] == role_actual) & (fit_quality_df['group'] == group)
            ]['participant_id'].nunique()
            prevalence_df = compute_peak_prevalence(
                all_peaks_df, CHANNEL_NAMES, [SLOW_FREQ_WINDOW, FAST_FREQ_WINDOW],
                role_actual, group=group, n_participants=n_participants,
            )
            validation = validate_roi_two_bands(
                prevalence_df, roi_channels, SLOW_FREQ_WINDOW, FAST_FREQ_WINDOW, MIN_PREVALENCE,
            )
            validations[(role_label, group)] = validation
            row[f'slow_prev_{role_label}_{group}'] = round(validation['slow_prevalence'], 3)
            row[f'fast_prev_{role_label}_{group}'] = round(validation['fast_prevalence'], 3)

    row['slow_viable'] = all(v['slow_viable'] for v in validations.values())
    row['fast_viable'] = all(v['fast_viable'] for v in validations.values())
    rows.append(row)

column_order = [
    'ROI', 'channels',
    'slow_prev_child_TD', 'slow_prev_child_ASD', 'slow_prev_cg_TD', 'slow_prev_cg_ASD',
    'fast_prev_child_TD', 'fast_prev_child_ASD', 'fast_prev_cg_TD', 'fast_prev_cg_ASD',
    'slow_viable', 'fast_viable',
]
roi_validation_df = pd.DataFrame(rows)[column_order]
roi_validation_df.to_csv(OUTPUT_DIR / 'roi_validation.csv', index=False)
print(roi_validation_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Save final ROI definitions with per-band viability and skip_rerun flag
# ---------------------------------------------------------------------------
roi_definitions = {}
for _, row in roi_validation_df.iterrows():
    roi_name = row['ROI']
    roi_channels = ROIS[roi_name]
    roi_definitions[roi_name] = {
        'channels': roi_channels,
        'slow_viable': bool(row['slow_viable']),
        'fast_viable': bool(row['fast_viable']),
        'skip_rerun': len(roi_channels) <= 1,
    }

with open(OUTPUT_DIR / 'roi_definitions.json', 'w') as f:
    json.dump(roi_definitions, f, indent=2)
print(f'Saved ROI definitions to {OUTPUT_DIR / "roi_definitions.json"}')

# ---------------------------------------------------------------------------
# Artifact 5a - ROI validation heatmap
# ---------------------------------------------------------------------------
fig = plot_roi_validation_heatmap(roi_validation_df, min_prevalence=MIN_PREVALENCE)
fig.savefig(OUTPUT_DIR / 'roi_validation_heatmap.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 5a to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 5b - ROI validation heatmap, individualized band windows
# ---------------------------------------------------------------------------
# Uses each participant's own slow_cf/slow_bw and fast_cf/fast_bw (from Step 4,
# looked up per ROI — not borrowed from the primary/parietal ROI) instead of
# the fixed group-level SLOW/FAST_FREQ_WINDOW used above.
individual_rows = []
for roi_name, roi_channels in ROIS.items():
    individual_presence_df = validate_roi_individual_bands(
        all_peaks_df, band_assignments_df, roi_channels, roi_name,
        slow_window=SLOW_FREQ_WINDOW, fast_window=FAST_FREQ_WINDOW,
    )
    row = {'roi': roi_name}
    for role_label, role_actual in ROLE_LABELS.items():
        for group in GROUPS:
            subset = individual_presence_df[
                (individual_presence_df['role'] == role_actual) & (individual_presence_df['group'] == group)
            ]
            row[f'slow_prev_{role_label}_{group}'] = round(float(subset['slow_present'].mean()), 3)
            row[f'fast_prev_{role_label}_{group}'] = round(float(subset['fast_present'].mean()), 3)
    individual_rows.append(row)

roi_validation_individual_df = pd.DataFrame(individual_rows)
roi_validation_individual_df.to_csv(OUTPUT_DIR / 'roi_validation_individual.csv', index=False)
print(roi_validation_individual_df.to_string(index=False))

fig = plot_roi_validation_heatmap_individual(roi_validation_individual_df, min_prevalence=MIN_PREVALENCE)
fig.savefig(OUTPUT_DIR / 'roi_validation_heatmap_individual.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 5b to {OUTPUT_DIR}')
