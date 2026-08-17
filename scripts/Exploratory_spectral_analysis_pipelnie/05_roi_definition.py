"""Step 5 - ROI definition: theory-driven groupings validated against peak prevalence.

Loads prevalence data from Step 3, defines theory-driven ROIs, validates each
against its associated rhythm's peak prevalence, decides whether to merge
parietal + occipital, and saves the final ROI definitions plus artifact 5a.
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
from src.roi import define_rois_theory, validate_roi_with_prevalence

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QUALITY_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '02_specparam_quality'
OUTPUT_DIR = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '05_roi_definition')

MIN_PREVALENCE = 0.5
# ROI -> target rhythm frequency band used for validation. Temporal has no
# associated rhythm in this design and is left unvalidated.
ROI_TARGET_BAND = {
    'frontal_midline': (3, 7),
    'frontal_lateral': (3, 7),
    'central': (7, 13),
    'parietal': (7, 13),
    'occipital': (7, 13),
}
ROI_TARGET_LABEL = {
    'frontal_midline': 'theta', 'frontal_lateral': 'theta',
    'central': 'mu', 'parietal': 'alpha', 'occipital': 'alpha',
}
MERGE_THRESHOLD_HZ = 1.0

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2',
]

# ---------------------------------------------------------------------------
# 1. Load peaks (substrate for prevalence, as computed/saved in Step 3)
# ---------------------------------------------------------------------------
all_peaks_df = pd.read_csv(QUALITY_DIR / 'all_peaks.csv')
# fit_quality_df has one row per channel for every participant regardless of
# whether a peak was detected, giving the true cohort size for prevalence
# denominators (all_peaks_df omits participants with zero peaks entirely).
fit_quality_df = pd.read_csv(QUALITY_DIR / 'fit_quality.csv')
n_participants_by_role = {
    role: fit_quality_df.loc[fit_quality_df['role'] == role, 'participant_id'].nunique()
    for role in ['child', 'caregiver']
}

# ---------------------------------------------------------------------------
# 2. Define theory-driven ROIs
# ---------------------------------------------------------------------------
theory_rois = define_rois_theory()

# ---------------------------------------------------------------------------
# 3. Validate each ROI against its target band's prevalence, per role
# ---------------------------------------------------------------------------
rows = []
for roi_name, channels in theory_rois.items():
    target_band = ROI_TARGET_BAND.get(roi_name)
    row = {
        'ROI': roi_name,
        'channels': ','.join(channels),
        'target_band': ROI_TARGET_LABEL.get(roi_name, 'not validated'),
    }
    if target_band is None:
        row['mean_prevalence_children'] = None
        row['mean_prevalence_caregivers'] = None
        row['passes_threshold'] = None
    else:
        child_prev = compute_peak_prevalence(all_peaks_df, CHANNEL_NAMES, [target_band], 'child', n_participants=n_participants_by_role['child'])
        cg_prev = compute_peak_prevalence(all_peaks_df, CHANNEL_NAMES, [target_band], 'caregiver', n_participants=n_participants_by_role['caregiver'])
        passes_child, child_vals = validate_roi_with_prevalence(child_prev, channels, target_band, MIN_PREVALENCE)
        passes_cg, cg_vals = validate_roi_with_prevalence(cg_prev, channels, target_band, MIN_PREVALENCE)
        row['mean_prevalence_children'] = round(float(child_vals.mean()), 3)
        row['mean_prevalence_caregivers'] = round(float(cg_vals.mean()), 3)
        row['passes_threshold'] = bool(passes_child and passes_cg)
        if not passes_child:
            print(f'{roi_name}: children below threshold at ' +
                  ', '.join(f'{ch}={v:.2f}' for ch, v in child_vals.items() if v < MIN_PREVALENCE))
        if not passes_cg:
            print(f'{roi_name}: caregivers below threshold at ' +
                  ', '.join(f'{ch}={v:.2f}' for ch, v in cg_vals.items() if v < MIN_PREVALENCE))
    rows.append(row)

roi_validation_df = pd.DataFrame(rows)
roi_validation_df.to_csv(OUTPUT_DIR / 'roi_validation_table.csv', index=False)
print(roi_validation_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Decide whether to merge parietal + occipital based on peak frequency similarity
# ---------------------------------------------------------------------------
alpha_peaks = all_peaks_df[(all_peaks_df['center_freq'] >= 7) & (all_peaks_df['center_freq'] <= 13)]
parietal_mean_cf = alpha_peaks.loc[alpha_peaks['channel'].isin(theory_rois['parietal']), 'center_freq'].mean()
occipital_mean_cf = alpha_peaks.loc[alpha_peaks['channel'].isin(theory_rois['occipital']), 'center_freq'].mean()
merge_diff = abs(parietal_mean_cf - occipital_mean_cf)
merge_posterior = merge_diff < MERGE_THRESHOLD_HZ

print(f'\nParietal alpha CF: {parietal_mean_cf:.2f} Hz | Occipital alpha CF: {occipital_mean_cf:.2f} Hz '
      f'| diff: {merge_diff:.2f} Hz -> merge={merge_posterior}')

final_rois = dict(theory_rois)
if merge_posterior:
    final_rois['posterior'] = final_rois.pop('parietal') + final_rois.pop('occipital')

with open(OUTPUT_DIR / 'final_rois.json', 'w') as f:
    json.dump(final_rois, f, indent=2)
print(f'Saved final ROI definitions to {OUTPUT_DIR / "final_rois.json"}')

# ---------------------------------------------------------------------------
# Artifact 5a - ROI validation table (CSV already saved; also render as figure)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 0.6 * len(roi_validation_df) + 1))
ax.axis('off')
display_df = roi_validation_df.fillna('-')
table = ax.table(
    cellText=display_df.values, colLabels=display_df.columns,
    cellLoc='center', loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
ax.set_title('ROI validation summary', fontsize=12, pad=20)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'roi_validation_table.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved artifact 5a to {OUTPUT_DIR}')
