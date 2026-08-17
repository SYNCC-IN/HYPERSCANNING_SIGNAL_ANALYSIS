"""Step 4 - Assign functional band labels per participant.

Loads the peak table from Step 2, applies the posterior-alpha / frontal-theta
/ mu decision sequence, saves the assignment table, and generates artifact 4a
(strip plot of posterior alpha frequency by role x group).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_dir
from src.bands import assign_bands
from src.viz import COLORS

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QUALITY_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '02_specparam_quality'
OUTPUT_DIR = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '04_band_assignment')

# ---------------------------------------------------------------------------
# 1. Load peaks and per-participant metadata
# ---------------------------------------------------------------------------
all_peaks_df = pd.read_csv(QUALITY_DIR / 'all_peaks.csv')
participant_meta = (
    all_peaks_df[['participant_id', 'role', 'group', 'age_months']]
    .drop_duplicates(subset='participant_id')
    .set_index('participant_id')
)

# ---------------------------------------------------------------------------
# 2. Run band assignment for each participant
# ---------------------------------------------------------------------------
assignments = []
for pid, meta in participant_meta.iterrows():
    age_years = meta['age_months'] / 12.0 if meta['role'] == 'child' and pd.notna(meta['age_months']) else None
    result = assign_bands(all_peaks_df, pid, age=age_years)
    result['role'] = meta['role']
    result['group'] = meta['group']
    result['age_months'] = meta['age_months']
    assignments.append(result)

band_assignments_df = pd.DataFrame(assignments)
band_assignments_df.to_csv(OUTPUT_DIR / 'band_assignments.csv', index=False)
print(f'Saved band assignments for {len(band_assignments_df)} participants to {OUTPUT_DIR}')

# ---------------------------------------------------------------------------
# Artifact 4a - Strip plot of posterior alpha frequency by role x group
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
rng = np.random.default_rng(42)

categories = [('child', 'TD'), ('child', 'ASD'), ('caregiver', 'TD'), ('caregiver', 'ASD')]
x_positions = {cat: i for i, cat in enumerate(categories)}

for cat in categories:
    role, group = cat
    subset = band_assignments_df[(band_assignments_df['role'] == role) & (band_assignments_df['group'] == group)]
    vals = subset['posterior_alpha_cf'].dropna()
    jitter = rng.uniform(-0.15, 0.15, size=len(vals))
    ax.scatter(np.full(len(vals), x_positions[cat]) + jitter, vals, color=COLORS.get(group, 'gray'), alpha=0.7, s=25)

for freq in [4, 8, 12]:
    ax.axhline(freq, color='gray', linestyle='--', lw=0.8, alpha=0.7)

ax.set_xticks(list(x_positions.values()))
ax.set_xticklabels([f'{role}\n{group}' for role, group in categories])
ax.set_ylabel('Posterior alpha center frequency (Hz)')
ax.set_title('Posterior dominant rhythm by role and group')

n_no_posterior = band_assignments_df['posterior_alpha_cf'].isna().sum()
n_theta_only = (
    band_assignments_df['frontal_theta_cf'].notna() & band_assignments_df['posterior_alpha_cf'].isna()
).sum()
n_mu = band_assignments_df['mu_cf'].notna().sum()
annotation = (
    f'No posterior peak: {n_no_posterior}\n'
    f'Theta-only (no posterior): {n_theta_only}\n'
    f'Mu detected: {n_mu}'
)
ax.text(0.98, 0.02, annotation, transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'posterior_alpha_strip_plot.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 4a to {OUTPUT_DIR}')
