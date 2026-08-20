"""Step 6 - ROI-level specparam rerun with two-band peak survival check.

Averages movie-averaged PSDs within each multi-channel ROI (from Step 5,
skipping single-channel ROIs), reruns specparam on both the ROI-averaged PSD
and each individual channel, and checks whether channel-level peaks in the
slow and fast rhythm windows survive ROI averaging. Generates artifacts 6a
(ROI vs. per-channel fit galleries, band windows marked) and 6b (peak
survival summary table and bar chart), plus QUALITY_GATE_NOTES.md.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import ensure_dir
from src.roi import average_psd_within_roi, check_peak_survival
from src.specparam_utils import extract_fit_quality, extract_peak_params, fit_specparam
from src.viz import plot_survival_rate_bars

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DERIVED_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / 'derived_data'
ROI_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '05_roi_definition'
BAND_ASSIGNMENT_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '04_band_assignment'
OUTPUT_FITS = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '06_roi_specparam_rerun')
OUTPUT_GATE = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '06_roi_quality_gate')

SPECPARAM_SETTINGS = {
    'freq_range': (3, 14),
    'peak_width_limits': (1, 3),
    'max_n_peaks': 4,
    'min_peak_height': 0.1,
    'aperiodic_mode': 'fixed',
}
SLOW_FREQ_WINDOW = (3, 7)
FAST_FREQ_WINDOW = (7, 14)
FREQ_TOLERANCE = 1.0  # Hz — max channel-to-ROI shift to count as "survived"
SURVIVAL_FLAG_THRESHOLD = 0.60

N_GALLERY_CHILDREN = 5
N_GALLERY_CAREGIVERS = 5
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 1. Load PSDs, ROI definitions (Step 5), and band assignments (Step 4)
# ---------------------------------------------------------------------------
with open(DERIVED_DIR / 'psd_data.pkl', 'rb') as f:
    psd_data = pickle.load(f)
freqs = psd_data['freqs']
channel_names = psd_data['channel_names']
psd_avg = psd_data['psd_avg']

with open(ROI_DIR / 'roi_definitions.json') as f:
    roi_definitions = json.load(f)
# Single-channel ROIs (Fz, Cz) skip the rerun — averaging one channel is trivial.
RERUN_ROIS = {name: info['channels'] for name, info in roi_definitions.items() if not info['skip_rerun']}

band_assignments_df = pd.read_csv(BAND_ASSIGNMENT_DIR / 'band_assignments.csv')
participant_meta = pd.read_csv(DERIVED_DIR / 'participant_metadata.csv').set_index('participant_id')

# ---------------------------------------------------------------------------
# 2. Fit specparam per participant x ROI (ROI-averaged + each channel),
#    then check whether channel-level slow/fast peaks survive ROI averaging
# ---------------------------------------------------------------------------
peak_survival_rows = []
models_cache = {}  # (pid, roi, 'ROI' or channel_name) -> fitted SpectralModel

for pid, psd in psd_avg.items():
    meta = participant_meta.loc[pid]
    for roi_name, roi_channels in RERUN_ROIS.items():
        roi_channels_avail = [ch for ch in roi_channels if ch in channel_names]

        roi_psd = average_psd_within_roi(psd, channel_names, roi_channels_avail)
        roi_model = fit_specparam(freqs, roi_psd, **SPECPARAM_SETTINGS)
        roi_peaks = extract_peak_params(roi_model)
        models_cache[(pid, roi_name, 'ROI')] = roi_model

        channel_peaks_list = []
        for ch in roi_channels_avail:
            ch_idx = channel_names.index(ch)
            ch_model = fit_specparam(freqs, psd[ch_idx, :], **SPECPARAM_SETTINGS)
            models_cache[(pid, roi_name, ch)] = ch_model
            for peak in extract_peak_params(ch_model):
                channel_peaks_list.append({'channel': ch, **peak})

        for band_name, band_window in [('slow', SLOW_FREQ_WINDOW), ('fast', FAST_FREQ_WINDOW)]:
            survival = check_peak_survival(channel_peaks_list, roi_peaks, band_window, freq_tolerance=FREQ_TOLERANCE)
            peak_survival_rows.append({
                'participant_id': pid, 'role': meta['role'], 'group': meta['group'],
                'roi': roi_name, 'band': band_name, **survival,
            })

peak_survival_df = pd.DataFrame(peak_survival_rows)
peak_survival_df.to_csv(OUTPUT_GATE / 'peak_survival.csv', index=False)
print(f'Fitted specparam for {len(psd_avg)} participants x {len(RERUN_ROIS)} ROIs; '
      f'{len(peak_survival_df)} participant x roi x band survival checks.')

# ---------------------------------------------------------------------------
# Artifact 6a - ROI-level fits vs. per-channel fits, band windows marked
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)


def _select_gallery_pids(roi_name):
    two_rhythm_pids = set(
        band_assignments_df.loc[
            (band_assignments_df['roi'] == roi_name) & (band_assignments_df['assignment_note'] == 'two_rhythms'),
            'participant_id',
        ]
    )
    selected = []
    for role, n in [('child', N_GALLERY_CHILDREN), ('caregiver', N_GALLERY_CAREGIVERS)]:
        role_pool = participant_meta.index[participant_meta['role'] == role].tolist()
        preferred = [p for p in role_pool if p in two_rhythm_pids]
        rest = [p for p in role_pool if p not in two_rhythm_pids]
        rng.shuffle(preferred)
        rng.shuffle(rest)
        selected.extend((preferred + rest)[:n])
    return selected


for roi_name, roi_channels in RERUN_ROIS.items():
    roi_channels_avail = [ch for ch in roi_channels if ch in channel_names]
    gallery_pids = _select_gallery_pids(roi_name)
    n_panels = len(roi_channels_avail) + 1

    for pid in gallery_pids:
        meta = participant_meta.loc[pid]
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
        for col, ch in enumerate(roi_channels_avail):
            model = models_cache[(pid, roi_name, ch)]
            model.plot(ax=axes[col], add_legend=(col == 0))
            axes[col].axvspan(*SLOW_FREQ_WINDOW, color='tab:blue', alpha=0.08)
            axes[col].axvspan(*FAST_FREQ_WINDOW, color='tab:orange', alpha=0.08)
            axes[col].set_title(f'{ch} (r2={extract_fit_quality(model)["r_squared"]:.2f})', fontsize=9)

        roi_model = models_cache[(pid, roi_name, 'ROI')]
        roi_model.plot(ax=axes[-1], add_legend=False)
        axes[-1].axvspan(*SLOW_FREQ_WINDOW, color='tab:blue', alpha=0.08, label='slow window')
        axes[-1].axvspan(*FAST_FREQ_WINDOW, color='tab:orange', alpha=0.08, label='fast window')
        axes[-1].set_title(f'ROI-averaged (r2={extract_fit_quality(roi_model)["r_squared"]:.2f})', fontsize=9)
        axes[-1].legend(fontsize=7)

        fig.suptitle(f"{pid} ({meta['role']}, {meta['group']}) — {roi_name}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUTPUT_FITS / f'{pid}_{roi_name}_bands.png', dpi=300)
        plt.close(fig)

print(f'Saved artifact 6a to {OUTPUT_FITS}')

# ---------------------------------------------------------------------------
# Artifact 6b - Peak survival summary table and bar chart
# ---------------------------------------------------------------------------
summary_rows = []
for roi_name in RERUN_ROIS:
    for band_name in ['slow', 'fast']:
        for role in ['child', 'caregiver']:
            for group in ['TD', 'ASD']:
                subset = peak_survival_df[
                    (peak_survival_df['roi'] == roi_name) & (peak_survival_df['band'] == band_name)
                    & (peak_survival_df['role'] == role) & (peak_survival_df['group'] == group)
                ]
                n_participants = subset['participant_id'].nunique()
                n_with_channel_peak = int(subset['channel_peak_present'].sum())
                n_survived = int(subset['survived'].sum())
                survival_rate = n_survived / n_with_channel_peak if n_with_channel_peak > 0 else np.nan
                summary_rows.append({
                    'roi': roi_name, 'band': band_name, 'role': role, 'group': group,
                    'n_participants': n_participants,
                    'n_with_channel_peak': n_with_channel_peak,
                    'n_survived_roi': n_survived,
                    'survival_rate': round(survival_rate, 3) if not np.isnan(survival_rate) else np.nan,
                    'mean_cf_shift': round(subset['cf_shift'].mean(), 3) if subset['cf_shift'].notna().any() else np.nan,
                })

peak_survival_summary_df = pd.DataFrame(summary_rows)
peak_survival_summary_df.to_csv(OUTPUT_GATE / 'peak_survival_summary.csv', index=False)
print(peak_survival_summary_df.to_string(index=False))

fig = plot_survival_rate_bars(peak_survival_summary_df, threshold=SURVIVAL_FLAG_THRESHOLD)
fig.savefig(OUTPUT_GATE / 'peak_survival_rate_bars.png', dpi=300)
plt.close(fig)
print(f'Saved artifact 6b to {OUTPUT_GATE}')

# ---------------------------------------------------------------------------
# QUALITY_GATE_NOTES.md
# ---------------------------------------------------------------------------
roi_band_rate = peak_survival_summary_df.groupby(['roi', 'band'])['survival_rate'].mean()
passing = roi_band_rate[roi_band_rate >= SURVIVAL_FLAG_THRESHOLD].sort_values(ascending=False)
failing = roi_band_rate[roi_band_rate < SURVIVAL_FLAG_THRESHOLD].sort_values()

lines = [
    '# Quality gate notes: ROI x band peak survival',
    '',
    f'Survival rate = n_survived_roi / n_with_channel_peak, averaged across role x group '
    f'subgroups. Threshold: {SURVIVAL_FLAG_THRESHOLD:.0%}.',
    '',
    '## Passing ROI x band combinations',
    '',
]
for (roi, band), rate in passing.items():
    lines.append(f'- **{roi} / {band}**: {rate:.0%} average survival rate')
if not len(passing):
    lines.append('(none)')

lines += ['', '## Failing ROI x band combinations', '']
for (roi, band), rate in failing.items():
    detail = peak_survival_summary_df[(peak_survival_summary_df['roi'] == roi) & (peak_survival_summary_df['band'] == band)]
    worst = detail.loc[detail['survival_rate'].idxmin()]
    lines.append(
        f"- **{roi} / {band}**: {rate:.0%} average survival rate — lowest in "
        f"{worst['role']} {worst['group']} ({worst['survival_rate']:.0%} survival, "
        f"n_with_channel_peak={worst['n_with_channel_peak']}, "
        f"mean_cf_shift={worst['mean_cf_shift']} Hz) — too much inter-channel frequency "
        f"variability for this band to be captured reliably by ROI averaging."
    )
if not len(failing):
    lines.append('(none)')

lines += ['', '## Recommendation', '']
if len(passing):
    best_roi, best_band = passing.index[0]
    lines.append(
        f'Use **{best_roi} / {best_band}** for DTF where possible (highest average survival '
        f'rate, {passing.iloc[0]:.0%}). Prefer other passing combinations above over failing ones.'
    )
else:
    lines.append('No ROI x band combination reaches the survival threshold; reconsider the ROI '
                  'averaging strategy before DTF.')

(OUTPUT_GATE / 'QUALITY_GATE_NOTES.md').write_text('\n'.join(lines) + '\n')
print(f'Saved QUALITY_GATE_NOTES.md to {OUTPUT_GATE}')
