"""Step 6 - ROI-level specparam rerun.

Averages movie-averaged PSDs within each final ROI, reruns specparam with the
same settings as Step 2, and checks that peaks survive averaging by comparing
against the best-fitting individual channel within each ROI. Generates
artifacts 6a (ROI vs. best-channel fit comparison) and 6b (ROI quality gate
table).
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
from src.roi import average_psd_within_roi
from src.specparam_utils import fit_specparam, extract_peak_params, extract_fit_quality

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DERIVED_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / 'derived_data'
ROI_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / '05_roi_definition'
OUTPUT_FITS = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '06_roi_specparam_rerun')
OUTPUT_GATE = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '06_roi_quality_gate')

FREQ_RANGE = (3, 25)
PEAK_WIDTH_LIMITS = (1, 6)
MAX_N_PEAKS = 4
MIN_PEAK_HEIGHT = 0.1
APERIODIC_MODE = 'fixed'

N_GALLERY_CHILDREN = 5
N_GALLERY_CAREGIVERS = 5
RANDOM_SEED = 42
PREVALENCE_FLAG_THRESHOLD = 0.60

# ---------------------------------------------------------------------------
# 1. Load PSDs and final ROI definitions
# ---------------------------------------------------------------------------
with open(DERIVED_DIR / 'psd_data.pkl', 'rb') as f:
    psd_data = pickle.load(f)
freqs = psd_data['freqs']
channel_names = psd_data['channel_names']
psd_avg = psd_data['psd_avg']

with open(ROI_DIR / 'final_rois.json') as f:
    final_rois = json.load(f)

participant_meta = pd.read_csv(DERIVED_DIR / 'participant_metadata.csv').set_index('participant_id')

# ---------------------------------------------------------------------------
# 2. Fit specparam per participant x ROI (ROI-averaged PSD) and best channel
# ---------------------------------------------------------------------------
roi_peak_rows = []
roi_quality_rows = []
best_channel_by_pid_roi = {}

for pid, psd in psd_avg.items():
    meta = participant_meta.loc[pid]
    for roi_name, roi_channels in final_rois.items():
        roi_channels_avail = [ch for ch in roi_channels if ch in channel_names]

        roi_psd = average_psd_within_roi(psd, channel_names, roi_channels_avail)
        roi_model = fit_specparam(freqs, roi_psd, FREQ_RANGE, PEAK_WIDTH_LIMITS,
                                   MAX_N_PEAKS, MIN_PEAK_HEIGHT, APERIODIC_MODE)
        roi_peaks = extract_peak_params(roi_model)
        roi_quality = extract_fit_quality(roi_model)

        # Best-fitting individual channel within the ROI (by r_squared)
        channel_fits = {}
        for ch in roi_channels_avail:
            ch_idx = channel_names.index(ch)
            ch_model = fit_specparam(freqs, psd[ch_idx, :], FREQ_RANGE, PEAK_WIDTH_LIMITS,
                                      MAX_N_PEAKS, MIN_PEAK_HEIGHT, APERIODIC_MODE)
            channel_fits[ch] = ch_model
        best_channel = max(channel_fits, key=lambda ch: extract_fit_quality(channel_fits[ch])['r_squared'])
        best_channel_by_pid_roi[(pid, roi_name)] = (best_channel, channel_fits[best_channel])

        for peak in roi_peaks:
            roi_peak_rows.append({
                'participant_id': pid, 'role': meta['role'], 'group': meta['group'],
                'roi': roi_name, **peak,
            })
        roi_quality_rows.append({
            'participant_id': pid, 'role': meta['role'], 'group': meta['group'], 'roi': roi_name,
            'r_squared': roi_quality['r_squared'], 'n_peaks': len(roi_peaks),
            'best_channel': best_channel, 'best_channel_r_squared': extract_fit_quality(channel_fits[best_channel])['r_squared'],
        })

roi_peaks_df = pd.DataFrame(roi_peak_rows)
roi_quality_df = pd.DataFrame(roi_quality_rows)
roi_peaks_df.to_csv(OUTPUT_GATE / 'roi_peaks.csv', index=False)
print(f'Fitted specparam for {len(psd_avg)} participants x {len(final_rois)} ROIs.')

# ---------------------------------------------------------------------------
# Artifact 6a - ROI-level fits vs. best individual channel (10 representative participants)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)
children = participant_meta.index[participant_meta['role'] == 'child'].tolist()
caregivers = participant_meta.index[participant_meta['role'] == 'caregiver'].tolist()
gallery_pids = list(rng.choice(children, size=min(N_GALLERY_CHILDREN, len(children)), replace=False)) + \
    list(rng.choice(caregivers, size=min(N_GALLERY_CAREGIVERS, len(caregivers)), replace=False))

for pid in gallery_pids:
    meta = participant_meta.loc[pid]
    psd = psd_avg[pid]
    n_rois = len(final_rois)
    fig, axes = plt.subplots(n_rois, 2, figsize=(9, 3 * n_rois))
    for row, (roi_name, roi_channels) in enumerate(final_rois.items()):
        roi_channels_avail = [ch for ch in roi_channels if ch in channel_names]
        roi_psd = average_psd_within_roi(psd, channel_names, roi_channels_avail)
        roi_model = fit_specparam(freqs, roi_psd, FREQ_RANGE, PEAK_WIDTH_LIMITS,
                                   MAX_N_PEAKS, MIN_PEAK_HEIGHT, APERIODIC_MODE)
        best_channel, best_model = best_channel_by_pid_roi[(pid, roi_name)]

        roi_model.plot(ax=axes[row, 0], add_legend=(row == 0))
        axes[row, 0].set_title(f'ROI: {roi_name} (r2={extract_fit_quality(roi_model)["r_squared"]:.2f})', fontsize=9)
        best_model.plot(ax=axes[row, 1], add_legend=False)
        axes[row, 1].set_title(f'Best channel: {best_channel} (r2={extract_fit_quality(best_model)["r_squared"]:.2f})', fontsize=9)
    fig.suptitle(f"{pid}  ({meta['role']}, {meta['group']})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_FITS / f'{pid}_roi_vs_channel.png', dpi=300)
    plt.close(fig)

print(f'Saved artifact 6a for {len(gallery_pids)} participants to {OUTPUT_FITS}')

# Dominant peak per participant x ROI (highest power), so mixed theta/alpha/beta
# peaks don't get blended into a meaningless average center frequency.
dominant_peaks_df = (
    roi_peaks_df.sort_values('power', ascending=False)
    .drop_duplicates(subset=['participant_id', 'roi'], keep='first')
)

# ---------------------------------------------------------------------------
# Artifact 6b - ROI quality gate table
# ---------------------------------------------------------------------------
summary_rows = []
for roi_name in final_rois:
    for role in ['child', 'caregiver']:
        for group in ['TD', 'ASD']:
            mask = (dominant_peaks_df['roi'] == roi_name) & (dominant_peaks_df['role'] == role) & (dominant_peaks_df['group'] == group)
            n_total = participant_meta[(participant_meta['role'] == role) & (participant_meta['group'] == group)].shape[0]
            subset = dominant_peaks_df[mask]
            n_with_peak = subset['participant_id'].nunique()
            prevalence = n_with_peak / n_total if n_total > 0 else np.nan
            summary_rows.append({
                'roi': roi_name, 'role': role, 'group': group,
                'mean_center_freq': subset['center_freq'].mean(),
                'sd_center_freq': subset['center_freq'].std(),
                'mean_power': subset['power'].mean(),
                'sd_power': subset['power'].std(),
                'prevalence_pct': round(100 * prevalence, 1) if not np.isnan(prevalence) else np.nan,
                'n_total': n_total,
                'flag_low_prevalence': bool(prevalence < PREVALENCE_FLAG_THRESHOLD) if not np.isnan(prevalence) else True,
            })

roi_quality_gate_df = pd.DataFrame(summary_rows)
roi_quality_gate_df.to_csv(OUTPUT_GATE / 'roi_quality_gate.csv', index=False)

n_flagged = roi_quality_gate_df['flag_low_prevalence'].sum()
print(f'ROI quality gate: {n_flagged}/{len(roi_quality_gate_df)} (roi x role x group) cells below '
      f'{PREVALENCE_FLAG_THRESHOLD:.0%} prevalence.')
print(roi_quality_gate_df.to_string(index=False))
