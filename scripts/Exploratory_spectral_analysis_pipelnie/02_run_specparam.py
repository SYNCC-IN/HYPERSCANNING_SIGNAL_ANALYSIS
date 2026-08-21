"""Step 2 - Run specparam on movie-averaged PSDs, channel-by-channel.

Loads PSDs saved in Step 1, fits specparam per participant x channel, saves
peak/quality tables, and generates artifacts 2a (fit gallery), 2b (fit
quality summary), and 2c (peak detection failure log).
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mne

from src.io_utils import ensure_dir
from src.specparam_utils import batch_fit_specparam, fit_specparam, extract_fit_quality, extract_peak_params
from src.viz import make_montage_info

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DERIVED_DIR = PROJECT_ROOT / 'Exploratory_spectral_analysis' / 'derived_data'
OUTPUT_GALLERY = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '02_specparam_fits_gallery')
OUTPUT_QUALITY = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '02_specparam_quality')
OUTPUT_NO_PEAKS = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '02_specparam_no_peaks')

FREQ_RANGE = (3, 15)
PEAK_WIDTH_LIMITS = (1, 3) #(1,6)
MAX_N_PEAKS = 4
MIN_PEAK_HEIGHT = 0.05
APERIODIC_MODE = 'fixed'

GALLERY_CHANNELS = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
N_GALLERY_CHILDREN = 61
N_GALLERY_CAREGIVERS = 61
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 1. Load PSDs from Step 1
# ---------------------------------------------------------------------------
with open(DERIVED_DIR / 'psd_data.pkl', 'rb') as f:
    psd_data = pickle.load(f)
freqs = psd_data['freqs']
channel_names = psd_data['channel_names']
psd_avg = psd_data['psd_avg']

participant_meta = pd.read_csv(DERIVED_DIR / 'participant_metadata.csv')
meta_by_pid = participant_meta.set_index('participant_id')

# ---------------------------------------------------------------------------
# 2. Batch-fit specparam per participant
# ---------------------------------------------------------------------------
all_peaks = []
all_quality = []

for pid, psd in psd_avg.items():
    peaks_df, quality_df = batch_fit_specparam(
        freqs, psd, channel_names,
        freq_range=FREQ_RANGE, peak_width_limits=PEAK_WIDTH_LIMITS,
        max_n_peaks=MAX_N_PEAKS, min_peak_height=MIN_PEAK_HEIGHT,
        aperiodic_mode=APERIODIC_MODE,
    )
    meta = meta_by_pid.loc[pid]
    for df in (peaks_df, quality_df):
        df['participant_id'] = pid
        df['role'] = meta['role']
        df['group'] = meta['group']
        df['age_months'] = meta['age_months']
    all_peaks.append(peaks_df)
    all_quality.append(quality_df)

all_peaks_df = pd.concat(all_peaks, ignore_index=True)
fit_quality_df = pd.concat(all_quality, ignore_index=True)

all_peaks_df.to_csv(OUTPUT_QUALITY / 'all_peaks.csv', index=False)
fit_quality_df.to_csv(OUTPUT_QUALITY / 'fit_quality.csv', index=False)
print(f'Saved {len(all_peaks_df)} peaks and {len(fit_quality_df)} channel-fit rows to {OUTPUT_QUALITY}')

# ---------------------------------------------------------------------------
# Artifact 2a - specparam fit gallery (10 children + 10 caregivers, reproducible)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)


def _sample_participants(role, n):
    pool = meta_by_pid.index[meta_by_pid['role'] == role].tolist()
    n = min(n, len(pool))
    return rng.choice(pool, size=n, replace=False)


gallery_pids = list(_sample_participants('child', N_GALLERY_CHILDREN)) + \
    list(_sample_participants('caregiver', N_GALLERY_CAREGIVERS))

for pid in gallery_pids:
    meta = meta_by_pid.loc[pid]
    psd = psd_avg[pid]
    fig, axes = plt.subplots(3,3, figsize=(9, 7))
    for ch_name, ax in zip(GALLERY_CHANNELS, axes.flat):
        ch_idx = channel_names.index(ch_name)
        sm = fit_specparam(freqs, psd[ch_idx, :], FREQ_RANGE, PEAK_WIDTH_LIMITS,
                            MAX_N_PEAKS, MIN_PEAK_HEIGHT, APERIODIC_MODE)
        quality = extract_fit_quality(sm)
        n_peaks = len(extract_peak_params(sm))
        sm.plot(ax=ax, add_legend=(ch_name == GALLERY_CHANNELS[0]))
        ax.set_title(
            f"{ch_name}  r2={quality['r_squared']:.2f}  n_peaks={n_peaks}",
            fontsize=9,
        )
    fig.suptitle(f"{pid}  ({meta['role']}, {meta['group']})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_GALLERY / f'{pid}_specparam_fits.png', dpi=300)
    plt.close(fig)

print(f'Saved specparam fit gallery for {len(gallery_pids)} participants to {OUTPUT_GALLERY}')

# ---------------------------------------------------------------------------
# Artifact 2b - Fit quality summary (r^2 histogram by role)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
bins = np.arange(0, 1.02, 0.02)
for role, color in [('child', '#2c3e50'), ('caregiver', '#7f8c8d')]:
    vals = fit_quality_df.loc[fit_quality_df['role'] == role, 'r_squared'].dropna()
    ax.hist(vals, bins=bins, alpha=0.55, label=role, color=color)
ax.axvline(0.90, color='red', linestyle='--', lw=1.5, label='r^2 = 0.90 threshold')
ax.set_xlabel('r^2')
ax.set_ylabel('Count (channel x participant fits)')
ax.set_title('specparam fit quality by role')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_QUALITY / 'fit_quality_r2_histogram.png', dpi=300)
plt.close(fig)

# ---------------------------------------------------------------------------
# Artifact 2c - Peak detection failure log + topomap of zero-peak fraction
# ---------------------------------------------------------------------------
no_peaks_df = fit_quality_df[fit_quality_df['n_peaks'] == 0]
no_peaks_df.to_csv(OUTPUT_NO_PEAKS / 'no_peaks_log.csv', index=False)
print(f'{len(no_peaks_df)} / {len(fit_quality_df)} channel fits had zero peaks.')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
info = make_montage_info(channel_names)
for ax, role in zip(axes, ['child', 'caregiver']):
    role_quality = fit_quality_df[fit_quality_df['role'] == role]
    zero_peak_fraction = (
        role_quality.assign(no_peak=role_quality['n_peaks'] == 0)
        .groupby('channel')['no_peak'].mean()
        .reindex(channel_names)
        .values
    )
    im, _ = mne.viz.plot_topomap(
        zero_peak_fraction, info, axes=ax, show=False, vlim=(0, 1), cmap='Reds', contours=0,
    )
    ax.set_title(f'{role}: fraction with 0 peaks', fontsize=10)
colorbar = plt.colorbar(im, ax=ax, shrink=0.7)
colorbar.set_label('Fraction of fits with 0 peaks', fontsize=9) 

fig.suptitle('Peak detection failure rate by channel')
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUTPUT_NO_PEAKS / 'no_peaks_topomap.png', dpi=300)
plt.close(fig)

print(f'Saved fit quality and no-peaks artifacts to {OUTPUT_QUALITY} and {OUTPUT_NO_PEAKS}')
