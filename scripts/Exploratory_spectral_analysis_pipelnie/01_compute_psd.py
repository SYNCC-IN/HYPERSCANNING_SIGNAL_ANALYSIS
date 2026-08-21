"""Step 1 - Compute multitaper PSD per participant x movie x channel.

Loads ICA-cleaned EEG NetCDF files, computes multitaper PSD for each
participant x movie, averages across movies per participant, and saves
everything for downstream steps. Generates artifacts 1a (individual PSD
plots) and 1b (grand average PSD by role).
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

from src.io_utils import get_participant_files, load_eeg_nc, trim_to_event_window, ensure_dir
from src.psd import compute_psd_multitaper, average_psd_across_conditions
from src.viz import COLORS, ROLE_LINESTYLE

plt.style.use('seaborn-v0_8-whitegrid')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# These paths work for JZygierwicz. Anyone using this script has to set his/her own paths.
FUW_path = Path('/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/'
                '.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/'
                'WP4          - Joint study/UniWAW Data collection/'
                'UNIWAW_EEG_exported_BY_TASKS/ICA_output/EEG_ICA_CLEANED') #FUW path
home_path=Path(
    "/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/"
    ".shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/"
    "WP4          - Joint study/UniWAW Data collection/UNIWAW_EEG_exported_BY_TASKS/"
    "ICA_output/EEG_ICA_CLEANED"
)


DATA_DIR = home_path
OUTPUT_INDIVIDUAL = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '01_psd_individual')
OUTPUT_GRAND_AVG = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / '01_psd_grand_average')
DERIVED_DIR = ensure_dir(PROJECT_ROOT / 'Exploratory_spectral_analysis' / 'derived_data')

FMIN, FMAX, BANDWIDTH = 1.0, 30.0, 2.0
CHANNELS_9 = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'O1']
CHANNELS_4 = ['Fz', 'Cz', 'Pz', 'O1']
MOVIE_COLORS = {'Peppa': '#1b9e77', 'Incredibles': '#d95f02', 'Brave': '#7570b3'}

# Dyad IDs excluded from analysis (both child and caregiver dropped), e.g. due to
# severe artifact contamination or dead/flat channels found during QC inspection.
#EXCLUDED_DYADS = ['W_027','W_042', 'W_044', 'W_072','W_114']  #['W_029', 'W_071','W_074','W_116']

EXCLUDED_DYADS = ['W_008', 'W_012', 'W_025', 'W_027', 'W_035', 'W_037', 'W_042', 'W_044', 'W_057', 'W_072', 'W_074','W_114']

# ---------------------------------------------------------------------------
# 1. Discover files, compute PSD per participant x movie
# ---------------------------------------------------------------------------
files_df = get_participant_files(DATA_DIR)
if EXCLUDED_DYADS:
    n_before = len(files_df)
    files_df = files_df[~files_df['dyad_id'].isin(EXCLUDED_DYADS)].reset_index(drop=True)
    print(f'Excluded {n_before - len(files_df)} participant(s) from {len(EXCLUDED_DYADS)} excluded dyad(s): {EXCLUDED_DYADS}')
print(f'Found {len(files_df)} EEG files across {files_df["dyad_id"].nunique()} dyads.')

freqs_common = None
channel_names = None
psd_by_movie = {}          # participant_id -> {movie: psd (n_ch, n_freq)}
participant_records = {}   # participant_id -> metadata dict

for _, row in files_df.iterrows():
    rec = load_eeg_nc(row['filepath'])
    participant_id = f"{rec['dyad_id']}_{rec['role_code']}"

    for movie in rec['movies']:
        data, time = trim_to_event_window(rec['data'], rec['time'], movie['duration_s'], start=movie['start_s'])
        freqs, psd = compute_psd_multitaper(data, rec['sfreq'], FMIN, FMAX, BANDWIDTH)

        if freqs_common is None:
            freqs_common = freqs
            channel_names = rec['channel_names']
        else:
            psd = np.vstack([np.interp(freqs_common, freqs, psd[ch, :]) for ch in range(psd.shape[0])])

        psd_by_movie.setdefault(participant_id, {})[movie['name']] = psd

    participant_records[participant_id] = {
        'participant_id': participant_id,
        'dyad_id': rec['dyad_id'],
        'role': rec['role'],
        'role_code': rec['role_code'],
        'group': rec['group'],
        'age_months': rec['age_months'],
        'sex': rec['sex'],
    }

participant_meta = pd.DataFrame(participant_records.values()).sort_values('participant_id').reset_index(drop=True)
print(f'Computed PSD for {len(psd_by_movie)} participants.')

# ---------------------------------------------------------------------------
# 2. Movie-averaged PSD per participant
# ---------------------------------------------------------------------------
psd_avg = {pid: average_psd_across_conditions(movies) for pid, movies in psd_by_movie.items()}

# ---------------------------------------------------------------------------
# 3. Save PSDs for downstream steps
# ---------------------------------------------------------------------------
with open(DERIVED_DIR / 'psd_data.pkl', 'wb') as f:
    pickle.dump({
        'freqs': freqs_common,
        'channel_names': channel_names,
        'psd_by_movie': psd_by_movie,
        'psd_avg': psd_avg,
    }, f)
participant_meta.to_csv(DERIVED_DIR / 'participant_metadata.csv', index=False)
print(f'Saved PSD data to {DERIVED_DIR / "psd_data.pkl"} and participant metadata CSV.')

# ---------------------------------------------------------------------------
# Group demographics: number of dyads and mean age (months) +/- SEM, by group
# ---------------------------------------------------------------------------
dyad_meta = participant_meta.drop_duplicates(subset='dyad_id')[['dyad_id', 'group', 'age_months']]
demographics_rows = []
for group in ['TD', 'ASD', 'P', 'ASD+P']:
    ages = dyad_meta.loc[dyad_meta['group'] == group, 'age_months'].dropna()
    demographics_rows.append({
        'group': group,
        'n_dyads': int((dyad_meta['group'] == group).sum()),
        'mean_age_months': ages.mean() if len(ages) else np.nan,
        'sem_age_months': ages.std(ddof=1) / np.sqrt(len(ages)) if len(ages) > 1 else np.nan,
    })
demographics_df = pd.DataFrame(demographics_rows)
demographics_df.to_csv(DERIVED_DIR / 'group_demographics.csv', index=False)
print(demographics_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Artifact 1a - Individual PSD plots (3x3 grid, one figure per participant)
# ---------------------------------------------------------------------------
for pid, movies in psd_by_movie.items():
    meta = participant_meta.set_index('participant_id').loc[pid]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True)
    for ch_name, ax in zip(CHANNELS_9, axes.flat):
        if ch_name not in channel_names:
            ax.set_visible(False)
            continue
        ch_idx = channel_names.index(ch_name)
        for movie, psd in movies.items():
            ax.plot(freqs_common, psd[ch_idx, :], label=movie, color=MOVIE_COLORS.get(movie), lw=1.2)
        ax.set_yscale('log')
        ax.set_title(ch_name, fontsize=9)
        ax.set_xlim(FMIN, FMAX)
    for ax in axes[-1, :]:
        ax.set_xlabel('Frequency (Hz)')
    for ax in axes[:, 0]:
        ax.set_ylabel('PSD (uV^2/Hz)')
    axes[0, 0].legend(fontsize=7, loc='upper right')
    fig.suptitle(f"{pid}  ({meta['role']}, {meta['group']})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_INDIVIDUAL / f'{pid}_psd.png', dpi=300)
    plt.close(fig)

print(f'Saved {len(psd_by_movie)} individual PSD plots to {OUTPUT_INDIVIDUAL}')

# ---------------------------------------------------------------------------
# Artifact 1b - Grand average PSD by role (children | caregivers)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, role in zip(axes, ['child', 'caregiver']):
    role_pids = participant_meta.loc[participant_meta['role'] == role, 'participant_id']
    stack = np.stack([psd_avg[pid] for pid in role_pids if pid in psd_avg], axis=0)  # (n_part, n_ch, n_freq)
    for ch_name in CHANNELS_4:
        ch_idx = channel_names.index(ch_name)
        vals = stack[:, ch_idx, :]
        mean = vals.mean(axis=0)
        sem = vals.std(axis=0, ddof=1) / np.sqrt(vals.shape[0])
        ax.plot(freqs_common, mean, label=ch_name, lw=1.5)
        ax.fill_between(freqs_common, mean - sem, mean + sem, alpha=0.25)
    ax.set_yscale('log')
    ax.set_xlim(FMIN, FMAX)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title(f'{role} (n={stack.shape[0]})')
axes[0].set_ylabel('PSD (uV^2/Hz)')
axes[0].legend(fontsize=8)
fig.suptitle('Grand average PSD +/- 1 SEM')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTPUT_GRAND_AVG / 'grand_average_psd_by_role.png', dpi=300)
plt.close(fig)

print(f'Saved grand average PSD figure to {OUTPUT_GRAND_AVG}')
