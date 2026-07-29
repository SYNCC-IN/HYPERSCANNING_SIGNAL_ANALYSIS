from pathlib import Path
import sys
import numpy as np

# Add project root to sys.path so imports work regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ICAPreprocessor

input_data_folder = "/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/WP4          - Joint study/UniWAW Data collection/UNIWAW_EEG_exported_BY_TASKS"
ica_folder = Path('/Users/admin/Documents/Hoza/PROJEKTY/SYNCC_IN_LOCAL_HOME/hyperscanning-signal-analysis/data/ica_output')

proc = ICAPreprocessor(
    export_folder=Path(input_data_folder),
    target_events=['passive_movies'],
)
proc.find_eeg_files(smoke_test=False)  # process all detected dyads


# Phase 1a — fit ICA, save models and sources
fs = 128
target_freqs = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15])

# quarter-, half-, and full-periods for each target frequency
lags_set = set()
for f in target_freqs:
    for fraction in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        lag = round(fs / f * fraction)
        if 1 <= lag <= fs * 2:
            lags_set.add(lag)

strategic_lags = sorted(lags_set)
print(strategic_lags)

proc.decompose_and_save_sobi(
    ica_folder,
    ica_n_components=15,
    n_lags=128,
    lags=strategic_lags,
    save_plot=False
)

# Phase 1b — compute PSD and FOOOF parameters

proc.compute_component_features_sobi(
    ica_folder,
    psd_fmax=35.0,
    fooof_peak_width_limits=(1.5, 12.0),
    fooof_max_n_peaks=4,
    fooof_peak_threshold=2.5,
    fooof_r_squared_threshold=0.90,
    fooof_aperiodic_mode='fixed',
)

# ---- generate overview plots for every processed dyad member ----
figs_dir = ica_folder / 'FIGS'
figs_dir.mkdir(parents=True, exist_ok=True)

component_files = sorted(ica_folder.rglob('*_components.nc'))
for comp_nc in component_files:
    fig_name = f"{comp_nc.stem.replace('_components', '')}_all_components.png"
    fig_path = figs_dir / fig_name
    proc.plot_all_components(
        comp_nc,
        n_cols=5,
        save_path=fig_path,
        show=False,
    )
    print(f"Saved overview plot: {fig_path}")