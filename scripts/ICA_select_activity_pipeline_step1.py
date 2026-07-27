from pathlib import Path
import sys

# Add project root to sys.path so imports work regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ICAPreprocessor

input_data_folder = "/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/WP4          - Joint study/UniWAW Data collection/UNIWAW_EEG_exported_BY_TASKS"
ica_folder = Path('data/ica_output')

proc = ICAPreprocessor(
    export_folder=Path(input_data_folder),
    target_events=['passive_movies'],
)
proc.find_eeg_files(smoke_test=True, smoke_dyads_n=2)



# Phase 1a — fit ICA, save models and sources
proc.decompose_and_save(
    ica_folder,
    ica_n_components=15,
    ica_max_iter=2000,
    save_plot=False
)

# Phase 1b — compute PSD and FOOOF parameters
proc.compute_component_features(ica_folder, psd_fmax=45.0)

#----
comp_nc = ica_folder / 'W_000' / 'W_000_EEG_ch_passive_movies_components.nc'

# Overview of all components
proc.plot_component_grid(comp_nc, n_cols=5,
                         save_path=ica_folder / 'W_000' / 'grid.png')

# Detailed view of a single component
fig = proc.plot_component(comp_nc, comp_id=3)
fig = proc.plot_component(comp_nc, comp_id='ICA007')