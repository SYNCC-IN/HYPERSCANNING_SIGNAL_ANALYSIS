from pathlib import Path
import sys
from importlib import reload

# Add project root to sys.path so imports work regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.ica_preprocessing as ica_preprocessing
reload(ica_preprocessing)
from src.ica_preprocessing import ICAPreprocessor

input_data_folder = "/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/WP4          - Joint study/UniWAW Data collection/UNIWAW_EEG_exported_BY_TASKS"
ica_folder = Path("/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/WP4          - Joint study/UniWAW Data collection/UNIWAW_EEG_exported_BY_TASKS/ICA_output")

proc = ICAPreprocessor(
    export_folder=Path(input_data_folder),
    target_events=['passive_movies'],
)
# Konkretna dyada (np. do debugowania Stage 2)
# proc.find_eeg_files(dyad_ids=['W_000'])

# Kilka konkretnych dyad
# proc.find_eeg_files(dyad_ids=['W_001', 'W_007'])
# Smoke test (bez zmian)
# proc.find_eeg_files(smoke_test=True, smoke_dyads_n=2)

# Pełne przetwarzanie
proc.find_eeg_files(smoke_test=False)

proc.fit_and_save_ica(ica_folder, n_components=15)
proc.classify_and_save_labels(ica_folder, iclabel_threshold=0.70, brain_threshold=0.4, amplitude_threshold=100.0, timecourse_seconds=200.0)
# → open quality check figuers and CSV in ICA_QC_FIGS_and_CSV , check and save
cleaned_folder = ica_folder / 'EEG_ICA_CLEANED'
cleaned_folder.mkdir(parents=True, exist_ok=True)
proc.apply_ica_and_save(ica_folder, cleaned_folder)
