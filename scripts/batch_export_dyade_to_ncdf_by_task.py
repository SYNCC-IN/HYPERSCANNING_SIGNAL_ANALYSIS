import sys
import os
import importlib
import pandas as pd

# Add the parent directory to the path to import src as a package
sys.path.insert(0, os.path.abspath('..'))
from src import dataloader
importlib.reload(dataloader)
from src import export
importlib.reload(export)

input_folder = "/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/WP4          - Joint study/UniWAW Data collection/UNIWAW_RAW_DATA"
export_folder = "/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/WP4          - Joint study/UniWAW Data collection/UNIWAW_EEG_exported_BY_TASKS"

# load the metadata file to get the list of all dyads and their corresponding movie durations
metadata_file = os.path.join(input_folder, "meta_data.csv")
metadata_df = pd.read_csv(metadata_file, sep=';')

# create dyades_to_export from metadata rows with EEG Passive == 1.0
dyades_to_export = (
    metadata_df.loc[metadata_df['EEG Passive'] == 1.0, 'ID']
    .astype(str)
    .sort_values()
    .tolist()
)
print(f"Exporting {len(dyades_to_export)} dyads: {dyades_to_export}")

# Loop through each dyad and export the data
failed_dyads = []
for dyad in dyades_to_export:
    try:
        print(f"Exporting {dyad}...")
        export.export_passive_and_talk_data(
                dyad_id_list=[dyad],
                load_eeg=True,
                load_et=False,
                load_meta=True,
                lowcut=1.0,
                highcut=40.0,
                eeg_filter_type='fir',
                decimate_factor=8,
                plot_flag=False,
                time_margin=20,
                input_data_path = input_folder,
                export_path = export_folder,
                verbose=False)
        export.check_exported_data_quality(dyad=dyad, modality='EEG', member='ch', task='passive_movies', export_folder=export_folder)
        export.check_exported_data_quality(dyad=dyad, modality='EEG', member='cg', task='passive_movies', export_folder=export_folder)
        print(f"Done: {dyad}")
    except Exception as e:
        failed_dyads.append((dyad, str(e)))
        print(f"Failed: {dyad} -> {e}")

print(f"Finished. Success: {len(dyades_to_export) - len(failed_dyads)}, Failed: {len(failed_dyads)}")
if failed_dyads:
    print("Failed dyads:")
    for dyad, err in failed_dyads:
        print(f"  - {dyad}: {err}")

os.makedirs(export_folder, exist_ok=True)
log_path = os.path.join(export_folder, "export.log")
with open(log_path, "a", encoding="utf-8") as log_file:
    log_file.write(
        f"Finished. Success: {len(dyades_to_export) - len(failed_dyads)}, Failed: {len(failed_dyads)}\n"
    )
    if failed_dyads:
        log_file.write("Failed dyads:\n")
        for dyad, err in failed_dyads:
            log_file.write(f"  - {dyad}: {err}\n")