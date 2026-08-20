# %% [markdown]
# # Batch export of dyads to NCDF by task
import sys
import os
import importlib
import pandas as pd

# Add the parent directory to the path to import src as a package
sys.path.insert(0, os.path.abspath('..'))
from src import dataloader
importlib.reload(dataloader)
from src import multimodal_io
importlib.reload(multimodal_io)
from src.export import export_passive_and_talk_data
from src.mne_bridge import check_exported_data_quality



# # Function to export data for a single dyad
def export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=None):
        print(f"Exporting {dyad}...")
        export_passive_and_talk_data(
                dyad_id_list=[dyad],
                load_eeg=True,
                load_et=False,
                load_meta=True,
                lowcut=1.0,
                highcut=64, # for ICLabel #40.0,
                eeg_filter_type='iir',#'fir',
                EEG_bad_channels=EEG_bad_channels,
                decimate_factor=8,
                plot_flag=False,
                time_margin=20,
                input_data_path = input_folder,
                export_path = export_folder,
                verbose=False)
        check_exported_data_quality(dyad=dyad, modality='EEG', member='ch', task='passive_movies', export_folder=export_folder)
        check_exported_data_quality(dyad=dyad, modality='EEG', member='cg', task='passive_movies', export_folder=export_folder)
        print(f"Done: {dyad}")



# # Setup folders and metadata
input_folder = "/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/WP4          - Joint study/UniWAW Data collection/UNIWAW_RAW_DATA"
export_folder = "/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/WP4          - Joint study/UniWAW Data collection/UNIWAW_EEG_exported_BY_TASKS"

# load the metadata file to get the list of all dyads and their corresponding movie durations
metadata_file = os.path.join(input_folder, "meta_data.csv")
metadata_df = pd.read_csv(metadata_file, sep=';')



# %% [markdown]
# #  Standard export for all dyads with EEG Passive == 1.0 
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
        export_one_dyade(dyad, input_folder, export_folder)
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

# %% [markdown]
# ## Special cases

dyad = "W_000"
EEG_bad_channels_W000 = ['Fp2_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W000)
print(f"Finished special case: {dyad}")

dyad = "W_001"
EEG_bad_channels_W001 =  ['P4_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W001)

dyad = "W_003"
EEG_bad_channels_W003 =  ['F4_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W003)

dyad = "W_005"
EEG_bad_channels_W005 =  ['Fp1_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W005)

dyad = "W_010"
EEG_bad_channels_W010 = ['C4_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W010)
# %%
dyad = "W_019"
EEG_bad_channels_W019 =  ['Fz_ch', 'C3_ch', 'Cz_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W019)
# %%
dyad = "W_020"
EEG_bad_channels_W020 =  ['P8_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W020)

dyad = "W_022"
EEG_bad_channels_W022 =  ['Fp2_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W022)

dyad = "W_024"
EEG_bad_channels_W024 = [ 'F3_ch', 'C3_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W024  )
# %%
dyad = "W_025"
EEG_bad_channels_W025 = ['Cz_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W025)


# %%
dyad = "W_026"
EEG_bad_channels_W026 = ['Fz_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W026)

dyad = "W_028"
EEG_bad_channels_W028 =   ['C3_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W028)

dyad = "W_029"
EEG_bad_channels_W029 =   ['C3_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W029)

dyad = "W_042"
EEG_bad_channels_W042 = ['T8_cg','F8_cg', 'C3_cg', 'F7_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W042)
# %%
dyad = "W_044"
EEG_bad_channels_W044 = ['F7_ch', 'T5_ch','T8_ch', 'F7_cg', 'Fz_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W044)

dyad = "W_047"
EEG_bad_channels_W047 = ['O2_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W047)

# %%
dyad = "W_048"
EEG_bad_channels_W048 = ['C4_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W048)

dyad = "W_053"
EEG_bad_channels_W053 =  ['Fp1_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W053)

dyad = "W_071"
EEG_bad_channels_W071 = ['F7_cg','F3_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W071)

dyad = "W_072"
EEG_bad_channels_W072 = ['P7_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W072)

dyad = "W_074"
EEG_bad_channels_W074 = ['P4_cg', 'T7_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W074)

dyad = "W_079"
EEG_bad_channels_W079 = ['C4_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W079)

dyad = "W_085"
EEG_bad_channels_W085 = ['T8_ch','P8_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W085)

dyad = "W_091"
EEG_bad_channels_W091 = ['Fz_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W091)

dyad = "W_115"
EEG_bad_channels_W115 = ['F7_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W115)

dyad = "W_116"
EEG_bad_channels_W116 =  ['F4_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W116)


# %% [markdown]
# # Optional: Export only a subset of dyads for testing/debugging
dyad = "W_025" #- W_025: More than 2 talks detected, something is wrong.
EEG_bad_channels_W025 = ['Cz_ch']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W025)

dyad = "W_114"# list index out of range
EEG_bad_channels_W114 = ['F7_cg','P8_cg']
export_one_dyade(dyad, input_folder, export_folder, EEG_bad_channels=EEG_bad_channels_W114)
