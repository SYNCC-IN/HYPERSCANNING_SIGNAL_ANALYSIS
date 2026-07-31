# %%
import matplotlib.pyplot as plt
import sys
import os
import importlib
import numpy as np

# Add the parent directory to the path to import src as a package
sys.path.insert(0, os.path.abspath('..'))
from src import dataloader

importlib.reload(dataloader)
from src import utils
importlib.reload(utils)
from src import export

importlib.reload(export)
from src.export import export_passive_and_talk_data


# %% [markdown]
# # Example of saving data for multiple dyads to NCDF while creating the folder structure
# This demo saves two chunks of data corresponding to the two tasks:
# - passive movie watching 
# - talks

# %%
input_data_path="/Users/admin/Library/CloudStorage/GoogleDrive-j.zygierewicz@uw.edu.pl/.shortcut-targets-by-id/1N4ySQ5GO6UE8fY2jnRkRUjBFm4XHrBRv/SYNCC-IN/WP4          - Joint study/UniWAW Data collection/UNIWAW_RAW_DATA"
export_path="../data/UNIWAW_imported"
dyad_id = "W_030"

export_passive_and_talk_data([dyad_id], input_data_path=input_data_path,  
                            export_path=export_path, 
                             load_et=False,eeg_filter_type='iir', 
                             time_margin=20, 
                             mounts_eeg_multimodal=False, export_mounted='CAR', 
                             EEG_bad_channels=['T4_ch','T6_ch', 'T3_ch','T5_ch'],
                             plot_flag=False, # flag for debug plots in the multimodal creation
                             mne_plot_flag=False, # flag for debug plots in the export via mne interpolation and re-referencing function
                             verbose=True
                             )

export.check_exported_data_quality(dyad=dyad_id, modality='EEG', member='ch', task='passive_movies', export_folder=export_path)

# %% [markdown]
# # Example of reading from ncdf files containing the chunks to xarray

# %%
### Load one exported `.nc` file to xarray (`EEG` / `ECG` / `IBI` / `RMSSD`, `ch`, `Peppa`)

from pprint import pprint
from src.export import load_xarray_from_netcdf, get_export_metadata

# Selection

selected_modality = "EEG"  # e.g. 'EEG', 'ECG', 'IBI', 'RMSSD'
selected_member = "ch"
selected_task= "passive_movies"  # e.g. 'passive_movies', 'talk'
selected_event = "Peppa"  # e.g. 'Peppa', 'Incred,ibles', 'Brave', 'talk_1', 'talk_2'

member_folder = {"ch": "child", "cg": "caregiver"}[selected_member]

nc_path = os.path.join("../data/UNIWAW_imported", selected_modality, dyad_id, member_folder, 
    f"{dyad_id}_{selected_modality}_{selected_member}_{selected_task}.nc"
)
nc_diode_path = os.path.join("../data/UNIWAW_imported", "diode", dyad_id, member_folder, 
    f"{dyad_id}_{'diode'}_{selected_member}_{selected_task}.nc"
)

# Load to xarray
data_xr = load_xarray_from_netcdf(str(nc_path))

# Optional: read structured metadata payload
metadata = get_export_metadata(data_xr)

print(f"Loaded: {nc_path}")
print(data_xr)
print("Metadata keys:", list(metadata.keys()))
pprint(metadata, sort_dicts=False)


# %% [markdown]
# # Plot loaded EEG data from the previous cell using utils.plot_signal_with_events.
# Additionally load diode data to plot it together with the other modalities and confirm correct cut of events

# %%
time = np.asarray(data_xr.coords["time"].values, dtype=float)
channels = [str(ch) for ch in data_xr.coords["channel"].values]
data = np.asarray(data_xr.values, dtype=float)  # shape: (n_samples, n_channels)
diode_xr = load_xarray_from_netcdf(str(nc_diode_path))
diode_signal = np.asarray(diode_xr.values, dtype=float)

# Add diode signal as an additional channel to the data array for plotting purposes
data = np.concatenate([data, 200*diode_signal], axis=1)
channels.append("diode_signal")
# Build marker channel from exported event-structure metadata.
raw_events_structure = data_xr.attrs.get("task_events_structure")

event_to_marker = {}
marker_channel = np.zeros(time.shape[0], dtype=int)
for idx, ev in enumerate(raw_events_structure, start=1):
    event_name = str(ev.get("name", f"event_{idx}"))
    start_rel = float(ev.get("start_rel_s", np.nan))
    duration_s = float(ev.get("duration_s", np.nan))
    end_rel = start_rel + duration_s
    if np.isfinite(start_rel) and np.isfinite(end_rel):
        event_to_marker[event_name] = idx
        marker_channel[(time >= start_rel) & (time <= end_rel)] = idx

selected_time = [float(time.min()), float(time.max())]
utils.plot_signal_with_events(
    time=time,
    data=data,
    channels=channels,
    marker_channel=marker_channel,
    event_to_marker=event_to_marker,
    selected_time=selected_time,
)

print(f"Plotted {len(channels)} channels over {selected_time[0]:.2f}s to {selected_time[1]:.2f}s")
print(f"Events shown: {list(event_to_marker.keys())}")
