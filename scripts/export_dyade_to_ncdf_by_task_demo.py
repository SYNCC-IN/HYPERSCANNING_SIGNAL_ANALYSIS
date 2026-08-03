# %%
import sys
import os
import importlib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Add the parent directory to the path to import src as a package
sys.path.insert(0, os.path.abspath('..'))
from src import dataloader
importlib.reload(dataloader)
from src import multimodal_io
importlib.reload(multimodal_io)
from src import utils
importlib.reload(utils)
from src import export
from src.plot_utils import plot_xarray_signals
importlib.reload(export)
from src.export import export_passive_and_talk_data
from src.mne_bridge import check_exported_data_quality, load_xarray_from_netcdf, get_export_metadata


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

check_exported_data_quality(dyad=dyad_id, modality='EEG', member='ch', task='passive_movies', export_folder=export_path)

# %% [markdown]
# # Example of reading from ncdf files containing the chunks to xarray

# %%
### Load one exported `.nc` file to xarray (`EEG` / `ECG` / `IBI` / `RMSSD`, `ch`, `Peppa`)

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



# %% [markdown]
# # Plot loaded EEG data from the previous cell using utils.plot_signal_with_events.
# Additionally load diode data to plot it together with the other modalities and confirm correct cut of events

# %%
time = np.asarray(data_xr.coords["time"].values, dtype=float)
diode_xr = load_xarray_from_netcdf(str(nc_diode_path))
diode_signal = np.asarray(diode_xr.values, dtype=float)

# Combine EEG channels with diode as an additional channel for visualization.
data_values = np.asarray(data_xr.values, dtype=float)
if diode_signal.ndim == 1:
    diode_signal = diode_signal.reshape(-1, 1)
elif diode_signal.ndim == 2 and diode_signal.shape[0] == 1:
    diode_signal = diode_signal.T

combined_values = np.concatenate([data_values, 200 * diode_signal], axis=1)
channel_names = [str(ch) for ch in data_xr.coords["channel"].values] + ["diode_signal"]

plot_data = xr.DataArray(
    combined_values,
    dims=["time", "channel"],
    coords={"time": time, "channel": channel_names},
    name="signals",
)
plot_data.attrs.update(data_xr.attrs)

regions = []
colors =["#72c2fa67", "#ff7e0e50", "#2ca02c80"]
for idx, ev in enumerate(data_xr.attrs.get("task_events_structure", []) or [], start=1):
    if not isinstance(ev, dict):
        continue
    start_rel = float(ev.get("start_rel_s", np.nan))
    duration_s = float(ev.get("duration_s", np.nan))
    if np.isfinite(start_rel) and np.isfinite(duration_s):
        regions.append({
            "span": (start_rel, start_rel + duration_s),
            "name": str(ev.get("name", f"event_{idx}")),
            "color": str(ev.get("color", colors[(idx - 1) % len(colors)])),
        })

fig, ax = plot_xarray_signals(
    plot_data,
    regions=regions,
    event_duration=float(data_xr.attrs.get("task_duration", np.nan)),
    time_margin_s=float(data_xr.attrs.get("time_margin_s", 0.0)),
    title=f"{selected_task} — {selected_member} ({selected_modality})",
)
plt.show()
print(f"Plotted {plot_data.sizes['channel']} channels over {float(time.min()):.2f}s to {float(time.max()):.2f}s")
print(f"Events shown: {[region['name'] for region in regions]}")
