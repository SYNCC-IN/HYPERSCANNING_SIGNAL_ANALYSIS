# Export to NCDF and Import from NCDF

This document describes how to export processed multimodal data to NetCDF (`.nc`) files and how to load it back into xarray.

## Table of contents

- [Overview](#overview)
- [Preferred export: by-task NCDF export](#preferred-export-by-task-ncdf-export)
  - [Output folder structure (by-task export)](#output-folder-structure-by-task-export)
- [Older per-event export (`write_dyad_to_uniwaw_imported` / `export_to_xarray`)](#older-per-event-export-write_dyad_to_uniwaw_imported--export_to_xarray)
- [Output folder structure](#output-folder-structure)
- [Naming conventions used in export](#naming-conventions-used-in-export)
  - [Dyad members](#dyad-members)
  - [Modalities](#modalities)
  - [Site and dyad IDs](#site-and-dyad-ids)
  - [Experimental session names](#experimental-session-names)
- [Structure of xarray data stored in exported NCDF](#structure-of-xarray-data-stored-in-exported-ncdf)
  - [Data variable](#data-variable)
  - [Dimensions and coordinates](#dimensions-and-coordinates)
  - [Attributes on `signals`](#attributes-on-signals)
  - [Structured metadata payload (`metadata_json`)](#structured-metadata-payload-metadata_json)
- [Metadata serialization format](#metadata-serialization-format)
  - [NetCDF serialization constraints](#netcdf-serialization-constraints)
- [Export/load data](#exportload-data)
  - [Export a full dyad to NCDF](#export-a-full-dyad-to-ncdf)
  - [Export one selection to xarray](#export-one-selection-to-xarray)
  - [Load one NCDF file back to xarray](#load-one-ncdf-file-back-to-xarray)
  - [Minimal round-trip example](#minimal-round-trip-example)
- [EEG quality checking](#eeg-quality-checking)
  - [Functions](#functions)
  - [Single-file interactive demo](#single-file-interactive-demo)
  - [Batch processing notebook](#batch-processing-notebook)
- [MVAR / DTF analysis helpers](#mvar--dtf-analysis-helpers)
  - [load_eeg_signals](#load_eeg_signals)
  - [compute_and_plot_mvar](#compute_and_plot_mvar-srcmtmvarpy)
- [MATLAB R2019b compatibility (channel names)](#matlab-r2019b-compatibility-channel-names)

---

## Overview

The export/import workflow is split across three modules:

- [src/export.py](../src/export.py):
  - `export_passive_and_talk_data(...)` — **preferred, current export path.** Exports two
    continuous per-task chunks (`passive_movies`, `talk`) per modality/member, instead of
    one file per individual event. See [Preferred export: by-task NCDF export](#preferred-export-by-task-ncdf-export) below.
  - `write_dyad_to_uniwaw_imported(...)` — older export path; exports a whole dyad into a
    folder tree with one `.nc` file per modality/member/**event**.
  - `export_to_xarray(...)` — older export path; exports one selected modality/member/event
    to a single `xarray.DataArray`.
- [src/ncdf.py](../src/ncdf.py):
  - `load_xarray_from_netcdf(...)` loads a saved `.nc` file back into `xarray.DataArray`.
  - `get_export_metadata(...)` reads the structured metadata payload from `metadata_json`.
- [src/mne_bridge.py](../src/mne_bridge.py): converts exported EEG NetCDF to MNE and runs
  quality checks — see [EEG quality checking](#eeg-quality-checking) below.

## Preferred export: by-task NCDF export

`export_passive_and_talk_data(...)` (in [src/export.py](../src/export.py)) is the
current, preferred way to export a dyad. For each modality and member it loads the
dyad via `dataloader.create_multimodal_data(...)` and writes **two chunks** instead of
one file per event:

- `passive_movies` — one continuous chunk covering the movie events (Peppa, Incredibles,
  Brave) back to back.
- `talk` — one continuous chunk covering every event whose name contains `"talk"`.

```python
from src.export import export_passive_and_talk_data
from src.mne_bridge import check_exported_data_quality

export_passive_and_talk_data(
    dyad_id_list=["W_030"],
    input_data_path="/path/to/UNIWAW_RAW_DATA",
    export_path="/path/to/UNIWAW_EEG_exported_BY_TASKS",
    load_eeg=True,
    load_et=False,
    lowcut=1.0,
    highcut=64,          # wide, to leave room for ICLabel-based ICA cleaning downstream
    eeg_filter_type="iir",
    decimate_factor=8,
    time_margin=20,
    verbose=True,
)

# Optional: run an AutoReject-based quality report and save it alongside the export
check_exported_data_quality(
    dyad="W_030", modality="EEG", member="ch", task="passive_movies",
    export_folder="/path/to/UNIWAW_EEG_exported_BY_TASKS",
)
```

[scripts/export_dyade_to_ncdf_by_task_batch.py](../scripts/export_dyade_to_ncdf_by_task_batch.py)
runs this for every dyad flagged `EEG Passive == 1.0` in the project's metadata file,
plus a list of special-case dyads with known bad channels
(`EEG_bad_channels=[...]`, interpolated before CAR referencing). See
[scripts/export_dyade_to_ncdf_by_task_demo.py](../scripts/export_dyade_to_ncdf_by_task_demo.py)
for a single-dyad walkthrough that also reloads and plots the exported chunk.

### Output folder structure (by-task export)

```
UNIWAW_EEG_exported_BY_TASKS/<MODALITY>/<dyad_id>/<child|caregiver>/<dyad_id>_<MODALITY>_<ch|cg>_<passive_movies|talk>.nc
```

Example:

- `UNIWAW_EEG_exported_BY_TASKS/EEG/W_030/child/W_030_EEG_ch_passive_movies.nc`

This is the folder tree that `src/io_utils.py` and downstream analysis pipelines (e.g.
[scripts/Exploratory_spectral_analysis_pipelnie/](../scripts/Exploratory_spectral_analysis_pipelnie/))
expect — after ICA cleaning (see the project [README](../README.md#current-data-pipeline)),
under `UNIWAW_EEG_exported_BY_TASKS/ICA_output/EEG_ICA_CLEANED/`.

The `signals` DataArray structure, attributes, and metadata payload written by this
function are the same as described below for the older per-event export, except that
`event_name` is the task name (`passive_movies` or `talk`) and the metadata payload's
`event_order`/embedded event structure lists the individual events making up that chunk
(see `task_events_structure` in the attrs, used by `src/io_utils.py` to find movie
boundaries within the `passive_movies` chunk).

---

## Older per-event export (`write_dyad_to_uniwaw_imported` / `export_to_xarray`)

The rest of this document describes the older, per-event export path. It is still
used for the small local demo dataset under `data/UNIWAW_imported/` (see the
project [README](../README.md)) but is not the current production pipeline.

## Output folder structure

A typical export path looks like this:

- `data/UNIWAW_imported/<MODALITY>/<DYAD_ID>/<member_folder>/<file>.nc`

Example:

- `data/UNIWAW_imported/EEG/W_030/child/W_030_EEG_ch_Peppa.nc`

Where:

- `member_folder` is `child` for `ch` and `caregiver` for `cg`.
- `<file>` follows `<DYAD_ID>_<MODALITY>_<member_code>_<EVENT>.nc`.

## Naming conventions used in export

This project uses the following conventions in NCDF export paths and filenames.

### Dyad members

- Full member names are: `child` and `caregiver`.
- Member codes used in filenames are:
    - `ch` = `child`
    - `cg` = `caregiver`

### Modalities

- Modality names are uppercase in paths and filenames, e.g.:
    - `EEG`
    - `ET`
    - `FNIRS`
    - `IBI`
  - `RMSSD`

### Site and dyad IDs

- Site codes:
    - `K` = Kopenhagen
    - `W` = Warsaw
    - `M` = Milan
    - `H` = Heidelberg
- Dyad numeric code uses three zero-padded digits, e.g. `003`, `030`, `125`.
- Practical dyad ID format used in files: `<SITE>_<NNN>` (example: `W_030`).

### Experimental session names

- Session/event names used in exported filenames:
    - `Secore`
    - `Talk1`
    - `Talk2`
    - `Peppa`
    - `Incredibles`
    - `Brave`

Example filename built from these conventions:

- `W_030_EEG_ch_Peppa.nc`

## Structure of xarray data stored in exported NCDF

Each exported file stores one `xarray.DataArray` named `signals`.

### Data variable

- Variable name: `signals`
- Shape: `[n_time, n_channel]`
- Meaning: signal amplitudes/samples for one selected modality, dyad member, and event.

### Dimensions and coordinates

- Dimensions:
    - `time`
    - `channel`
- Coordinates:
    - `time`: relative time in seconds (event start is shifted to `0.0`)
    - `channel`: channel labels (e.g., `Fp1`, `Fp2`, `Cz`)

### Attributes on `signals`

Common scalar/string attributes written during export:

- `dyad_id`
- `who` (`ch` or `cg`)
- `sampling_freq`
- `event_name`
- `event_start` (relative start in exported window; currently `0.0`)
- `event_duration`
- `time_margin_s`
- `channel_names_csv` (MATLAB-friendly comma-separated channel names)
- `channel_names_json` (MATLAB-friendly JSON array of channel names)
- `metadata_json` (serialized structured metadata)

### Structured metadata payload (`metadata_json`)

- JSON object containing, depending on modality:
    - `notes`
    - `child_info`
    - `event_order` (chronological order by event start time for available target events:
      `Peppa`, `Incredibles`, `Brave`)
    - for EEG additionally: `eeg.filtration` and `eeg.references`

For EEG exports, `eeg.filtration` includes nested dictionaries for `notch`, `low_pass`, and `high_pass`.
In particular, `low_pass` and `high_pass` store:

- `type`: high-level filter family used in pipeline (`fir` or `iir`)
- `cut_f`: cutoff frequency in Hz
- `order`: filter order
- `f_type`: concrete design function (`firwin` or `butter`)
- `a`, `b`: filter coefficients
- `applied`: whether the filter was applied

Use `get_export_metadata(...)` to decode and access this payload safely.

#### Example: reading `event_order`

```python
from src.ncdf import load_xarray_from_netcdf, get_export_metadata

da = load_xarray_from_netcdf("data/UNIWAW_imported/EEG/W_030/child/W_030_EEG_ch_Peppa.nc")
meta = get_export_metadata(da)

event_order = meta.get("event_order", [])
print(event_order or "event order not available")
# e.g. ['Peppa', 'Brave', 'Incredibles']
```

## Metadata serialization format

Exported DataArrays include:

- compact scalar attrs (for quick filtering), e.g. `dyad_id`, `event_name`, `who`, `sampling_freq`, `event_start`, `event_duration`
- structured metadata serialized to `metadata_json`

Use helper API to access structured metadata safely:

```python
from src.ncdf import get_export_metadata

metadata = get_export_metadata(data_xr)
print(metadata.keys())
```
### NetCDF serialization constraints

NetCDF attributes do not support all Python object types directly.

Current export behavior sanitizes attrs before writing:

- `None` -> empty string
- `dict` and nested structures -> JSON string
- non-serializable objects -> string representation

This avoids runtime errors during `to_netcdf(...)`.

---
## Export/load data
### Export a full dyad to NCDF

```python
from src.export import write_dyad_to_uniwaw_imported

write_dyad_to_uniwaw_imported(
    dyad_id_list=["W_030"],
    input_data_path="data",
    export_path="data/UNIWAW_imported",
    load_eeg=True,
    load_et=True,
    load_meta=True,
    eeg_filter_type="iir",
    time_margin=10,
    verbose=True,
)
```

#### Notes

- Use `verbose=True` to see progress logs.
- The function exports all events for all available modalities/members in the dyad.

### Export one selection to xarray

```python
from src import dataloader
from src.export import export_to_xarray

mmd = dataloader.create_multimodal_data(
    data_base_path="data",
    dyad_id="W_030",
    load_eeg=True,
    load_et=False,
    load_meta=False,
    decimate_factor=8,
)

data_xr = export_to_xarray(
    multimodal_data=mmd,
    selected_event="Peppa",
    selected_channels=["Fp1", "Fp2", "F3"],
    selected_modality="EEG",
    member="ch",
    time_margin=10,
    verbose=False,
)
```

### Load one NCDF file back to xarray

```python
from pathlib import Path
from src.ncdf import load_xarray_from_netcdf

dyad_id = "W_030"
selected_modality = "EEG"
selected_member = "ch"
selected_event = "Peppa"

member_folder = {"ch": "child", "cg": "caregiver"}[selected_member]

nc_path = Path("data/UNIWAW_imported") / selected_modality / dyad_id / member_folder / (
    f"{dyad_id}_{selected_modality}_{selected_member}_{selected_event}.nc"
)

data_xr = load_xarray_from_netcdf(str(nc_path))
print(data_xr)
```

---

### Minimal round-trip example

```python
from src.ncdf import load_xarray_from_netcdf, get_export_metadata

path = "data/UNIWAW_imported/EEG/W_030/child/W_030_EEG_ch_Peppa.nc"
da = load_xarray_from_netcdf(path)
meta = get_export_metadata(da)

print(type(da).__name__)          # DataArray
print("child_info" in meta)       # True for newly exported files
```

---

## EEG quality checking

The AutoReject-based quality pipeline for exported EEG NCDF files lives in
[src/mne_bridge.py](../src/mne_bridge.py) (not `src/export.py`).

### Functions

#### `load_eeg_ncdf_as_mne_raw(ncdf_path, montage, scale_to_volts, data_xr)`

Loads an EEG NCDF file and returns an `mne.io.RawArray` object.

- Reads the `signals` DataArray and transposes it to `[channel, time]`.
- Infers sampling frequency from the `sampling_freq` attr; raises `ValueError` if it is missing.
- Applies `scale_to_volts` (default `1e-6`, i.e. µV → V).
- Attaches `montage` (default `"standard_1020"`); unknown channels are silently ignored.
- `data_xr`: optional pre-loaded `xarray.DataArray`; when provided the file is not read from disk again (avoids duplicate I/O when the caller already holds the DataArray).

This function is also used by `src/ica_preprocessing.py`'s `ICAPreprocessor` (see the
project [README](../README.md#current-data-pipeline)) to load `passive_movies` NCDF
files as MNE `Raw` objects for ICA fitting/classification/application.

> **Note:** `plot_eeg_with_rejected_segments`, described in earlier versions of this
> guide, is currently commented out in `src/plot_utils.py` and not callable. Use
> `plot_xarray_signals` (`src/plot_utils.py`) for plotting exported signals instead.

#### `run_eeg_autoreject_quality_report(ncdf_path, epoch_duration_s, n_interpolate, cv, random_state, n_jobs, montage, scale_to_volts, verbose)`

Full pipeline: NCDF → MNE → AutoReject → tabular summaries + visualization.

| Parameter | Default | Description |
|---|---|---|
| `ncdf_path` | — | Path to EEG NCDF file |
| `epoch_duration_s` | `2.0` | Fixed epoch length in seconds |
| `n_interpolate` | `(1, 2, 4)` | AutoReject grid for max interpolated channels |
| `cv` | `5` | Cross-validation folds for AutoReject |
| `random_state` | `42` | Reproducibility seed |
| `n_jobs` | `-1` | Parallel jobs (`-1` = all CPUs) |
| `montage` | `"standard_1020"` | MNE montage name |
| `scale_to_volts` | `1e-6` | µV → V conversion factor |
| `verbose` | `True` | Print MNE / AutoReject progress |

Returns a `dict` with keys:

| Key | Content |
|---|---|
| `raw` | `mne.io.RawArray` |
| `epochs` | `mne.Epochs` (fixed-length) |
| `autoreject` | Fitted `AutoReject` object |
| `reject_log` | `RejectLog` from `ar.transform(return_log=True)` |
| `epoch_summary` | `pd.DataFrame` — one row per epoch: `epoch_idx`, `start_s`, `end_s`, `interpolated_channels`, `rejected`, `in_margin` |
| `channel_summary` | `pd.DataFrame` — one row per channel: `channel`, `interpolated_epochs`, `bad_labels`, `interpolated_pct`, `bad_labels_pct`, sorted by `bad_labels` descending |
| `global_summary` | `dict` — `ncdf_path`, `n_channels`, `n_epochs`, `epoch_duration_s`, `rejected_epochs`, `rejected_epochs_pct`, `total_interpolations` |
| `figure` | `matplotlib.Figure` — stacked EEG plot with rejection and margin overlays |
| `axis` | `matplotlib.Axes` |

**Note on `in_margin`:** An epoch is flagged `in_margin=True` when it lies entirely before t = 0 or entirely after t = `event_duration`. Rejected windows shown in the plot and saved to the CSV report **exclude** margin epochs.

**Dependency:** requires `mne` and `autoreject` (both listed in `requirements.txt`).

#### `check_exported_data_quality(dyad, modality, member, task, export_folder)`

Thin wrapper used by the by-task export pipeline (see
[Preferred export: by-task NCDF export](#preferred-export-by-task-ncdf-export)) to gate
each export: for `modality='EEG'` it loads `<export_folder>/EEG/<dyad>/<child|caregiver>/<dyad>_EEG_<member>_<task>.nc`,
runs `run_eeg_autoreject_quality_report` on it, and saves the resulting figure to
`<export_folder>/EEG/Quality_reports/<dyad>_EEG_<member>_<task>_quality_report.png`.
Called once per member right after `export_passive_and_talk_data(...)` in
[scripts/export_dyade_to_ncdf_by_task_batch.py](../scripts/export_dyade_to_ncdf_by_task_batch.py).

### Single-file interactive demo

[scripts/eeg_quality_report_demo.ipynb](../scripts/eeg_quality_report_demo.ipynb) walks through the full pipeline for one EEG NCDF file:

1. Auto-discovers the first EEG file in the export folder.
2. Runs `run_eeg_autoreject_quality_report`.
3. Displays `global_summary`, `channel_summary`, and `epoch_summary` tables.
4. Prints a human-readable channel interpretation per channel, e.g.:  
   `- Fp1 was problematic in 65% of epochs: 0% fixable + 65% still bad`
5. Saves a TOML-style CSV report and a PNG plot (same folder as the NCDF file).

The TOML-style CSV format:

```
section,key,value
global,ncdf_file,"W_000_EEG_cg_Brave.nc"
global,rejected_epochs,2
rejected_windows,epoch_2,{start_s=4.000, end_s=6.000, interpolated_channels=10}
global,rejected_epochs_pct,5.0
global,epoch_duration_s,2.0
global,total_interpolations,47
top_channels,ch_1,{name="Fp1", problematic_pct=65, fixable_pct=0, still_bad_pct=65}
```

Output artefacts follow the NCDF basename:

- `<stem>_quality_report.csv`
- `<stem>_quality_plot.png`

### Batch processing notebook

[scripts/eeg_quality_report_batch.ipynb](../scripts/eeg_quality_report_batch.ipynb) processes all EEG NCDF files found under the export folder:

- **Cell 3** — crawls the tree with `rglob("*.nc")`, filtering on `"_EEG_"` in the filename (538 files for the UNIWAW dataset).
- **Cell 4** — smoke-test toggle:
  ```python
  smoke_test = True   # set False for full batch
  subset_size = 12
  ```
- **Cell 5** — helper functions: `_toml_scalar`, `_extract_dyad_name`, `_get_top_bad_channels`, `_build_report_rows`, `_save_report_artifacts`.
- **Cell 6** — per-file loop: shows dyad heading, runs AutoReject (`verbose=False`), displays the TOML table and plot, saves CSV + PNG artefacts, collects results.
- **Cell 7** — final summary table saved to `EEG_quality_summary_report.csv` in the export folder root.

Summary columns: `dyad`, `ncdf_file`, `status`, `rejected_epochs`, `top_bad_channels` (channels with `bad_labels_pct > 10%`). An `error` column is appended automatically when any file fails.

---

## MVAR / DTF analysis helpers

`load_eeg_signals` supports loading exported EEG NCDF files directly for the MVAR
pipeline, without going through MNE. It lives in `src/mne_bridge.py`; the MVAR/DTF
computation itself lives in `src/mtmvar.py`.

### `load_eeg_signals`

```python
from src.mne_bridge import load_eeg_signals

signals, ch_names, fs, time_s, event_duration_s = load_eeg_signals(
    ncdf_path="data/UNIWAW_imported/EEG/W_030/child/W_030_EEG_ch_Peppa.nc",
    channel_subset=["F3", "Fz", "F4", "C3", "Cz", "C4"],  # None = all channels
    low_cutoff_hz=1.0,    # high-pass; None = skip
    high_cutoff_hz=None,  # low-pass;  None = skip
)
# signals: np.ndarray (n_chan, n_samp), z-scored per channel
# time_s:  1-D array in seconds (0 = event start)
```

Behaviour:

- Filtering (Butterworth 4th order, zero-phase `filtfilt`) is applied to the **full
  signal** (including margins) before trimming.
- A 50 Hz IIR notch (Q = 15) is applied automatically when `fs > 100 Hz`.
- After trimming the time margin (`t in [0, event_duration]`), `M1` and `M2`
  mastoid reference channels are **always** removed.
- Signals are **z-score normalised per channel** after trimming.

| Return value | Type | Description |
|---|---|---|
| `signals` | `np.ndarray (n_chan, n_samp)` | z-scored EEG |
| `channel_names` | `list[str]` | ordered channel labels |
| `fs` | `float` | sampling frequency (Hz) |
| `time_s` | `np.ndarray (n_samp,)` | time axis (0 = event start) |
| `event_duration_s` | `float` | event window length (s) |

> **Known issue:** `compute_and_plot_mvar` below (`src/mtmvar.py`) internally does
> `from .multimodal_io import load_eeg_signals, plot_loaded_eeg_signals`, but
> `src/multimodal_io.py` only defines `save_to_file`/`load_output_data` — neither
> `load_eeg_signals` nor `plot_loaded_eeg_signals` exists there (`load_eeg_signals`
> is in `src/mne_bridge.py`, as used above; `plot_loaded_eeg_signals` is not
> currently implemented anywhere — see the note under
> [EEG quality checking](#eeg-quality-checking)). As a result, calling
> `compute_and_plot_mvar(...)` currently raises `ImportError`. Use `src.mtmvar`'s
> lower-level functions (`mvar_criterion`, `full_freq_dtf`, `multivariate_spectra`,
> `mvar_plot`) directly on a signal array loaded via `src.mne_bridge.load_eeg_signals`
> until this is fixed.

### `compute_and_plot_mvar` (`src/mtmvar.py`)

```python
from src.mtmvar import compute_and_plot_mvar

ff_dtf, spectra, chan_names, crit, model_order_range, p_opt = compute_and_plot_mvar(
    ncdf_path="data/UNIWAW_imported/EEG/W_030/child/W_030_EEG_ch_Peppa.nc",
    channel_subset=None,          # None = all (minus M1/M2)
    max_model_order=15,
    optimal_model_order=None,     # None = auto-select via crit_type
    crit_type="HQ",               # 'AIC', 'HQ', or 'SC'
    freq_min=1.0,
    freq_max=40.0,
    freq_step=0.5,
    low_cutoff_hz=1.0,
    high_cutoff_hz=None,
    plot=True,                    # MVAR matrix plot
    plot_loaded_signal=True,      # stacked EEG preview
)
```

High-level pipeline:

1. `load_eeg_signals` — load, filter, trim and normalise.
2. *(optional)* `plot_loaded_eeg_signals` — preview the loaded traces.
3. `mvar_criterion` — select optimal model order (skipped when
   `optimal_model_order` is set; `crit` and `model_order_range` are then
   returned as empty arrays).
4. `full_freq_dtf` — compute full-frequency DTF.
5. `multivariate_spectra` — compute multivariate power spectra.
6. *(optional)* `mvar_plot` — display the spectra / connectivity matrix.

| Return value | Description |
|---|---|
| `ff_dtf` | Full-frequency DTF `(n_chan, n_chan, n_freq)` |
| `spectra` | Multivariate spectra `(n_chan, n_chan, n_freq)` |
| `chan_names` | Channel labels |
| `crit` | Information-criterion values per model order |
| `model_order_range` | Tested model orders |
| `p_opt` | Selected model order |

See [scripts/ESCan_drfat.ipynb](../scripts/ESCan_drfat.ipynb) for a full batch
example that processes a whole export folder and saves composite MVAR figures.

---

## MATLAB R2019b compatibility (channel names)

To make channel names easy to read in MATLAB R2019b, export now stores channel labels
as variable attributes on `signals`:

- `channel_names_csv` (comma-separated text)
- `channel_names_json` (JSON array text)

Example in MATLAB:

```matlab
ncFile = 'data/UNIWAW_imported/EEG/W_030/child/W_030_EEG_ch_Peppa.nc';

% easiest option
csvNames = ncreadatt(ncFile, 'signals', 'channel_names_csv');
channels = strsplit(csvNames, ',');

% alternative: JSON payload
jsonNames = ncreadatt(ncFile, 'signals', 'channel_names_json');
channelsFromJson = jsondecode(jsonNames);
```


