# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Python tools to load, process, and analyze multimodal hyperscanning data (EEG, ECG/IBI, eye-tracking) recorded from caregiver-child dyads in the SYNCC-IN project (EU Horizon Europe). Data comes from experiments run at the University of Warsaw, consisting of three parts: SECORE (H10 chest-strap IBI task), passive MOVIE viewing (EEG/ET), and free TALK (EEG/ET).

## Current data flow

The real, current pipeline — verified against the actual scripts, not just the docs — has four stages. Each stage reads the previous stage's output and writes to its own folder.

1. **Raw import.** Raw SVAROG EEG/ECG and Pupil-Labs eye-tracking files live in `UNIWAW_RAW_DATA` (Google-Drive-mounted on the researcher's machine). `src/dataloader.py`'s `create_multimodal_data(data_base_path=<UNIWAW_RAW_DATA>, dyad_id=...)`, backed by the `MultimodalData` class in `src/data_structures.py`, reads them into the project's internal, unified in-memory format (one `pandas.DataFrame` per dyad, all modalities, one shared time base). See [docs/data_structure_spec.md](docs/data_structure_spec.md).
2. **Per-task NetCDF export — preferred pipeline input.** `src/export.py`'s `export_passive_and_talk_data(...)` loads a dyad via stage 1 and exports two continuous chunks per modality/member — `passive_movies` (Peppa, Incredibles, Brave back to back) and `talk` — instead of one file per individual event. Run in batch by [scripts/export_dyade_to_ncdf_by_task_batch.py](scripts/export_dyade_to_ncdf_by_task_batch.py). Output tree:
   ```
   UNIWAW_EEG_exported_BY_TASKS/<MODALITY>/<dyad_id>/<child|caregiver>/<dyad_id>_<MODALITY>_<ch|cg>_<passive_movies|talk>.nc
   ```
   **`UNIWAW_EEG_exported_BY_TASKS` is the preferred input for all analysis pipelines** — prefer it over the older per-event export (`write_dyad_to_uniwaw_imported` / `export_to_xarray`, still used only for the small local `data/UNIWAW_imported/` demo dataset).
3. **EEG ICA cleaning.** [scripts/EEG_ICA_clean.py](scripts/EEG_ICA_clean.py) drives `src.ica_preprocessing.ICAPreprocessor` over the `passive_movies` chunks from stage 2, in three stages: fit ICA (`fit_and_save_ica`), classify components with ICLabel (`classify_and_save_labels`, followed by a **manual QC review** of the saved CSV/figures before proceeding), and apply the reviewed exclusions (`apply_ica_and_save`).
   **Clean EEG for downstream pipelines lives in:**
   ```
   ~/SYNCC-IN/WP4          - Joint study/UniWAW Data collection/UNIWAW_EEG_exported_BY_TASKS/ICA_output/EEG_ICA_CLEANED
   ```
   (i.e. `<UNIWAW_EEG_exported_BY_TASKS>/ICA_output/EEG_ICA_CLEANED/<dyad_id>/<dyad_id>_EEG_<ch|cg>_passive_movies_cleaned.nc`).
4. **Analysis pipelines.** Read cleaned EEG via `src/io_utils.py` (`get_participant_files`, `load_eeg_nc`, `trim_to_event_window`). [scripts/Exploratory_spectral_analysis_pipelnie/](scripts/Exploratory_spectral_analysis_pipelnie/) — 7 numbered steps deriving individualized frequency bands/ROIs — is the current reference example of how a pipeline is built on this data; use it as the template for new hyperscanning analyses.

`data/` in this repository (e.g. `data/W_030/`, `data/UNIWAW_imported/`) is a small local sample dataset used only for unit tests, demos, and docs examples — it is **not** the production dataset. All the paths above are hard-coded at the top of the relevant scripts as researcher-specific Google Drive mounts; anyone else running them must point those constants at their own mount.

Full details and verified function locations: [README.md](README.md#current-data-pipeline), [docs/architecture_diagram.md](docs/architecture_diagram.md#current-production-pipeline), [docs/export_ncdf_guide.md](docs/export_ncdf_guide.md#preferred-export-by-task-ncdf-export).

## Target architecture

The codebase is moving toward a clean split:

- **`src/`** — a library of simple, reusable, well-documented functions. No pipeline orchestration, no environment-specific literals, no hard-coded paths.
- **`scripts/`** — sets of scripts that compose those library functions into pipelines implementing specific data-analysis protocols, in particular hyperscanning analyses. A script owns its configuration and orchestration; the library owns the logic.

[scripts/Exploratory_spectral_analysis_pipelnie/](scripts/Exploratory_spectral_analysis_pipelnie/) is the closest thing in the repo today to this target pattern — each numbered script has a `# Configuration` block at the top (paths, constants) followed by calls into `src/io_utils.py`, `src/psd.py`, `src/specparam_utils.py`, `src/peaks.py`, `src/bands.py`, `src/roi.py`, `src/viz.py`. Use it as the template when writing or restructuring pipelines.

## Coding conventions (must follow)

These are the project owner's explicit conventions for any code written or modified in this repo — follow them even where existing code does not:

- **Simplicity over defensiveness.** Functions and scripts must be simple, with no unnecessary safeguards. Prefer letting a runtime error surface over catching and masking it. An error is usually a sign that something needs to be rethought, not something to paper over. Do not add catch-all `try/except`, silent fallbacks, or "just in case" validation unless it is genuinely warranted by the problem at hand.
- **Every function must have a docstring.**
- **Functions must be simple, maintainable, and reusable**, so that pipelines can be composed from them easily — this is the whole point of the `src/` + `scripts/` split above.
- **All paths and constants are defined at the SCRIPT level**, as a configuration block near the top of the script (see the numbered `Exploratory_spectral_analysis_pipelnie/` scripts for the pattern). Functions in `src/` must NOT contain hard-coded paths or constants — those values are always passed in as arguments. This keeps `src/` free of environment-specific literals and keeps all configuration visible in one place per script.

## Environment

- A `.venv` virtualenv exists at the repo root (`.venv/bin/python`); `.vscode/settings.json` points the editor at it.
- Install dependencies with `pip install -r requirements.txt` (pinned versions; also installs `specparam`, `mne`, `neurokit2`, `xarray`/`netCDF4`, etc). Note `requirements.txt` does not include `pytest` — install it separately if missing. Some scripts additionally need `mne-icalabel` (`classify_and_save_labels` in `src/ica_preprocessing.py`) and `autoreject` (EEG quality checking in `src/mne_bridge.py`).

## Common commands

```bash
# Run the full test suite (unit tests only skip cleanly if sample data is absent)
.venv/bin/python -m pytest

# Run a single test file / test / class
.venv/bin/python -m pytest tests/test_dataloader.py
.venv/bin/python -m pytest tests/test_dataloader.py::TestLoadEegDataIntegration::test_load_eeg_data_returns_multimodal_data
.venv/bin/python -m pytest -k "consistency"

# Exclude integration tests that require real raw data files under data/
.venv/bin/python -m pytest -m "not integration"
```

Tests are configured via `pytest.ini` (`testpaths = tests`). Tests marked `@pytest.mark.integration` require real SVAROG/ET data files (e.g. under `data/W_030/` or `data/eeg/`) and `pytest.skip` themselves when the data is missing — this is expected and not a failure.

There is no lint/format command configured in this repo.

## Core architecture

### Data model: `MultimodalData` (unified DataFrame) — Stage 1

`src/data_structures.py` (`MultimodalData`) + `src/dataloader.py` (`create_multimodal_data()`) store all signal modalities (EEG, ECG, IBI, eye-tracking, diode, events) for one dyad together in a single `pandas.DataFrame` (`MultimodalData.data`) sharing one time base and one sampling frequency (`fs`).

Column naming convention (must be followed when adding new columns/readers):
- `time`, `time_idx`
- `EEG_ch_{channel}` / `EEG_cg_{channel}` (child / caregiver)
- `ECG_ch`, `ECG_cg`, `IBI_ch`, `IBI_cg`
- `ET_ch_x`, `ET_ch_y`, `ET_ch_pupil`, `ET_ch_blinks` (and `ET_cg_*`)
- `diode`, `events` (unified), `EEG_events`, `ET_event`

Methods on `MultimodalData` prefixed with `_` (e.g. `_set_eeg_data`, `_decimate_signals`, `_create_events_column`) are populated exclusively by the `create_multimodal_data()` pipeline during construction and are not meant to be called directly by analysis code. Public methods (`get_signals()`, `get_eeg_data_ch/cg()`, `to_mne_raw()`, `get_events_as_marker_channel()`, `print_events()`) are the supported read API.

Full field/method reference: [docs/data_structure_spec.md](docs/data_structure_spec.md). Visual pipeline diagrams: [docs/architecture_diagram.md](docs/architecture_diagram.md).

An automatic consistency check (`check_consistency_of_multimodal_data()` in `src/dataloader.py`) validates that `modalities`, the `events` column, and EEG-vs-ET event start times agree; it runs by default inside `create_multimodal_data()`.

### Export layer — Stage 2

Split across three modules (verified against imports, not just prior docs):
- `src/export.py` — `export_passive_and_talk_data(...)` (preferred, by-task) and the older `write_dyad_to_uniwaw_imported(...)` / `export_to_xarray(...)` (per-event).
- `src/ncdf.py` — `load_xarray_from_netcdf(...)`, `get_export_metadata(...)`, `load_ncdf(...)`, `task_regions(...)`.
- `src/mne_bridge.py` — `load_eeg_ncdf_as_mne_raw(...)`, `load_eeg_signals(...)`, `run_eeg_autoreject_quality_report(...)`, `check_exported_data_quality(...)` (EEG NCDF <-> MNE bridge and AutoReject-based quality gating, used by the batch export script and by `src/ica_preprocessing.py`).

Naming conventions (member codes `ch`/`cg`, site codes `K`/`W`/`M`/`H`, session/task names) are documented in [docs/export_ncdf_guide.md](docs/export_ncdf_guide.md) — follow them exactly when adding new export paths, since `src/mne_bridge.py`, `src/ica_preprocessing.py`, `src/io_utils.py`, and `matlab_utils/ncdf_test_read_demo.m` all parse these names/paths rather than reading structured metadata alone.

### EEG ICA cleaning — Stage 3

`src/ica_preprocessing.py`'s `ICAPreprocessor` class: `find_eeg_files()` discovers `passive_movies` NCDF files under an export folder, then `fit_and_save_ica()` / `classify_and_save_labels()` (requires `mne-icalabel`; writes a user-editable `exclude` column CSV for manual QC) / `apply_ica_and_save()` run the three ICA stages described above.

### Analysis pipelines — Stage 4

`src/io_utils.py` reads cleaned EEG NetCDF files produced by stage 3. [scripts/Exploratory_spectral_analysis_pipelnie/](scripts/Exploratory_spectral_analysis_pipelnie/) is the current reference pipeline (see [Target architecture](#target-architecture) above); its supporting library functions live in `src/psd.py`, `src/specparam_utils.py`, `src/peaks.py`, `src/bands.py`, `src/roi.py`, `src/viz.py`.

### Other subsystems

- **SECORE (cardiac) pipeline** — `src/secore_loader.py`: loads Polar H10 IBI CSVs, corrects ectopic beats (neurokit2/Kubios), interpolates, computes RMSSD, aligns to EEG-derived IBI, packages as `xarray.DataArray`. Independent of the EEG stages above; operates on raw H10 CSVs directly. Details: [docs/secore_loader_guide.md](docs/secore_loader_guide.md).
- **Connectivity / decomposition** — `src/mtmvar.py`: MVAR/DTF connectivity analysis and FAD (Frequency-Amplitude-Damping) decomposition, an AR-model-based alternative to specparam. See [docs/fad_specparam_guide.md](docs/fad_specparam_guide.md). Note: `compute_and_plot_mvar()` currently has a broken internal import (`from .multimodal_io import load_eeg_signals, plot_loaded_eeg_signals` — neither function is defined in `src/multimodal_io.py`); use the lower-level `mvar_criterion`/`full_freq_dtf`/`multivariate_spectra`/`mvar_plot` functions directly instead until this is fixed.
- **Envelopes** — `src/envelopes.py`: instantaneous-amplitude envelope utilities (band-pass + Hilbert transform), demoed in `scripts/demo_envelopes.py` (reads from `EEG_ICA_CLEANED`).
- **ET / ICA bridges** — `src/eyetracker.py` (Pupil Labs gaze/pupil/blink processing), `src/multimodal_io.py` (joblib save/load of `MultimodalData` objects — unrelated to the NCDF export layer).
- **Legacy** — `src/warsaw_pilot_data.py` is an older, standalone example pipeline (HRV/EEG/combined DTF analysis) that predates the pipeline above and operates on the local `data/` sample only; not part of the current production flow.

## Working conventions

- Follow the DataFrame column-naming and NCDF export/path-naming conventions above exactly; multiple modules parse filenames/column names rather than structured metadata alone.
- When changing `MultimodalData`'s schema, update [docs/data_structure_spec.md](docs/data_structure_spec.md)'s version history section — this doc is actively kept in sync with the implementation.
- Treat `_`-prefixed `MultimodalData` methods as internal to `create_multimodal_data()`; add new data-population logic there rather than calling/extending them from analysis scripts.
- Before trusting any doc's description of where a function lives, grep for it — several functions have moved between `src/export.py`, `src/ncdf.py`, and `src/mne_bridge.py` over time, and doc updates have lagged (see the corrections in [docs/export_ncdf_guide.md](docs/export_ncdf_guide.md)).
