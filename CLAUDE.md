# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Python tools to load, process, and analyze multimodal hyperscanning data (EEG, ECG/IBI, eye-tracking) recorded from caregiver-child dyads in the SYNCC-IN project (EU Horizon Europe). Data comes from experiments run at the University of Warsaw, consisting of three parts: SECORE (H10 chest-strap IBI task), passive MOVIE viewing (EEG/ET), and free TALK (EEG/ET). The code is written to be adaptable to other partner sites' paradigms.

## Environment

- A `.venv` virtualenv exists at the repo root (`.venv/bin/python`); `.vscode/settings.json` points the editor at it.
- Install dependencies with `pip install -r requirements.txt` (pinned versions; also installs `specparam`, `mne`, `neurokit2`, `xarray`/`netCDF4`, etc). Note `requirements.txt` does not include `pytest` — install it separately if missing.

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

### Data model: `MultimodalData` (unified DataFrame)

The center of the codebase is `src/data_structures.py` (`MultimodalData`) + `src/dataloader.py` (`create_multimodal_data()`, the main entry point). All signal modalities (EEG, ECG, IBI, eye-tracking, diode, events) for one dyad are stored together in a single `pandas.DataFrame` (`MultimodalData.data`) sharing one time base and one sampling frequency (`fs`), rather than as separate per-modality objects.

Column naming convention (must be followed when adding new columns/readers):
- `time`, `time_idx`
- `EEG_ch_{channel}` / `EEG_cg_{channel}` (child / caregiver)
- `ECG_ch`, `ECG_cg`, `IBI_ch`, `IBI_cg`
- `ET_ch_x`, `ET_ch_y`, `ET_ch_pupil`, `ET_ch_blinks` (and `ET_cg_*`)
- `diode`, `events` (unified), `EEG_events`, `ET_event`

Methods on `MultimodalData` prefixed with `_` (e.g. `_set_eeg_data`, `_decimate_signals`, `_create_events_column`) are populated exclusively by the `DataLoader`/`create_multimodal_data()` pipeline during construction and are not meant to be called directly by analysis code. Public methods (`get_signals()`, `get_eeg_data_ch/cg()`, `to_mne_raw()`, `get_events_as_marker_channel()`, `print_events()`) are the supported read API.

Full field/method reference: [docs/data_structure_spec.md](docs/data_structure_spec.md). Visual pipeline diagrams (load → filter → merge → decimate → access → analyze): [docs/architecture_diagram.md](docs/architecture_diagram.md).

Loading pipeline in one line: raw SVAROG EEG/ECG (`.obci`/`.xml`/`.tag`) + Pupil-Labs ET CSVs → `create_multimodal_data()` → per-modality loaders (`load_eeg_data`, `load_et_data`) → filtering/mounting/event detection → merged into the DataFrame → optional `_decimate_signals()` → unified `events` structure → consumed via `get_signals()` / `to_mne_raw()` / `export_to_xarray()`.

An automatic consistency check (`check_consistency_of_multimodal_data()` in `src/dataloader.py`) validates that `modalities`, the `events` column, and EEG-vs-ET event start times agree; it runs by default inside `create_multimodal_data()` (`run_consistency_check=True`, non-strict by default).

### Export layer: xarray / NetCDF

`src/export.py` and `src/ncdf.py` convert slices of `MultimodalData` into `xarray.DataArray` objects and persist them as NetCDF (`.nc`), which is the interchange format used for downstream analysis and for MATLAB interop.

- `export_to_xarray(...)` — one modality/member/event slice → `xarray.DataArray` (dims `['time', 'channel']`), with structured metadata in `.attrs` (`metadata_json`, filtration info for EEG, etc).
- `write_dyad_to_uniwaw_imported(...)` — exports a whole dyad to `data/UNIWAW_imported/<MODALITY>/<DYAD_ID>/<child|caregiver>/<DYAD_ID>_<MODALITY>_<ch|cg>_<EVENT>.nc`.
- `load_xarray_from_netcdf(...)` / `get_export_metadata(...)` — load a `.nc` file back and read its structured metadata.
- Naming conventions (member codes `ch`/`cg`, site codes `K`/`W`/`M`/`H`, session names `Secore`/`Talk1`/`Talk2`/`Peppa`/`Incredibles`/`Brave`) are documented in [docs/export_ncdf_guide.md](docs/export_ncdf_guide.md) — follow them exactly when adding new export paths, since MATLAB tooling (`matlab_utils/`) and other pipeline stages parse these names.

### SECORE (cardiac) pipeline

`src/secore_loader.py` loads Polar H10 IBI CSVs for both dyad members, corrects ectopic beats (neurokit2/Kubios), interpolates to a uniform grid, computes sliding-window RMSSD, aligns to EEG-derived IBI via cross-correlation, and packages everything as one `xarray.DataArray` with event-window annotations. Entry point: `build_h10_ibi_rmssd_xarray_auto(dyad_nr=..., preferred_dev_ch=..., preferred_dev_cg=...)`. Details: [docs/secore_loader_guide.md](docs/secore_loader_guide.md).

### Connectivity / spectral decomposition: `src/mtmvar.py`

MVAR modeling and Directed Transfer Function (DTF) analysis for EEG connectivity, plus FAD (Frequency-Amplitude-Damping) decomposition — an AR-model-based alternative to specparam that decomposes a signal into exponentially damped oscillators via partial-fraction expansion of the AR transfer function (`fad_decomposition()`, `fad_components_table()`). Based on Kamiński & Blinowska DTF methodology and Blinowska & Żygierewicz FAD methodology — see module docstring and [docs/fad_specparam_guide.md](docs/fad_specparam_guide.md) for the underlying math and comparison against `specparam`/FOOOF peak fitting.

### Exploratory spectral parameterization pipeline

`scripts/Exploratory_spectral_analysis_pipelnie/` is a numbered batch pipeline (`01_compute_psd.py` → `02_run_specparam.py`/`02_individual_specparam.py` → `03_peak_inventory.py` → `04_band_assignment.py` → `05_roi_definition.py` → `06_roi_specparam_rerun.py` → `07_movie_stability_check.py`) that derives individualized EEG frequency bands and ROIs per participant, since fixed adult bands (theta 4-8 Hz, alpha 8-13 Hz) do not hold for the 3-6 y.o. cohort. It reads ICA-cleaned per-participant NetCDF files (`<dyad_id>_EEG_<ch|cg>_passive_movies_cleaned.nc`, produced by `scripts/EEG_ICA_clean.py` / `src/ica_preprocessing.py`) via `src/io_utils.py`, and writes intermediate/derived artifacts to `Exploratory_spectral_analysis/` (git-ignored, regenerable — see comment in `.gitignore`). Supporting modules: `src/psd.py` (multitaper PSD), `src/specparam_utils.py` (specparam fit/extract/QC), `src/peaks.py` (peak inventory/clustering), `src/bands.py` (slow/fast band assignment via seeded k-means), `src/roi.py` (ROI channel groupings), `src/viz.py` (shared plotting conventions for this pipeline). Findings and rationale for each pipeline stage's parameter choices are recorded in `Exploratory_spectral_analysis/spectral_parameterization_report.md`.

### Envelopes

`src/envelopes.py` provides instantaneous-amplitude envelope utilities for narrow-band signals (zero-phase Butterworth band-pass + Hilbert transform), demoed in `scripts/demo_envelopes.py`.

### MNE / eye-tracking / ICA bridges

- `src/mne_bridge.py` — converts exported EEG NetCDF back into `mne.io.Raw` objects (standard 10-20 montage) for use with MNE tooling.
- `src/ica_preprocessing.py` — ICA-based EEG cleaning pipeline (loads via `mne_bridge`, applies `mne.preprocessing.ICA`).
- `src/eyetracker.py` — Pupil Labs eye-tracking signal processing (gaze/pupil filtering, blink interpolation, common time-vector construction across movies/talk tasks).

## Data layout conventions

Raw per-dyad data lives under `data/<dyad_id>/` (e.g. `data/W_030/`):
```
data/<dyad_id>/
    eeg/  (or EEG/)   <dyad_id>.obci{,.raw}, .xml, .tag, H10 IBI/ECG CSVs, stage-timing txt
    et/   (or ET/)    child/{000,001,002}, caregiver/{000,001,002}   (movies, talk1, talk2)
```
Exported/processed data lands in `data/UNIWAW_imported/<MODALITY>/<dyad_id>/<child|caregiver>/`. `data/_meta_data.csv` holds per-participant metadata (age, group T/ASD/P, sex, device IDs, etc). Site codes: `K`=Kopenhagen, `W`=Warsaw, `M`=Milan, `H`=Heidelberg; dyad numeric code is 3 zero-padded digits (e.g. `W_030`).

Notebooks in `scripts/` prefixed `deprecated_` are kept for reference but superseded by newer scripts/notebooks — don't build on them for new work.

## Working conventions

- Follow the DataFrame column-naming and NCDF export-naming conventions above exactly; several pipeline stages, `src/mne_bridge.py`, and `matlab_utils/ncdf_test_read_demo.m` parse filenames/column names rather than structured metadata alone.
- When changing `MultimodalData`'s schema (new fields, renamed filtration keys, etc.), update [docs/data_structure_spec.md](docs/data_structure_spec.md)'s version history section — this doc is the authoritative spec and is actively kept in sync with the implementation (see its version history for the pattern to follow).
- Treat `_`-prefixed `MultimodalData` methods as DataLoader-internal; add new data-population logic there rather than calling/extending them from analysis scripts.
