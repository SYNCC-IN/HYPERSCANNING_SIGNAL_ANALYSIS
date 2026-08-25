![Funded by the European Union](EN_FundedbytheEU_RGB_POS.png)
The work was conducted as a part of the SYNCC-IN project funded by the European Union (EU) under the Horizon Europe programme (agreement No 101159414).  



# HYPERSCANNING_SIGNAL_ANALYSIS
A set of tools to analyze multimodal data recorded in hyperscanning experiments of diads in SYNCC-IN project.


In this repo, we develop Python tools to operate and analyze multimodal data, i.e.:
- EEG
- ECG
- IBI - Inter bit intervals
- ET - eye-trackers
  
Currently, the tools are tailored for the experimental setup executed at the University of Warsaw as a part of the SYNCC-IN project.

These experiments consist of three major parts: SECORE, passive MOVIE viewing, and free TALK.

We hope that they can be adapted to the paradigms of other Partners.

## Current data pipeline

The repository is organized around a linear processing chain. Each stage reads the
previous stage's output and writes to its own folder, so a stage can be re-run without
repeating earlier ones.

1. **Raw import.** Raw SVAROG EEG/ECG and Pupil-Labs eye-tracking files recorded at
   UW live in `UNIWAW_RAW_DATA`. `src/dataloader.py`'s `create_multimodal_data(...)`
   (backed by the `MultimodalData` class in `src/data_structures.py`) reads them into
   the project's internal, unified in-memory format — see
   [docs/data_structure_spec.md](docs/data_structure_spec.md).
2. **Per-task NCDF export (preferred pipeline input).** `src/export.py`'s
   `export_passive_and_talk_data(...)` loads each dyad via step 1 and exports two
   continuous chunks per modality/member — `passive_movies` (Peppa, Incredibles,
   Brave back to back) and `talk` — to NetCDF. This is run in batch by
   [scripts/export_dyade_to_ncdf_by_task_batch.py](scripts/export_dyade_to_ncdf_by_task_batch.py)
   (per-dyad example: [scripts/export_dyade_to_ncdf_by_task_demo.py](scripts/export_dyade_to_ncdf_by_task_demo.py)),
   producing the `UNIWAW_EEG_exported_BY_TASKS` folder tree:
   `UNIWAW_EEG_exported_BY_TASKS/<MODALITY>/<dyad_id>/<child|caregiver>/<dyad_id>_<MODALITY>_<ch|cg>_<passive_movies|talk>.nc`.
   **`UNIWAW_EEG_exported_BY_TASKS` is the preferred input for all downstream
   analysis pipelines** — prefer it over the older per-event export described under
   [Older single-event export API](#older-single-event-export-api) below.
3. **EEG ICA cleaning.** [scripts/EEG_ICA_clean.py](scripts/EEG_ICA_clean.py) drives
   `src.ica_preprocessing.ICAPreprocessor` over the `passive_movies` NCDF files from
   step 2, in three stages: fit ICA (`fit_and_save_ica`), classify components with
   ICLabel (`classify_and_save_labels`, requires manual QC review of the saved
   CSV/figures), and apply the reviewed exclusions (`apply_ica_and_save`). Cleaned EEG
   is written under `UNIWAW_EEG_exported_BY_TASKS/ICA_output/EEG_ICA_CLEANED/<dyad_id>/<dyad_id>_EEG_<ch|cg>_passive_movies_cleaned.nc`.
   **This `EEG_ICA_CLEANED` folder is the clean-EEG input that downstream analyses
   should use.**
4. **Analysis pipelines.** Downstream scripts read cleaned EEG NetCDF files via
   `src/io_utils.py` (`get_participant_files`, `load_eeg_nc`, `trim_to_event_window`).
   [scripts/Exploratory_spectral_analysis_pipelnie/](scripts/Exploratory_spectral_analysis_pipelnie/)
   is the reference example of a full pipeline built this way — see its section under
   [docs/architecture_diagram.md](docs/architecture_diagram.md) for how it is composed
   from `src/` functions.

`data/` in this repository (e.g. `data/W_030/`, `data/UNIWAW_imported/`) is a small
local sample dataset used for unit tests, demos, and the code examples further down in
this README — it is **not** the production dataset, which lives on the
SYNCC-IN Google Drive under `UNIWAW_RAW_DATA` / `UNIWAW_EEG_exported_BY_TASKS`
(paths hard-coded at the top of the scripts above; anyone else running them needs to
point those paths at their own mount of that Drive folder).

`src/warsaw_pilot_data.py` is an older, standalone example pipeline (HRV/EEG/combined
DTF analysis) that predates the pipeline above and operates on the small local `data/`
sample only; it is not part of the current production data flow.

## Data structure update (v2.4)

`MultimodalData.eeg_filtration` uses nested dictionaries instead of flat fields.

- `eeg_filtration.notch`: `{"Q", "freq", "a", "b", "applied"}`
- `eeg_filtration.low_pass`: `{"type", "a", "b", "applied"}`
- `eeg_filtration.high_pass`: `{"type", "a", "b", "applied"}`

Example:

```python
notch_freq = multimodal_data.eeg_filtration.notch["freq"]
high_pass_type = multimodal_data.eeg_filtration.high_pass["type"]
is_low_pass_applied = multimodal_data.eeg_filtration.low_pass["applied"]
```

For full details, see [docs/data_structure_spec.md](docs/data_structure_spec.md).

For NetCDF export/import usage, see [docs/export_ncdf_guide.md](docs/export_ncdf_guide.md).

For MVAR/DTF analysis of exported EEG NCDF files, see the [MVAR helpers section](docs/export_ncdf_guide.md#mvar--dtf-analysis-helpers) and the batch notebook [scripts/ESCan_drfat.ipynb](scripts/ESCan_drfat.ipynb).

## Multimodal consistency checker

The loader and dataloader module include a consistency validator:

- `check_consistency_of_multimodal_data(multimodal_data, start_error=0.35, event_time_error=None, verbose=True)`

It validates:

- consistency between `multimodal_data.modalities` and actually present DataFrame columns,
- consistency between `multimodal_data.events` and the `events` column,
- consistency of matching EEG/ET event start times (`EEG_events` vs `ET_event`) within `start_error`.

Important:

- EEG/ET start-time validation runs only when both `EEG` and `ET` are present in `multimodal_data.modalities`.

`create_multimodal_data(...)` can run this check automatically via:

- `run_consistency_check=True` (default),
- `consistency_strict=False` (if `True`, raises `ValueError` when inconsistent),
- `consistency_start_error=0.35`.

Example:

```python
from src.dataloader import create_multimodal_data, check_consistency_of_multimodal_data

md = create_multimodal_data(
	data_base_path='./data',
	dyad_id='W_030',
	load_eeg=True,
	load_et=True,
	run_consistency_check=True,
	consistency_strict=False,
)

report = check_consistency_of_multimodal_data(md, start_error=0.35, verbose=False)
print(report['is_consistent'])
print(report['eeg_et_start_consistency'])
```

## Older single-event export API

The example below uses `export_to_xarray(...)`, which exports one modality/member/
**event** at a time (e.g. just `'Incredibles'`) to the local `data/UNIWAW_imported/`
demo folder. This is the older, lower-level export API described in full in
[docs/export_ncdf_guide.md](docs/export_ncdf_guide.md); for real analysis work, use
the per-task export described in [Current data pipeline](#current-data-pipeline)
above instead.

```python
from src.dataloader import create_multimodal_data
from src.export import export_to_xarray

md = create_multimodal_data(
	data_base_path='./data',
	dyad_id='W030',
	load_eeg=True,
	load_et=True,
	decimate_factor=8,
)

data_xr = export_to_xarray(
	multimodal_data=md,
	selected_event='Incredibles',
	selected_channels=['Fz', 'Cz', 'Pz'],
	selected_modality='EEG',
	member='ch',
	time_margin=10,
)

print(data_xr)
print(data_xr.attrs)
data_xr.plot.line(x='time', hue='channel')

# optional: z-score per channel over time
# from scipy.stats import zscore
# data_xr.data = zscore(data_xr.data, axis=0, nan_policy='omit')
```
