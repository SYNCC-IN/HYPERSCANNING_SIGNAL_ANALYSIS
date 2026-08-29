# Function Reference (`src/` library)

Catalog of the reusable, public API in `src/`, grouped by pipeline stage / theme (see
[CLAUDE.md](../CLAUDE.md) for the stage numbering and the `src/` = library,
`scripts/` = pipelines split this catalog follows). Intended to be pasted into a
prompt so a new pipeline or feature is designed to reuse existing functions instead of
duplicating them.

Scope and conventions:

- Only public (non-`_`-prefixed) functions/methods are listed; private helpers are
  internal implementation details and are omitted.
- Signatures are auto-extracted from the current source (default values included);
  purpose lines are the first sentence of each function's docstring, or a short manual
  note where the source has no docstring (marked *(no docstring)*).
- `class Name(init_args)` lines describe the constructor; methods are indented under
  the class.
- This file can go stale — for anything you're about to build on, grep the signature
  in the actual `src/*.py` file before relying on it (per [CLAUDE.md](../CLAUDE.md)'s
  "before trusting any doc" note).

---

## Stage 1 — Raw import / internal data model

### `src/dataloader.py`
Loads raw SVAROG EEG/ECG + Pupil-Labs ET files into the internal `MultimodalData` format.

- `create_multimodal_data(data_base_path, dyad_id, load_eeg=True, load_et=True, load_meta=False, lowcut=4.0, highcut=40.0, eeg_filter_type='fir', mounts_eeg=False, interpolate_et_during_blinks_threshold=0, median_filter_size=64, low_pass_et_order=351, et_pos_cutoff=128, et_pupil_cutoff=4, pupil_model_confidence=0.9, window_size=30, decimate_factor=1, plot_flag=False, run_consistency_check=True, consistency_strict=False, consistency_start_error=0.35)` — **Main entry point.** Creates and populates a `MultimodalData` instance by loading EEG and ET data from `data_base_path/<dyad_id>/EEG|eeg/...` and `.../ET|et/child|caregiver/000|001|002/`.
- `check_consistency_of_multimodal_data(multimodal_data, start_error=0.35, event_time_error=None, verbose=True)` — Validate consistency of a `MultimodalData` object (modalities vs. columns, `events` vs. structure, EEG/ET event start times).
- `load_eeg_data(multimodal_data=None, dyad_id=None, folder_eeg=None, lowcut=4.0, highcut=40.0, eeg_filter_type='fir', window_size=30, mounts_eeg=False, plot_flag=False)` — Load and filter EEG data from SVAROG-format files into a `MultimodalData` instance.
- `load_et_data(multimodal_data, dyad_id, folder_et, interpolate_et_during_blinks_threshold=0, median_filter_size=64, low_pass_et_order=351, et_pos_cutoff=128, et_pupil_cutoff=1, pupil_model_confidence=0.9, plot_flag=False)` — Load eye-tracking data from CSV files and integrate it into the `MultimodalData` instance.
- `save_to_file(multimodal_data, output_dir)` / `load_output_data(filename)` / `get_eeg_data(df, who)` / `export_eeg_to_mne_raw(multimodal_data, who, times=None, event=None, margin_around_event=0)` — thin re-export wrappers; the real implementations are `src.multimodal_io.save_to_file`/`load_output_data`, `MultimodalData.get_eeg_data` (static), and `MultimodalData.to_mne_raw` respectively. Prefer calling those directly in new code.
- `to_status(value)` *(no docstring)* — normalizes a raw metadata value (`None`/NaN/string) to a `WhoEnum` status (`ch`/`cg`/`both`/`Neither`).

### `src/data_structures.py`
Defines `MultimodalData`, the unified per-dyad DataFrame container, and its supporting dataclasses. Full field reference: [docs/data_structure_spec.md](data_structure_spec.md).

- `class MultimodalData()` — one dyad's EEG/ECG/IBI/ET/events/diode, unified in a single `pandas.DataFrame` (`.data`) sharing one time base and `fs`. Populated by `create_multimodal_data()`, not constructed directly.
  - `get_signals(mode='EEG', member='ch', selected_channels=None, selected_events=None, selected_times=None)` — retrieve `(time, channel_names, data)` filtered by mode/member/channels/events/times.
  - `get_events_as_marker_channel(selected_times=None)` — events as a time-aligned integer marker channel + name→marker map.
  - `eeg_channel_names_all()` — combined child + caregiver EEG channel name list.
  - `get_eeg_data_ch()` / `get_eeg_data_cg()` — child/caregiver EEG as `[n_channels x n_samples]` array.
  - `get_eeg_data(df, who)` *(static)* — EEG array + channel names for `who` (`'ch'`/`'cg'`) from any DataFrame following the column-naming convention.
  - `to_mne_raw(who, times=None, event=None, margin_around_event=0)` — export to `mne.io.RawArray` with annotations, montage, and filter-info set; `event` takes precedence over `times` if both given.
  - `from_mne_raw(raw, time, who)` — import cleaned EEG data from an MNE `Raw` object back into the `MultimodalData` structure (inverse of `to_mne_raw`).
  - `print_events()` — pretty-print the `events` dict as a table.
- `class Filtration()`, `class Paths()`, `class Tasks()` (+ `DualHRV`/`DualEEG`/`DualFnirs`/`DualET`), `class ChildInfo()`, `class WhoEnum()` — supporting dataclasses/enum for filter params, file paths, task flags, and child metadata; see [docs/data_structure_spec.md](data_structure_spec.md) for field-level detail.

### `src/eyetracker.py`
Pupil-Labs eye-tracking signal processing, called internally by `load_et_data`.

- `process_time_et(child_pos_df_0, caregiver_pos_df_0, child_pupil_df_0, caregiver_pupil_df_0, child_pupil_df_1, caregiver_pupil_df_1, child_pupil_df_2, caregiver_pupil_df_2, Fs=1024)` — build a common time vector spanning movies/talk1/talk2 from gaze + pupil dataframes.
- `process_pos(pos_df, df, who, median_filter_size=64, order=351, cutoff=128, Fs=1024)` — median-filter + low-pass-filter (with delay correction) gaze x/y onto the common time vector.
- `process_pupil(pupil_df, df, who, model_confidence=0.9, median_size=10, order=351, cutoff=1, Fs=1024, plot_flag=False)` — confidence-filter, median-filter, interpolate, and low-pass-filter 3D pupil diameter.
- `process_event_et(annotations, df, event_name=None)` — mark events from ET annotations into the `ET_event` column.
- `process_blinks(blinks, df, who)` — mark blink confidence into the `ET_{who}_blinks` column.

---

## Stage 2 — Per-task NCDF export & MNE bridge

### `src/export.py`
Exports `MultimodalData` to `xarray`/NetCDF. See [docs/export_ncdf_guide.md](export_ncdf_guide.md).

- `export_passive_and_talk_data(dyad_id_list=None, load_eeg=True, load_et=True, load_meta=True, lowcut=1.0, highcut=40.0, eeg_filter_type='fir', decimate_factor=8, plot_flag=False, time_margin=20, input_data_path='../data', export_path='../data/UNIWAW_imported', mounts_eeg_multimodal=False, export_mounted='CAR', EEG_bad_channels=None, verbose=False, mne_plot_flag=False, logger=None)` — **preferred, current export path.** Exports two continuous chunks per modality/member (`passive_movies`, `talk`) instead of one file per event.
- `write_dyad_to_uniwaw_imported(dyad_id_list=None, load_eeg=True, load_et=True, load_meta=True, lowcut=1.0, highcut=40.0, eeg_filter_type='fir', decimate_factor=8, plot_flag=False, time_margin=10, input_data_path='../data', export_path='../data/UNIWAW_imported', verbose=False, logger=None)` — older per-event export: one `.nc` file per modality/member/event for a whole dyad.
- `export_chunk_to_xarray(multimodal_data, selected_events, selected_channels, selected_modality, member, time_margin, chunk_name, EEG_montage=None, EEG_bad_channels=None, verbose=True, mne_plot_flag=False, logger=None)` — lower-level building block used by both functions above: exports one chunk spanning multiple events to a single `xarray.DataArray` (handles bad-channel interpolation + re-referencing via MNE).

### `src/ncdf.py`
Load exported NetCDF back into `xarray`.

- `load_xarray_from_netcdf(filename, decode_json_attrs=True)` — load a `.nc` file into `xarray.DataArray`, optionally JSON-decoding structured attrs.
- `get_export_metadata(data_xr)` — decode the `metadata_json` attrs payload into a dict.
- `load_ncdf(path)` — one-line convenience loader for scripts (wraps `load_xarray_from_netcdf`).
- `task_regions(data_xr)` — build plotting "regions" (name/span/color) from the `task_events_structure` attr, for use with `plot_xarray_signals(regions=...)`.

### `src/mne_bridge.py`
Bridges exported EEG NetCDF to MNE and runs AutoReject-based quality checks.

- `load_eeg_ncdf_as_mne_raw(ncdf_path, montage='standard_1020', scale_to_volts=1e-06, data_xr=None)` — load an EEG `.nc` file as `mne.io.RawArray` (10-20 channels typed `'eeg'`, M1/M2 typed `'misc'`); pass a pre-loaded `data_xr` to avoid re-reading from disk.
- `load_eeg_signals(ncdf_path, channel_subset=None, low_cutoff_hz=None, high_cutoff_hz=None)` — load, optionally Butterworth-filter, trim the time margin, drop M1/M2, and z-score-normalize an EEG `.nc` file; returns `(signals, channel_names, fs, time_s, event_duration_s)`.
- `run_eeg_autoreject_quality_report(ncdf_path, epoch_duration_s=2.0, n_interpolate=(1, 2, 4), cv=5, random_state=42, n_jobs=-1, montage='standard_1020', scale_to_volts=1e-06, verbose=True)` — full NCDF → MNE → AutoReject pipeline; returns `raw`, `epochs`, `autoreject`, `reject_log`, `epoch_summary`, `channel_summary`, `global_summary`, `figure`, `axis`.
- `check_exported_data_quality(dyad, modality, member, task, export_folder)` — thin QC wrapper used by the batch export pipeline: runs `run_eeg_autoreject_quality_report` on one exported EEG chunk and saves the figure under `<export_folder>/EEG/Quality_reports/`.

### `src/multimodal_io.py`
joblib persistence of `MultimodalData` objects (unrelated to the NCDF export layer).

- `save_to_file(multimodal_data, output_dir)` — save a `MultimodalData` instance to a joblib file.
- `load_output_data(filename)` — load a `MultimodalData` instance from a joblib file.

---

## Stage 3 — EEG ICA cleaning

### `src/ica_preprocessing.py`
Three-stage ICA cleaning pipeline over `passive_movies` NCDF chunks.

- `class ICAPreprocessor(export_folder, target_events)` — driven by `scripts/EEG_ICA_clean.py`.
  - `find_eeg_files(smoke_test=True, smoke_dyads_n=2, dyad_ids=None)` — populate `self.eeg_files` by scanning `export_folder` for `*_{target_event}.nc` files, grouped by dyad; supports smoke-test / specific-dyad-list / full modes.
  - `fit_and_save_ica(ica_folder, n_components=15, max_iter=2000)` — **Stage A.** Fit `infomax`-extended ICA per file (1 Hz high-pass, rank-limited component count), save `.fif` under `<ica_folder>/ICA_COMPs/<dyad_id>/`.
  - `classify_and_save_labels(ica_folder, eog_channels=None, iclabel_threshold=0.7, neural_threshold=0.5, amplitude_threshold=100, exclude_labels=None, timecourse_seconds=30.0)` — **Stage B.** Classify components with ICLabel (requires `mne-icalabel`), save a user-editable `exclude` column CSV + QC topomap/timecourse figure under `<ica_folder>/ICA_QC_FIGS_and_CSV/`. **Manual review of the CSV is expected before Stage C.**
  - `apply_ica_and_save(ica_folder, cleaned_folder, save_plots=True)` — **Stage C.** Apply the (reviewed) `exclude` flags, save cleaned EEG to `<cleaned_folder>/<dyad_id>/<label>_cleaned.nc`.

---

## Stage 4 — Analysis pipeline library (exploratory spectral pipeline)

### `src/io_utils.py`
Loads cleaned EEG NetCDF for analysis pipelines.

- `ensure_dir(path)` — create a directory (+ parents) if missing; returns it as `Path`.
- `get_participant_files(data_dir)` — scan a directory of `*_passive_movies_cleaned.nc` files, return one row per participant (`filepath`, `dyad_id`, `role_code`, `role`).
- `load_eeg_nc(filepath)` — load one cleaned EEG `.nc` file into a dict: `data` (chan x time, µV), `channel_names`, `sfreq`, `time`, `dyad_id`, `role_code`, `role`, `movies` (per-movie boundaries), `age_months`, `group`, `sex`.
- `trim_to_event_window(data, time, duration, start=0.0)` — slice `(data, time)` to `[start, start+duration]`, e.g. one movie within the full recording.

### `src/psd.py`
- `compute_psd_multitaper(data, sfreq, fmin, fmax, bandwidth)` — multitaper PSD for multichannel EEG (`mne.time_frequency.psd_array_multitaper`).
- `average_psd_across_conditions(psd_dict)` — average PSD arrays across conditions (e.g. movies) for one participant.

### `src/specparam_utils.py`
specparam (FOOOF) fitting, extraction, and quality checks.

- `fit_specparam(freqs, psd_1d, freq_range, peak_width_limits, max_n_peaks, min_peak_height, aperiodic_mode)` — fit a `specparam.SpectralModel` to one power spectrum.
- `batch_fit_specparam(freqs, psd_2d, channel_names, freq_range, peak_width_limits, max_n_peaks, min_peak_height, aperiodic_mode)` — fit specparam across all channels of one participant's PSD.
- `extract_peak_params(model)` — detected peak params (CF, power, bandwidth) from a fitted model.
- `extract_aperiodic_params(model)` — aperiodic (1/f) params from a fitted model.
- `extract_fit_quality(model)` — goodness-of-fit metrics (R², error) from a fitted model.
- `find_peak_in_window(peaks, freq_window)` — strongest peak within a fixed frequency window.
- `find_peak_in_individual_window(peaks, cf, bw)` — strongest peak within an individualized band window (center freq ± bandwidth).

### `src/peaks.py`
Peak inventory: prevalence across scalp/frequency, and channel clustering.

- `classify_channel_cluster(channel_name, channel_clusters)` — map a channel name to a scalp cluster label.
- `compute_peak_prevalence(peaks_df, channel_names, freq_bins, role, group=None, n_participants=None)` — fraction of participants with a detected peak per freq-bin x channel.
- `cluster_peaks_across_channels(peaks_df, freq_tolerance=1.0, min_channels=3)` — cluster detected peaks across channels by center-frequency proximity.
- `cluster_peaks_within_roi(peaks_df, roi_channels, freq_tolerance=1.0)` — cluster peaks within a single ROI by center-frequency proximity.
- `cluster_peaks_all_rois(peaks_df, channel_clusters, freq_tolerance=1.0)` — run within-ROI clustering for all ROIs, one participant.
- `count_peaks_per_roi(peaks_df, roi_channels, freq_range=(3, 14))` — count peaks per participant within an ROI + frequency range.

### `src/bands.py`
Slow/fast rhythm band assignment and IAF metrics.

- `assign_two_bands_kmeans(roi_peaks_df, slow_cf_range=(3.0, 7.5), fast_cf_range=(7.5, 13.0), min_gap=1.5, max_iter=50)` — assign slow/fast bands via seeded, power-weighted k-means over specparam peaks for one participant x ROI.
- `assign_bands_all_rois(participant_peaks_df, participant_id, role, roi_channels, min_gap=1.5, slow_cf_range=(3.0, 7.5), fast_cf_range=(7.5, 13.0), max_iter=50)` — run `assign_two_bands_kmeans` for every ROI, one participant.
- `compute_iaf_metrics(band_assignments_df)` — individual alpha frequency metrics + dyadic (child-caregiver) distances.

### `src/roi.py`
ROI definitions, validation, and averaging.

- `define_rois_theory()` — theory-driven ROI channel groupings for the 19-channel 10-20 montage.
- `validate_roi_with_prevalence(prevalence_df, roi_channels, freq_bin, min_prevalence=0.5)` — check whether all channels in an ROI meet a minimum peak-prevalence threshold.
- `validate_roi_two_bands(prevalence_df, roi_channels, slow_window, fast_window, min_prevalence=0.5)` — validate ROI viability separately for slow and fast bands.
- `validate_roi_individual_bands(peaks_df, band_assignments_df, roi_channels, roi_label, slow_window=(3, 7), fast_window=(7, 14))` — validate ROI prevalence using each participant's individualized band windows.
- `check_peak_survival(channel_peaks_list, roi_peaks, band_window, freq_tolerance=1.0)` — check whether a peak seen at individual channels survives ROI averaging.
- `average_psd_within_roi(psd_2d, channel_names, roi_channels)` — average PSD across the channels belonging to an ROI.

### `src/viz.py`
Plotting/figure-generation shared across the exploratory spectral pipeline (consistent color/linestyle conventions: `COLORS`, `ROLE_LINESTYLE`, `PARTICIPANT_BAND_COLORS`).

- `make_montage_info(channel_names, sfreq=128.0, montage_name='standard_1020')` — build an `mne.Info` with standard 10-20 positions, for topomap plotting.
- `plot_peak_prevalence_topomap(prevalence_series, channel_names, title, vmin=0, vmax=1, ax=None)` — topomap of peak-detection prevalence.
- `plot_peak_freq_histogram(peaks_df, channel_cluster, role, group=None, bin_width=0.5, freq_range=(3, 25), ax=None)` — histogram of detected peak center frequencies for a channel cluster.
- `plot_peak_freq_topomap_individual(peaks_df, channel_names, participant_id, f_range, ax=None)` — topomap of the strongest peak's center frequency per channel, one participant.
- `plot_peak_freq_vs_age(peaks_df, roi_channels, role='child', ax=None)` — scatter of strongest peak CF vs. age, colored by group.
- `plot_peak_cluster_topomaps(cluster_df, peaks_df, channel_names, participant_id, role, group, max_clusters=6)` — topomaps per peak cluster, one participant.
- `plot_roi_cluster_scatter(clusters_df, group_colors, roi_order=None, freq_range=(3, 14))` — beeswarm scatter of within-ROI peak clusters.
- `plot_peak_count_histogram(peak_counts_df, rois)` — histogram of peak count per participant per ROI.
- `plot_two_peak_scatter(two_peak_df)` — scatter of peak-1 freq vs. peak-2 freq for participants with exactly 2 peaks in an ROI.
- `plot_individual_peak_profiles(peaks_df, roi_channels, roi_label, role='child')` — strip plot of each participant's detected peaks within one ROI.
- `plot_band_assignment_strips(band_assignments_df, rois)` — strip plot of each participant's slow/fast (or single-rhythm) assignment.
- `plot_iaf_distance_by_group(iaf_metrics_df)` — boxplot + strip overlay of dyadic IAF / slow-rhythm distance by group.
- `plot_gap_distribution(band_assignments_df, min_gap)` — histogram of slow/fast frequency gap for two-rhythm participants.
- `plot_band_cf_histograms(band_assignments_df, rois, bin_width=0.5, slow_color=None, fast_color=None)` — histograms of merged slow/fast CF per ROI/role/group.
- `plot_dyad_gap_scatter(iaf_metrics_df)` — scatter of each dyad's slow-fast gap, caregiver vs. child.
- `plot_roi_validation_heatmap(roi_validation_df, min_prevalence=0.5)` — heatmap of ROI prevalence validation, per band and subgroup.
- `plot_roi_validation_heatmap_individual(prevalence_df, min_prevalence=0.5, title_suffix='(individualized bands)')` — same, using individualized band windows.
- `plot_survival_rate_bars(summary_df, threshold=0.6)` — bar chart of ROI x band peak-survival rate, by role/group.
- `plot_stability_heatmap(summary_df, threshold=60.0, title_suffix='')` — heatmap of % participants with a peak detected in all 3 movies.
- `plot_detection_consistency_heatmap(movie_band_peaks_df, movies, bands=('slow', 'fast'), title=None)` — per-participant heatmap of peak detection across movies/bands.
- `plot_participant_spectral_overview(psd_by_roi, freqs, band_assignments_participant, participant_id, role, group, specparam_settings, roi_layout)` — per-participant grid of ROI spectra with specparam fits + individualized bands.

---

## Envelope extraction

### `src/envelopes.py`
Instantaneous-amplitude envelope utilities for narrow-band EEG/HRV signals.

- `bandpass_filter(signal, sfreq, low, high, order)` — zero-phase Butterworth band-pass filter, 1-D signal.
- `filter_individual_band(signal, sfreq, center_freq, bandwidth, order)` — band-pass around an individualized rhythm (center ± bandwidth/2).
- `hilbert_envelope(signal)` — instantaneous amplitude envelope via the analytic (Hilbert-transform) signal.
- `downsample(signal, sfreq, target_sfreq)` — anti-alias + resample a 1-D signal.
- `eeg_band_envelope(signal, sfreq, center_freq, bandwidth, order, target_sfreq)` — filter + Hilbert-envelope + downsample, one call, for an individualized EEG band.
- `hrv_hf_envelope(ibi_signal, ibi_sfreq, hf_low, hf_high, order, target_sfreq)` — same, for the HRV high-frequency band of an IBI signal.
- `average_channels(signals)` — average signals across channels (e.g. within an ROI).
- `plot_signal_filtered_envelope(raw, filtered, envelope, sfreq, title)` — plot raw + filtered + envelope together.
- `plot_dyad_envelopes(env_child, env_caregiver, sfreq, title, labels)` — child + caregiver envelopes on a shared time axis.
- `plot_eeg_hrv_envelopes(env_eeg, eeg_sfreq, env_hrv, hrv_sfreq, title)` — EEG-band and HRV-HF envelopes on separate shared-time panels.

---

## Connectivity: MVAR / DTF / FAD

### `src/mtmvar.py`
MVAR modeling, DTF-family connectivity, and FAD decomposition. See [docs/fad_specparam_guide.md](fad_specparam_guide.md).

- `ar_coeff(data, model_order=5)` — estimate MVAR coefficients for multivariate/multi-trial data.
- `mvar_criterion(data, max_model_order, crit_type='AIC', plot=False)` — AIC/HQ/SC model-order selection criteria.
- `mvar_transfer_function(ar_coeffs, freqs, fs)` — transfer function H from MVAR coefficients.
- `multivariate_spectra(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC')` — multivariate power spectra for all channels.
- `full_freq_dtf(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC')` — full-frequency DTF (ffDTF).
- `dtf_multivariate(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC', comment=None)` — (plain, non-full-frequency) multivariate DTF.
- `direct_dtf(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC')` — direct DTF (dDTF).
- `gen_partial_directed_coherence(signals, freqs, fs, max_model_order=20, optimal_model_order=None, crit_type='AIC')` — generalized partial directed coherence (GPDC).
- `partial_coherence(spectra)` — partial coherence from a multivariate spectra array.
- `mvar_plot(on_diag, off_diag, freqs, x_label, y_label, chan_names, top_title, scale='linear')` — bar-plot grid of diagonal (auto) + off-diagonal (cross) connectivity terms.
- `graph_plot(connectivity_matrix, ax, freqs, freq_range, chan_names, title)` — plot a connectivity matrix as a directed graph (`networkx`); returns the `DiGraph`.
- `fad_decomposition(signal, fs, model_order=None, max_model_order=20, crit_type='AIC', plot=False, pair_conjugates=True, imag_tol=1e-08)` — FAD (Frequency-Amplitude-Damping) decomposition of a univariate AR model into damped oscillators.
- `fad_components_table(fad_params, output='dataframe', decimals=None)` — compact table (one row per FAD component) for export.
- `compute_and_plot_mvar(ncdf_path, channel_subset=None, max_model_order=20, optimal_model_order=None, crit_type='AIC', freq_min=1.0, freq_max=40.0, freq_step=0.5, low_cutoff_hz=None, high_cutoff_hz=None, plot=True, plot_loaded_signal=False, loaded_signal_max_channels=19, loaded_signal_spacing=8.0, loaded_signal_figsize=(16.0, 9.0))` — high-level pipeline chaining `load_eeg_signals` → `mvar_criterion` → `full_freq_dtf` → `multivariate_spectra` → `mvar_plot`. **Currently broken** (bad internal import — see [docs/export_ncdf_guide.md](export_ncdf_guide.md#mvar--dtf-analysis-helpers)); call the lower-level functions above directly instead.

---

## SECORE (cardiac / H10) sub-pipeline

Independent of the EEG stages above — reads raw H10 CSVs directly. See [docs/secore_loader_guide.md](secore_loader_guide.md).

### `src/secore_loader.py`
- `build_h10_ibi_rmssd_xarray_auto(dyad_nr, video_timings, data_base_path='../data', fs_ibi=8, window_size_rmssd_s=30, decimate_factor_loader=8, decimate_factor_align=16, selected_time=(0, 220), lowcut=1.0, highcut=40.0, eeg_filter_type='iir', plot=False, save_dir=None, preferred_dev_ch=None, preferred_dev_cg=None)` — **entry point.** Auto-detects recording date/time and device IDs, builds the aligned IBI/RMSSD `xarray.DataArray` for one dyad.
- `build_h10_ibi_rmssd_xarray(dyad_nr, date, time_of_recording, dev_ch, dev_cg, video_timings, data_base_path='../data', fs_ibi=8, window_size_rmssd_s=30, decimate_factor_loader=8, decimate_factor_align=16, selected_time=(0, 220), lowcut=1.0, highcut=40.0, eeg_filter_type='iir', plot=False, save_dir=None)` — same, with explicit date/time/device IDs (no auto-detection).
- `load_h10_ibi(path)` — load one H10 IBI CSV, return `(stage, computer_timestamps_s, ibi_ms)`.
- `fix_and_interpolate_ibi(ibi_cum_s, stage, fs_out=8, samp_rate=1024, window_size=30)` — Kubios ectopic-beat correction + cubic-spline interpolation to a uniform grid + sliding-window RMSSD.
- `compute_signal_lag(signal1, signal2, fs, plot=False, label1='', label2='')` — integer-sample lag maximizing cross-correlation (used to align H10 and EEG clocks).

### `src/secore_utils.py`
- `export_h10_to_secore_ncdf(h10_xarray, dyad_id, export_root)` *(no docstring)* — write the 4 SECORE channels (`IBI_CH/CG`, `RMSSD_CH/CG`) to separate NetCDF files under `<export_root>/Secore_IBI|Secore_RMSSD/<dyad_id>/...`.
- `save_secore_QC_figures(dyad_id, export_root)` *(no docstring)* — save SECORE alignment/event QC figures.
- `sec_ms_str_to_float(value)` *(no docstring)* — parse a seconds/milliseconds string field (from the timing file) to `float`, `NaN`-safe.

---

## General plotting utilities

### `src/plot_utils.py`
- `plot_xarray_signals(data_xr, regions=None, stacked=None, max_channels=30, spacing=8.0, normalize=True, figsize=(16.0, 9.0), event_duration=None, time_margin_s=None, title='', xlabel='Time (s)  (0 = event start)', ylabel=None, line_color='#1f4f8b', linewidth=0.6)` — the standard plot for any exported `xarray.DataArray` (used across the export/quality/ICA/envelope code); optional highlighted event `regions` (see `src.ncdf.task_regions`).
- `plot_filter_characteristics(b, a, f, T, Fs, f_lim=None, db_lim=None)` — magnitude response, group delay, impulse response, and step response of a digital filter, in one figure.

### `src/utils.py`
Plotly-based interactive plotting (older / alternative to `plot_utils.py`'s matplotlib plots).

- `plot_eeg_channels_pl(mmd, selected_events, selected_channels, title='Filtered EEG Channels', renderer='auto')` — interactive stacked-channel EEG plot with event highlighting.
- `overlay_eeg_channels_hyperscanning_pl(data_ch, data_cg, event, selected_channels_ch, selected_channels_cg, title='Filtered EEG Channels - Hyperscanning', renderer='auto')` — child vs. caregiver EEG side-by-side subplots.
- `save_figure_to_html(fig, title, event=None)` *(no docstring)* — save a Plotly figure to a standalone `.html` file.

---

## Legacy / standalone example pipelines

Not part of the current production data flow (see [CLAUDE.md](../CLAUDE.md)); listed for
completeness, but prefer the Stage 1-4 library functions above when building something new.

### `src/warsaw_pilot_data.py`
Older, standalone HRV/EEG/combined-DTF example analysis, operating on the small local `data/` sample only.

- `main(plot_debug=False, analyze_hrv_dtf=False, analyze_eeg_dtf=False, analyze_eeg_hrv_dtf=False)` — example analysis entry point.
- `analyze_hrv_dtf_for_event(mmd, selected_event)` / `analyze_eeg_dtf_for_events(mmd, selected_events)` / `analyze_eeg_hrv_dtf_for_events(mmd, selected_events)` — per-modality DTF analysis for one dyad's `MultimodalData`.

### `src/eeg_alpha_ibi_ffdtf.py`
A full pipeline packaged as one class (pre-dates the `src`=library / `scripts`=pipeline split — treat as a pattern to extract functions from, not to extend in place).

- `class EEG_IBI_FFDTF_Pipeline(cleaned_signals_folder, output_ffDTF_folder, target_events, smoke_test=False, smoke_dyads_n=1, left_frontal_eeg_channel='F3', right_frontal_eeg_channel='F4', fs_downsampled=8.0, n_windows=3, window_size=None, ar_p=5, plot_global_enabled=True, save_global_enabled=True, plot_windowed_enabled=True, save_windowed_enabled=True)` — dyadic EEG-IBI ffDTF pipeline: frontal alpha asymmetry (FAA) from EEG, IBI preprocessing, synchronization/resampling, multivariate ffDTF between EEG-alpha-envelope and IBI channels, global and windowed. Scans `cleaned_signals_folder` for EEG+IBI `.nc` files at construction time.
  - `run_pipeline()` — run the full pipeline over all discovered dyads/events.
