import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import Markdown, display
import mne
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine as cosine_distance

try:
    from fooof import FOOOF
    _FOOOF_BACKEND = 'fooof'
except ImportError:
    try:
        from specparam import SpectralModel as FOOOF
        _FOOOF_BACKEND = 'specparam'
    except ImportError:
        FOOOF = None
        _FOOOF_BACKEND = None

from src.export import load_eeg_ncdf_as_mne_raw, plot_loaded_eeg_signals


class ICAPreprocessor:
    def __init__(self, export_folder: Path, target_events: list):
        self.export_folder = export_folder
        self.target_events = target_events
        self.eeg_files: list = []
    

    def find_eeg_files(self, smoke_test: bool = True, smoke_dyads_n: int = 2):
        self.smoke_test = smoke_test
        self.smoke_dyads_n = smoke_dyads_n

        all_eeg_files = sorted([
            p for p in self.export_folder.rglob("*.nc")
            if "_EEG_" in p.name
            and any(p.stem.endswith(f"_{ev}") for ev in self.target_events)
        ])

        if not all_eeg_files:
            raise FileNotFoundError(f"No EEG NetCDF files found for events {self.target_events} under: {self.export_folder}")

        files_by_dyad = {}
        for p in all_eeg_files:
            parts = p.stem.split('_')
            dyad_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else p.stem
            files_by_dyad.setdefault(dyad_id, []).append(p)

        all_dyads = sorted(files_by_dyad.keys())

        if self.smoke_test:
            dyads_to_process = all_dyads[:self.smoke_dyads_n]
            mode = f"SMOKE TEST (first {self.smoke_dyads_n} dyads)"
        else:
            dyads_to_process = all_dyads
            mode = "FULL ICA PREPROCESSING"

        self.eeg_files = []
        for dyad in dyads_to_process:
            self.eeg_files.extend(sorted(files_by_dyad[dyad]))

        print(f"Mode: {mode}")
        print(f"Dyads selected: {len(dyads_to_process)} / {len(all_dyads)}")
        print(f"Files selected: {len(self.eeg_files)} / {len(all_eeg_files)}")
        print("Dyads:")
        for dyad in dyads_to_process:
            print(f"  - {dyad}")


    def _extract_provenance(self, attrs: dict, stem: str) -> dict:
        attrs = attrs or {}
        parts = stem.split('_')

        def _nonempty(val) -> bool:
            return val is not None and str(val).strip() != ''

        dyad_id = attrs.get('dyad_id', '')
        if not _nonempty(dyad_id):
            dyad_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else stem

        who = attrs.get('who', '')
        if not _nonempty(who):
            who = parts[3] if len(parts) >= 4 else ''

        site = attrs.get('site', '')
        if not _nonempty(site):
            site = parts[0] if parts else ''

        return {'dyad_id': str(dyad_id), 'who': str(who), 'site': str(site)}

    def _sanitize_attrs(self, attrs):
        if attrs is None:
            return {}

        sanitized = {}
        for key, value in attrs.items():
            if value is None:
                sanitized[key] = ''
            elif isinstance(value, (dict, list)):
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (str, int, float, np.integer, np.floating)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    def _extract_spectral_fit_outputs(self, model):
        """Return (aperiodic, peaks, r_squared, error) for fooof or specparam backends."""
        # FOOOF backend (legacy style attributes).
        if hasattr(model, 'aperiodic_params_'):
            aperiodic = np.asarray(getattr(model, 'aperiodic_params_', []), dtype=float)
            peaks = np.asarray(getattr(model, 'peak_params_', []), dtype=float)
            r_squared = float(getattr(model, 'r_squared_', np.nan))
            error = float(getattr(model, 'error_', np.nan))
            return aperiodic, peaks, r_squared, error

        # specparam backend (results object API).
        if hasattr(model, 'results'):
            results = model.results
            aperiodic = np.asarray(
                results.params.aperiodic.asdict().get('aperiodic_fit', []),
                dtype=float,
            )
            peaks = np.asarray(
                results.params.periodic.asdict().get('peak_fit', []),
                dtype=float,
            )

            r_squared = np.nan
            error = np.nan
            if hasattr(results, 'metrics'):
                categories = getattr(results.metrics, 'categories', [])
                if 'gof' in categories:
                    r_squared = float(results.metrics.get_metrics('gof'))
                if 'error' in categories:
                    error = float(results.metrics.get_metrics('error'))

            return aperiodic, peaks, r_squared, error

        raise AttributeError("Unsupported spectral model backend output format")

    def preprocess_and_save(self, cleaned_signals_folder: Path, ica_n_components: int = 15, ica_max_iter: int = 2000, eog_channels: list = ['Fp1', 'Fp2'], eog_threshold: float = 3.0, save_plots: bool = True):
        '''
        Preprocess EEG signals using ICA to remove blink artifacts, then save the cleaned signals to NetCDF files with enriched metadata.
        Optionally saves plot previews of the cleaned signals.
        '''
        if not self.eeg_files:
            raise RuntimeError("No EEG files loaded. Call find_eeg_files() first.")
        
        cleaned_signals_folder.mkdir(parents=True, exist_ok=True)

        for ncdf_path in self.eeg_files:
            label = ncdf_path.stem
            display(Markdown(f"## {label}"))
            print(f"Processing & Saving: {ncdf_path.name}")
            
            # Load original metadata for later enrichment when saving cleaned signals
            with xr.open_dataarray(ncdf_path) as da_original:
                original_attrs = da_original.attrs.copy()
                original_name = da_original.name

            raw_signal = load_eeg_ncdf_as_mne_raw(str(ncdf_path), montage="standard_1020", scale_to_volts=1e-6)

            # Get signal data and metadata
            signals = raw_signal.get_data() * 1e6
            channel_names = raw_signal.ch_names
            fs = raw_signal.info['sfreq']
            time_s = raw_signal.times
            event_duration_s = time_s[-1] if len(time_s) > 0 else 0.0

            # ICA preprocessing
            raw_for_ica = raw_signal.copy()
            raw_for_ica.filter(l_freq=1.0, h_freq=None, verbose='ERROR')

            ica = ICA(n_components=ica_n_components, random_state=42, max_iter=ica_max_iter)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
                old_log_level = mne.set_log_level('ERROR', return_old_level=True)
                try:
                    ica.fit(raw_for_ica)
                finally:
                    mne.set_log_level(old_log_level)

            available_eog = [ch for ch in eog_channels if ch in channel_names]
            if not available_eog:
                eog_indices = []
            else:
                eog_indices, _ = ica.find_bads_eog(raw_for_ica, ch_name=available_eog, threshold=eog_threshold)
            
            ica.exclude = eog_indices
            
            raw_cleaned = raw_signal.copy()
            ica.apply(raw_cleaned)
            signals_after_ica = raw_cleaned.get_data() * 1e6

            # Saving cleaned signals to NetCDF with enriched metadata
            output_dir = cleaned_signals_folder / label[:5]
            output_dir.mkdir(parents=True, exist_ok=True)
            export_path = output_dir / f"{label}_cleaned.nc"

            new_attrs = original_attrs.copy()
            old_desc = new_attrs.get('description', 'EEG signals')
            new_attrs['description'] = f"{old_desc} | Cleaned with ICA (blink artifacts removed)"
            new_attrs['processing_history'] = new_attrs.get('processing_history', '') + " -> ICA_cleaned"
            new_attrs['sampling_freq'] = fs
            new_attrs['event_duration_s'] = event_duration_s

            da = xr.DataArray(
                data=signals_after_ica.T,
                dims=['time', 'channel'],
                coords={'time': time_s, 'channel': channel_names},
                name=original_name,
                attrs=self._sanitize_attrs(new_attrs)
            )
            da.to_netcdf(export_path, engine='netcdf4')
            print(f"-> Saved NetCDF to: {export_path}")

            # Save plot preview of cleaned signals if requested
            if save_plots:
                plot_loaded_eeg_signals(
                    time_s=time_s, 
                    signals=signals_after_ica, 
                    channel_names=channel_names, 
                    event_duration_s=event_duration_s,
                    title=f"{label} - Cleaned with ICA"
                )
                plot_export_path = output_dir / f"{label}_cleaned_ica_plot.png"
                plt.savefig(plot_export_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved plot preview to: {plot_export_path}")

    def decompose_and_save(
        self,
        output_folder: Path,
        ica_n_components: int = 15,
        ica_max_iter: int = 2000,
        save_plot: bool = False,
    ) -> None:
        """
        Decompose EEG signals using ICA and save raw signals, ICA model, and sources to disk.
        """
        if not self.eeg_files:
            raise RuntimeError("No EEG files loaded. Call find_eeg_files() first.")

        output_folder.mkdir(parents=True, exist_ok=True)

        for ncdf_path in self.eeg_files:
            label = ncdf_path.stem
            display(Markdown(f"## {label}"))
            print(f"Processing: {ncdf_path.name}")

            with xr.open_dataarray(ncdf_path) as da_original:
                original_attrs = da_original.attrs.copy()

            raw = load_eeg_ncdf_as_mne_raw(str(ncdf_path), montage='standard_1020', scale_to_volts=1e-6)
            fs = raw.info['sfreq']
            times = raw.times
            ch_names = raw.ch_names
            event_duration_s = times[-1] if len(times) > 0 else 0.0
            prov = self._extract_provenance(original_attrs, label)

            out_dir = output_folder / prov['dyad_id']
            out_dir.mkdir(parents=True, exist_ok=True)

            raw_attrs = self._sanitize_attrs({
                **original_attrs,
                **prov,
                'file_stem': label,
                'sampling_freq': fs,
                'event_duration_s': event_duration_s,
                'processing_history': original_attrs.get('processing_history', '') + ' -> stored_raw_for_ICA',
            })
            raw_data = raw.get_data() * 1e6
            raw_da = xr.DataArray(
                data=raw_data.T,
                dims=['time', 'channel'],
                coords={'time': times, 'channel': ch_names},
                name='signals',
                attrs=raw_attrs,
            )
            raw_path = out_dir / f"{label}_raw.nc"
            raw_da.to_netcdf(raw_path, engine='netcdf4')
            print(f"  Saved raw:        {raw_path.name}")

            raw_for_ica = raw.copy()
            raw_for_ica.set_channel_types({'M1': 'misc', 'M2': 'misc'})

            raw_for_ica.filter(l_freq=1.0, h_freq=None, verbose='ERROR')

            ica = ICA(n_components=ica_n_components, random_state=42, max_iter=ica_max_iter)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
                old_log_level = mne.set_log_level('ERROR', return_old_level=True)
                try:
                    ica.fit(raw_for_ica, picks='eeg', verbose='ERROR')
                finally:
                    mne.set_log_level(old_log_level)

            n_components_actual = int(getattr(ica, 'n_components_', ica_n_components))
            ica_path = out_dir / f"{label}-ica.fif"
            ica.save(ica_path, overwrite=True)
            print(f"  Saved ICA model:  {ica_path.name}")

            sources_raw = ica.get_sources(raw)
            sources_data = sources_raw.get_data()
            comp_names = [f'ICA{i:03d}' for i in range(n_components_actual)]
            sources_attrs = self._sanitize_attrs({
                **original_attrs,
                **prov,
                'file_stem': label,
                'sampling_freq': fs,
                'event_duration_s': event_duration_s,
                'ica_n_components': n_components_actual,
                'ica_method': 'fastica',
                'ica_random_state': 42,
                'processing_history': original_attrs.get('processing_history', '') + ' -> ICA_decomposition',
            })
            sources_da = xr.DataArray(
                data=sources_data.T,
                dims=['time', 'component'],
                coords={'time': times, 'component': comp_names},
                name='sources',
                attrs=sources_attrs,
            )
            sources_path = out_dir / f"{label}_sources.nc"
            sources_da.to_netcdf(sources_path, engine='netcdf4')
            print(f"  Saved sources:    {sources_path.name}  ({n_components_actual} components)")

            if save_plot:
                figs = ica.plot_components(show=False, title=f"{label} — ICA topographies")
                if not isinstance(figs, list):
                    figs = [figs]
                for i, fig in enumerate(figs):
                    suffix = f"_{i}" if i > 0 else ""
                    plot_path = out_dir / f"{label}_ica_topographies{suffix}.png"
                    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                print(f"  Saved ICA plots")

    def compute_component_features(
        self,
        decomposition_folder: Path,
        psd_fmin: float = 1.0,
        psd_fmax: float = 45.0,
        psd_n_fft: int | None = None,
        fooof_max_n_peaks: int = 4,
        fooof_peak_threshold: float = 2.0,
        fooof_r_squared_threshold: float = 0.90,
        fooof_aperiodic_mode: str = 'fixed',
    ) -> None:
        """
        Load ICA decomposition outputs and compute PSD/FOOOF-based component features.
        """
        if FOOOF is None:
            raise ImportError("FOOOF or specparam is required for component feature computation")
        if not self.eeg_files:
            raise RuntimeError("No EEG files loaded. Call find_eeg_files() first.")

        for ncdf_path in self.eeg_files:
            label = ncdf_path.stem
            display(Markdown(f"## {label}"))
            print(f"Computing features: {ncdf_path.name}")

            with xr.open_dataarray(ncdf_path) as da:
                original_attrs = da.attrs.copy()

            prov = self._extract_provenance(original_attrs, label)
            out_dir = decomposition_folder / prov['dyad_id']
            ica_fif_path = out_dir / f"{label}-ica.fif"
            sources_path = out_dir / f"{label}_sources.nc"
            components_path = out_dir / f"{label}_components.nc"

            if not ica_fif_path.exists() or not sources_path.exists():
                print(f"  Warning: missing ICA outputs for {label}; skipping")
                continue

            ica = mne.preprocessing.read_ica(ica_fif_path)

            with xr.open_dataarray(sources_path) as da_src:
                sources_data = da_src.values.T
                comp_names = list(da_src.coords['component'].values)
                fs = float(da_src.attrs['sampling_freq'])

            raw = load_eeg_ncdf_as_mne_raw(str(ncdf_path), montage='standard_1020', scale_to_volts=1e-6)
            ch_names = raw.ch_names

            A = ica.get_components()
            topomap_raw = A.T
            topomap_norm = np.zeros_like(topomap_raw, dtype=float)
            for j in range(topomap_raw.shape[0]):
                col = topomap_raw[j].astype(float)
                if np.linalg.norm(col) == 0:
                    continue
                sign = np.sign(col[np.argmax(np.abs(col))])
                if sign == 0:
                    sign = 1.0
                col = col * sign
                norm = np.linalg.norm(col)
                if norm > 0:
                    col = col / norm
                topomap_norm[j] = col

            src_var = np.var(sources_data, axis=1)
            total_var = src_var.sum()
            if total_var > 0:
                explained_var_ratio = src_var / total_var
            else:
                explained_var_ratio = np.ones(len(comp_names)) / max(len(comp_names), 1)

            n_times = sources_data.shape[1]
            if psd_n_fft is None:
                n_fft = min(int(4.0 * fs), n_times)
                n_fft = int(2 ** np.floor(np.log2(n_fft)))
                n_fft = max(n_fft, 256)
            else:
                n_fft = min(psd_n_fft, n_times)
            n_overlap = n_fft // 2

            psd_power, freqs = psd_array_welch(
                sources_data,
                sfreq=fs,
                fmin=psd_fmin,
                fmax=psd_fmax,
                n_fft=n_fft,
                n_overlap=n_overlap,
                verbose=False,
            )

            n_components = len(comp_names)
            fooof_aperiodic = np.full((n_components, 2), np.nan)
            fooof_peaks_arr = np.full((n_components, fooof_max_n_peaks, 3), np.nan)
            fooof_r_squared = np.full(n_components, np.nan)
            fooof_error = np.full(n_components, np.nan)
            fooof_valid = np.zeros(n_components, dtype=np.int8)

            for j in range(n_components):
                if explained_var_ratio[j] < 1e-4:
                    print(f"  {comp_names[j]}: skipped (negligible variance)")
                    continue

                try:
                    fm = FOOOF(
                        peak_width_limits=[0.5, 12.0],
                        max_n_peaks=fooof_max_n_peaks,
                        min_peak_height=0.0,
                        peak_threshold=fooof_peak_threshold,
                        aperiodic_mode=fooof_aperiodic_mode,
                        verbose=False,
                    )
                    fm.fit(freqs, psd_power[j], [psd_fmin, psd_fmax])

                    aperiodic_params, peaks, r_squared, fit_error = self._extract_spectral_fit_outputs(fm)

                    if aperiodic_params.size >= 2:
                        fooof_aperiodic[j] = aperiodic_params[:2]

                    if peaks is not None and len(peaks) > 0:
                        n_found = min(len(peaks), fooof_max_n_peaks)
                        fooof_peaks_arr[j, :n_found, :] = peaks[:n_found]

                    fooof_r_squared[j] = r_squared
                    fooof_error[j] = fit_error
                    fooof_valid[j] = bool(np.isfinite(r_squared) and r_squared >= fooof_r_squared_threshold)
                    status = '✓' if fooof_valid[j] else '✗'
                    print(
                        f"  {comp_names[j]}: offset={fooof_aperiodic[j, 0]:.2f}  exp={fooof_aperiodic[j, 1]:.2f}  "
                        f"peaks={len(peaks) if peaks is not None else 0}  r²={fooof_r_squared[j]:.2f}  {status}"
                    )
                    cf_list = [f"{p[0]:.1f}Hz" for p in peaks] if peaks is not None and len(peaks) > 0 else []
                    print(f"    peaks={cf_list}")
                except Exception as exc:
                    print(f"  {comp_names[j]}: FOOOF failed — {exc}")

            ds = xr.Dataset(
                data_vars={
                    'topomap_raw': (['component', 'channel'], topomap_raw),
                    'topomap': (['component', 'channel'], topomap_norm),
                    'psd_power': (['component', 'frequency'], psd_power),
                    'fooof_aperiodic': (['component', 'ap_param'], fooof_aperiodic),
                    'fooof_peaks': (['component', 'peak_idx', 'peak_param'], fooof_peaks_arr),
                    'fooof_r_squared': (['component'], fooof_r_squared),
                    'fooof_error': (['component'], fooof_error),
                    'fooof_valid': (['component'], fooof_valid),
                    'explained_var_ratio': (['component'], explained_var_ratio),
                },
                coords={
                    'component': comp_names,
                    'channel': ch_names,
                    'frequency': freqs,
                    'peak_idx': list(range(fooof_max_n_peaks)),
                    'peak_param': ['CF', 'PW', 'BW'],
                    'ap_param': ['offset', 'exponent'],
                },
                attrs=self._sanitize_attrs({
                    **prov,                             # dyad_id, who, site
                    'file_stem': label,
                    'source_ncdf': ncdf_path.name,
                    'ica_method': 'fastica',
                    'ica_n_components': n_components,
                    'psd_method': 'welch',
                    'psd_n_fft': n_fft,
                    'psd_n_overlap': n_overlap,
                    'psd_fmin': psd_fmin,
                    'psd_fmax': psd_fmax,
                    'psd_units': 'arbitrary^2/Hz',
                    'fooof_freq_min': psd_fmin,
                    'fooof_freq_max': psd_fmax,
                    'fooof_max_n_peaks': fooof_max_n_peaks,
                    'fooof_peak_threshold': fooof_peak_threshold,
                    'fooof_r_squared_threshold': fooof_r_squared_threshold,
                    'fooof_aperiodic_mode': fooof_aperiodic_mode,
                    'topomap_normalization': 'L2',
                    'topomap_sign_convention': 'max_abs_positive',
                    'sampling_freq': fs,
                    'processing_history': original_attrs.get('processing_history', '') + ' -> ICA_features',
                }),
            )
            ds.to_netcdf(components_path, engine='netcdf4')
            valid_count = int(fooof_valid.sum())
            print(f"  Saved: {components_path.name}  (FOOOF valid: {valid_count}/{n_components})")

    def _nanmean_peaks(self, df_mask: pd.DataFrame, prefix: str) -> float:
        cols = [c for c in df_mask.columns if c.startswith(prefix)]
        vals = df_mask[cols].values.flatten()
        valid = vals[~np.isnan(vals)]
        return float(valid.mean()) if len(valid) > 0 else float('nan')

    def _template_vector(self, tmpl: dict, features: str) -> np.ndarray:
        parts = []
        if features in ('topomap', 'both'):
            parts.append(np.asarray(tmpl['mean_topomap'], dtype=float))
        if features in ('fooof', 'both'):
            fooof_vec = np.array([
                tmpl['mean_fooof_offset'],
                tmpl['mean_fooof_exponent'],
                tmpl['mean_cf'] if not np.isnan(tmpl['mean_cf']) else 0.0,
                tmpl['mean_pw'] if not np.isnan(tmpl['mean_pw']) else 0.0,
            ], dtype=float)
            parts.append(fooof_vec)
        if not parts:
            return np.array([], dtype=float)
        return np.concatenate(parts)

    def _signed_cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        sim_pos = 1.0 - cosine_distance(a, b)
        sim_neg = 1.0 - cosine_distance(a, -b)
        return float(max(sim_pos, sim_neg))

    def collect_features(
        self,
        decomposition_folder: Path,
        member_filter: str | None = None,
        valid_only: bool = True,
    ) -> pd.DataFrame:
        """Collect component-level feature rows for all ICA decomposition outputs."""
        if not self.eeg_files:
            raise RuntimeError("No EEG files loaded. Call find_eeg_files() first.")

        rows = []
        n_files = 0

        for ncdf_path in self.eeg_files:
            with xr.open_dataarray(ncdf_path) as da:
                attrs = da.attrs.copy()
            prov = self._extract_provenance(attrs, ncdf_path.stem)
            if member_filter is not None and prov.get('who') != member_filter:
                continue

            components_path = decomposition_folder / prov['dyad_id'] / f"{ncdf_path.stem}_components.nc"
            if not components_path.exists():
                print(f"  [SKIP] {components_path.name}")
                continue

            n_files += 1
            with xr.open_dataset(components_path) as ds:
                fooof_valid = ds['fooof_valid'].values.astype(bool)
                fooof_ap = ds['fooof_aperiodic'].values
                fooof_peaks = ds['fooof_peaks'].values
                topomap_norm = ds['topomap'].values
                exp_var = ds['explained_var_ratio'].values
                r_sq = ds['fooof_r_squared'].values
                comp_names = list(ds.coords['component'].values)
                max_peaks = len(ds.coords['peak_idx'])
                file_stem = ds.attrs.get('file_stem', components_path.stem)
                dyad_id = ds.attrs.get('dyad_id', '')
                who = ds.attrs.get('who', '')
                site = ds.attrs.get('site', '')

                for j, comp_name in enumerate(comp_names):
                    if valid_only and not fooof_valid[j]:
                        continue
                    row = {
                        'file_stem': file_stem,
                        'dyad_id': dyad_id,
                        'who': who,
                        'site': site,
                        'component': comp_name,
                        'fooof_valid': fooof_valid[j],
                        'fooof_r_squared': r_sq[j],
                        'explained_var_ratio': exp_var[j],
                        'fooof_offset': fooof_ap[j, 0],
                        'fooof_exponent': fooof_ap[j, 1],
                    }
                    for k in range(max_peaks):
                        CF, PW, BW = fooof_peaks[j, k]
                        row[f'fooof_cf_{k}'] = CF
                        row[f'fooof_pw_{k}'] = PW
                        row[f'fooof_bw_{k}'] = BW
                    for c, val in enumerate(topomap_norm[j]):
                        row[f'topo_{c}'] = val
                    rows.append(row)

        df = pd.DataFrame(rows)
        print(f"Collected {len(df)} components from {n_files} files.")
        return df

    def cluster_components(
        self,
        features_df: pd.DataFrame,
        n_clusters: int = 5,
        features: str = 'both',
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, dict]:
        """Cluster ICA components using either topographic, fooof, or combined features."""
        topo_cols = [c for c in features_df.columns if c.startswith('topo_')]
        fooof_cols = ['fooof_offset', 'fooof_exponent'] + [
            c for c in features_df.columns if c.startswith('fooof_cf_') or c.startswith('fooof_pw_') or c.startswith('fooof_bw_')
        ]

        if features == 'topomap':
            selected_cols = topo_cols
        elif features == 'fooof':
            selected_cols = fooof_cols
        else:
            selected_cols = topo_cols + fooof_cols

        X = features_df[selected_cols].fillna(0.0).values.astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = km.fit_predict(X_scaled)

        df_out = features_df.copy().reset_index(drop=True)
        df_out['cluster_label'] = labels

        templates = {}
        for label in sorted(set(labels)):
            mask = labels == label
            df_mask = df_out.loc[mask]
            templates[int(label)] = {
                'n_components': int(mask.sum()),
                'mean_topomap': df_mask.loc[:, topo_cols].values.mean(axis=0) if topo_cols else np.array([], dtype=float),
                'mean_fooof_offset': float(df_mask['fooof_offset'].mean()) if 'fooof_offset' in df_mask.columns else float('nan'),
                'mean_fooof_exponent': float(df_mask['fooof_exponent'].mean()) if 'fooof_exponent' in df_mask.columns else float('nan'),
                'mean_cf': self._nanmean_peaks(df_mask, 'fooof_cf_'),
                'mean_pw': self._nanmean_peaks(df_mask, 'fooof_pw_'),
                'members': df_out.loc[mask, ['file_stem', 'component', 'who']].to_dict('records'),
            }

        print(f"Clustering: {n_clusters} clusters, features='{features}', n={len(df_out)}")
        for lbl, tmpl in templates.items():
            print(f"  Cluster {lbl}: {tmpl['n_components']} components, "
                  f"mean_exp={tmpl['mean_fooof_exponent']:.2f}, "
                  f"mean_cf={tmpl['mean_cf']:.1f} Hz")

        return df_out, templates

    def find_cross_group_equivalents(
        self,
        templates_a: dict,
        templates_b: dict,
        features: str = 'topomap',
    ) -> pd.DataFrame:
        """Find cross-group cluster equivalents using cosine similarity with sign-invariant matching."""
        rows = []
        for lbl_a, tmpl_a in templates_a.items():
            vec_a = self._template_vector(tmpl_a, features)
            for lbl_b, tmpl_b in templates_b.items():
                vec_b = self._template_vector(tmpl_b, features)
                sim = self._signed_cosine_sim(vec_a, vec_b)
                rows.append({
                    'cluster_a': lbl_a,
                    'cluster_b': lbl_b,
                    'similarity': round(sim, 4),
                    'n_a': tmpl_a['n_components'],
                    'n_b': tmpl_b['n_components'],
                    'mean_cf_a': tmpl_a['mean_cf'],
                    'mean_cf_b': tmpl_b['mean_cf'],
                    'features_used': features,
                })

        result = pd.DataFrame(rows).sort_values('similarity', ascending=False)
        return result

    def export_assignment_template(
        self,
        features_df: pd.DataFrame,
        output_csv: Path,
        suggested_label_col: str = 'cluster_label',
    ) -> None:
        """Export a CSV template for manual rhythm-label assignment."""
        cols_out = ['file_stem', 'component', 'dyad_id', 'who', 'fooof_exponent', 'fooof_cf_0', 'fooof_pw_0', 'explained_var_ratio', 'fooof_r_squared']
        if suggested_label_col in features_df.columns:
            cols_out.append(suggested_label_col)

        template = features_df[[c for c in cols_out if c in features_df.columns]].copy()
        template['rhythm_label'] = ''
        template['confidence'] = ''
        template['notes'] = ''
        template.to_csv(output_csv, index=False, float_format='%.4f')
        print(f"Assignment template saved: {output_csv}  ({len(template)} rows)")

    def load_user_assignments(
        self,
        csv_path: Path,
    ) -> dict[str, dict[str, str]]:
        """Load user-provided rhythm assignments from the export template CSV."""
        df = pd.read_csv(csv_path, dtype=str).fillna('')

        assignments: dict[str, dict[str, str]] = {}
        skipped = 0

        for _, row in df.iterrows():
            label = row.get('rhythm_label', '').strip()
            if not label:
                skipped += 1
                continue
            stem = row['file_stem'].strip()
            comp = row['component'].strip()
            assignments.setdefault(stem, {})[comp] = label

        n_total = len(df)
        n_assigned = n_total - skipped
        print(f"Loaded assignments: {n_assigned}/{n_total} rows assigned "
              f"across {len(assignments)} files.")
        return assignments

    def reconstruct_with_assignments(
        self,
        assignments: dict[str, dict[str, str]],
        decomposition_folder: Path,
        output_folder: Path,
        rhythm_labels: list[str] | None = None,
    ) -> None:
        """Reconstruct EEG signals for user-assigned ICA components grouped by rhythm label."""
        files_by_stem = {p.stem: p for p in self.eeg_files}
        output_folder.mkdir(parents=True, exist_ok=True)

        for file_stem, comp_label_map in assignments.items():
            display(Markdown(f"## {file_stem}"))

            if file_stem not in files_by_stem:
                print(f"  [SKIP] {file_stem} — not in self.eeg_files")
                continue

            ncdf_path = files_by_stem[file_stem]

            with xr.open_dataarray(ncdf_path) as da:
                original_attrs = da.attrs.copy()
            prov = self._extract_provenance(original_attrs, file_stem)

            dyad_dir = decomposition_folder / prov['dyad_id']
            ica_fif_path = dyad_dir / f"{file_stem}-ica.fif"
            raw_nc_path = dyad_dir / f"{file_stem}_raw.nc"
            out_dir = output_folder / prov['dyad_id']
            out_dir.mkdir(parents=True, exist_ok=True)

            if not ica_fif_path.exists():
                print(f"  [SKIP] missing {ica_fif_path.name}")
                continue
            if not raw_nc_path.exists():
                print(f"  [SKIP] missing {raw_nc_path.name}")
                continue

            ica = mne.preprocessing.read_ica(ica_fif_path)
            ica.exclude = []

            raw = load_eeg_ncdf_as_mne_raw(
                str(raw_nc_path),
                montage='standard_1020',
                scale_to_volts=1e-6,
            )
            fs = raw.info['sfreq']
            times = raw.times
            ch_names = raw.ch_names
            event_dur = times[-1] if len(times) > 0 else 0.0

            label_to_indices: dict[str, list[int]] = {}
            for comp_name, r_label in comp_label_map.items():
                if rhythm_labels is not None and r_label not in rhythm_labels:
                    continue
                try:
                    idx = int(comp_name.replace('ICA', ''))
                except ValueError:
                    print(f"  [WARN] cannot parse index from '{comp_name}'; skipping")
                    continue
                label_to_indices.setdefault(r_label, []).append(idx)

            for r_label, comp_indices in label_to_indices.items():
                comp_indices_sorted = sorted(comp_indices)
                print(f"  Reconstructing '{r_label}': components {comp_indices_sorted}")

                raw_recon = raw.copy()
                ica.apply(raw_recon, include=comp_indices_sorted)
                signals_uv = raw_recon.get_data() * 1e6

                out_attrs = self._sanitize_attrs({
                    **original_attrs,
                    **prov,
                    'file_stem': file_stem,
                    'rhythm_label': r_label,
                    'ica_components': str(comp_indices_sorted),
                    'sampling_freq': fs,
                    'event_duration_s': event_dur,
                    'processing_history': original_attrs.get('processing_history', '')
                                          + f' -> reconstructed_{r_label}',
                })

                da_out = xr.DataArray(
                    data=signals_uv.T,
                    dims=['time', 'channel'],
                    coords={'time': times, 'channel': ch_names},
                    name='signals',
                    attrs=out_attrs,
                )

                out_path = out_dir / f"{file_stem}_reconstructed_{r_label}.nc"
                da_out.to_netcdf(out_path, engine='netcdf4')
                print(f"    Saved: {out_path.name}")

    def plot_component(
        self,
        components_path: Path,
        comp_id: str | int,
        *,
        topomap_cmap: str = 'RdBu_r',
        show: bool = True,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """Plot a single ICA component as a topomap + FOOOF PSD fit."""
        with xr.open_dataset(components_path) as ds:
            if isinstance(comp_id, int):
                comp_name = f'ICA{comp_id:03d}'
            else:
                comp_name = str(comp_id)

            available = [str(x) for x in ds.coords['component'].values]
            if comp_name not in available:
                raise ValueError(f"Component '{comp_name}' not found. Available: {available}")

            topo = ds['topomap_raw'].sel(component=comp_name).values
            psd = ds['psd_power'].sel(component=comp_name).values
            freqs = ds.coords['frequency'].values
            ch_names = list(ds.coords['channel'].values)

            fooof_ap = ds['fooof_aperiodic'].sel(component=comp_name).values
            fooof_peaks = ds['fooof_peaks'].sel(component=comp_name).values
            r_sq = float(ds['fooof_r_squared'].sel(component=comp_name).values)
            is_valid = bool(ds['fooof_valid'].sel(component=comp_name).values)
            exp_var = float(ds['explained_var_ratio'].sel(component=comp_name).values)
            fooof_mode = ds.attrs.get('fooof_aperiodic_mode', 'fixed')
            file_stem = ds.attrs.get('file_stem', components_path.stem)

            info = mne.create_info(ch_names=ch_names, sfreq=500., ch_types='eeg')
            montage = mne.channels.make_standard_montage('standard_1020')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                info.set_montage(montage, on_missing='ignore', verbose=False)

            log_freqs = np.log10(freqs)
            ap_fit = fooof_ap[0] - fooof_ap[1] * log_freqs
            peak_fit = np.zeros(len(freqs), dtype=float)
            valid_peaks = []
            for peak in fooof_peaks:
                if np.isnan(peak[0]):
                    break
                CF, PW, BW = peak
                sigma = BW / 2.0
                peak_fit += PW * np.exp(-((freqs - CF) ** 2) / (2 * sigma ** 2))
                valid_peaks.append((CF, PW, BW))

            full_fit = ap_fit + peak_fit

            fig = plt.figure(figsize=(13, 5))
            fig.suptitle(
                f"{file_stem}  —  {comp_name}  "
                f"({'✓ valid' if is_valid else '✗ invalid'}, "
                f"r²={r_sq:.2f}, exp_var={100 * exp_var:.1f}%)",
                fontsize=11,
                fontweight='bold',
                color='black' if is_valid else 'gray',
            )

            gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.35)
            ax_topo = fig.add_subplot(gs[0])
            ax_psd = fig.add_subplot(gs[1])

            im, _ = mne.viz.plot_topomap(
                topo,
                info,
                axes=ax_topo,
                cmap=topomap_cmap,
                show=False,
                contours=6,
                extrapolate='head',
                sphere='auto',
                vlim=(None, None),
            )
            plt.colorbar(im, ax=ax_topo, fraction=0.046, pad=0.04, label='µV (a.u.)')
            ax_topo.set_title(comp_name, fontsize=10)

            EEG_BANDS = [
                ('delta', 1, 4, '#EBF5FB'),
                ('theta', 4, 8, '#EAF7EF'),
                ('alpha', 8, 13, '#FDFDE7'),
                ('beta', 13, 30, '#FEF5E7'),
                ('gamma', 30, 45, '#FDEDEC'),
            ]

            ax_psd.plot(freqs, psd, color='#AAAAAA', lw=0.9, alpha=0.8, label='PSD', zorder=2)
            ax_psd.plot(freqs, 10**ap_fit, color='#2471A3', lw=1.2, ls='--', alpha=0.85, label='aperiodic fit', zorder=3)
            ax_psd.plot(freqs, 10**full_fit, color='#1A5276', lw=2.0, label='FOOOF model', zorder=4)
            ax_psd.set_yscale('log')

            for band_name, f_lo, f_hi, color in EEG_BANDS:
                f_hi_clipped = min(f_hi, freqs.max())
                f_lo_clipped = max(f_lo, freqs.min())
                if f_lo_clipped < f_hi_clipped:
                    ax_psd.axvspan(f_lo_clipped, f_hi_clipped, color=color, alpha=1.0, zorder=0)

            for CF, PW, BW in valid_peaks:
                ax_psd.axvline(CF, color='#E74C3C', lw=1.0, ls=':', alpha=0.8, zorder=5)
                ax_psd.axvspan(CF - BW / 2, CF + BW / 2, color='#FADBD8', alpha=0.45, zorder=1)
                idx = np.argmin(np.abs(freqs - CF))
                y_pos = 10 ** full_fit[idx]
                ax_psd.annotate(
                    f'{CF:.1f} Hz\nPW={PW:.2f}',
                    xy=(CF, y_pos),
                    xytext=(8, 8),
                    textcoords='offset points',
                    fontsize=7,
                    color='#C0392B',
                    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.8),
                )

            ax_psd.set_xlabel('Frequency (Hz)', fontsize=9)
            ax_psd.set_ylabel('Power (a.u.²/Hz)', fontsize=9)
            ax_psd.set_xlim(freqs[0], freqs[-1])
            ax_psd.legend(fontsize=8, loc='upper right')
            ax_psd.grid(True, lw=0.4, alpha=0.4)
            ax_psd.set_title(
                f"offset={fooof_ap[0]:.2f}  exp={fooof_ap[1]:.2f}  r²={r_sq:.3f}  [{fooof_mode}]",
                fontsize=8,
                color='black' if is_valid else 'gray',
            )

            for band_name, f_lo, f_hi, _color in EEG_BANDS:
                f_hi_clipped = min(f_hi, freqs.max())
                f_lo_clipped = max(f_lo, freqs.min())
                if f_lo_clipped < f_hi_clipped:
                    ax_psd.text(
                        (f_lo_clipped + f_hi_clipped) / 2,
                        0.98,
                        band_name,
                        transform=ax_psd.get_xaxis_transform(),
                        ha='center',
                        va='top',
                        fontsize=7,
                        color='#888888',
                    )

            if save_path is not None:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"  Saved: {save_path}")

            if show:
                plt.show()

            return fig

    def plot_component_grid(
        self,
        components_path: Path,
        *,
        n_cols: int = 5,
        topomap_cmap: str = 'RdBu_r',
        show: bool = True,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """Plot a grid of ICA component topomaps with FOOOF summary annotations."""
        import math

        with xr.open_dataset(components_path) as ds:
            comp_names = [str(x) for x in ds.coords['component'].values]
            ch_names = list(ds.coords['channel'].values)
            n_components = len(comp_names)
            file_stem = ds.attrs.get('file_stem', components_path.stem)

            topomap_raw = ds['topomap_raw'].values
            fooof_peaks = ds['fooof_peaks'].values
            fooof_ap = ds['fooof_aperiodic'].values
            fooof_valid = ds['fooof_valid'].values.astype(bool)
            exp_var = ds['explained_var_ratio'].values
            r_sq = ds['fooof_r_squared'].values

            info = mne.create_info(ch_names=ch_names, sfreq=500., ch_types='eeg')
            montage = mne.channels.make_standard_montage('standard_1020')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                info.set_montage(montage, on_missing='ignore', verbose=False)

            n_rows = math.ceil(n_components / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 4.0 * n_rows))
            axes = np.array(axes).flatten()

            fig.suptitle(
                f"{file_stem}  —  ICA components overview  "
                f"({int(fooof_valid.sum())}/{n_components} valid)",
                fontsize=12,
                fontweight='bold',
                y=1.01,
            )

            for j, comp_name in enumerate(comp_names):
                ax = axes[j]
                valid = bool(fooof_valid[j])
                topo = topomap_raw[j]

                try:
                    mne.viz.plot_topomap(
                        topo,
                        info,
                        axes=ax,
                        cmap=topomap_cmap,
                        show=False,
                        contours=4,
                        extrapolate='head',
                        sphere='auto',
                    )
                except Exception:
                    ax.text(0.5, 0.5, 'topomap\nerror', ha='center', va='center', transform=ax.transAxes, fontsize=8, color='red')

                for spine in ax.spines.values():
                    spine.set_edgecolor('#27AE60' if valid else '#BDC3C7')
                    spine.set_linewidth(2.0 if valid else 0.8)

                ax.set_title(
                    comp_name,
                    fontsize=8,
                    fontweight='bold' if valid else 'normal',
                    color='black' if valid else '#95A5A6',
                    pad=2,
                )

                if valid:
                    peaks_j = fooof_peaks[j]
                    cf_labels = []
                    for peak in peaks_j:
                        if np.isnan(peak[0]):
                            break
                        cf_labels.append(f'{peak[0]:.1f}')
                    cf_str = ', '.join(cf_labels) + ' Hz' if cf_labels else '—'
                    exp_str = f'exp={fooof_ap[j, 1]:.2f}'
                    var_str = f'{100 * exp_var[j]:.1f}%'
                    subtitle = f'{cf_str}\n{exp_str}  var={var_str}'
                    color = '#1A5276'
                else:
                    subtitle = f'[invalid]  r²={r_sq[j]:.2f}\nvar={100 * exp_var[j]:.1f}%'
                    color = '#95A5A6'

                ax.text(
                    0.5,
                    -0.08,
                    subtitle,
                    transform=ax.transAxes,
                    ha='center',
                    va='top',
                    fontsize=6.5,
                    color=color,
                    linespacing=1.4,
                )

            for k in range(n_components, len(axes)):
                axes[k].set_visible(False)

            fig.tight_layout()

            if save_path is not None:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"  Saved grid: {save_path}")

            if show:
                plt.show()

            return fig