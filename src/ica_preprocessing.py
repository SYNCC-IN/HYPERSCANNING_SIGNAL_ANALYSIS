from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

import mne
from mne.preprocessing import ICA

from src.export import load_eeg_ncdf_as_mne_raw, plot_loaded_eeg_signals


class ICAPreprocessor:
    def __init__(self, export_folder: Path, target_events: list):
        self.export_folder = export_folder
        self.target_events = target_events
        self.eeg_files: list = []

    def find_eeg_files(
        self,
        smoke_test: bool = True,
        smoke_dyads_n: int = 2,
        dyad_ids: str | list[str] | None = None,
    ):
        """Populate self.eeg_files from the export folder."""
        self.smoke_test = smoke_test
        self.smoke_dyads_n = smoke_dyads_n

        all_eeg_files = sorted([
            p for p in self.export_folder.rglob("*.nc")
            if "_EEG_" in p.name
            and any(p.stem.endswith(f"_{ev}") for ev in self.target_events)
        ])

        if not all_eeg_files:
            raise FileNotFoundError(
                f"No EEG NetCDF files found for events {self.target_events} under: {self.export_folder}"
            )

        files_by_dyad: dict[str, list[Path]] = {}
        for p in all_eeg_files:
            parts = p.stem.split('_')
            dyad_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else p.stem
            files_by_dyad.setdefault(dyad_id, []).append(p)

        all_dyads = sorted(files_by_dyad.keys())

        if dyad_ids is not None:
            requested = [dyad_ids] if isinstance(dyad_ids, str) else list(dyad_ids)
            missing = [d for d in requested if d not in files_by_dyad]
            if missing:
                raise ValueError(
                    f"Dyad(s) not found in export folder: {missing}\nAvailable: {all_dyads}"
                )
            dyads_to_process = requested
            mode = f"SELECTED DYADS: {', '.join(requested)}"
        elif smoke_test:
            dyads_to_process = all_dyads[:smoke_dyads_n]
            mode = f"SMOKE TEST (first {smoke_dyads_n} dyads)"
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
            print(f"  - {dyad}  ({len(files_by_dyad[dyad])} files)")

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

    def fit_and_save_ica(
        self,
        ica_folder: Path,
        n_components: int = 15,
        max_iter: int = 2000,
    ) -> None:
        """Fit ICA on EEG channels and save the model plus raw signal snapshots."""
        if not self.eeg_files:
            raise RuntimeError("No EEG files loaded. Call find_eeg_files() first.")

        ica_folder.mkdir(parents=True, exist_ok=True)

        for ncdf_path in self.eeg_files:
            label = ncdf_path.stem
            display(Markdown(f"## {label}"))
            print(f"[Stage 1] Fitting ICA: {ncdf_path.name}")

            try:
                with xr.open_dataarray(ncdf_path) as da:
                    original_attrs = da.attrs.copy()

                raw = load_eeg_ncdf_as_mne_raw(
                    str(ncdf_path), montage='standard_1020', scale_to_volts=1e-6
                )
                fs = raw.info['sfreq']
                times = raw.times
                prov = self._extract_provenance(original_attrs, label)

                out_dir = ica_folder / prov['dyad_id']
                out_dir.mkdir(parents=True, exist_ok=True)

                raw_path = out_dir / f"{label}_raw.nc"
                xr.DataArray(
                    data=raw.get_data() * 1e6,
                    dims=['channel', 'time'],
                    coords={'channel': raw.ch_names, 'time': times},
                    name='signals',
                    attrs=self._sanitize_attrs({
                        **original_attrs,
                        **prov,
                        'file_stem': label,
                        'sampling_freq': fs,
                        'processing_history': original_attrs.get('processing_history', '') + ' -> raw_stored_for_ICA',
                    }),
                ).to_netcdf(raw_path, engine='netcdf4')

                raw_hp = raw.copy()
                raw_hp.filter(l_freq=1.0, h_freq=None, verbose='ERROR')

                ica = ICA(
                    n_components=n_components,
                    method='infomax',
                    random_state=42,
                    max_iter=max_iter,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')
                    old_lvl = mne.set_log_level('ERROR', return_old_level=True)
                    try:
                        ica.fit(raw_hp, picks='eeg', verbose='ERROR')
                    finally:
                        mne.set_log_level(old_lvl)

                ica_path = out_dir / f"{label}-ica.fif"
                ica.save(ica_path, overwrite=True)
                n_actual = int(getattr(ica, 'n_components_', n_components))
                print(f"  Saved: {raw_path.name}  |  {ica_path.name}  ({n_actual} components)")
            except Exception as exc:
                print(f"  [ERROR] {label}: {exc!r} — skipping")

    def _plot_ica_classification(
        self,
        ica: "mne.preprocessing.ICA",
        raw_hp: "mne.io.BaseRaw",
        labels_list: list[str],
        full_proba: np.ndarray,          # shape (n_comp, 7) — full probability matrix
        iclabel_classes: list[str],
        exclude_labels: list[str],
        iclabel_threshold: float,
        label: str,
        save_path: Path,
        timecourse_seconds: float = 30.0,
    ) -> None:
        """
        Save a QC figure with one row per ICA component:
          left  (narrow) : scalp topography
          right (wide)   : source time course, title shows all 7 ICLabel probs

        Parameters
        ----------
        full_proba : ndarray, shape (n_comp, 7)
            Full per-class probability matrix from iclabel_label_components().
            Column order must match iclabel_classes.
        """
        from matplotlib.gridspec import GridSpec

        SHORT = {
            'brain':          'brain',
            'muscle artifact':'muscle',
            'eye blink':      'eye',
            'heart beat':     'heart',
            'line noise':     'line',
            'channel noise':  'chan',
            'other':          'other',
        }

        n_comp = len(labels_list)
        fs     = raw_hp.info['sfreq']
        A      = ica.get_components()                      # (n_eeg_ch, n_comp)

        src_data   = ica.get_sources(raw_hp).get_data()   # (n_comp, n_times)
        n_show     = min(int(timecourse_seconds * fs), src_data.shape[1])
        src_show   = src_data[:, :n_show]
        times_show = raw_hp.times[:n_show]

        info_eeg = mne.create_info(ch_names=ica.ch_names, sfreq=fs, ch_types='eeg')
        montage  = mne.channels.make_standard_montage('standard_1020')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            info_eeg.set_montage(montage, on_missing='ignore', verbose=False)

        fig = plt.figure(figsize=(50, n_comp * 2.6))
        gs  = GridSpec(n_comp, 2, width_ratios=[1, 10], hspace=0.60, wspace=0.12)
        fig.suptitle(
            f"{label}  —  ICA component classification  (ICLabel)",
            fontsize=11, fontweight='bold', y=1.002,
        )

        for j in range(n_comp):
            ax_topo = fig.add_subplot(gs[j, 0])
            ax_ts   = fig.add_subplot(gs[j, 1])

            predicted = labels_list[j]

            # auto_exclude: predicted class is unwanted AND its probability
            # meets the threshold (read directly from the full matrix)
            pred_idx          = iclabel_classes.index(predicted)
            pred_prob         = float(full_proba[j, pred_idx])
            auto_exclude      = predicted in exclude_labels and pred_prob >= iclabel_threshold

            color = (
                '#C0392B' if auto_exclude        # red   — excluded
                else '#1E8449' if predicted == 'brain'  # green — brain
                else '#D35400'                          # orange — uncertain
            )

            comp_name = f'ICA{j:03d}'

            # ── Topomap ──────────────────────────────────────────────────────
            try:
                mne.viz.plot_topomap(
                    A[:, j], info_eeg,
                    axes=ax_topo, cmap='RdBu_r', show=False,
                    contours=4, extrapolate='head', sphere='auto',
                )
            except Exception:
                ax_topo.text(
                    0.5, 0.5, 'topo\nerror',
                    ha='center', va='center',
                    transform=ax_topo.transAxes, fontsize=7, color='red',
                )

            ax_topo.set_title(comp_name, fontsize=8, pad=2,
                              color=color, fontweight='bold')
            for spine in ax_topo.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.0)

            # ── Time course ───────────────────────────────────────────────────
            ax_ts.plot(times_show, src_show[j], color='#2C3E50', lw=0.55, alpha=0.85)
            ax_ts.axhline(0, color='#AAAAAA', lw=0.4)
            ax_ts.set_xlim(times_show[0], times_show[-1])
            ax_ts.tick_params(labelsize=6)
            ax_ts.grid(True, lw=0.3, alpha=0.35)
            if j < n_comp - 1:
                ax_ts.set_xticklabels([])
            else:
                ax_ts.set_xlabel('Time (s)', fontsize=7)
            ax_ts.set_ylabel('a.u.', fontsize=6)

            for spine in ax_ts.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.4)

            # ── Title: predicted class + all 7 class probabilities ────────────
            exclude_marker = '✗ EXCLUDE' if auto_exclude else '✓ keep'
            prob_parts = [
                f"{SHORT.get(cls, cls)}: {float(full_proba[j, k]):.2f}"
                for k, cls in enumerate(iclabel_classes)
            ]
            prob_str = '   '.join(prob_parts)
            ax_ts.set_title(
                f"{predicted}  [{exclude_marker}]     {prob_str}",
                fontsize=6.5, pad=3, color=color, loc='left',
            )

        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved classification plot: {save_path.name}")

    # ══════════════════════════════════════════════════════════════════════════

    def classify_and_save_labels(
        self,
        ica_folder: Path,
        eog_channels: list[str] | None = None,
        eog_threshold: float = 3.0,
        iclabel_threshold: float = 0.70,
        exclude_labels: list[str] | None = None,
        timecourse_seconds: float = 30.0,
    ) -> None:
        """
        Classify ICA components with ICLabel, save labels CSV and QC figures.

        Calls ``iclabel_label_components`` directly (not the wrapper
        ``label_components``) to obtain the full (n_comp, 7) probability
        matrix — the wrapper returns only the per-component maximum, which
        is insufficient for displaying all class probabilities.

        Requires:  pip install mne-icalabel

        Saves per recording
        -------------------
        <label>_ica_labels.csv          one row per component (user-editable)
        <label>_ica_classification.png  topomap + time course QC figure
        """
        try:
            from mne_icalabel.iclabel import iclabel_label_components
        except ImportError as exc:
            raise ImportError("pip install mne-icalabel") from exc

        if eog_channels is None:
            eog_channels = ['Fp1', 'Fp2']
        if exclude_labels is None:
            exclude_labels = [
                'muscle artifact', 'eye blink', 'heart beat',
                'line noise', 'channel noise',
            ]

        # Canonical ICLabel class order (matches iclabel_label_components output)
        ICLABEL_CLASSES = [
            'brain', 'muscle artifact', 'eye blink', 'heart beat',
            'line noise', 'channel noise', 'other',
        ]
        COL_MAP = {
            'brain':           'prob_brain',
            'muscle artifact': 'prob_muscle',
            'eye blink':       'prob_eye',
            'heart beat':      'prob_heart',
            'line noise':      'prob_line_noise',
            'channel noise':   'prob_channel_noise',
            'other':           'prob_other',
        }

        if not self.eeg_files:
            raise RuntimeError("No EEG files loaded. Call find_eeg_files() first.")

        for ncdf_path in self.eeg_files:
            label = ncdf_path.stem
            display(Markdown(f"## {label}"))
            print(f"[Stage 2] Classifying components: {ncdf_path.name}")

            try:
                with xr.open_dataarray(ncdf_path) as da:
                    original_attrs = da.attrs.copy()
                prov    = self._extract_provenance(original_attrs, label)
                out_dir = ica_folder / prov['dyad_id']

                ica_path = out_dir / f"{label}-ica.fif"
                if not ica_path.exists():
                    print(f"  [SKIP] missing {ica_path.name} — run fit_and_save_ica first")
                    continue

                ica = mne.preprocessing.read_ica(ica_path)
                raw = load_eeg_ncdf_as_mne_raw(
                    str(ncdf_path), montage='standard_1020', scale_to_volts=1e-6
                )
                raw_hp = raw.copy()
                raw_hp.filter(l_freq=1.0, h_freq=None, verbose='ERROR')

                # ── Full (n_comp, 7) probability matrix ───────────────────────
                # iclabel_label_components returns the raw softmax output,
                # shape (n_components, n_classes), before argmax reduction.
                full_proba = iclabel_label_components(raw_hp, ica)   # (n_comp, 7)

                # Derive predicted labels from argmax (same logic as the wrapper)
                pred_indices = np.argmax(full_proba, axis=1)          # (n_comp,)
                labels_list  = [ICLABEL_CLASSES[i] for i in pred_indices]
                n_comp       = len(labels_list)
                comp_names   = [f'ICA{i:03d}' for i in range(n_comp)]

                # ── EOG correlation scores ────────────────────────────────────
                eog_scores    = np.full(n_comp, np.nan)
                available_eog = [ch for ch in eog_channels if ch in raw.ch_names]
                if available_eog:
                    try:
                        ica.find_bads_eog(
                            raw_hp, ch_name=available_eog, threshold=eog_threshold
                        )
                        scores_raw = getattr(ica, 'labels_scores_', {}).get('eog', None)
                        if scores_raw is not None:
                            arr = np.asarray(scores_raw).flatten()
                            eog_scores[:len(arr)] = arr
                    except Exception:
                        pass

                # ── QC figure (before DataFrame, uses full_proba) ─────────────
                plot_path = out_dir / f"{label}_ica_classification.png"
                self._plot_ica_classification(
                    ica=ica,
                    raw_hp=raw_hp,
                    labels_list=labels_list,
                    full_proba=full_proba,
                    iclabel_classes=ICLABEL_CLASSES,
                    exclude_labels=exclude_labels,
                    iclabel_threshold=iclabel_threshold,
                    label=label,
                    save_path=plot_path,
                    timecourse_seconds=timecourse_seconds,
                )

                # ── Build DataFrame ───────────────────────────────────────────
                rows = []
                for j, comp in enumerate(comp_names):
                    predicted = labels_list[j]
                    pred_prob = float(full_proba[j, pred_indices[j]])
                    auto_exclude = (
                        predicted in exclude_labels
                        and pred_prob >= iclabel_threshold
                    )
                    rows.append({
                        'component':    comp,
                        'iclabel':      predicted,
                        'iclabel_prob': pred_prob,         # max prob (predicted class)
                        **{COL_MAP[cls]: float(full_proba[j, k])
                           for k, cls in enumerate(ICLABEL_CLASSES)},
                        'eog_score':    float(eog_scores[j]),
                        'auto_exclude': auto_exclude,
                        'exclude':      auto_exclude,      # user edits this column
                        'notes':        '',
                    })

                df = pd.DataFrame(rows)
                labels_csv = out_dir / f"{label}_ica_labels.csv"
                df.to_csv(labels_csv, index=False, float_format='%.4f')

                n_auto = int(df['auto_exclude'].sum())
                print(f"  Saved: {labels_csv.name}  (auto_exclude: {n_auto}/{n_comp})")
                print(
                    df[['component', 'iclabel', 'iclabel_prob', 'auto_exclude']]
                    .to_string(index=False)
                )

            except Exception as exc:
                print(f"  [ERROR] {label}: {exc!r} — skipping")
    def apply_ica_and_save(
        self,
        ica_folder: Path,
        cleaned_folder: Path,
        save_plots: bool = True,
    ) -> None:
        """Apply reviewed ICA exclusions and save cleaned EEG signals."""
        if not self.eeg_files:
            raise RuntimeError("No EEG files loaded. Call find_eeg_files() first.")

        cleaned_folder.mkdir(parents=True, exist_ok=True)

        for ncdf_path in self.eeg_files:
            label = ncdf_path.stem
            display(Markdown(f"## {label}"))
            print(f"[Stage 3] Applying ICA: {ncdf_path.name}")

            try:
                with xr.open_dataarray(ncdf_path) as da:
                    original_attrs = da.attrs.copy()
                    original_name = da.name
                prov = self._extract_provenance(original_attrs, label)

                out_dir_ica = ica_folder / prov['dyad_id']
                raw_nc_path = out_dir_ica / f"{label}_raw.nc"
                ica_path = out_dir_ica / f"{label}-ica.fif"
                labels_path = out_dir_ica / f"{label}_ica_labels.csv"

                for p in (raw_nc_path, ica_path, labels_path):
                    if not p.exists():
                        raise FileNotFoundError(f"Missing: {p.name}")

                ica = mne.preprocessing.read_ica(ica_path)
                df_labels = pd.read_csv(labels_path)

                excluded_components = df_labels.index[df_labels['exclude'].astype(bool)].tolist()
                excluded_names = df_labels.loc[excluded_components, 'component'].tolist()
                excluded_reasons = df_labels.loc[excluded_components, 'iclabel'].tolist()

                print(f"  Excluding {len(excluded_components)} components: " + ', '.join(f"{n} ({r})" for n, r in zip(excluded_names, excluded_reasons)))

                raw = load_eeg_ncdf_as_mne_raw(str(raw_nc_path), montage='standard_1020', scale_to_volts=1e-6)
                ica.exclude = excluded_components
                raw_cleaned = raw.copy()
                ica.apply(raw_cleaned)

                signals_clean = raw_cleaned.get_data() * 1e6
                fs = raw.info['sfreq']
                times = raw.times

                out_dir_clean = cleaned_folder / prov['dyad_id']
                out_dir_clean.mkdir(parents=True, exist_ok=True)
                export_path = out_dir_clean / f"{label}_cleaned.nc"

                xr.DataArray(
                    data=signals_clean.T,
                    dims=['time', 'channel'],
                    coords={'time': times, 'channel': raw.ch_names},
                    name=original_name,
                    attrs=self._sanitize_attrs({
                        **original_attrs,
                        **prov,
                        'file_stem': label,
                        'sampling_freq': fs,
                        'ica_method': 'infomax_extended_picard',
                        'ica_excluded': str(excluded_names),
                        'ica_excluded_labels': str(excluded_reasons),
                        'processing_history': original_attrs.get('processing_history', '') + ' -> ICA_cleaned',
                    }),
                ).to_netcdf(export_path, engine='netcdf4')
                print(f"  Saved cleaned: {export_path.name}")

                if save_plots:
                    plot_loaded_eeg_signals(
                        time_s=times,
                        signals=signals_clean,
                        channel_names=raw.ch_names,
                        event_duration_s=times[-1] if len(times) > 0 else 0.0,
                        title=f"{label} — cleaned (excluded: {excluded_names})",
                    )
                    plot_path = out_dir_clean / f"{label}_cleaned_plot.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved plot:    {plot_path.name}")
            except Exception as exc:
                print(f"  [ERROR] {label}: {exc!r} — skipping")
