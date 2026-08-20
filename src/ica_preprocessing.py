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

try:
    from .mne_bridge import load_eeg_ncdf_as_mne_raw
    from .ncdf import load_xarray_from_netcdf
    from .plot_utils import plot_xarray_signals
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.mne_bridge import load_eeg_ncdf_as_mne_raw
    from src.ncdf import load_xarray_from_netcdf
    from src.plot_utils import plot_xarray_signals


def _format_component_probabilities(row: pd.Series, short_names: dict | None = None) -> str:
    """Format per-component probability columns into a compact title fragment."""
    short_names = short_names or {}
    prob_cols = [col for col in row.index if str(col).startswith("prob_")]
    parts = []
    for col in prob_cols:
        class_name = str(col).removeprefix("prob_")
        label = short_names.get(class_name, class_name)
        parts.append(f"{label}: {float(row[col]):.2f}")
    return "   ".join(parts)





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
                raw, original_attrs = load_eeg_ncdf_as_mne_raw(
                    str(ncdf_path), montage='standard_1020', scale_to_volts=1e-6
                )
                prov = self._extract_provenance(original_attrs, label)

                out_dir = ica_folder / 'ICA_COMPs' / prov['dyad_id']
                out_dir.mkdir(parents=True, exist_ok=True)

                raw_hp = raw.copy()
                raw_hp.filter(l_freq=1.0, h_freq=100, verbose='ERROR')
                rank = mne.compute_rank(raw_hp, rank='info')['eeg']  # data-based SVD catches interp + CAR
                ica = ICA(
                    n_components=min(n_components, rank),
                    method='infomax',
                    fit_params=dict(extended=True),
                    random_state=42,
                    max_iter=max_iter,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning, module='mne')
                    old_lvl = mne.set_log_level('ERROR', return_old_level=True)
                    try:
                        ica.fit(raw_hp, picks='eeg', reject=dict(eeg=500e-6), reject_by_annotation=True, verbose='ERROR')
                    finally:
                        mne.set_log_level(old_lvl)

                ica_path = out_dir / f"{label}-ica.fif"
                ica.save(ica_path, overwrite=True)
                n_actual = int(getattr(ica, 'n_components_', n_components))
                print(f"  Saved:   {ica_path.name}  ({n_actual} components)") # {raw_path.name}  |
            except Exception as exc:
                print(f"  [ERROR] {label}: {exc!r} — skipping")

    def _plot_ica_classification(
        self,
        ica: "mne.preprocessing.ICA",
        raw_hp: "mne.io.BaseRaw",
        df: pd.DataFrame,
        label: str,
        save_path: Path,
        masked_time_events: np.ndarray | None = None,
        times: np.ndarray | None = None,
    ) -> None:
        """
        Save a QC figure with one row per ICA component:
          left  (narrow) : scalp topography
          right (wide)   : source time course, title shows all 7 ICLabel probs

        Parameters
        ----------
        df : pd.DataFrame from labels classification, must contain columns:
            - 'iclabel': predicted ICLabel class for each component
            - proba_<class>: predicted probability for each of the 7 ICLabel classes
            - 'auto_exclude': boolean flag indicating whether the component should be automatically excluded
        masked_time_events : np.ndarray | None
            Boolean array indicating which time points correspond to task events.
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

        n_comp = len(df)
        fs     = raw_hp.info['sfreq']
        A      = ica.get_components()                      # (n_eeg_ch, n_comp)

        src_data   = ica.get_sources(raw_hp).get_data()   # (n_comp, n_times)
        #n_show     = min(int(timecourse_seconds * fs), src_data.shape[1])
        src_show   = src_data #[:, :n_show]
        src_show /= np.max(np.abs(src_show[:, masked_time_events]), axis=1)[:, None]  # normalize each component to [-1, 1] in the task event time window
        
        for j in range(n_comp):
            # n_pca_components=ica.n_components_ restricts reconstruction to exactly the
            # requested component; without it, MNE's default (full PCA rank) also adds
            # back the residual PCA dimensions beyond n_components_, which is identical
            # regardless of j and inflates the displayed amplitude for every component.
            only_j = ica.apply(raw_hp.copy(), include=[j], n_pca_components=ica.n_components_)   # zeruje pozostałe składowe
            projected_j = only_j.get_data() *1e6                # wkład komp. j w jednostkach danych
            max_projected_amp = float(np.max(np.abs(projected_j[:, masked_time_events])))  # max abs amplitude of the component in the EEG space, only during task events
            src_show[j] *= max_projected_amp  # scale time course to match projected amplitude
        
        times_show = times # [:n_show]

        info_eeg = mne.create_info(ch_names=ica.ch_names, sfreq=fs, ch_types='eeg')
        montage  = mne.channels.make_standard_montage('standard_1020')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            info_eeg.set_montage(montage, on_missing='ignore', verbose=False)

        fig = plt.figure(figsize=(50, n_comp * 2.6))
        gs  = GridSpec(
            n_comp, 2,
            width_ratios=[1, 10],
            hspace=0.60,
            wspace=0.01,    # ← near-zero gap between topo and timecourse columns
            top=0.98,       # ← pulls top of grid up, removes whitespace above row 0
            bottom=0.02,
            left=0.03,
            right=0.99,
        )
        fig.suptitle(
            f"{label}  —  ICA component classification  (ICLabel)",
            fontsize=11, fontweight='bold',
            y=0.995,        # ← just inside the top margin
        )

        for j in range(n_comp):
            ax_topo = fig.add_subplot(gs[j, 0])
            ax_ts   = fig.add_subplot(gs[j, 1])

            predicted = df.loc[j, 'iclabel']
            auto_exclude = bool(df.loc[j, 'auto_exclude'])
            color = (
                "#F61F07" if auto_exclude        # red   — excluded
                else '#1E8449' if predicted == 'brain' or predicted == 'other'  # green — brain
                else "#F19A61"                          # orange — uncertain
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
            ax_ts.set_ylabel('µV', fontsize=6)

            for spine in ax_ts.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.4)

            # ── Title: predicted class + all 7 class probabilities ────────────
            exclude_marker = '✗ EXCLUDE' if auto_exclude else '✓ keep'
            prob_str = _format_component_probabilities(df.iloc[j], short_names=SHORT)
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
        # eog_threshold: float = 3.0,
        iclabel_threshold: float = 0.70,
        neural_threshold: float = 0.50,
        # keep_threshold: float = 0.50,
        amplitude_threshold: float = 100, # in microvolts, max abs amplitude of the component in the EEG space
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
        brain_idx = ICLABEL_CLASSES.index('brain')
        other_idx = ICLABEL_CLASSES.index('other')
        artifacts_idx = [ICLABEL_CLASSES.index(lbl) for lbl in exclude_labels if lbl in ICLABEL_CLASSES]    
        if not self.eeg_files:
            raise RuntimeError("No EEG files loaded. Call find_eeg_files() first.")

        for ncdf_path in self.eeg_files:
            label = ncdf_path.stem
            display(Markdown(f"## {label}"))
            print(f"[Stage 2] Classifying components: {ncdf_path.name}")

            try:
                data_xr = load_xarray_from_netcdf(str(ncdf_path))
                original_attrs = data_xr.attrs.copy()
                original_times = np.asarray(data_xr.coords['time'].values, dtype=float)

                masked_time_events = np.any(
                    [(original_times > ev['start_rel_s']) & (original_times <= ev['start_rel_s'] + ev['duration_s'])
                     for ev in original_attrs.get('task_events_structure', [])],
                    axis=0
                )
                prov = self._extract_provenance(original_attrs, label)
                raw, _ = load_eeg_ncdf_as_mne_raw(str(ncdf_path), montage='standard_1020', scale_to_volts=1e-6, data_xr=data_xr)
                out_dir = ica_folder / 'ICA_COMPs' / prov['dyad_id']
                ica_path = out_dir / f"{label}-ica.fif"
                if not ica_path.exists():
                    print(f"  [SKIP] missing {ica_path.name} — run fit_and_save_ica first")
                    continue

                ica = mne.preprocessing.read_ica(ica_path)
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



                # ── Build DataFrame ───────────────────────────────────────────
                rows = []
                for j, comp in enumerate(comp_names):
                    predicted = labels_list[j]
                    pred_prob = float(full_proba[j, pred_indices[j]])
                    max_projected_amp = _get_max_projected_amplitude(ica, raw_hp, j, masked_time_events)
                    p_brain = float(full_proba[j, brain_idx])
                    p_other = float(full_proba[j, other_idx])
                    p_artifacts = float(np.sum(full_proba[j,artifacts_idx]))  # mass ICLabel puts on "leave it in"
                    if p_brain < p_artifacts:
                        p_neural = p_brain  
                        p_artifacts += p_other  # if it is rather artifacts than brain  add other to artifacts mass
                    else:
                        p_neural = p_brain + p_other        # if it is rather brain than artifacts  add other to brain mass

                    # c1 — confident artifact: argmax is an unwanted class AND ICLabel is sure
                    c1 = (predicted in exclude_labels) and (pred_prob >= iclabel_threshold)

                    # c2 — not confidently neural: too little combined brain+other mass to trust
                    c2 = (p_neural < neural_threshold)

                    # c3 — amplitude VETO, not an override: abnormally large back-projection
                    c3 = (max_projected_amp > amplitude_threshold)
                    
                    auto_exclude = c1 or c2 or c3 
                    rows.append({
                        'component':    comp,
                        'iclabel':      predicted,
                        'iclabel_prob': pred_prob,         # max prob (predicted class)
                        **{COL_MAP[cls]: float(full_proba[j, k])
                           for k, cls in enumerate(ICLABEL_CLASSES)},
                        'max_projected_amp': max_projected_amp,
                        'auto_exclude': auto_exclude,
                        'exclude':      auto_exclude,      # user edits this column
                        'notes':        '',
                    })

                df = pd.DataFrame(rows)
                labels_csv = ica_folder /'ICA_QC_FIGS_and_CSV' / f"{label}_ica_labels.csv"
                labels_csv.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(labels_csv, index=False, float_format='%.4f')

                n_auto = int(df['auto_exclude'].sum())
                print(f"  Saved: {labels_csv.name}  (auto_exclude: {n_auto}/{n_comp})")
                print(
                    df[['component', 'iclabel', 'iclabel_prob', 'auto_exclude']]
                    .to_string(index=False)
                )
                # ── QC figure (after DataFrame, it uses info from the DataFrame) ─────────────
                plot_path = ica_folder / 'ICA_QC_FIGS_and_CSV' / f"{label}_ica_classification.png"
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                self._plot_ica_classification(
                    ica=ica,
                    raw_hp=raw_hp,
                    df=df,
                    label=label,
                    save_path=plot_path,
                    masked_time_events=masked_time_events, #timecourse_seconds,
                    times=original_times
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
                data_xr = load_xarray_from_netcdf(str(ncdf_path))
                original_attrs = data_xr.attrs.copy()
                original_times = np.asarray(data_xr.coords['time'].values, dtype=float)
                raw, _ = load_eeg_ncdf_as_mne_raw(str(ncdf_path), montage='standard_1020', scale_to_volts=1e-6, data_xr=data_xr)
                prov = self._extract_provenance(original_attrs, label)

                out_dir_ica = ica_folder / 'ICA_COMPs' / prov['dyad_id']
                ica_path = out_dir_ica / f"{label}-ica.fif"
                labels_path = ica_folder / 'ICA_QC_FIGS_and_CSV' / f"{label}_ica_labels.csv"

                ica = mne.preprocessing.read_ica(ica_path)
                df_labels = pd.read_csv(labels_path)

                excluded_components = df_labels.index[df_labels['exclude'].astype(bool)].tolist()
                excluded_names = df_labels.loc[excluded_components, 'component'].tolist()
                excluded_reasons = df_labels.loc[excluded_components, 'iclabel'].tolist()

                print(f"  Excluding {len(excluded_components)} components: " + ', '.join(f"{n} ({r})" for n, r in zip(excluded_names, excluded_reasons)))

                
                ica.exclude = excluded_components
                raw_cleaned = raw.copy()
                ica.apply(raw_cleaned)

                signals_clean = raw_cleaned.get_data() * 1e6
                times = original_times  # Use original times from the NetCDF file

                # Save cleaned signals to NetCDF
                out_dir_clean = cleaned_folder / prov['dyad_id']
                out_dir_clean.mkdir(parents=True, exist_ok=True)
                export_path = out_dir_clean / f"{label}_cleaned.nc"

                xr_sig = xr.DataArray(
                    data=signals_clean.T,
                    dims=['time', 'channel'],
                    coords={'time': times, 'channel': raw.ch_names},
                    name='signals',
                    attrs=self._sanitize_attrs({
                        **original_attrs,
                        **prov,
                        'file_stem': label,
                        'ica_method': 'infomax_extended_picard',
                        'ica_excluded': str(excluded_names),
                        'ica_excluded_labels': str(excluded_reasons),
                        'processing_history': original_attrs.get('processing_history', '') + ' -> ICA_cleaned',
                    }),
                )
                xr_sig.to_netcdf(export_path, engine='netcdf4')
                print(f"  Saved cleaned: {export_path.name}")

                if save_plots:
                    plot_dir = cleaned_folder / 'FIGS'
                    plot_dir.mkdir(parents=True, exist_ok=True)
                    plot_path = plot_dir / f"{label}_cleaned_plot.png"

                    fig, _ = plot_xarray_signals(
                        xr_sig,
                        event_duration=float(xr_sig.attrs.get("task_duration", xr_sig.attrs.get("event_duration", np.nan))),
                        time_margin_s=float(xr_sig.attrs.get("time_margin_s", 0.0)),
                        title=f"{label} — cleaned (excluded: {excluded_names})",
                    )
                    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  Saved plot:    {plot_path.name}")
            except Exception as exc:
                print(f"  [ERROR] {label}: {exc!r} — skipping")

def _get_max_projected_amplitude(ica, raw_hp, j, mask_time_events):
    '''
    Compute the maximum absolute amplitude of the j-th ICA component's projection
    only during the time events specified by mask_time_events.
    '''
    only_j = ica.apply(raw_hp.copy(), include=[j], n_pca_components=ica.n_components_)   # zeruje pozostałe składowe
    projected_j = only_j.get_data() * 1e6                # wkład komp. j w jednostkach danych
    max_projected_amp = float(np.max(np.abs(projected_j[:, mask_time_events])))  # max abs amplitude of the component in the EEG space, only during task events
    return max_projected_amp