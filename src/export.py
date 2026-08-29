import json
import os
import warnings
from dataclasses import asdict, is_dataclass
from typing import Optional

import mne
import numpy as np
import xarray as xr

from .netcdf_io import sanitize_netcdf_attrs_inplace
from . import dataloader

# ── Old-style 10-20 names that MNE's standard_1020 montage does not recognise ──
_OLD_TO_MNE = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
_MASTOIDS   = frozenset({'M1', 'M2'})


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def export_chunk_to_xarray(
    multimodal_data,
    selected_events,
    selected_channels,
    selected_modality,
    member,
    time_margin,
    chunk_name,
    EEG_montage=None,
    EEG_bad_channels=None,
    verbose=True,
    mne_plot_flag=False,
    logger=None,
) -> xr.DataArray:
    """Export a chunk spanning multiple events from a MultimodalData instance to xarray.

    The exported time axis is reset to 0 at the start of the first event in
    ``selected_events``.  See module docstring for full attribute description.
    """
    # ── 1. Validation + time window ───────────────────────────────────────────
    ordered_events, chunk_start, chunk_end, selected_time = _compute_time_window(
        multimodal_data, selected_events, time_margin, verbose, logger, chunk_name
    )

    # ── 2. Extract signals and strip modality prefix from channel names ────────
    time, channels, data = _extract_and_strip(
        multimodal_data, selected_modality, member,
        selected_channels, selected_time, chunk_start, chunk_name, ordered_events
    )

    # ── 3. EEG-specific MNE processing (interpolation + rereferencing) ─────────
    metadata = _build_export_metadata(multimodal_data, selected_modality)
    if selected_modality == 'EEG' and EEG_montage is not None:
        data, channels, metadata = _apply_eeg_montage(
            data, channels, multimodal_data.fs,
            EEG_montage, EEG_bad_channels, member, metadata,
            mne_plot_flag, verbose, logger, chunk_name
        )

    # ── 4. Build and annotate DataArray ───────────────────────────────────────
    return _build_dataarray(
        time, channels, data,
        multimodal_data, ordered_events, chunk_start, chunk_end,
        chunk_name, time_margin, member, selected_modality, metadata
    )


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_time_window(
    multimodal_data, selected_events, time_margin, verbose, logger, chunk_name
) -> tuple[list, float, float, list]:
    """Validate events and compute the time window for the chunk.

    Returns
    -------
    ordered_events : list[str]
    chunk_start    : float   — time of first event start (s)
    chunk_end      : float   — time of last event end (s)
    selected_time  : [float, float]  — actual window including margins
    """
    if not selected_events:
        raise ValueError("selected_events must be a non-empty list.")

    missing = [e for e in selected_events if e not in multimodal_data.events]
    if missing:
        raise ValueError(
            f"Events not found: {missing}. "
            f"Available: {list(multimodal_data.events.keys())}"
        )

    ordered_events = _sort_events_by_start(multimodal_data, selected_events)
    first_event    = multimodal_data.events[ordered_events[0]]
    last_event     = multimodal_data.events[ordered_events[-1]]
    chunk_start    = float(first_event["start"])
    chunk_end      = float(last_event["start"] + last_event["duration"])

    t_min = multimodal_data.data["time"].min()
    t_max = multimodal_data.data["time"].max()
    selected_time  = [
        max(t_min, chunk_start - time_margin),
        min(t_max, chunk_end   + time_margin),
    ]

    if verbose:
        _log(logger,
             f"Chunk '{chunk_name}' spans {ordered_events} "
             f"from {chunk_start:.2f}s to {chunk_end:.2f}s")
        _log(logger,
             f"Time window ±{time_margin}s: "
             f"{selected_time[0]:.2f}s → {selected_time[1]:.2f}s")

    return ordered_events, chunk_start, chunk_end, selected_time


def _extract_and_strip(
    multimodal_data, selected_modality, member,
    selected_channels, selected_time, chunk_start, chunk_name, ordered_events
) -> tuple[np.ndarray, list, np.ndarray]:
    """Extract signals from MultimodalData and strip the modality/member prefix.

    Returns
    -------
    time     : (n_times,)          seconds, reset so 0 = chunk_start
    channels : list[str]           clean channel names (no prefix)
    data     : (n_times, n_ch)     signal values
    """
    if selected_modality in ('EEG', 'ET') and not selected_channels:
        raise ValueError(
            f"selected_channels required for modality='{selected_modality}' "
            f"in chunk '{chunk_name}'."
        )

    signals = multimodal_data.get_signals(
        mode=selected_modality,
        member=member,
        selected_channels=selected_channels,
        selected_times=selected_time,
    )
    if signals is None:
        raise ValueError(
            f"No signals for modality='{selected_modality}', member='{member}', "
            f"channels={selected_channels}, chunk='{chunk_name}', events={ordered_events}."
        )

    time, channels, data = signals
    time = time - chunk_start   # reset so 0 = event start

    # Strip modality/member prefix from channel names
    prefix_map = {
        'EEG':   f'EEG_{member}_',
        'ET':    f'ET_{member}_',
        'IBI':   f'IBI_{member}',
        'RMSSD': f'RMSSD_{member}',
        'ECG':   f'ECG_{member}',
        'diode': None,
    }
    prefix = prefix_map.get(selected_modality)
    if prefix:
        channels = [ch.replace(prefix, '') for ch in channels]
    elif selected_modality == 'diode':
        channels = ['diode']

    channels = [str(ch) for ch in channels]
    return time, channels, data


def _apply_eeg_montage(
    data, channels, fs,
    EEG_montage, EEG_bad_channels, member, metadata,
    mne_plot_flag, verbose, logger, chunk_name
) -> tuple[np.ndarray, list, dict]:
    """Apply MNE montage, interpolate bad channels and optionally CAR.

    Parameters
    ----------
    data     : (n_times, n_ch) μV
    channels : list[str]  10-20 labels after prefix stripping

    Returns
    -------
    data     : (n_times, n_ch) μV   corrected
    channels : list[str]            updated (M1/M2 dropped after CAR)
    metadata : dict                 enriched with interpolation/reference notes
    """
    bad_channels = _find_bad_channels(EEG_bad_channels, channels, member)
    channels_mne = [_OLD_TO_MNE.get(ch, ch) for ch in channels]
    bad_mne      = [_OLD_TO_MNE.get(ch, ch) for ch in bad_channels]

    if verbose:
        rename = {ch: channels_mne[i]
                  for i, ch in enumerate(channels) if ch != channels_mne[i]}
        if rename:
            _log(logger, f"  '{member}' {chunk_name}: MNE rename {rename}")
        if bad_channels:
            _log(logger, f"  '{member}' {chunk_name}: bad channels {bad_channels} → interpolate")

    # ── Build RawArray (data is μV, MNE expects V) ────────────────────────────
    ch_types = ['misc' if ch in _MASTOIDS else 'eeg' for ch in channels_mne]
    info     = mne.create_info(ch_names=channels_mne, sfreq=fs, ch_types=ch_types)
    raw      = mne.io.RawArray(data.T * 1e-6, info, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rename = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
        raw.rename_channels({k: v for k, v in rename.items() if k in raw.ch_names})
        raw.set_montage('standard_1020', on_missing='ignore', verbose=False)

    # ── Step 1: interpolate bad channels ─────────────────────────────────────
    if bad_channels:
        raw.info['bads'] = bad_mne
        if mne_plot_flag:
            _plot_raw(raw.copy(), f"{member} BEFORE interpolation")
        raw.interpolate_bads(reset_bads=True)
        if mne_plot_flag:
            _plot_raw(raw, f"{member} AFTER interpolation", block=True)
        metadata['interpolation'] = f'Interpolated: {bad_mne}'

    # ── Step 2: CAR (optional) ────────────────────────────────────────────────
    if EEG_montage == 'CAR':
        if mne_plot_flag:
            _plot_raw(raw.copy(), f"{member} {chunk_name} BEFORE CAR")
        raw.set_eeg_reference('average', projection=False, verbose=False)
        mastoids = [ch for ch in raw.ch_names if ch in _MASTOIDS]
        if mastoids:
            raw.drop_channels(mastoids)
        if mne_plot_flag:
            _plot_raw(raw, f"{member} {chunk_name} AFTER CAR", block=True)
        metadata['references'] = (
            "CAR applied; M1/M2 excluded from average and dropped"
        )
        if verbose:
            _log(logger, f"  '{member}' {chunk_name}: CAR applied, M1/M2 dropped")
    else:
        metadata['references'] = "Original reference retained"

    # ── Retrieve corrected data (V → μV, (n_times, n_ch)) ────────────────────
    data     = raw.get_data().T * 1e6
    channels = list(raw.ch_names)
    return data, channels, metadata


def _build_dataarray(
    time, channels, data,
    multimodal_data, ordered_events, chunk_start, chunk_end,
    chunk_name, time_margin, member, modality, metadata
) -> xr.DataArray:
    """Assemble the final annotated xarray DataArray."""
    events_structure = _build_events_structure(
        multimodal_data, ordered_events, chunk_start
    )
    da = xr.DataArray(
        data,
        coords=[time, channels],
        dims=['time', 'channel'],
        name='signals',
    )
    units = "unknown"
    if modality == 'EEG' or modality == 'ECG':
        units = "μV"
    elif modality == 'ET':
        units = "px"
    elif modality == 'IBI' or modality == 'RMSSD':  
        units = "ms"

  
    da.attrs.update({
        'dyad_id':               multimodal_data.id,
        'who':                   member,
         "modality":             modality,
         "units":                units,
        'sampling_freq':         float(multimodal_data.fs),
        'task_name':             chunk_name,
        'task_start':            0.0,
        'task_duration':         float(chunk_end - chunk_start),
        'time_margin_s':         float(time_margin),
        'channel_names_csv':     ','.join(channels),
        'channel_names_json':    json.dumps(channels, ensure_ascii=True),
        'metadata_json':         json.dumps(metadata, ensure_ascii=False, default=str),
        'task_event_names_csv':  ','.join(ordered_events),
        'task_event_names_json': json.dumps(ordered_events, ensure_ascii=True),
        'task_events_structure': events_structure,
    })
    sanitize_netcdf_attrs_inplace(da.attrs)
    return da


# ══════════════════════════════════════════════════════════════════════════════
# Micro-helpers (single responsibility, no branching)
# ══════════════════════════════════════════════════════════════════════════════

def _find_bad_channels(
    EEG_bad_channels: list | None,
    channels: list[str],
    member: str,
) -> list[str]:
    """Return base channel names (without _ch/_cg suffix) that are bad for this member."""
    if not EEG_bad_channels:
        return []
    suffix = f'_{member}'
    return [
        ch[:-len(suffix)]
        for ch in EEG_bad_channels
        if ch.endswith(suffix) and ch[:-len(suffix)] in channels
    ]


def _plot_raw(raw, title: str, block: bool = False) -> None:
    raw.plot(
        block=block, show=True, scalings='auto', verbose=False,
    ).suptitle(title, fontsize=10, fontweight='bold')


def _log(logger, msg: str) -> None:
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


# ── Unchanged helpers (kept here, near the functions that use them) ────────────



def _build_events_structure(multimodal_data, ordered_event_names, chunk_start):
    events = []
    for name in ordered_event_names:
        ev = multimodal_data.events.get(name, {})
        start_abs = float(ev.get("start", 0.0))
        duration  = float(ev.get("duration", 0.0))
        events.append({
            "name":        name,
            "start_s":     start_abs,
            "start_rel_s": start_abs - chunk_start,
            "duration_s":  duration,
        })
    return events


def _build_export_metadata(multimodal_data, selected_modality):
    def _dataclass_or_dict(value):
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        return value

    # Filmy posortowane chronologicznie — zachowane z oryginalnej wersji
    movie_events = ["Peppa", "Incredibles", "Brave"]
    event_order = sorted(
        (name for name in movie_events if name in multimodal_data.events
         and "start" in multimodal_data.events[name]),
        key=lambda name: multimodal_data.events[name]["start"],
    )

    metadata = {
        "notes":       getattr(multimodal_data, "notes", ""),
        "child_info":  _dataclass_or_dict(getattr(multimodal_data, "child_info", {})),
        "event_order": event_order,
    }
    if selected_modality == 'EEG':
        metadata["eeg"] = {
            "filtration": _dataclass_or_dict(getattr(multimodal_data, "eeg_filtration", None)),
            "references": getattr(multimodal_data, "references", ""),
        }
    return metadata

def _sort_events_by_start(multimodal_data, event_names):
    return sorted(
        event_names,
        key=lambda name: multimodal_data.events[name]["start"],
    )


def write_dyad_to_uniwaw_imported(dyad_id_list=None, load_eeg=True, load_et=True, load_meta=True, lowcut=1.0, highcut=40.0, eeg_filter_type='fir',decimate_factor=8, plot_flag=False, time_margin=10, input_data_path="../data", export_path="../data/UNIWAW_imported", verbose=False, logger: Optional[object] = None):
    '''Export signals from a specified dyad to xarray DataArrays and save them as NetCDF files in a structured directory format compatible with UNIWAW_imported.
    Args:
        dyad_id_list: List of the IDs of the dyads to export (e.g., ['W_003']). If None, a ValueError is raised'
        load_eeg: Whether to load EEG data for the dyad.
        load_et: Whether to load eye-tracking data for the dyad.
        load_meta: Whether to load metadata for the dyad.
        lowcut: The low cut frequency for EEG filtering.
        highcut: The high cut frequency for EEG filtering.
        eeg_filter_type: The type of EEG filter to use ('fir' or 'iir').
        decimate_factor: The factor by which to decimate the EEG data.
        plot_flag: Whether to plot the data during processing.
        time_margin: The time margin to include around events.
        input_data_path: The path to the input data directory.
        export_path: The path to the export directory.
        verbose: If True, emit progress messages during export.
        logger: Optional logger-like object with .info(str). If provided and verbose=True,
            messages are sent to logger.info instead of print.
        '''
    def _log(message: str) -> None:
        if not verbose:
            return
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    if dyad_id_list is None:
        raise ValueError("dyad_id_list must be provided")
    if isinstance(dyad_id_list, str):
        dyad_id_list = [dyad_id_list]
    if not isinstance(dyad_id_list, list) or len(dyad_id_list) == 0:
        raise ValueError("dyad_id_list must be a non-empty list of dyad IDs to export (e.g., ['W_003']).")
    members = {'ch': 'child', 'cg': 'caregiver'}
    selected_channels = {
        'EEG': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'M1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'M2', 'T5', 'P3', 'Pz',
                'P4', 'T6', 'O1', 'O2'],
        'ET': ['x', 'y', 'pupil', 'blinks'],
        'ECG': ['ECG'],
        'IBI': ['IBI'],
        'RMSSD': ['RMSSD']}
    for dyad_id in dyad_id_list:
        _log(f"Loading dyad '{dyad_id}' from '{input_data_path}'")
        multimodal_data = dataloader.create_multimodal_data(data_base_path = input_data_path,
                                                    dyad_id = dyad_id,
                                                    load_eeg=load_eeg,
                                                    load_et=load_et,
                                                    load_meta=load_meta,
                                                    lowcut=lowcut,
                                                    highcut=highcut,
                                                    eeg_filter_type=eeg_filter_type,
                                                    interpolate_et_during_blinks_threshold=0.3,
                                                    median_filter_size=64,
                                                    low_pass_et_order=351,
                                                    et_pos_cutoff=128,
                                                    et_pupil_cutoff=4,
                                                    pupil_model_confidence=0.9,
                                                    decimate_factor=decimate_factor,
                                                    plot_flag=plot_flag)
        
        _log(f"Loaded dyad '{multimodal_data.id}'. Export root: '{export_path}'")
        _event_order = _build_export_metadata(multimodal_data, 'EEG').get("event_order", [])
        _log(f"Event order: {_event_order}")
        for modality in multimodal_data.modalities:
            path_modality = os.path.join(export_path, modality,str(multimodal_data.id))
            if not os.path.exists(path_modality):
                os.makedirs(path_modality)
            for who, member in members.items():
                path_member = os.path.join(path_modality, member)
                if not os.path.exists(path_member):
                    os.makedirs(path_member)
                for event in multimodal_data.events.keys():
                    _log(f"Exporting modality='{modality}', member='{who}', event='{event}'")
                    data_xr = export_chunk_to_xarray(
                                multimodal_data=multimodal_data,
                                selected_events=[event],          # ← lista z jednym eventem
                                selected_channels=selected_channels.get(modality),
                                selected_modality=modality,
                                member=who,
                                time_margin=time_margin,
                                chunk_name=event,                 # ← nazwa chunka = nazwa eventu
                                verbose=False,
                                logger=logger,
                            )
                    file_path = os.path.join(path_member, f'{multimodal_data.id}_{modality}_{who}_{event}.nc')
                    data_xr.to_netcdf(file_path, engine='netcdf4', format='NETCDF4_CLASSIC')
                    _log(f"Saved: {file_path}")

        _log(f"Finished export for dyad '{multimodal_data.id}'")


def export_passive_and_talk_data(
    dyad_id_list=None,
    load_eeg=True,
    load_et=True,
    load_meta=True,
    lowcut=1.0,
    highcut=40.0,
    eeg_filter_type='fir',
    decimate_factor=8,
    plot_flag=False,
    time_margin=20,
    input_data_path="../data",
    export_path="../data/UNIWAW_imported",
    mounts_eeg_multimodal=False, # wether to mount EEG channels to M1/M2 or not (if False, EEG channels are exported as recorded)
    export_mounted = 'CAR', # wether to export EEG data as CAR (common average reference) or as recorded (None)
    EEG_bad_channels=None, # list of bad EEG channels to interpolate (e.g., ['T3_ch', 'Fp1_cg'])
    verbose=False,
    mne_plot_flag=False,
    logger: Optional[object] = None,
):
    '''Export two chunk types per modality/member: passive movies and talk.

    Instead of exporting every event separately, this function exports:
      1) one continuous chunk covering movie events, i.e., it corresponds to the passive task: Peppa, Incredibles, Brave
      2) one continuous chunk covering all events whose names include "talk"

    The exported file structure is compatible with UNIWAW_imported.
    '''

    def _log(message: str) -> None:
        if not verbose:
            return
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    if dyad_id_list is None:
        raise ValueError("dyad_id_list must be provided")
    if isinstance(dyad_id_list, str):
        dyad_id_list = [dyad_id_list]
    if not isinstance(dyad_id_list, list) or len(dyad_id_list) == 0:
        raise ValueError("dyad_id_list must be a non-empty list of dyad IDs to export (e.g., ['W_003']).")

    members = {'ch': 'child', 'cg': 'caregiver'}
    selected_channels = {
        'EEG': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'M1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'M2', 'T5', 'P3', 'Pz',
                'P4', 'T6', 'O1', 'O2'],
        'ET': ['x', 'y', 'pupil', 'blinks'],
        'ECG': ['ECG'],
        'IBI': ['IBI'],
        'RMSSD': ['RMSSD'],
        'diode': ['diode'],
    }

    movie_events_target = ["Peppa", "Incredibles", "Brave"]

    for dyad_id in dyad_id_list:
        _log(f"Loading dyad '{dyad_id}' from '{input_data_path}'")
        multimodal_data = dataloader.create_multimodal_data(
            data_base_path=input_data_path,
            dyad_id=dyad_id,
            load_eeg=load_eeg,
            load_et=load_et,
            load_meta=load_meta,
            lowcut=lowcut,
            highcut=highcut,
            eeg_filter_type=eeg_filter_type,
            interpolate_et_during_blinks_threshold=0.3,
            median_filter_size=64,
            low_pass_et_order=351,
            et_pos_cutoff=128,
            et_pupil_cutoff=4,
            pupil_model_confidence=0.9,
            decimate_factor=decimate_factor,
            mounts_eeg=mounts_eeg_multimodal,
            plot_flag=plot_flag,
        )

        _log(f"Loaded dyad '{multimodal_data.id}'. Export root: '{export_path}'")

        available_event_names = list(multimodal_data.events.keys())
        movie_events = [name for name in movie_events_target if name in multimodal_data.events]
        talk_events = [name for name in available_event_names if "talk" in name.lower()]

        chunks = [
            ("passive_movies", movie_events),
            ("talk", talk_events),
        ]

        _log(f"Available events: {available_event_names}")
        _log(f"Passive movie events selected: {movie_events}")
        _log(f"Talk events selected: {talk_events}")

        modalities_to_export = list(multimodal_data.modalities)
        if 'diode' in multimodal_data.data.columns and 'diode' not in modalities_to_export:
            modalities_to_export.append('diode')
            _log("Detected diode signal in multimodal_data; adding 'diode' modality to chunk export.")

        for modality in modalities_to_export:
            path_modality = os.path.join(export_path, modality, str(multimodal_data.id))
            if not os.path.exists(path_modality):
                os.makedirs(path_modality)

            for who, member in members.items():
                path_member = os.path.join(path_modality, member)
                if not os.path.exists(path_member):
                    os.makedirs(path_member)

                for chunk_name, chunk_events in chunks:
                    if not chunk_events:
                        _log(
                            f"Skipping chunk '{chunk_name}' for dyad='{multimodal_data.id}' "
                            f"because no matching events were found."
                        )
                        continue

                    _log(
                        f"Exporting chunk='{chunk_name}', modality='{modality}', member='{who}', "
                        f"events={chunk_events}"
                    )
                    data_xr = export_chunk_to_xarray(
                        multimodal_data=multimodal_data,
                        selected_events=chunk_events,
                        selected_channels=selected_channels.get(modality),
                        selected_modality=modality,
                        member=who,
                        time_margin=time_margin,
                        chunk_name=chunk_name,
                        EEG_montage=export_mounted if modality == 'EEG' else None,
                        EEG_bad_channels=EEG_bad_channels if modality == 'EEG' else None,
                        verbose=verbose,
                        mne_plot_flag=mne_plot_flag,
                        logger=logger,
                    )
                    file_path = os.path.join(
                        path_member,
                        f"{multimodal_data.id}_{modality}_{who}_{chunk_name}.nc",
                    )
                    data_xr.to_netcdf(file_path, engine='netcdf4', format='NETCDF4_CLASSIC')
                    _log(f"Saved: {file_path}")

        _log(f"Finished chunk export for dyad '{multimodal_data.id}'")

#def export_to_xarray(multimodal_data, selected_event, selected_channels, selected_modality, member, time_margin, verbose=True, logger: Optional[object] = None):
    # '''Export selected signals from a MultimodalData instance to an xarray DataArray.
    # Args:
    #     multimodal_data: The MultimodalData instance containing the data.
    #     selected_event: The name of the event to select (e.g., 'Incredibles').
    #     selected_channels: List of channel names to include in the export (e.g., ['Fp1', 'Fp2'] for EEG).
    #     selected_modality: The modality to export (e.g., 'EEG', 'ECG', 'ET', 'IBI', 'RMSSD', or 'diode').
    #     member: The member to select ('ch' or 'cg').
    #     time_margin: Margin in seconds to include before and after the event.
    #     verbose: If True, emit export progress messages.
    #     logger: Optional logger-like object with .info(str). If provided and verbose=True,
    #         messages are sent to logger.info instead of print.
    # Returns:
    #     An xarray DataArray containing the selected signals for the specified event and modality, with time reset to 0 at the start of the event and metadata included as attributes.
    #     The DataArray will have dimensions 'time' and 'channel', and coordinates corresponding to the time points and channel names.
    #     Metadata attributes include information about dyad, member, sampling frequency, event details, and ``metadata_json``.
    #     The ``metadata_json`` payload contains ``notes`` and ``child_info`` and additionally
    #     ``event_order`` with the chronological order (by start time) of available target events:
    #     ``Peppa``, ``Incredibles``, and ``Brave``. For EEG exports (``selected_modality == 'EEG'``),
    #     ``metadata_json`` also includes an ``eeg`` object with details about signal ``filtration`` and
    #     channel ``references``.
    # '''
    # if selected_event not in multimodal_data.events:
    #     raise ValueError(f"Event '{selected_event}' not found. Available events: {list(multimodal_data.events.keys())}")

    # ev = multimodal_data.events[selected_event]
    # event_start = ev['start']
    # event_end = ev['start'] + ev['duration']

    # # find time range covering selected event with margin on both sides
    # recording_start = multimodal_data.data['time'].min()
    # recording_end = multimodal_data.data['time'].max()

    # selected_time = [
    #     max(recording_start, event_start - time_margin),
    #     min(recording_end, event_end + time_margin),
    # ]

    # if verbose:
    #     msg_1 = f"Event '{selected_event}' starts at {event_start:.2f}s and ends at {event_end:.2f}s"
    #     msg_2 = f"Selected time range with ±{time_margin}s margin: {selected_time[0]:.2f}s to {selected_time[1]:.2f}s"
    #     if logger is not None:
    #         logger.info(msg_1)
    #         logger.info(msg_2)
    #     else:
    #         print(msg_1)
    #         print(msg_2)

    # signals = multimodal_data.get_signals(
    #     mode=selected_modality,
    #     member=member,
    #     selected_channels=selected_channels,
    #     selected_times=selected_time
    # )
    # if signals is None:
    #     raise ValueError(
    #         f"No signals available for modality='{selected_modality}', member='{member}', "
    #         f"channels={selected_channels}, event='{selected_event}'."
    #     )
    # time, channels, data = signals

    # # convert the retrieved data to xarray DataArray, resetting time to 0 at event start
    # time = time - event_start
    # # strip channel names to remove EEG_{member}_ prefix if modality is EEG
    # if selected_modality == 'EEG':
    #     channels = [ch.replace(f'EEG_{member}_', '') for ch in channels]
    # elif selected_modality == 'ET':
    #     channels = [ch.replace(f'ET_{member}_', '') for ch in channels]
    # elif selected_modality == 'IBI':
    #     channels = [ch.replace(f'IBI_{member}', 'IBI') for ch in channels]
    # elif selected_modality == 'RMSSD':
    #     channels = [ch.replace(f'RMSSD_{member}', 'RMSSD') for ch in channels]
    # elif selected_modality == 'ECG':
    #     channels = [ch.replace(f'ECG_{member}', 'ECG') for ch in channels]
    # elif selected_modality == 'diode':
    #     channels = ['diode']

    # channels = [str(ch) for ch in channels]

    # data_xr = xr.DataArray(
    #     data,
    #     coords=[time, channels],
    #     dims=['time', 'channel'],
    #     name='signals'
    # )

    # metadata = _build_export_metadata(multimodal_data, selected_modality)

    # data_xr.attrs.update({
    #     'dyad_id': multimodal_data.id,
    #     'who': member,
    #     'sampling_freq': float(multimodal_data.fs),
    #     'event_name': selected_event,
    #     'event_start': 0.0,
    #     'event_duration': float(event_end - event_start),
    #     'time_margin_s': float(time_margin),
    #     'channel_names_csv': ','.join(channels),
    #     'channel_names_json': json.dumps(channels, ensure_ascii=True),
    #     'metadata_json': json.dumps(metadata, ensure_ascii=False, default=str),
    # })

    # sanitize_netcdf_attrs_inplace(data_xr.attrs)
    # return data_xr
