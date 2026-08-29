"""Per-dyad data assembly for the interbrain ffDTF + HRV pipeline (Stage 1).

Composes the existing NetCDF loaders (`src.ncdf`) into one continuous,
per-dyad container: EEG (ROI-subset, channels kept separate) plus IBI, both
on the shared EEG time grid, plus label-keyed film windows and per-child
metadata. Pure composition -- no filtering, no alignment, no Hilbert
envelopes, and no per-film cutting (film windows are handed off as metadata
so Stage 2 can segment *after* filtering, keeping edge transients out of the
kept data). See `scripts/stage01_coverage.py` for the orchestration built on
top of this module.
"""

import re
from pathlib import Path

import numpy as np

try:
    from .ncdf import get_export_metadata, load_xarray_from_netcdf, task_regions
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.ncdf import get_export_metadata, load_xarray_from_netcdf, task_regions

ROLE_CODE_OF = {'child': 'ch', 'caregiver': 'cg'}
ROLE_DIR_OF = {'ch': 'child', 'cg': 'caregiver'}


def ibi_path_for(ibi_root, dyad_id, role_code):
    """Build the expected per-task IBI NetCDF path for one dyad/role.

    Parameters
    ----------
    ibi_root : str or pathlib.Path
        Root of the IBI-by-task export tree, with files at
        ``<ibi_root>/<dyad_id>/<child|caregiver>/<dyad_id>_IBI_<ch|cg>_passive_movies.nc``.
    dyad_id : str
    role_code : str
        ``'ch'`` or ``'cg'``.

    Returns
    -------
    pathlib.Path
        Expected file path. Existence is not checked here.
    """
    role_dir = ROLE_DIR_OF[role_code]
    return Path(ibi_root) / dyad_id / role_dir / f"{dyad_id}_IBI_{role_code}_passive_movies.nc"


def select_roi_channels(data, channel_names, roi_channels):
    """Select a channel subset from a (channel, time) array, in `roi_channels` order.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
        Signal array indexed along axis 0 by `channel_names`.
    channel_names : sequence of str
        Channel label for each row of `data`.
    roi_channels : sequence of str
        Channel labels to select, in the desired output order.

    Returns
    -------
    tuple
        ``(data_roi, channels_found)`` -- `data_roi` has shape
        ``(len(channels_found), n_times)``; `channels_found` is the subset
        of `roi_channels` actually present in `channel_names`, in
        `roi_channels` order. A channel absent from `channel_names` is
        simply omitted, not an error -- the caller decides what a partial
        match means (e.g. `roi_ok` in `assemble_dyad`).
    """
    channel_names = list(channel_names)
    channels_found = [ch for ch in roi_channels if ch in channel_names]
    indices = [channel_names.index(ch) for ch in channels_found]
    return data[indices, :], channels_found


def parse_interpolated_channels(interpolation_note):
    """Parse the export pipeline's free-text interpolated-channel note.

    Parameters
    ----------
    interpolation_note : str or None
        Value of ``metadata_json['interpolation']`` as written by the ICA
        cleaning export, e.g. ``"Interpolated: ['P4']"``. ``None`` (or the
        key being absent) means no channel was interpolated.

    Returns
    -------
    list of str
        Channel names listed as interpolated (empty if none).
    """
    if not interpolation_note:
        return []
    match = re.search(r"\[(.*)\]", interpolation_note)
    if match is None:
        return []
    return [name.strip(" '\"") for name in match.group(1).split(",") if name.strip()]


def assemble_dyad(dyad_id, eeg_files, ibi_root, roi_channels):
    """Assemble one dyad's continuous EEG/IBI signals, film windows, and metadata.

    IBI is exported on the same time grid and event structure as EEG, so no
    re-alignment happens here -- film windows come from the EEG event
    structure (child preferred, caregiver used as a cross-check) and are
    reused as-is for IBI.

    Parameters
    ----------
    dyad_id : str
    eeg_files : pandas.DataFrame
        Rows from `io_utils.get_participant_files`, filtered to this dyad
        (columns: filepath, dyad_id, role_code, role). A role missing from
        `eeg_files` is treated as absent, not an error.
    ibi_root : str or pathlib.Path
        Root of the IBI-by-task export tree, see `ibi_path_for`.
    roi_channels : sequence of str
        EEG channel labels making up the ROI (e.g. ``['P7', 'P8']``).

    Returns
    -------
    dict
        ``{'dyad_id', 'group', 'meta': {'age_months', 'sex'},
        'roi_channels_expected', 'film_windows': {film_label: (start_s, end_s)},
        'eeg': {'child': {...} or None, 'caregiver': {...} or None},
        'ibi': {'child': {...} or None, 'caregiver': {...} or None},
        'notes': [str, ...]}``.

        Each present ``eeg[role]`` entry has keys ``data`` (ndarray,
        ``(n_roi_found, n_times)``), ``channel_names`` (== ``roi_found``),
        ``sfreq``, ``time``, ``roi_found``, ``roi_interpolated``, ``roi_ok``
        (all expected ROI channels found and none interpolated). Each
        present ``ibi[role]`` entry has ``data`` (ndarray, ``(n_times,)``),
        ``sfreq``, ``time``, ``grid_matches_eeg`` (time axis identical to
        the corresponding EEG file). A missing file leaves the role's entry
        as ``None``. ``notes`` collects dyad-level anomalies (e.g.
        child/caregiver film-window or group disagreement).
    """
    notes = []
    eeg = {'child': None, 'caregiver': None}
    ibi = {'child': None, 'caregiver': None}
    role_film_windows = {}
    role_child_info = {}

    eeg_path_of_role = dict(zip(eeg_files['role'], eeg_files['filepath']))

    for role, role_code in ROLE_CODE_OF.items():
        filepath = eeg_path_of_role.get(role)
        if filepath is None:
            continue

        data_xr = load_xarray_from_netcdf(filepath, decode_json_attrs=True)
        data = data_xr.transpose('channel', 'time').values.astype(float)
        channel_names = list(data_xr.coords['channel'].values)
        time = np.asarray(data_xr.coords['time'].values, dtype=float)
        sfreq = float(data_xr.attrs['sampling_freq'])
        metadata = get_export_metadata(data_xr)

        data_roi, channels_found = select_roi_channels(data, channel_names, roi_channels)
        interpolated = parse_interpolated_channels(metadata.get('interpolation'))
        roi_interpolated = sorted(set(channels_found) & set(interpolated))
        roi_ok = set(channels_found) == set(roi_channels) and not roi_interpolated

        eeg[role] = {
            'data': data_roi,
            'channel_names': channels_found,
            'sfreq': sfreq,
            'time': time,
            'roi_found': channels_found,
            'roi_interpolated': roi_interpolated,
            'roi_ok': roi_ok,
        }

        role_film_windows[role] = {r['name']: r['span'] for r in task_regions(data_xr)}
        child_info = metadata.get('child_info', {})
        role_child_info[role] = child_info if isinstance(child_info, dict) else {}

        ibi_path = ibi_path_for(ibi_root, dyad_id, role_code)
        if ibi_path.exists():
            ibi_xr = load_xarray_from_netcdf(ibi_path, decode_json_attrs=True)
            ibi_time = np.asarray(ibi_xr.coords['time'].values, dtype=float)
            grid_matches_eeg = len(ibi_time) == len(time) and np.array_equal(ibi_time, time)
            if not grid_matches_eeg:
                notes.append(f"IBI time grid does not match EEG for {dyad_id} {role}")
            ibi[role] = {
                'data': np.asarray(ibi_xr.values, dtype=float).reshape(-1),
                'sfreq': float(ibi_xr.attrs['sampling_freq']),
                'time': ibi_time,
                'grid_matches_eeg': grid_matches_eeg,
            }

    film_windows = role_film_windows.get('child') or role_film_windows.get('caregiver') or {}
    if 'child' in role_film_windows and 'caregiver' in role_film_windows:
        if role_film_windows['child'] != role_film_windows['caregiver']:
            notes.append(f"caregiver film windows differ from child for {dyad_id}")

    child_info = role_child_info.get('child') or role_child_info.get('caregiver') or {}
    group = child_info.get('group')
    meta = {'age_months': child_info.get('age_months'), 'sex': child_info.get('sex')}
    if 'child' in role_child_info and 'caregiver' in role_child_info:
        child_group = role_child_info['child'].get('group')
        cg_group = role_child_info['caregiver'].get('group')
        if child_group is not None and cg_group is not None and child_group != cg_group:
            notes.append(f"group disagreement between child/caregiver files for {dyad_id}")

    return {
        'dyad_id': dyad_id,
        'group': group,
        'meta': meta,
        'roi_channels_expected': list(roi_channels),
        'film_windows': film_windows,
        'eeg': eeg,
        'ibi': ibi,
        'notes': notes,
    }
