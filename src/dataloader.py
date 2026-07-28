import csv
import math
import os
from collections import deque
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import xmltodict
from matplotlib import pyplot as plt
from scipy.signal import (
    filtfilt,
    butter,
    sosfiltfilt,
    iirnotch,
    firwin,
    lfilter,
)
import mne

from . import eyetracker as et
from .data_structures import MultimodalData, Tasks, WhoEnum
from .utils import plot_filter_characteristics
from . import export  # For backwards compatibility
# --------------- ENUM status handler
def to_status(value):
    if value is None:
        return WhoEnum.Neither
    try:
        if isinstance(value, float) and math.isnan(value):
            return WhoEnum.Neither
    except TypeError:
        pass

    if isinstance(value, str):
        v = value.strip().lower()
        if v == "" or v in {"nan", "none"}:
            return WhoEnum.Neither
        v = v.replace(",", ".")
        try:
            v = int(float(v))
        except ValueError:
            return None
    else:
        try:
            v = int(value)
        except (TypeError, ValueError):
            return None
    # 0 - nikt, 1 - obydwoje, 3 - tylko mama
    if v == 0:
        return WhoEnum.Neither
    elif v == 1:
        return WhoEnum.Both
    elif v == 3:
        return WhoEnum.CG_Only
    else:
        return None

# --------------  Create multimodal data instance and populate it with dat


def _resolve_modality_directory(base_path: str, dyad_id: str, names: List[str]) -> str:
    """Return the first existing dyad modality directory for provided candidate names.

    If none exists, return the first candidate path to preserve previous behavior
    and allow downstream code to raise a clear file-not-found error.
    """
    for name in names:
        candidate = os.path.join(base_path, dyad_id, name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(base_path, dyad_id, names[0])

def create_multimodal_data(
    data_base_path,
    dyad_id,
    load_eeg=True,
    load_et=True,
    load_meta=False,
    lowcut=4.0,
    highcut=40.0,
    eeg_filter_type="fir",
    interpolate_et_during_blinks_threshold=0,
    median_filter_size=64,
    low_pass_et_order=351,
    et_pos_cutoff=128,
    et_pupil_cutoff=4,
    pupil_model_confidence=0.9,
    window_size=30,
    decimate_factor=1,
    plot_flag=False,
    run_consistency_check=True,
    consistency_strict=False,
    consistency_start_error=0.35,
):
    """Create and populate a MultimodalData instance by loading EEG and ET data.
    directory structure assumed is:
    data_base_path/
    <dyad_id>/
        EEG/ or eeg/
            <dyad_id>.obci
            <dyad_id>.xml
        ET/ or et/
            child/
                000/
                001/
                002/
            caregiver/
                000/
                001/
                002/

    Args:
        data_base_path (str): Base path to the data directory.
        dyad_id (str): Identifier for the dyad.
        load_eeg (bool, optional): Whether to load EEG data. Defaults to True.
        load_et (bool, optional): Whether to load eye-tracker data. Defaults to True.
        load_meta (bool, optional): Whether to load metadata. Defaults to True.
        lowcut (float, optional): Low cut-off frequency for EEG filtering. Defaults to 4.0 Hz.
        highcut (float, optional): High cut-off frequency for EEG filtering. Defaults to 40.0 Hz.
        eeg_filter_type (str, optional): Type of filter to use for EEG data ('fir' or 'iir'). Defaults to 'fir'.
        interpolate_et_during_blinks_threshold (float, optional): Confidence threshold for interpolating ET data during blinks. 0 means no interpolation. Defaults to 0.
        median_filter_size (int, optional): Size of the median filter for ET data processing. Defaults to 64.
        low_pass_et_order (int, optional): Order of the low-pass filter for ET data processing. Defaults to 351.
        et_pos_cutoff (float, optional): Cutoff frequency for ET position data low-pass filter. Defaults to 128 Hz.
        et_pupil_cutoff (float, optional): Cutoff frequency for ET pupil data low-pass filter. Defaults to 4 Hz.
        pupil_model_confidence (float, optional): Confidence level for 3D pupil model. Defaults to 0.9.
        window_size (float, optional): Sliding window size in seconds used for
            RMSSD computation from ECG. Defaults to 30.
        plot_flag (bool, optional): Whether to plot intermediate results for debugging/visualization. Defaults to False.
        run_consistency_check (bool, optional): Whether to run consistency checks
            after loading and event structure creation. Defaults to True.
        consistency_strict (bool, optional): If True, raise ValueError when
            consistency check fails. Defaults to False.
        consistency_start_error (float, optional): Allowed start-time mismatch
            between EEG and ET corresponding events. Defaults to 0.35 seconds.

    Returns:
        MultimodalData: An instance populated with EEG and ET data.
    """
    multimodal_data = MultimodalData()
    multimodal_data.id = dyad_id
    if load_meta:
        meta_path = os.path.join(data_base_path, 'meta_data.csv')
        df_meta = pd.read_csv(meta_path,sep=";",quoting=csv.QUOTE_NONE)
        df_meta.columns = df_meta.columns.str.strip()
        df_meta["ID"] = df_meta["ID"].astype(str).str.strip()
        dyad_id = str(dyad_id).strip()
        df_meta = df_meta.set_index("ID")
        if dyad_id not in df_meta.index:
            raise ValueError(f"No {dyad_id} found in meta_data.csv")
        # tasks
        row = df_meta.loc[dyad_id]
        multimodal_data.tasks.dual_hrv.secore = to_status(row["Active ECG during SECORE"])
        multimodal_data.tasks.dual_hrv.movies = to_status(row["Passive ECG during EEG&ET"])
        multimodal_data.tasks.dual_hrv.conversation = None # brak kolumny w WarsawID

        multimodal_data.tasks.dual_eeg.movies = to_status(row["EEG Passive"])
        multimodal_data.tasks.dual_eeg.conversation = to_status(row["EEG Active during Conversation"])

        multimodal_data.tasks.dual_et.movies = to_status(row["ET Passive ET"])
        multimodal_data.tasks.dual_et.conversation = to_status(row["ET Active ET during Conversation"])

        multimodal_data.tasks.dual_fnirs.movies = to_status(row["Fnirs Passive"])
        multimodal_data.tasks.dual_fnirs.conversation = to_status(row["Fnirs Active during Conversation"])

        # child info
        multimodal_data.child_info.age_months = row["Wiek [m]"]
        multimodal_data.child_info.sex = row["Płeć"]
        multimodal_data.child_info.group = row["Grupa"]

        # notes
        multimodal_data.notes = row["Comments"]

    if load_eeg:
        folder_eeg = _resolve_modality_directory(
            data_base_path, dyad_id, ["EEG", "eeg"]
        )
        multimodal_data = load_eeg_data(
            multimodal_data,
            dyad_id=dyad_id,
            folder_eeg=folder_eeg,
            lowcut=lowcut,
            highcut=highcut,
            eeg_filter_type=eeg_filter_type,
            window_size=window_size,
            plot_flag=plot_flag,
        )
    if load_et:
        folder_et = _resolve_modality_directory(
            data_base_path, dyad_id, ["ET", "et"]
        )
        if multimodal_data.fs is None:
            # default EEG sampling frequency common to all signals if EEG data
            # not loaded or set before ET data
            multimodal_data.fs = 1024
            print("Setting default EEG sampling frequency to 1024 Hz used also in ET data.")
        multimodal_data = load_et_data(
            multimodal_data,
            dyad_id=dyad_id,
            folder_et=folder_et,
            interpolate_et_during_blinks_threshold=interpolate_et_during_blinks_threshold,
            median_filter_size=median_filter_size,
            low_pass_et_order=low_pass_et_order,
            et_pos_cutoff=et_pos_cutoff,
            et_pupil_cutoff=et_pupil_cutoff,
            pupil_model_confidence=pupil_model_confidence,
            plot_flag=plot_flag,
        )
    if decimate_factor > 1:
        multimodal_data = multimodal_data._decimate_signals(q=decimate_factor)
    multimodal_data._create_events_column()
    multimodal_data._create_event_structure()
    if run_consistency_check:
        consistency_report = check_consistency_of_multimodal_data(
            multimodal_data,
            start_error=consistency_start_error,
            verbose=plot_flag,
        )
        if consistency_strict and not consistency_report["is_consistent"]:
            raise ValueError(
                f"Inconsistent multimodal data for dyad {dyad_id}: {consistency_report}"
            )
    return multimodal_data


def check_consistency_of_multimodal_data(
    multimodal_data: MultimodalData,
    start_error: float = 0.35,
    event_time_error: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validate consistency of a MultimodalData object.

    Checks performed:
    1. Whether modalities listed in ``multimodal_data.modalities`` are consistent
       with modalities inferred from DataFrame column names.
    2. Whether ``multimodal_data.events`` matches the ``events`` column content.
    3. Whether corresponding events in ``EEG_events`` and ``ET_event`` start
       within ``start_error`` seconds.

    Args:
        multimodal_data: Object to validate.
        start_error: Max allowed absolute start-time difference for matching
            EEG/ET events.
        event_time_error: Max allowed absolute error for matching event start
            and duration between ``events`` column and ``multimodal_data.events``.
            If None, defaults to one sample period (``1/fs``) when available.
        verbose: Print human-readable consistency messages.

    Returns:
        dict: Structured validation report with ``is_consistent`` flag and
        detailed results per check.
    """
    if event_time_error is None:
        if multimodal_data.fs is not None and multimodal_data.fs > 0:
            event_time_error = 1.0 / multimodal_data.fs
        else:
            event_time_error = 1e-6

    data_columns = set(multimodal_data.data.columns)
    listed_modalities = set(multimodal_data.modalities or [])

    modality_prefixes = {
        "EEG": ("EEG_",),
        "ECG": ("ECG_",),
        "IBI": ("IBI_",),
        "RMSSD": ("RMSSD_",),
        "ET": ("ET_",),
    }

    inferred_modalities = set()
    for modality, prefixes in modality_prefixes.items():
        if any(col.startswith(prefixes) for col in data_columns):
            inferred_modalities.add(modality)

    unknown_modalities = sorted(
        [m for m in listed_modalities if m not in modality_prefixes]
    )
    listed_without_columns = sorted(
        [
            m
            for m in listed_modalities
            if m in modality_prefixes
            and not any(col.startswith(modality_prefixes[m]) for col in data_columns)
        ]
    )
    present_but_not_listed = sorted(list(inferred_modalities - listed_modalities))

    modalities_ok = (
        len(unknown_modalities) == 0
        and len(listed_without_columns) == 0
        and len(present_but_not_listed) == 0
    )

    events_column_present = "events" in data_columns
    events_from_column = set()
    if events_column_present:
        events_from_column = set(
            multimodal_data.data["events"].dropna().astype(str).unique().tolist()
        )

    events_struct = multimodal_data.events if isinstance(multimodal_data.events, dict) else {}
    events_from_struct = set(events_struct.keys())

    events_missing_in_struct = sorted(list(events_from_column - events_from_struct))
    events_missing_in_column = sorted(list(events_from_struct - events_from_column))

    event_timing_mismatches = []
    if events_column_present:
        for ev_name in sorted(events_from_column & events_from_struct):
            mask = multimodal_data.data["events"] == ev_name
            if not mask.any():
                continue
            start_col = multimodal_data.data.loc[mask, "time"].min()
            end_col = multimodal_data.data.loc[mask, "time"].max()
            duration_col = end_col - start_col

            struct_ev = events_struct.get(ev_name, {})
            start_struct = struct_ev.get("start")
            duration_struct = struct_ev.get("duration")

            if isinstance(start_struct, (int, float)):
                start_diff = abs(start_col - float(start_struct))
                if start_diff > event_time_error:
                    event_timing_mismatches.append(
                        {
                            "event": ev_name,
                            "field": "start",
                            "column_value": float(start_col),
                            "structure_value": float(start_struct),
                            "abs_diff": float(start_diff),
                        }
                    )
            else:
                event_timing_mismatches.append(
                    {
                        "event": ev_name,
                        "field": "start",
                        "column_value": float(start_col),
                        "structure_value": start_struct,
                        "abs_diff": None,
                    }
                )

            if isinstance(duration_struct, (int, float)):
                duration_diff = abs(duration_col - float(duration_struct))
                if duration_diff > event_time_error:
                    event_timing_mismatches.append(
                        {
                            "event": ev_name,
                            "field": "duration",
                            "column_value": float(duration_col),
                            "structure_value": float(duration_struct),
                            "abs_diff": float(duration_diff),
                        }
                    )
            else:
                event_timing_mismatches.append(
                    {
                        "event": ev_name,
                        "field": "duration",
                        "column_value": float(duration_col),
                        "structure_value": duration_struct,
                        "abs_diff": None,
                    }
                )

    events_ok = (
        events_column_present
        and len(events_missing_in_struct) == 0
        and len(events_missing_in_column) == 0
        and len(event_timing_mismatches) == 0
    )

    def _events_from_column(column_name: str) -> List[Dict[str, Any]]:
        if column_name not in multimodal_data.data.columns:
            return []

        event_dicts = []
        event_names = multimodal_data.data[column_name].dropna().unique().tolist()
        for ev_name in event_names:
            mask = multimodal_data.data[column_name] == ev_name
            start_time = multimodal_data.data.loc[mask, "time"].min()
            end_time = multimodal_data.data.loc[mask, "time"].max()
            event_dicts.append(
                {
                    "name": ev_name,
                    "start": float(start_time),
                    "duration": float(end_time - start_time),
                }
            )
        return event_dicts

    eeg_et_modalities_present = {"EEG", "ET"}.issubset(listed_modalities)
    eeg_events_dicts = _events_from_column("EEG_events")
    et_events_dicts = _events_from_column("ET_event")

    eeg_et_mismatches = []
    shared_event_names = []
    eeg_et_check_possible = (
        eeg_et_modalities_present
        and len(eeg_events_dicts) > 0
        and len(et_events_dicts) > 0
    )

    if eeg_et_check_possible:
        eeg_start_by_name = {ev["name"]: ev["start"] for ev in eeg_events_dicts}
        et_start_by_name = {ev["name"]: ev["start"] for ev in et_events_dicts}
        shared_event_names = sorted(
            list(set(eeg_start_by_name) & set(et_start_by_name))
        )

        for ev_name in shared_event_names:
            diff = abs(eeg_start_by_name[ev_name] - et_start_by_name[ev_name])
            if diff > start_error:
                eeg_et_mismatches.append(
                    {
                        "event": ev_name,
                        "eeg_start": float(eeg_start_by_name[ev_name]),
                        "et_start": float(et_start_by_name[ev_name]),
                        "abs_diff": float(diff),
                    }
                )
                if verbose:
                    print(
                        f'\033[91mEvent {ev_name} differ in start times by: abs({diff}) seconds.\033[0m'
                    )
            elif verbose:
                print(
                    f'\033[92mEvent {ev_name} start times are consistent within {start_error} seconds.\033[0m'
                )

    eeg_et_ok = eeg_et_check_possible and len(eeg_et_mismatches) == 0

    report = {
        "is_consistent": modalities_ok
        and events_ok
        and (not eeg_et_check_possible or eeg_et_ok),
        "modalities": {
            "ok": modalities_ok,
            "listed_modalities": sorted(list(listed_modalities)),
            "inferred_from_columns": sorted(list(inferred_modalities)),
            "unknown_modalities": unknown_modalities,
            "listed_without_columns": listed_without_columns,
            "present_but_not_listed": present_but_not_listed,
        },
        "events_structure": {
            "ok": events_ok,
            "events_column_present": events_column_present,
            "events_missing_in_structure": events_missing_in_struct,
            "events_missing_in_column": events_missing_in_column,
            "timing_mismatches": event_timing_mismatches,
            "event_time_error": event_time_error,
        },
        "eeg_et_start_consistency": {
            "ok": eeg_et_ok,
            "check_possible": eeg_et_check_possible,
            "modalities_present": eeg_et_modalities_present,
            "start_error": start_error,
            "shared_events": shared_event_names,
            "mismatches": eeg_et_mismatches,
        },
    }

    if verbose:
        print(f"Modalities consistency: {modalities_ok}")
        print(f"Events structure consistency: {events_ok}")
        if not eeg_et_check_possible:
            print(
                "EEG/ET event-start consistency: skipped "
                "(missing EEG/ET modality or missing EEG_events/ET_event data)."
            )
        else:
            print(f"EEG/ET event-start consistency: {eeg_et_ok}")
        print(f"Overall consistency: {report['is_consistent']}")

    return report


# --------------  Load EEG and ECG data form SVAROG files -----------------


def load_eeg_data(
    multimodal_data=None,
    dyad_id=None,
    folder_eeg=None,
    lowcut=4.0,
    highcut=40.0,
    eeg_filter_type="fir",
    window_size=30,
    plot_flag=False,
):
    """Load and filter EEG data from SVAROG format files into MultimodalData instance.
    Assumes data were recorded as multiplexed signals in SVAROG system format.
    Assumes specific channel names for child and caregiver EEG data.

    Args:
        multimodal_data (MultimodalData, optional): An existing MultimodalData instance to populate.
            If None, a new instance is created. Defaults to None.
        dyad_id (str, optional): Identifier for the dyad. Defaults to None.
        folder_eeg (str, optional): Path to the folder containing the EEG data files. Defaults to None.
        lowcut (float, optional): Low cut-off frequency for EEG filtering. Defaults to 4.0 Hz.
        highcut (float, optional): High cut-off frequency for EEG filtering. Defaults to 40.0 Hz.
        eeg_filter_type (str, optional): Type of filter to use ('fir' or 'iir'). Defaults to 'fir'.
        window_size (float, optional): Sliding window size in seconds used for
            RMSSD computation from ECG. Defaults to 30.
        plot_flag (bool, optional): Whether to plot intermediate results for debugging/visualization. Defaults to False.

    Returns:
        MultimodalData: The populated multimodal data instance with EEG, ECG, and event data.
    """
    if multimodal_data is None:
        multimodal_data = MultimodalData()
        multimodal_data.id = dyad_id
    multimodal_data.paths.eeg_directory = folder_eeg
    multimodal_data.eeg_channel_names_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                                            'M1', 'T3', 'C3', 'Cz','C4', 'T4','M2',
                                            'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    multimodal_data.eeg_channel_names_cg = ['Fp1_cg', 'Fp2_cg', 'F7_cg', 'F3_cg', 'Fz_cg', 'F4_cg', 'F8_cg',
                                            'M1_cg', 'T3_cg', 'C3_cg', 'Cz_cg', 'C4_cg', 'T4_cg', 'M2_cg', 
                                            'T5_cg', 'P3_cg', 'Pz_cg', 'P4_cg', 'T6_cg', 'O1_cg', 'O2_cg']

    raw_eeg_data = _read_raw_svarog_data(multimodal_data, plot_flag)

    # extract diode signal for event detection filtering
    diode = raw_eeg_data[multimodal_data.eeg_channel_mapping["Diode"], :]

    # set the ECG modality with ECG signals (in place)
    _extract_ecg_data(multimodal_data, raw_eeg_data, window_size=window_size)
    
    # scan for events
    events_list, thresholded_diode = _scan_for_events(
        diode, multimodal_data.fs, plot_flag, threshold=0.75
    )
    # Convert to dict-of-dicts, filtering out incomplete events
    multimodal_data.events = {
        ev['name']: ev for ev in events_list
        if 'start' in ev and 'duration' in ev
    }
    print(f"Detected events: {multimodal_data.events}")

    # mount EEG data to M1 and M2 channels and filter the data (in place)
    _mount_eeg_data(multimodal_data, raw_eeg_data)
    filters = _design_eeg_filters(
        multimodal_data,
        lowcut=lowcut,
        highcut=highcut,
        filter_type=eeg_filter_type,
        notch_freq=50,
        notch_q=30,
        plot_flag=plot_flag,
    )
    _apply_filters(multimodal_data, filters, raw_eeg_data, plot_flag=plot_flag)

    # Store EEG data in DataFrame with each channel as a column and set time
    # column if not set yet
    multimodal_data._set_eeg_data(
        raw_eeg_data, multimodal_data.eeg_channel_mapping
    )
    # Set EEG events column
    multimodal_data._set_EEG_events_column()

    # Store diode in DataFrame
    multimodal_data._set_diode(thresholded_diode)

    if "EEG" not in multimodal_data.modalities:
        multimodal_data.modalities.append("EEG")



    # reset time column to be consistent with the first movie event start at time zero; this is needed to align with ET data later; accordingly reset time_idx
    # in the column 'EEG_events' find the first occurance of one of 'Brave',
    # 'Peppa', 'Incredibles'; reset to the corresponding time
    first_movie_event = multimodal_data.data[
        multimodal_data.data["EEG_events"].isin(["Brave", "Peppa", "Incredibles"])
    ]["time"].min()
    print(
        f"Reseting the EEG time to the start of {multimodal_data.data[multimodal_data.data['time'] == first_movie_event]['EEG_events'].iloc[0]}"
    )
    multimodal_data.data["time"] = (
        multimodal_data.data["time"] - first_movie_event
    )
    multimodal_data.data["time_idx"] = (
        multimodal_data.data["time"] * multimodal_data.fs
    ).astype(int)

    return multimodal_data


def _read_raw_svarog_data(multimodal_data: MultimodalData, plot_flag):
    file = multimodal_data.id + ".obci"  # SVAROG files have .obci extension
    # read meta information from xml file
    with open(
        os.path.join(multimodal_data.paths.eeg_directory, f"{file}.xml")
    ) as fd:
        xml = xmltodict.parse(fd.read())

    n_channels = int(xml["rs:rawSignal"]["rs:channelCount"])
    fs = int(float(xml["rs:rawSignal"]["rs:samplingFrequency"]))
    channel_names = xml["rs:rawSignal"]["rs:channelLabels"]["rs:label"]

    for i, name in enumerate(channel_names):
        multimodal_data.eeg_channel_mapping[name] = i

    # if debug print N_chan, Fs_EEG, chan_names
    if plot_flag:
        print(
            f"n_channels: {n_channels},\n fs_EEG: {fs},\n chan_names: {channel_names}"
        )

    multimodal_data.fs = fs
    raw_eeg_data = (
        np.fromfile(
            os.path.join(multimodal_data.paths.eeg_directory, f"{file}.raw"),
            dtype="float32",
        )
        .reshape((-1, n_channels))
        .T
    )  # transpose to have channels in rows and samples in columns

    # scale the signal to microvolts
    raw_eeg_data *= 0.0715

    return raw_eeg_data


def _mount_eeg_data(multimodal_data, raw_eeg_data):
    channel_mapping = multimodal_data.eeg_channel_mapping

    # Compute references before modifying any channels
    ref_ch = 0.5 * (
        raw_eeg_data[channel_mapping["M1"], :].copy()
        + raw_eeg_data[channel_mapping["M2"], :].copy()
    )
    ref_cg = 0.5 * (
        raw_eeg_data[channel_mapping["M1_cg"], :].copy()
        + raw_eeg_data[channel_mapping["M2_cg"], :].copy()
    )

    # Rereferuj wszystkie kanały child OPRÓCZ M1 i M2
    for channel in multimodal_data.eeg_channel_names_ch:
        if channel in channel_mapping and channel not in ("M1", "M2"):
            raw_eeg_data[channel_mapping[channel], :] -= ref_ch

    # Rereferuj wszystkie kanały caregiver OPRÓCZ M1_cg i M2_cg
    for channel in multimodal_data.eeg_channel_names_cg:
        if channel in channel_mapping and channel not in ("M1_cg", "M2_cg"):
            raw_eeg_data[channel_mapping[channel], :] -= ref_cg

    multimodal_data.references = (
        "linked ears montage: (M1+M2)/2; "
        "M1, M2, M1_cg, M2_cg retain original (pre-reference) values — "
        "reconstruction: ch_original[i] = ch_rereferenced[i] + 0.5*(M1 + M2)"
    )


def _design_eeg_filters(
    multimodal_data: MultimodalData,
    lowcut,
    highcut,
    notch_freq=50,
    notch_q=30,
    filter_type="fir",
    plot_flag=False,
):
    """
    Design notch, low-pass, and high-pass filters for EEG data.

    Args:
        multimodal_data (MultimodalData): The multimodal data instance containing sampling frequency.
        lowcut (float): Low cut-off frequency for high-pass filtering.
        highcut (float): High cut-off frequency for low-pass filtering.
        notch_freq (float, optional): Notch filter frequency. Defaults to 50 Hz.
        notch_q (float, optional): Quality factor for notch filter. Defaults to 30.
        filter_type (str, optional): Type of filter ('fir' or 'iir'). Defaults to 'fir'.
        plot_flag (bool, optional): Whether to plot filter characteristics. Defaults to False.

    Returns:
        tuple: ((b_notch, a_notch), (b_low, a_low), (b_high, a_high), filter_type)
               Filter coefficients and filter type.
    """
    b_notch, a_notch = iirnotch(notch_freq, notch_q, fs=multimodal_data.fs)

    if filter_type == "fir":
        numtaps_low = 201
        b_low = firwin(
            numtaps_low, highcut, fs=multimodal_data.fs, pass_zero="lowpass"
        )
        numtaps_high = 3049
        b_high = firwin(
            numtaps_high, lowcut, fs=multimodal_data.fs, pass_zero="highpass"
        )
        a_low = a_high = 1.0
        low_order = numtaps_low - 1
        high_order = numtaps_high - 1
        f_type = "firwin"
    else:
        butter_order = 4
        b_low, a_low = butter(
            N=butter_order, Wn=highcut, btype="low", fs=multimodal_data.fs
        )
        b_high, a_high = butter(
            N=butter_order, Wn=lowcut, btype="high", fs=multimodal_data.fs
        )
        low_order = butter_order
        high_order = butter_order
        f_type = "butter"

    if plot_flag:
        print("---- Notch filter characteristics: --------")
        f_max = 60.0
        plot_filter_characteristics(
            b_notch,
            a_notch,
            f=np.arange(0, f_max, 0.01),
            T=0.5,
            Fs=multimodal_data.fs,
            f_lim=(30, f_max),
            db_lim=(-300, 0.1),
        )
        print("---- Low-pass filter characteristics: --------")
        plot_filter_characteristics(
            b_low,
            a=[1],
            f=np.arange(0, multimodal_data.fs / 2, 0.1),
            T=0.5,
            Fs=multimodal_data.fs,
            f_lim=(0, 50),
            db_lim=(-60, 0.1),
        )
        print("---- High-pass filter characteristics: --------")
        plot_filter_characteristics(
            b_high,
            a=[1],
            f=np.arange(0, multimodal_data.fs / 2, 0.01),
            T=1.0,
            Fs=multimodal_data.fs,
            f_lim=(0, 10),
            db_lim=(-60, 0.1),
        )
        # add info about filtering to the multimodal data
    multimodal_data.eeg_filtration.low_pass["type"] = filter_type
    multimodal_data.eeg_filtration.low_pass["cut_f"] = highcut
    multimodal_data.eeg_filtration.low_pass["order"] = low_order
    multimodal_data.eeg_filtration.low_pass["f_type"] = f_type
    multimodal_data.eeg_filtration.low_pass["a"] = a_low
    multimodal_data.eeg_filtration.low_pass["b"] = b_low
    multimodal_data.eeg_filtration.high_pass["type"] = filter_type
    multimodal_data.eeg_filtration.high_pass["cut_f"] = lowcut
    multimodal_data.eeg_filtration.high_pass["order"] = high_order
    multimodal_data.eeg_filtration.high_pass["f_type"] = f_type
    multimodal_data.eeg_filtration.high_pass["a"] = a_high
    multimodal_data.eeg_filtration.high_pass["b"] = b_high
    multimodal_data.eeg_filtration.notch["Q"] = notch_q
    multimodal_data.eeg_filtration.notch["freq"] = notch_freq
    multimodal_data.eeg_filtration.notch["a"] = a_notch
    multimodal_data.eeg_filtration.notch["b"] = b_notch

    return (b_notch, a_notch), (b_low, a_low), (b_high, a_high), filter_type


def _apply_filters(
    multimodal_data: MultimodalData, filters, raw_eeg_data, plot_flag=False
):
    """
    Apply designed filters to raw EEG data in place.

    Args:
        multimodal_data (MultimodalData): The multimodal data instance containing channel mapping.
        filters (tuple): Filter coefficients from _design_eeg_filters.
        raw_eeg_data (np.ndarray): Raw EEG data array to filter in place.
        plot_flag (bool, optional): Whether to plot filtered signals. Defaults to False.

    Returns:
        None: Modifies raw_eeg_data in place.
    """
    (b_notch, a_notch), (b_low, a_low), (b_high, a_high), filter_type = filters
    print(f"Applying {filter_type} filters to EEG data.")

    # Filter and separate each channel
    for idx, ch in enumerate(multimodal_data.eeg_channel_names_all()):
        signal = raw_eeg_data[multimodal_data.eeg_channel_mapping[ch], :]
        signal = signal - np.mean(signal)  # remove DC offset
        if filter_type == "iir":
            signal = filtfilt(b_notch, a_notch, signal, axis=0)
            signal = filtfilt(b_low, a_low, signal, axis=0)
            signal = filtfilt(b_high, a_high, signal, axis=0)
        else:
            signal = lfilter(b_notch, a_notch, signal, axis=0)
            signal = lfilter(b_low, a_low, signal, axis=0)
            signal = lfilter(b_high, a_high, signal, axis=0)

            delay = (len(b_low) - 1) // 2 + (len(b_high) - 1) // 2
            signal = np.roll(signal, -delay)
            # zero-pad the end to account for the delay introduced by filtering
            signal[-delay:] = 0.0

        raw_eeg_data[multimodal_data.eeg_channel_mapping[ch], :] = signal
        # plot filtered signals for debugging
        if plot_flag:
            plt.figure(figsize=(12, 4))
            plt.plot(signal)
            plt.title(f"Filtered signal for channel {ch}")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude (uV)")
            plt.show()
    multimodal_data.eeg_filtration.notch["applied"] = True
    multimodal_data.eeg_filtration.low_pass["applied"] = True
    multimodal_data.eeg_filtration.high_pass["applied"] = True


def _extract_ecg_data(multimodal_data: MultimodalData, raw_eeg_data, window_size=30):
    """
    Extract and filter ECG data from raw EEG recording.

    Args:
        multimodal_data (MultimodalData): The multimodal data instance to populate with ECG.
        raw_eeg_data (np.ndarray): Raw EEG data containing ECG channels.
        window_size (float, optional): Sliding window size in seconds used for
            RMSSD computation from ECG. Defaults to 30.

    Returns:
        None: Modifies multimodal_data in place, adding ECG data and modality.
    """
    channel_mapping = multimodal_data.eeg_channel_mapping

    # extract and filter the ECG data
    ecg_ch = (
        raw_eeg_data[channel_mapping["EKG1"], :]
        - raw_eeg_data[channel_mapping["EKG2"], :]
    )
    ecg_cg = (
        raw_eeg_data[channel_mapping["EKG1_cg"], :]
        - raw_eeg_data[channel_mapping["EKG2_cg"], :]
    )

    # design filters:
    b_notch, a_notch = iirnotch(50, 30, fs=multimodal_data.fs)
    sos_ecg = butter(5, 0.5, btype="high", output="sos", fs=multimodal_data.fs)
    ecg_ch_filtered = sosfiltfilt(sos_ecg, ecg_ch)
    ecg_ch_filtered = filtfilt(b_notch, a_notch, ecg_ch_filtered)
    ecg_cg_filtered = sosfiltfilt(sos_ecg, ecg_cg)
    ecg_cg_filtered = filtfilt(b_notch, a_notch, ecg_cg_filtered)

    # Store ECG data in DataFrame
    multimodal_data._set_ecg_data(ecg_ch_filtered, ecg_cg_filtered)
    if "ECG" not in multimodal_data.modalities:
        multimodal_data.modalities.append("ECG")

    # compute IBI and add IBI modality
    multimodal_data._set_ibi()
    if "IBI" not in multimodal_data.modalities:
        multimodal_data.modalities.append("IBI")
    # compute the RMSSD feature and add it to the multimodal data
    multimodal_data._set_RMSSD_from_ECG(window_size=window_size)

def _scan_for_events(diode, eeg_fs, plot_flag, threshold=0.75):
    """Scan the diode signal to detect and identify experimental events.

    Processes the raw diode signal to find periods corresponding to
    specific experimental events, such as watching movies or engaging in conversation.
    Binarizes the signal based on a threshold to identify "on" and "off" states,
    then analyzes durations and intervals to classify into predefined event categories.

    The detection logic is tailored to a specific experimental design, expecting
    three movie sessions followed by two conversation sessions.

    Args:
        diode (np.ndarray): The diode signal array.
        eeg_fs (float): Sampling frequency of the EEG/diode signal.
        plot_flag (bool): Whether to plot the detected events for debugging.
        threshold (float, optional): The threshold for binarizing the diode signal,
            relative to its maximum value. Defaults to 0.75.

    Returns:
        tuple: (events, thresholded_diode)
            - events (list[dict]): List of dictionaries with 'name', 'start', 'duration' keys.
            - thresholded_diode (np.ndarray): Binary array of thresholded diode signal.
    """
    # Binarize the diode signal: values above the threshold become 1, others 0.
    thresholded_diode = ((diode / (threshold * np.max(diode))) > 1).astype(
        float
    )

    # Find rising (1) and falling (-1) edges in the binarized signal.
    # Collect the sample indices of all rising and falling edges.
    up_down_events = np.where(np.abs(np.diff(thresholded_diode)) == 1)[
        0
    ].tolist() + [len(diode)]

    events = [
        {"name": name}
        for name in ["Brave", "Peppa", "Incredibles", "Talk_1", "Talk_2"]
    ]

    found_movies = found_talks = 0
    queue = deque(maxlen=100)

    # Process pairs of up/down events to identify event durations and
    # intervals.
    for i in range(len(up_down_events) // 2):
        start = up_down_events[2 * i]
        duration = up_down_events[2 * i + 1] - up_down_events[2 * i]
        # Calculate the time until the next event starts.
        following_space = up_down_events[2 * i + 2] - up_down_events[2 * i + 1]
        queue.append(start)
        # Maintain a queue of recent event start times
        while queue[0] < start - 4 * eeg_fs:  # last 4 seconds
            queue.popleft()
        # Detect movie events based on their duration and number of recent
        # spikes
        if (
            duration > 55 * eeg_fs and len(queue) > 1
        ):  # movie events longer than 0:55
            events[len(queue) - 2]["start"] = (
                queue[0] + 1
            ) / eeg_fs  # add 1 sample due to shift caused by diff
            events[len(queue) - 2]["duration"] = (
                up_down_events[2 * i + 1] - queue[0]
            ) / eeg_fs
            found_movies += 1
        if found_movies > 3:
            raise ValueError(
                "More than 3 events detected, something is wrong."
            )
        # Detect talk events based on their position relative to movie events
        if (
            found_movies == 3
            and duration < 2 * eeg_fs
            and following_space > 175 * eeg_fs
        ):  # talk events longer than 2:55
            if found_talks < 2:
                event_index = found_movies + found_talks
                # add 1 sample due to shift caused by diff
                events[event_index]["start"] = (
                    up_down_events[2 * i + 1] + 1
                ) / eeg_fs
                events[event_index]["duration"] = following_space / eeg_fs
                found_talks += 1
            else:
                raise ValueError(
                    "More than 2 talks detected, something is wrong."
                )

    if plot_flag:
        _plot_scanned_events(
            threshold,
            diode,
            thresholded_diode,
            np.diff(thresholded_diode),
            events,
            eeg_fs,
        )

    return events, thresholded_diode


def _plot_scanned_events(
    threshold, diode, thresholded_diode, derivative, events, eeg_fs
):
    plt.figure(figsize=(12, 6))
    plt.plot(
        diode / (threshold * np.max(diode)),
        "b",
        label="Diode Signal normalized by threshold",
    )
    plt.plot(thresholded_diode, "r", label="Diode Signal Thresholded")
    plt.title("Diode Signal with events")
    plt.xlabel("Samples")
    plt.ylabel("Signal Value")
    plt.plot((derivative == 1).astype(int), "g", label="Up Events")
    plt.plot((derivative == -1).astype(int), "m", label="Down Events")
    for event in events:
        if "start" in event:
            plt.plot(event["start"] * eeg_fs, 1.2, "ko", markersize=10)
            plt.text(event["start"] * eeg_fs, 1.25, event["name"], rotation=45)
    plt.legend()
    plt.show()


# ==============================================================================
# I/O and Data Access Functions
# ==============================================================================
# For backwards compatibility, these functions are available:
#   - save_to_file() and load_output_data() -> now in export module
#   - get_eeg_data() -> now a static method of MultimodalData class
#   - export_eeg_to_mne_raw() -> now multimodal_data.to_mne_raw() method

# Backwards compatibility wrappers
def save_to_file(multimodal_data, output_dir):
    return export.save_to_file(multimodal_data, output_dir)

def load_output_data(filename):
    return export.load_output_data(filename)

def get_eeg_data(df, who):
    return MultimodalData.get_eeg_data(df, who)

def export_eeg_to_mne_raw(multimodal_data, who, times=None, event=None, margin_around_event=0):
    return multimodal_data.to_mne_raw(who, times, event, margin_around_event)
# --------------  Load eye-tracking data -----------------


def _build_et_file_paths(et_path: str, task_id: str, member: str) -> dict:
    """
    Build file paths for eye-tracker data for a specific task and dyad member.

    Args:
        et_path: Base path to ET data directory
        task_id: Task identifier ('000', '001', '002')
        member: Dyad member ('child' or 'caregiver')

    Returns:
        Dictionary mapping data types to file paths
    """
    base_export_path = os.path.join(et_path, member, task_id, "exports", "000")
    prefix = "ch" if member == "child" else "cg"

    paths = {
        f"annotations_{task_id}": os.path.join(
            base_export_path, "annotations.csv"
        ),
        f"{prefix}_pupil_{task_id}": os.path.join(
            base_export_path, "pupil_positions.csv"
        ),
    }

    # Movies task (000) has additional gaze position and blinks data
    if task_id == "000":
        paths[f"{prefix}_pos_{task_id}"] = os.path.join(
            base_export_path,
            "surfaces",
            "gaze_positions_on_surface_Surface 1.csv",
        )
        paths[f"{prefix}_blinks_{task_id}"] = os.path.join(
            base_export_path, "blinks.csv"
        )

    return paths


def _check_et_files_exist(file_paths: dict) -> tuple[bool, list]:
    """
    Check if ET data files exist.

    Args:
        file_paths: Dictionary mapping data types to file paths

    Returns:
        Tuple of (all_exist: bool, missing_files: list)
    """
    missing_files = [
        name for name, path in file_paths.items() if not os.path.exists(path)
    ]
    return len(missing_files) == 0, missing_files


def _load_et_task_data(
    file_paths: dict, task_id: str, member: str, min_max_times: list
) -> dict:
    """
    Load ET data files for a specific task and dyad member.

    Args:
        file_paths: Dictionary mapping data types to file paths
        task_id: Task identifier ('000', '001', '002')
        member: Dyad member ('child' or 'caregiver')
        min_max_times: List to append (min_time, max_time) tuples for time alignment

    Returns:
        Dictionary containing loaded DataFrames
    """
    prefix = "ch" if member == "child" else "cg"
    loaded_data = {}

    # Load annotations
    ann_key = f"annotations_{task_id}"
    if ann_key in file_paths:
        loaded_data[ann_key] = pd.read_csv(file_paths[ann_key])

    # Load pupil data
    pupil_key = f"{prefix}_pupil_{task_id}"
    if pupil_key in file_paths:
        pupil_df = pd.read_csv(file_paths[pupil_key])
        loaded_data[pupil_key] = pupil_df
        min_max_times.append(
            (
                pupil_df["pupil_timestamp"].min(),
                pupil_df["pupil_timestamp"].max(),
            )
        )

    # Load gaze position data (movies task only)
    pos_key = f"{prefix}_pos_{task_id}"
    if pos_key in file_paths:
        pos_df = pd.read_csv(file_paths[pos_key])
        loaded_data[pos_key] = pos_df
        min_max_times.append(
            (pos_df["gaze_timestamp"].min(), pos_df["gaze_timestamp"].max())
        )

    # Load blinks data (movies task only)
    blinks_key = f"{prefix}_blinks_{task_id}"
    if blinks_key in file_paths:
        loaded_data[blinks_key] = pd.read_csv(file_paths[blinks_key])

    return loaded_data


def _process_et_data_to_dataframe(
    et_df: pd.DataFrame,
    loaded_data: dict,
    task_flags: dict,
    task_names: list,
    median_filter_size=64,
    low_pass_et_order=351,
    et_pos_cutoff=128,
    et_pupil_cutoff=1,
    pupil_model_confidence=0.9,
    Fs=1024,
) -> None:
    """
    Process loaded ET data into the main ET dataframe.

    Args:
        et_df: DataFrame to populate with ET data
        loaded_data: Dictionary containing all loaded ET DataFrames
        task_flags: Dictionary with flags indicating which tasks/members have data
        task_names: List of task identifiers ['000', '001', '002']
        median_filter_size: Size of the median filter for smoothing
        low_pass_et_order: Order of the low-pass filter for ET data
        et_pos_cutoff: Cutoff frequency for position data low-pass filter
        et_pupil_cutoff: Cutoff frequency for pupil data low-pass filter
        Fs: Sampling frequency of ET data
    """
    # Process movies task (000) if available
    if task_flags.get("movies_ch") or task_flags.get("movies_cg"):
        if "annotations_000" in loaded_data:
            et.process_event_et(loaded_data["annotations_000"], et_df)

    if task_flags.get("movies_ch"):
        if "ch_pos_000" in loaded_data:
            et.process_pos(
                loaded_data["ch_pos_000"],
                et_df,
                "ch",
                median_filter_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pos_cutoff,
                Fs=Fs,
            )
        if "ch_pupil_000" in loaded_data:
            et.process_pupil(
                loaded_data["ch_pupil_000"],
                et_df,
                "ch",
                model_confidence=pupil_model_confidence,
                median_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pupil_cutoff,
                Fs=Fs,
            )
        if "ch_blinks_000" in loaded_data:
            et.process_blinks(loaded_data["ch_blinks_000"], et_df, "ch")

    if task_flags.get("movies_cg"):
        if "cg_pos_000" in loaded_data:
            et.process_pos(
                loaded_data["cg_pos_000"],
                et_df,
                "cg",
                median_filter_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pos_cutoff,
                Fs=Fs,
            )
        if "cg_pupil_000" in loaded_data:
            et.process_pupil(
                loaded_data["cg_pupil_000"],
                et_df,
                "cg",
                model_confidence=pupil_model_confidence,
                median_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pupil_cutoff,
                Fs=Fs,
            )
        if "cg_blinks_000" in loaded_data:
            et.process_blinks(loaded_data["cg_blinks_000"], et_df, "cg")

    # Process talk1 task (001)
    if task_flags.get("talk1_ch") or task_flags.get("talk1_cg"):
        if "annotations_001" in loaded_data:
            et.process_event_et(loaded_data["annotations_001"], et_df, "Talk1")

    if task_flags.get("talk1_ch"):
        if "ch_pupil_001" in loaded_data:
            et.process_pupil(
                loaded_data["ch_pupil_001"],
                et_df,
                "ch",
                model_confidence=pupil_model_confidence,
                median_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pupil_cutoff,
                Fs=Fs,
            )

    if task_flags.get("talk1_cg"):
        if "cg_pupil_001" in loaded_data:
            et.process_pupil(
                loaded_data["cg_pupil_001"],
                et_df,
                "cg",
                model_confidence=pupil_model_confidence,
                median_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pupil_cutoff,
                Fs=Fs,
            )

    # Process talk2 task (002)
    if task_flags.get("talk2_ch") or task_flags.get("talk2_cg"):
        if "annotations_002" in loaded_data:
            et.process_event_et(loaded_data["annotations_002"], et_df, "Talk2")

    if task_flags.get("talk2_ch"):
        if "ch_pupil_002" in loaded_data:
            et.process_pupil(
                loaded_data["ch_pupil_002"],
                et_df,
                "ch",
                model_confidence=pupil_model_confidence,
                median_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pupil_cutoff,
                Fs=Fs,
            )

    if task_flags.get("talk2_cg"):
        if "cg_pupil_002" in loaded_data:
            et.process_pupil(
                loaded_data["cg_pupil_002"],
                et_df,
                "cg",
                model_confidence=pupil_model_confidence,
                median_size=median_filter_size,
                order=low_pass_et_order,
                cutoff=et_pupil_cutoff,
                Fs=Fs,
            )


def load_et_data(
    multimodal_data,
    dyad_id,
    folder_et,
    interpolate_et_during_blinks_threshold=0,
    median_filter_size=64,
    low_pass_et_order=351,
    et_pos_cutoff=128,
    et_pupil_cutoff=1,
    pupil_model_confidence=0.9,
    plot_flag=False,
):
    """Load eye-tracking data from CSV files and integrate into the MultimodalData instance.

    Args:
        multimodal_data (MultimodalData): Instance of MultimodalData to populate with ET data.
        dyad_id (str): Identifier for the dyad.
        folder_et (str): Base path to ET data directory.
        interpolate_et_during_blinks_threshold (float, optional): Confidence threshold for interpolating ET data during blinks. 0 means no interpolation. Defaults to 0.
        median_filter_size (int, optional): Size of the median filter for ET data processing. Defaults to 64.
        low_pass_et_order (int, optional): Order of the low-pass filter for ET data processing. Defaults to 351.
        et_pos_cutoff (float, optional): Cutoff frequency for ET position data low-pass filter. Defaults to 128.
        et_pupil_cutoff (float, optional): Cutoff frequency for ET pupil data low-pass filter. Defaults to 1.
        pupil_model_confidence (float, optional): Confidence level for 3D pupil model. Defaults to 0.9.
        plot_flag (bool, optional): Whether to plot intermediate results for debugging/visualization. Defaults to False.

    Returns:
        MultimodalData: The multimodal data instance with integrated ET data.
    """
    if multimodal_data is None:
        multimodal_data = MultimodalData()
        multimodal_data.id = dyad_id
    # Configuration for tasks: 000=movies, 001=talk1, 002=talk2
    tasks = [
        {"id": "000", "name": "movies"},
        {"id": "001", "name": "talk1"},
        {"id": "002", "name": "talk2"},
    ]
    members = ["child", "caregiver"]

    # Build file paths and check availability for each task and member
    task_flags = {}
    min_max_times = []
    loaded_data = {}

    for task in tasks:
        for member in members:
            # Build file paths
            file_paths = _build_et_file_paths(folder_et, task["id"], member)

            # Check if files exist
            all_exist, missing = _check_et_files_exist(file_paths)

            # Set flag for this task/member combination
            flag_key = f"{task['name']}_{'ch' if member == 'child' else 'cg'}"
            task_flags[flag_key] = all_exist

            if not all_exist:
                print(
                    f"Warning: Missing ET files for {multimodal_data.id} {member} {task['name']}: {missing}"
                )
            else:
                # Load data if all files exist
                task_data = _load_et_task_data(
                    file_paths, task["id"], member, min_max_times
                )
                loaded_data.update(task_data)

    # Skip ET processing if no data was loaded
    if not min_max_times:
        print(f"Warning: No ET data available for {multimodal_data.id}")
        return

    # Construct dataframe for ET data
    et_df = pd.DataFrame()

    # Set sampling frequency if not already set
    if multimodal_data.fs is None:
        multimodal_data.fs = 1024  # default sampling rate for UW EEG data
        print(f"Setting fs to the default Fs of EEG: {multimodal_data.fs}")

    # Find the overall min and max times across all tasks and members
    overall_min_time = min([t[0] for t in min_max_times])
    overall_max_time = max([t[1] for t in min_max_times])

    print(f"ET time range: {overall_min_time:.2f}s to {overall_max_time:.2f}s")

    # Create time vector
    et_df["time"] = np.arange(
        overall_min_time, overall_max_time, 1 / multimodal_data.fs
    )
    et_df["time_idx"] = (et_df["time"] * multimodal_data.fs).astype(int)

    # Process loaded data into the dataframe
    _process_et_data_to_dataframe(
        et_df,
        loaded_data,
        task_flags,
        [t["id"] for t in tasks],
        median_filter_size=median_filter_size,
        low_pass_et_order=low_pass_et_order,
        et_pos_cutoff=et_pos_cutoff,
        et_pupil_cutoff=et_pupil_cutoff,
        pupil_model_confidence=pupil_model_confidence,
        Fs=multimodal_data.fs,
    )

    # Align ET time to EEG time by subtracting the time of the first event;
    # We consider the time of first event in all data series to be 0;
    # reset time_idx accordingly

    # reset time column to be consistent with the first movie event start at time zero; this is needed to align with EEG data later; accordingly reset time_idx
    # in the column 'ET_event' find the first occurance of one of 'Incredibles','Peppa','Brave',
    # 'm3'; reset to the corresponding time
    first_movie_event = et_df[et_df["ET_event"].isin(["Incredibles", "Peppa", "Brave"])]["time"].min()
    print(
        f"Reseting the ET time to the start of {et_df[et_df['time'] == first_movie_event]['ET_event'].iloc[0]}"
    )
    et_df["time"] = et_df["time"] - first_movie_event
    et_df["time_idx"] = (et_df["time"] * multimodal_data.fs).astype(int)

    #  merging ET data into the main dataframe
    if multimodal_data.data.empty:
        multimodal_data.data = et_df.copy()
    else:
        multimodal_data.data = pd.merge(
            multimodal_data.data, et_df, how="outer", on="time_idx"
        )
        # After merge, time_x and time_y are created; use time_x (from EEG) as
        # primary, fill missing with time_y (from ET)
        multimodal_data.data["time"] = multimodal_data.data["time_x"].fillna(
            multimodal_data.data["time_y"]
        )
        multimodal_data.data = multimodal_data.data.drop(
            columns=["time_x", "time_y"]
        )
        multimodal_data.data = multimodal_data.data.replace(np.nan, None)

    # correct alignment of x, y, blinks and pupil columns by delta_time
    # estimated to be -0.3s
    delta_time = -0.3
    for col in multimodal_data.data.columns:
        if any(
            keyword in col for keyword in ["x", "y", "blinks", "diameter3d"]
        ):
            multimodal_data.data[col] = multimodal_data.data[col].shift(
                int(delta_time * multimodal_data.fs)
            )
    if interpolate_et_during_blinks_threshold > 0:
        # correct x, y, and diameter3d columns by interpolating the values
        # during blinks, separately for child and caregiver
        for member in ["ch", "cg"]:
            blink_col = f"ET_{member}_blinks"
            print(f"Processing member: {member}, blink column: {blink_col}")
            for col in multimodal_data.data.columns:
                if any(
                    keyword in col
                    for keyword in [
                        f"ET_{member}_x",
                        f"ET_{member}_y",
                        f"ET_{member}_pupil",
                    ]
                ):
                    # Convert to numeric dtype to enable interpolation
                    multimodal_data.data[col] = pd.to_numeric(
                        multimodal_data.data[col], errors="coerce"
                    )
                    blink_indices = multimodal_data.data.index[
                        multimodal_data.data[blink_col]
                        > interpolate_et_during_blinks_threshold
                    ].tolist()
                    multimodal_data.data.loc[blink_indices, col] = np.nan
                    multimodal_data.data[col] = multimodal_data.data[
                        col
                    ].interpolate(method="linear", limit_direction="both")

    multimodal_data.modalities.append("ET")
    return multimodal_data

