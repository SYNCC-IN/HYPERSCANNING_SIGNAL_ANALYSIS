# Data Structure Specification

### EEGLAB-style multimodal structure for EEG, ET, and IBI signals

**Version:** 2.5  
**Last updated:** 2026-02-26  
**Author:** Joanna Duda-Goławska/Jarosław Żygierewicz

This document defines the unified **Python data structure** for handling multimodal child-caregiver data recorded with:

- EEG (electroencephalography),
- ET (eye-tracking),
- IBI (inter-beat interval, derived from ECG).

The structure is designed for transparent integration across different signal types, maintaining synchronisation and
consistent metadata.

`MultimodalData` is the **internal format** produced by Stage 1 of the project's data
pipeline (raw import). It is not consumed directly by analysis pipelines — those read
NetCDF files exported from it in Stage 2 and cleaned in Stage 3. See the project
[README](../README.md#current-data-pipeline) and
[docs/architecture_diagram.md](architecture_diagram.md#current-production-pipeline)
for the full pipeline, and [docs/export_ncdf_guide.md](export_ncdf_guide.md) for how
`MultimodalData` is exported.

In order to popoulate the data-structure, the expected directory-structure with raw data is (`<dyad_id>` here is
one dyad's subfolder under `UNIWAW_RAW_DATA`, or under the small local `data/` sample used for tests/demos):

```
data_base_path/
    <dyad_id>/
        EEG/ (or eeg/)
            <dyad_id>.obci
            <dyad_id>.xml
        ET/ (or et/)
            child/
                000/
                001/
                002/
            caregiver/
                000/
                001/
                002/
```

## Core Architecture

The data are stored in a `MultimodalData` class that uses a **DataFrame-based approach** where all signal data is combined in a single pandas DataFrame called `data`. This approach ensures:
- Unified sampling frequency across all signals
- Shared time indexing
- Easy data alignment and merging
- Consistent column naming conventions

## MultimodalData Class Structure

### Core Attributes

```python
class MultimodalData:
    # Core data storage
    data: pd.DataFrame                      # All signal data in single DataFrame
    fs: float                               # Unified sampling frequency (Hz) - typically EEG sampling rate
    
    # Metadata
    id: str                                 # Dyad ID

    
    # EEG-specific metadata
    eeg_channel_names: list[str]            # List of all channel names in order
    eeg_channel_mapping: dict[str, int]     # Mapping: channel name → index in raw data
    references: str                         # Information about reference electrodes (e.g., "linked ears montage: (M1+M2)/2")
    eeg_filtration: Filtration              # Filtration parameters (see Filtration dataclass below)
    eeg_channel_names_ch: list[str]         # Child EEG channels after montage
    eeg_channel_names_cg: list[str]         # Caregiver EEG channels after montage
    
    # Events
    events: dict[str, dict]                 # Dictionary of event dictionaries: {event_name: {'name': str, 'start': float, 'duration': float}}
    epoch: list or None                     # Epoch information (if applicable)
    
    # Organizational structures
    paths: Paths                            # File and directory paths (see Paths dataclass)
    tasks: Tasks                            # Task information (see Tasks dataclass)
    modalities: list[str]                   # List of available modalities (e.g., ['EEG', 'ECG', 'IBI', 'ET'])
    child_info: ChildInfo                   # Child participant information (see ChildInfo dataclass)
    
    # Notes
    notes: str or None                      # Experimental notes
```

### DataFrame Column Naming Conventions

The `data` DataFrame uses consistent naming patterns for different signal types:

- **Time columns**: `time` (seconds), `time_idx` (sample indices)
- **EEG channels**: 
  - Child: `EEG_ch_{channel}` (e.g., `EEG_ch_Fp1`, `EEG_ch_Cz`)
  - Caregiver: `EEG_cg_{channel}` (e.g., `EEG_cg_Fp1`, `EEG_cg_Cz`)
- **ECG signals**: `ECG_ch` (child), `ECG_cg` (caregiver)
- **IBI signals**: `IBI_ch` (child), `IBI_cg` (caregiver)
- **Eye-tracking**:
  - Position: `ET_ch_x`, `ET_ch_y` (child), `ET_cg_x`, `ET_cg_y` (caregiver)
  - Pupil: `ET_ch_pupil` (child), `ET_cg_pupil` (caregiver)
  - Blinks: `ET_ch_blinks` (child), `ET_cg_blinks` (caregiver)
  - Events: `ET_event`
- **Diode signal**: `diode`
- **Event columns**: `events` (unified), `EEG_events`, `ET_event` (modality-specific)

### Supporting Dataclasses

#### Filtration

Stores EEG signal filtration parameters:

```python
@dataclass
class Filtration:
    notch: dict[str, Any]               # {'Q', 'freq', 'a', 'b', 'applied'}
    low_pass: dict[str, Any]            # {'type', 'cut_f', 'order', 'f_type', 'a', 'b', 'applied'}
    high_pass: dict[str, Any]           # {'type', 'cut_f', 'order', 'f_type', 'a', 'b', 'applied'}
```

Nested dictionary keys:

- `notch['Q']`: Quality factor for notch filter
- `notch['freq']`: Notch filter frequency (Hz, typically 50 or 60)
- `notch['a']`, `notch['b']`: Notch denominator/numerator coefficients
- `notch['applied']`: Whether notch filter was applied
- `low_pass['type']`, `high_pass['type']`: Filter design type (`'fir'` or `'iir'`)
- `low_pass['cut_f']`, `high_pass['cut_f']`: Cutoff frequencies (Hz)
- `low_pass['order']`, `high_pass['order']`: Filter order
- `low_pass['f_type']`, `high_pass['f_type']`: Concrete design function (`'firwin'` or `'butter'`)
- `low_pass['a']`, `low_pass['b']`, `high_pass['a']`, `high_pass['b']`: filter coefficients
- `low_pass['applied']`, `high_pass['applied']`: Whether each filter was applied

#### Paths

Stores file and directory paths:

```python
@dataclass
class Paths:
    eeg_directory: str or None          # Path to EEG raw data
    et_directory: str or None           # Path to eye-tracking files
    hrv_directory: str or None          # Path to HRV/IBI files
    output_dir: str or None             # Path for saving results/figures
```

#### Tasks

Container for task-related information:

```python
@dataclass
class Tasks:
    dual_hrv: DualHRV                   # HRV task information
    dual_eeg: DualEEG                   # EEG task information
    dual_et: DualET                     # ET task information

@dataclass
class DualHRV:
    secore: bool or None                # Active HRV during SECORE
    movies: bool or None                # Passive HRV during movies
    conversation: bool or None          # Active HRV during conversation

@dataclass
class DualEEG:
    movies: bool or None                # Passive EEG during movies
    conversation: bool or None          # Active EEG during conversation

@dataclass
class DualET:
    movies: bool or None                # Passive ET during movies
    conversation: bool or None          # Active ET during conversation
```

#### ChildInfo

Stores child participant information:

```python
@dataclass
class ChildInfo:
    age_months: int or None             # Age in months at recording
    group: str or None                  # Group: 'T' (Typical), 'ASD', 'P' (Premature)
    sex: str or None                    # Sex: 'M' (male), 'F' (female)
```

## Key Methods

### Data Population Methods (Private - Called by DataLoader)

**Important Note**: The following methods are private (prefixed with `_`) and should only be called by the `DataLoader` class during data initialization. They are not intended for direct use by end users.

```python
_set_eeg_data(eeg_data: np.ndarray, channel_mapping: Dict[str, int])
    # Store EEG data with each channel as a DataFrame column
    # Creates time and time_idx columns if fs is set
    # Args:
    #   eeg_data: 2D array [n_channels x n_samples]
    #   channel_mapping: Dict mapping channel names to indices in eeg_data
    # Note: Private method - only called by DataLoader
    
_set_ecg_data(ecg_ch: np.ndarray, ecg_cg: np.ndarray)
    # Store ECG signals for child and caregiver
    # Creates columns 'ECG_ch' and 'ECG_cg'
    # Note: Private method - only called by DataLoader
    
_set_diode(diode: np.ndarray)
    # Store diode signal for event detection
    # Creates column 'diode'
    # Note: Private method - only called by DataLoader
    
_set_EEG_events_column(events: List[dict])
    # Populate EEG_events column based on event timing and duration
    # Events should have format: [{'name': str, 'start': float, 'duration': float}, ...]
    # Creates or updates 'EEG_events' column in the DataFrame
    # Note: Private method - only called by DataLoader

_create_events_column(start_error: float = 0.0)
    # Creates unified 'events' column from 'EEG_events' and 'ET_event' columns
    # Checks consistency between EEG and ET event timing
    # Args:
    #   start_error: Maximum acceptable difference in event start times (seconds)
    # Process:
    #   - Extracts event information from both EEG_events and ET_event columns
    #   - Checks for timing consistency between corresponding events
    #   - Prioritizes EEG_events timing when events overlap
    #   - Includes unique events from both sources
    #   - Populates 'events' column in the DataFrame
    # Note: Private method - only called by DataLoader

_create_event_structure()
    # Creates event structure dictionary from the 'events' column in the DataFrame
    # Populates self.events attribute as dict[str, dict] mapping event names to event info
    # Each event dict contains: {'name': str, 'start': float, 'duration': float}
    # Note: Requires 'events' column to exist in self.data
    # Note: Private method - only called by DataLoader

_interpolate_ibi_signals(who: str, label: str = '', plot_flag: bool = False) -> tuple[np.ndarray, np.ndarray]
    # Extracts R-peaks and interpolates IBI signals from ECG data using neurokit2
    # Merges interpolated IBI into the main data DataFrame using time-based alignment
    # Args:
    #   who: 'ch' (child) or 'cg' (caregiver)
    #   label: Optional label for plotting
    #   plot_flag: Whether to plot the interpolated IBI signal
    # Returns:
    #   Tuple of (ibi_interp, ecg_times) where:
    #   - ibi_interp: Interpolated IBI signal in milliseconds
    #   - ecg_times: Time vector for the interpolated IBI signal
    # Note: Creates or updates 'IBI_ch' or 'IBI_cg' column in the DataFrame
    # Note: Private method - only called by DataLoader

_set_ibi()
    # Convenience method to interpolate IBI signals for both child and caregiver
    # Calls _interpolate_ibi_signals for both 'ch' and 'cg'
    # Note: Private method - only called by DataLoader

_decimate_signals(q: int = 8) -> MultimodalData
    # Decimates all signal columns by factor q while maintaining synchronization across modalities
    # Args:
    #   q: Decimation factor (default=8) - reduce sampling rate by this factor
    # Returns:
    #   MultimodalData: New instance with decimated signals and updated sampling frequency (fs/q)
    # Process:
    #   - Applies anti-aliasing FIR filter before decimation to prevent frequency folding
    #   - Handles NaN values by forward filling before decimation to maintain signal continuity
    #   - Decimates time, time_idx columns by selecting every q-th sample
    #   - Event columns (events, ET_event, EEG_events) are downsampled by selecting every q-th sample
    #   - Diode column is downsampled by selecting every q-th sample
    #   - Signal columns (EEG, ECG, IBI, ET) are properly anti-aliased filtered then decimated
    #   - Updates fs (sampling frequency) by dividing by q
    #   - Copies all metadata (id, channel names, filtration, paths, tasks, child_info, notes)
    # Note: Creates a new MultimodalData object; does not modify in place
    # Note: Private method - only called by DataLoader

_ensure_data_length(length: int)
    # Internal method to ensure DataFrame has enough rows
    # Creates or extends DataFrame as needed
    # Note: Private method - only called by DataLoader
```

## Consistency Validation

The dataloader module provides `check_consistency_of_multimodal_data(...)` to validate coherence of `MultimodalData` content.

### What is checked

1. **Modalities list vs columns**
- Compares `multimodal_data.modalities` with modality presence inferred from DataFrame columns (`EEG_*`, `ECG_*`, `IBI_*`, `ET_*`).

2. **Event structure vs events column**
- Compares `multimodal_data.events` with the `events` DataFrame column (event names, start, duration).

3. **EEG vs ET event starts**
- Compares matching event start times between `EEG_events` and `ET_event` columns.
- Uses tolerance `start_error` (default `0.35` seconds).
- This check runs **only** if both `EEG` and `ET` are present in `multimodal_data.modalities`.

### Return value

The function returns a report dictionary with top-level keys:

- `is_consistent`
- `modalities`
- `events_structure`
- `eeg_et_start_consistency`

`eeg_et_start_consistency` includes:

- `ok`
- `check_possible`
- `modalities_present`
- `start_error`
- `shared_events`
- `mismatches`

### Integration in create_multimodal_data

`create_multimodal_data(...)` supports automatic validation with:

- `run_consistency_check=True`
- `consistency_strict=False`
- `consistency_start_error=0.35`

When `consistency_strict=True`, inconsistent data raises `ValueError`.

### Data Retrieval Methods (Public)

### Data Retrieval Methods (Public)

```python
get_signals(mode: str = 'EEG', member: str = 'ch', selected_channels: List[str] = None,
            selected_events: List[str] = None, selected_times: Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray] or None
    # Retrieve signals from the DataFrame based on mode, member, selected channels, events, and times
    # Args:
    #   mode: Signal type - 'EEG', 'ECG', 'IBI', 'ET' (eye-tracking), or 'diode'
    #   member: 'ch' (child) or 'cg' (caregiver)
    #   selected_channels: List of channel names to retrieve (None = all channels for that mode/member)
    #                      Required for 'EEG' and 'ET' modes. Ignored for other modes.
    #   selected_events: List of event names to filter by (None = no event filtering)
    #   selected_times: Time range tuple (start, end) in seconds (None = all times)
    # Returns:
    #   None if no matching columns found, otherwise tuple of:
    #   - time_vector: 1D array of timestamps in seconds
    #   - channel_names: List of column names corresponding to retrieved channels
    #   - data: 2D array [n_samples x n_channels] containing signal data
    # Note: If both selected_events and selected_times are provided, data is filtered by events first, then by times.

get_eeg_data_ch() -> np.ndarray or None
    # Returns child EEG data as 2D array [n_channels x n_samples]
    
get_eeg_data_cg() -> np.ndarray or None
    # Returns caregiver EEG data as 2D array [n_channels x n_samples]

@staticmethod
get_eeg_data(df: pd.DataFrame, who: str) -> tuple[np.ndarray or None, list]
    # Static method that returns EEG data and channel names for specified participant from any DataFrame
    # Args:
    #   df: DataFrame containing EEG data following MultimodalData column naming convention
    #   who: 'ch' for child, 'cg' for caregiver
    # Returns:
    #   Tuple of (eeg_data, channel_names) where:
    #   - eeg_data: 2D array [n_channels x n_samples] or None if no data
    #   - channel_names: List of clean channel names (e.g., ['Fp1', 'Fp2', ...])
    
eeg_channel_names_all() -> list[str]
    # Returns combined list of child and caregiver channel names

get_events_as_marker_channel(selected_times: tuple or None = None) -> tuple or None
    # Retrieve events as a marker channel aligned with the time column
    # Args:
    #   selected_times: Optional time window (start_time, end_time) to filter events
    # Returns:
    #   Tuple of (time_vector, marker_channel, event_to_marker) where:
    #   - time_vector: 1D array of time points in seconds
    #   - marker_channel: 1D array of event markers (integers) aligned with time_vector
    #   - event_to_marker: Dict mapping event names to integer markers
```

### Export and Visualization Methods (Public)

```python
to_mne_raw(who: str, times: tuple[float, float] or None = None, 
           event: str or None = None, margin_around_event: float = 0) -> tuple
    # Export EEG data to MNE Raw object with optional time/event filtering
    # Args:
    #   who: 'ch' for child, 'cg' for caregiver
    #   times: Optional time range (start_time, end_time) in seconds
    #   event: Optional event name to extract (e.g., 'Brave', 'Incredibles')
    #   margin_around_event: Time margin (seconds) to add before/after event
    # Returns:
    #   Tuple of (raw, time) where:
    #   - raw: MNE Raw object with EEG data and annotations
    #   - time: np.ndarray of time points in seconds
    # Note: If both event and times are provided, event takes precedence
    #       Adds event annotations to the MNE Raw object
    #       Sets montage to standard 10-20 electrode positions
    #       Marks data as pre-filtered based on low_pass/high_pass applied flags

print_events()
    # Prints formatted table of all events from self.events dictionary
    # Displays event name, start time, and duration for each event
```

### Export to Xarray

The `export` module provides helper functions for exporting selected signal fragments to `xarray.DataArray`.

```python
from src.export import export_to_xarray

export_to_xarray(multimodal_data, selected_event, selected_channels,
                 selected_modality, member, time_margin) -> xr.DataArray
```

Behavior:

- Validates that `selected_event` exists in `multimodal_data.events`
- Computes a time window covering event start/end with `time_margin` on both sides
- Slices data via `MultimodalData.get_signals(...)`
- Returns `xr.DataArray` with dimensions `['time', 'channel']`
- Normalizes channel labels by stripping modality/member prefixes (e.g., `EEG_ch_`)

Supported modalities:

- `EEG`
- `ET`
- `ECG`
- `IBI`
- `diode`

Default metadata stored in `DataArray.attrs`:

- `dyad_id`
- `who`
- `fs_hz`
- `event_name`
- `event_start_s`
- `event_end_s`
- `time_margin_s`
- `notes`
- `child_info`
- for EEG only: `filtration`, `references`

Example:

```python
from src.export import export_to_xarray

data_xr = export_to_xarray(
    multimodal_data=md,
    selected_event='Incredibles',
    selected_channels=['Fz', 'Cz', 'Pz'],
    selected_modality='EEG',
    member='ch',
    time_margin=10,
)

print(data_xr)
data_xr.plot.line(x='time', hue='channel')
```

Optional post-processing example (z-score over time per channel):

```python
from scipy.stats import zscore
data_xr.data = zscore(data_xr.data, axis=0, nan_policy='omit')
```

### Utility Methods (Public)

```python
eeg_channel_names_all() -> list[str]
    # Returns combined list of child and caregiver EEG channel names
```

## Example Usage

```python
from src.dataloader import create_multimodal_data
from src.data_structures import MultimodalData
import numpy as np

# Load data using DataLoader (recommended approach)
md = create_multimodal_data(
    data_base_path='./data',
    dyad_id='W003',
    load_eeg=True,
    load_et=True,
    lowcut=4.0,
    highcut=40.0,
    decimate_factor=8
)

# Access basic information
print(f"Dyad ID: {md.id}")
print(f"Sampling frequency: {md.fs} Hz")
print(f"Available modalities: {md.modalities}")
print(f"Data shape: {md.data.shape}")

# View events
md.print_events()

# Access specific columns
print(md.data.columns)  # All column names
print(md.data['EEG_ch_Fp1'])  # Specific channel data

# Get EEG data as arrays
ch_eeg = md.get_eeg_data_ch()  # All child EEG as 2D array [n_channels x n_samples]
cg_eeg = md.get_eeg_data_cg()  # All caregiver EEG as 2D array [n_channels x n_samples]

# Get filtered signals
time_vec, chan_names, eeg_data = md.get_signals(
    mode='EEG',
    member='ch',
    selected_channels=['Fp1', 'Fp2', 'F3', 'F4'],
    selected_events=['Brave'],
    selected_times=None
)

# Get ECG signals
time_vec, chan_names, ecg_data = md.get_signals(mode='ECG', member='ch')

# Export to MNE for advanced analysis
raw, time = md.to_mne_raw(who='ch', event='Brave', margin_around_event=2.0)
raw.plot()

# Export selected signals to xarray
from src.export import export_to_xarray
eeg_xr = export_to_xarray(
    multimodal_data=md,
    selected_event='Brave',
    selected_channels=['Fz', 'Cz', 'Pz'],
    selected_modality='EEG',
    member='ch',
    time_margin=5,
)
print(eeg_xr.attrs)

# Get events as marker channel for external tools
time_vec, markers, event_map = md.get_events_as_marker_channel()
```

## DataLoader Usage

The `DataLoader` module provides the `create_multimodal_data()` function to load and preprocess data:

```python
from src.dataloader import create_multimodal_data

md = create_multimodal_data(
    data_base_path='path/to/data',
    dyad_id='W003',
    load_eeg=True,              # Load EEG data
    load_et=True,               # Load eye-tracking data
    lowcut=4.0,                 # High-pass filter cutoff (Hz)
    highcut=40.0,               # Low-pass filter cutoff (Hz)
    eeg_filter_type='fir',      # 'fir' or 'iir'
    interpolate_et_during_blinks_threshold=0,  # ET blink interpolation (0 = no interpolation)
    median_filter_size=64,      # ET median filter size
    low_pass_et_order=351,      # ET low-pass filter order
    et_pos_cutoff=128,          # ET position cutoff frequency (Hz)
    et_pupil_cutoff=4,          # ET pupil cutoff frequency (Hz)
    pupil_model_confidence=0.9, # 3D pupil model confidence threshold
    decimate_factor=1,          # Decimation factor (1 = no decimation)
    plot_flag=False             # Show diagnostic plots
)
```

The DataLoader:
1. Loads EEG data from SVAROG format (.obci, .xml files)
2. Extracts ECG signals and processes them to obtain IBI
3. Detects events from diode channel
4. Applies EEG filters (notch, high-pass, low-pass)
5. Loads eye-tracking data from Pupil Labs format
6. Processes and filters ET signals (position, pupil diameter, blinks)
7. Synchronizes all modalities to common time base
8. Optionally decimates signals to reduce data size
9. Creates unified event structure from EEG and ET events

**Note**: Direct manipulation of MultimodalData using private methods (prefixed with `_`) is not recommended. Use the DataLoader instead.

## Version History

- **2.5 (2026-02-26)**: Added Xarray export documentation
    - Added `export_to_xarray()` API description and behavior
    - Documented supported modalities and `DataArray.attrs` metadata
    - Added end-to-end Xarray export examples and z-score post-processing snippet

- **2.4 (2026-02-26)**: Filtration schema refactor
    - Replaced flat `Filtration` fields with nested dictionaries: `notch`, `low_pass`, `high_pass`
    - Added per-filter `applied` flags inside each nested dictionary
    - Updated filter metadata documentation for `to_mne_raw()` behavior

- **2.2 (2026-01-30)**: Major documentation update to reflect actual implementation
  - Marked data population methods as private (prefixed with `_`) - should only be called by DataLoader
  - Updated `get_signals()` return signature to tuple of 3 items: (time_vector, channel_names, data)
  - Clarified ET column naming: `ET_ch_x`, `ET_ch_y`, `ET_ch_pupil`, `ET_ch_blinks` for child
  - Added separate sections for private (DataLoader-only) and public methods
  - Added comprehensive DataLoader usage section
  - Reorganized methods into logical groups: Data Population (Private), Data Retrieval (Public), Export/Visualization (Public)
  - Added warning against direct use of private methods
  - Updated example usage to show recommended DataLoader approach
  - Clarified that `_decimate_signals()` is private and called by DataLoader
  - Enhanced event column documentation to distinguish `events`, `EEG_events`, and `ET_event`
  - Changed `events` from `list[dict]` to `dict[str, dict]` with event names as keys
  - Added filter coefficient fields to `Filtration` dataclass (notch_a, notch_b, low_pass_a, low_pass_b, high_pass_a, high_pass_b)
  - Renamed `set_events_column()` to `set_EEG_events_column()` for clarity
  - Added `get_eeg_data()` static method for extracting EEG from any DataFrame
  - Added `get_events_as_marker_channel()` for event marker channel export
  - Added `create_events_column()` for merging EEG_events and ET_event into unified events column
  - Added `create_event_structure()` for creating events dictionary from DataFrame
  - Added `print_events()` for formatted event display
  - Added `to_mne_raw()` for MNE Raw object export with event annotations
  - Enhanced `interpolate_ibi_signals()` documentation to clarify return values and DataFrame merging
  - Enhanced `decimate_signals()` documentation with complete metadata copying details
  - Removed `align_time_to_first_event()` (not implemented in current code)

- **2.1 (2026-01-01)**: Enhanced signal retrieval and decimation methods
  - Added comprehensive `get_signals()` method for flexible signal retrieval with event/time filtering
  - Enhanced `decimate_signals()` documentation to clarify return behavior and NaN handling
  - Improved method documentation with detailed parameter and return value descriptions

- **2.0 (2025-12-30)**: Updated to reflect DataFrame-based architecture
  - Changed from separate arrays to unified DataFrame storage
  - Unified sampling frequency (`fs` instead of separate `eeg_fs`, `ecg_fs`, `ibi_fs`)
  - Updated filtration structure (`eeg_filtration` with `notch_Q` and `notch_freq`)
  - Added modalities tracking
  - Updated column naming conventions

- **1.0 (2025-10-19)**: Initial specification with separate data structures
