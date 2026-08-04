import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
from plotly.subplots import make_subplots
import pandas as pd

    

def plot_eeg_channels_pl(mmd, selected_events, selected_channels, title='Filtered EEG Channels', renderer='auto'):
    """
    Plot the filtered EEG channels with events highlighted using Plotly.
    Replicates the matplotlib version with vertical offsets in a single plot.
    Features interactive hover information and zooming capabilities.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing EEG data, channels, and time vectors
    events : dict
        Dictionary containing event timings
    selected_channels : list
        List of channel names to plot
    title : str, optional
        Title for the plot (default: 'Filtered EEG Channels')
    renderer : str, optional
        Plotly renderer to use: 'auto', 'browser', 'notebook', 'html' (default: 'auto')
    """
    colors = ['red', 'green', 'blue', 'orange', 'purple']  # colors for different events

    # Create a single figure
    fig = go.Figure()

    offset = 0
    spacing = 200  # vertical spacing between channels
    y_ticks = []
    y_tick_labels = []

    # Plot each channel with vertical offset
    for i, ch in enumerate(selected_channels):
        x_ch= mmd.data[ch].values
        # clip the amplitudes
        x_ch = np.clip(x_ch, -100, 100)

        # Add trace for this channel with offset
        fig.add_trace(go.Scatter(
            x=mmd.data.time,
            y=x_ch + offset,
            mode='lines',
            name=ch,
            line={'width': 1},
            showlegend=True
        ))

        y_ticks.append(offset)
        y_tick_labels.append(ch)
        offset += spacing

    # Add event highlights as vertical rectangles spanning all channels based on 'events' column
    # Group consecutive rows with the same event to create continuous background regions
    event_color_map = {}
    
    # Get unique events and assign colors
    unique_events = mmd.data['events'].dropna().unique()
    for i, event in enumerate(unique_events):
        color_idx = i % len(colors)
        event_color_map[event] = colors[color_idx]
    
    # Find continuous segments using vectorised diff instead of row-by-row loop
    events_series = mmd.data['events']
    time_series = mmd.data['time']
    # Detect where event value changes, but treat consecutive NaNs as a single segment
    value_changed = events_series.ne(events_series.shift())
    is_na = events_series.isna()
    shifted_is_na = is_na.shift(fill_value=False)
    # Do not mark positions where both current and previous values are NaN as changes
    changed = value_changed & ~(is_na & shifted_is_na)
    change_idx = np.flatnonzero(changed.values)
    # Append sentinel for the last segment
    change_idx = np.append(change_idx, len(events_series))

    for seg_i in range(len(change_idx) - 1):
        start_idx = change_idx[seg_i]
        end_idx = change_idx[seg_i + 1] - 1
        ev = events_series.iat[start_idx]
        if pd.notna(ev) and ev in event_color_map:
            fig.add_vrect(
                x0=time_series.iat[start_idx],
                x1=time_series.iat[end_idx],
                fillcolor=event_color_map[ev],
                opacity=0.2,
                layer="below",
                line_width=0
            )
    
    # Add legend entries for events
    for event, color in event_color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker={'size': 10, 'color': color},
            name=f'{event}',
            showlegend=True
        ))

    #event_colors_used.append(color_idx)

    # Update layout to match matplotlib appearance with enhanced interactivity
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'font': {'size': 16}},
        xaxis_title="Time [s]",
        yaxis_title="EEG Channels",
        yaxis={'tickvals': y_ticks, 'ticktext': y_tick_labels, 'showgrid': False, 'zeroline': False},
        xaxis={'showgrid': True, 'gridwidth': 1, 'gridcolor': 'lightgray'},
        height=600,
        width=1200,
        showlegend=True,
        legend={'orientation': "v", 'yanchor': "top", 'y': 1, 'xanchor': "left", 'x': 1.01, 'font': {'size': 10}},
        plot_bgcolor='white'
    )

    # Add range selector for time navigation
    fig.update_layout(
        xaxis={'rangeselector': {'buttons': [
            {'count': 30, 'label': "30s", 'step': "second", 'stepmode': "backward"},
            {'count': 60, 'label': "1m", 'step': "second", 'stepmode': "backward"},
            {'count': 300, 'label': "5m", 'step': "second", 'stepmode': "backward"},
            {'step': "all"}
        ]}, 'rangeslider': {'visible': True, 'thickness': 0.05}, 'type': "linear"}
    )

    # Show the figure based on renderer preference
    if renderer == 'html':
        save_figure_to_html(fig, title)
    elif renderer == 'auto':
        # Try different renderers in order of preference
        try:
            fig.show(renderer="browser")
        except Exception as e:
            print(f"Failed to display with renderer 'browser': {e}")
            try:
                fig.show(renderer="notebook")
            except Exception as e:
                print(f"Failed to display with renderer 'notebook': {e}")
                save_figure_to_html(fig, title)
    else:
        # Use specified renderer
        try:
            fig.show(renderer=renderer)
        except Exception as e:
            print(f"Failed to display with renderer '{renderer}': {e}")
            save_figure_to_html(fig, title)


def overlay_eeg_channels_hyperscanning_pl(data_ch, data_cg, 
                                          event,
                                          selected_channels_ch,
                                          selected_channels_cg, 
                                          title='Filtered EEG Channels - Hyperscanning',
                                          renderer='auto'):
    """
    Plot child and caregiver EEG channels for hyperscanning analysis using Plotly.
    Creates two subplots: one for child channels and one for caregiver channels.
    
    Parameters:
    -----------
    data_ch : numpy.ndarray
        Child EEG data (channels x samples)
    data_cg : numpy.ndarray  
        Caregiver EEG data (channels x samples)
    event : str
        Name of the event being plotted
    selected_channels_ch : list
        List of selected child channel names
    selected_channels_cg : list
        List of selected caregiver channel names
    title : str, optional
        Title for the plot (default: 'Filtered EEG Channels - Hyperscanning')
    renderer : str, optional
        Plotly renderer to use: 'auto', 'browser', 'notebook', 'html' (default: 'auto')
    """

    # Create subplots with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Child EEG channels for {event}', f'Caregiver EEG channels for {event}'),
        shared_xaxes=True,
        vertical_spacing=0.1
    )

    # Color palette for channels
    colors = px.colors.qualitative.Set3

    # Plot child EEG channels
    for i, ch in enumerate(selected_channels_ch):
        color_idx = i % len(colors)
        fig.add_trace(
            go.Scatter(
                x=list(range(data_ch.shape[1])),
                y=data_ch[:,i],
                mode='lines',
                name=ch,
                line={'color': colors[color_idx], 'width': 1.5},
                legendgroup='child',
                legendgrouptitle={'text': "Child Channels"}
            ),
            row=1, col=1
        )

    # Plot caregiver EEG channels  
    for i, ch in enumerate(selected_channels_cg):
        color_idx = i % len(colors)
        fig.add_trace(
            go.Scatter(
                x=list(range(data_cg.shape[1])),
                y=data_cg[:,i],
                mode='lines',
                name=ch,
                line={'color': colors[color_idx], 'width': 1.5},
                legendgroup='caregiver',
                legendgrouptitle={'text': "Caregiver Channels"}
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'font': {'size': 16}},
        height=800,
        width=1200,
        showlegend=True,
        legend={'orientation': "v", 'yanchor': "top", 'y': 1, 'xanchor': "left", 'x': 1.01, 'font': {'size': 10},
                'groupclick': "toggleitem"},
        plot_bgcolor='white'
    )

    # Update axes labels
    fig.update_xaxes(title_text="Samples", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (µV)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude (µV)", row=2, col=1)

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Show the figure based on renderer preference
    if renderer == 'html':
        save_figure_to_html(fig, title, event)
    elif renderer == 'auto':
        # Try different renderers in order of preference
        try:
            fig.show(renderer="browser")
        except Exception as e:
            print(f"Failed to display with renderer 'browser': {e}")
            try:
                fig.show(renderer="notebook")
            except Exception as e:
                print(f"Failed to display with renderer 'notebook': {e}")
                save_figure_to_html(fig, title, event)
    else:
        # Use specified renderer
        try:
            fig.show(renderer=renderer)
        except Exception as e:
            print(f"Failed to display with renderer '{renderer}': {e}")
            save_figure_to_html(fig, title, event)




# ==================================

def save_figure_to_html(fig, title, event=None):
    html_file = f"{title.replace(' ', '_').replace('(', '').replace(')', '').lower()}{'' if event is None else '_' + event}.html"
    fig.write_html(html_file)
    print(f"Plot saved as HTML file: {os.path.abspath(html_file)}")
    print("Open this file in your web browser to view the interactive plot.")

# Outdated functions - commented out for potential future reference
# def get_ibi_signal_from_ecg_for_selected_event(filtered_data, events, selected_event):
#     """Get IBI signal from ECG data for a specific event.
#     Args:
#         filtered_data (dict): Filtered data with the structure returned by filter_warsaw_pilot_data function.
#         events (dict): Dictionary containing the start and end time of detected events.
#         selected_event (str): Name of the event to extract IBI signal for
#     Returns:
#         IBI_ch_interp (np.ndarray): Interpolated IBI signal for the child.
#         IBI_cg_interp (np.ndarray): Interpolated IBI signal for the caregiver.
#         t_ECG (np.ndarray): Time vector for the interpolated IBI signal.
#     """
#     if selected_event not in events:
#         raise ValueError(f"Event '{selected_event}' not found in events dictionary.")
#     ibi_ch_interp = filtered_data['IBI_ch_interp']
#     ibi_cg_interp = filtered_data['IBI_cg_interp']
#     t_ibi = filtered_data['t_IBI']
#     t_idx = events[selected_event]  # get the time of the event in the data
#     if t_idx is not None:
#         # extract 60 seconds after the event
#         # find the index in t_IBI
#         start_idx = np.searchsorted(t_ibi, t_idx)  # find the index in t_IBI
#         end_idx = start_idx + int(60 * filtered_data['Fs_IBI'])  # extract 60 seconds after the event
#         # check if the start and end indices are within the bounds of the data
#         if start_idx < 0 or end_idx > filtered_data['data'].shape[1]:
#             raise ValueError(f"Event '{selected_event}' is out of bounds.")
#     else:
#         raise ValueError(f"Event '{selected_event}' is None.")

#     # cut the IBI signal of the selected event
#     ibi_ch_interp = ibi_ch_interp[start_idx:end_idx]
#     ibi_cg_interp = ibi_cg_interp[start_idx:end_idx]
#     t_ibi = t_ibi[start_idx:end_idx]

#     return ibi_ch_interp, ibi_cg_interp, t_ibi


# def get_data_for_selected_channel_and_event(filtered_data, selected_channels, events, selected_event):
#     """Get data for selected channels and event from the filtered data.
#     Args:
#         filtered_data (dict): Filtered data with the structure returned by filter_warsaw_pilot_data function.
#         selected_channels (list): List of channel names to extract data for.
#         events (dict): Dictionary containing the start and end time of detected events.
#         selected_event (str): Name of the event to extract data for.
#     Returns:
#         data_selected (np.ndarray): Data array with the shape (N_samples, N_channels) for the selected channels and event.
#     """
#     if selected_event not in events:
#         raise ValueError(f"Event '{selected_event}' not found in events dictionary.")
#     idx = events[selected_event]
#     if idx is not None:
#         # extract 60 seconds after the event
#         data_selected = np.zeros((len(selected_channels), int(60 * filtered_data['Fs_EEG'])))
#         start_idx = int(idx * filtered_data['Fs_EEG'])  # convert the event time to the index in the filtered data
#         end_idx = start_idx + int(60 * filtered_data['Fs_EEG'])  # extract 60 seconds after the event
#         # check if the start and end indices are within the bounds of the data
#         if start_idx < 0 or end_idx > filtered_data['data'].shape[1]:
#             raise ValueError(f"Event '{selected_event}' is out of bounds.")
#     else:
#         raise ValueError(f"Event '{selected_event}' is None.")
#     for i, ch in enumerate(selected_channels):
#         if ch in filtered_data['channels']:
#             idx_ch = filtered_data['channels'][ch]
#             data_selected[i, :] = filtered_data['data'][idx_ch, start_idx:end_idx]
#     return data_selected


# def clean_data_with_ica(data, selected_channels, event):
#     """Clean data with ICA to remove artifacts.
#     Args:
#         data (np.ndarray): Data array with the shape (N_channels, N_samples) for the selected channels and event.
#         selected_channels (list): List of channel names to extract data for.
#         event (str): Name of the event to extract data for.
#     Returns:
#         data_cleaned (np.ndarray): Cleaned data array with the shape (N_channels, N_samples) for the selected channels and event.
#     """
#     ica = FastICA(n_components=len(selected_channels), max_iter=1000, whiten="unit-variance")
#     ica_components = ica.fit_transform(data.T)  # get components

#     _, ax = plt.subplots(len(selected_channels), 1, figsize=(12, 8), sharex=True)
#     for i, ch in enumerate(selected_channels):
#         ax[i].plot(ica_components[:, i])
#         ax[i].set_ylabel(ch)
#     plt.tight_layout()
#     plt.show()
#     idx_to_remove = input(f'Event {event}: select components to remove and press Enter to continue...  ')
#     if idx_to_remove != '':
#         idx_to_remove = [int(i) for i in idx_to_remove.split(',')]
#         ica_components[:, idx_to_remove] = 0  # set the selected components to zero
#         print('Selected components to remove: ', idx_to_remove)
#     data_cleaned = ica.inverse_transform(ica_components).T  # reconstruct the data from the components
#     return data_cleaned