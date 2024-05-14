import streamlit as st

def convert(seconds):
    """
    Convert seconds to HH:MM:SS format.

    Args:
        seconds (int): Total number of seconds.

    Returns:
        str: Time in HH:MM:SS format.
    """
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)

def backward_moving_avg(samples, start, window):
    """
    Calculate backward moving average of samples.

    Args:
        samples (list): List of samples.
        start (int): Start index for calculating moving average.
        window (int): Size of the window for calculating moving average.

    Returns:
        float: Backward moving average.
    """
    if len(samples) < window:
        return 0
    assert start <= len(samples)
    return sum(samples[start-window:start]) / window

def apply_threshold(samples, threshold):
    """
    Apply threshold to samples.

    Args:
        samples (list): List of samples.
        threshold (float): Threshold value.

    Returns:
        int: 1 if samples exceed threshold, else 0.
    """
    return 1 if samples[-1] >= threshold else 0

def timestamps_func(arg0, arg1):
    """
    Generate and display timestamps.

    Args:
        arg0 (list): List of timestamps.
        arg1 (str): Additional string for display.

    Returns:
        None
    """
    start_timestamp = list(set(arg0))[-1][0]
    end_timestamp = list(set(arg0))[-1][1]
    st.subheader(f"{arg1}{start_timestamp} to {end_timestamp}")

def GetTimestamps(frame_list, prediction_list):
    """
    Generate timestamps based on frame list and prediction list.

    Args:
        frame_list (list): List of frame numbers.
        prediction_list (list): List of predictions.

    Returns:
        list: List of tuples containing start and end timestamps.
    """
    timestamps = []
    
    for frame, prediction in zip(frame_list, prediction_list):
        if prediction == 1:
            move_starting_frame = frame
            move_ending_frame = frame + (30 * 3)
            timestamps.append((convert(move_starting_frame / 30), convert((move_ending_frame) / 30)))
    
    return timestamps

def smoothen(samples, window=5, continuous_positives_window=3, threshold=0.8):
    """
    Smooth the samples using backward moving average and continuous positives window.

    Args:
        samples (list): List of samples.
        window (int): Length of window for label smoothening. Default is 5.
        continuous_positives_window (int): Length of window that should have all positive labels. Default is 3.
        threshold (float): Threshold value for smoothening. Default is 0.8.

    Returns:
        list: Smoothed samples.
    """
    windowed_samples, smoothed_samples = [], []

    for i in range(len(samples)):
        window_avg = backward_moving_avg(samples, start=i + 1, window=window)
        windowed_samples.append(window_avg)
        smoothed_samples.append(apply_threshold(windowed_samples, threshold))

    return smoothed_samples

def operation(frame_list, prediction_list, window=5, continuous_positives_window=3, threshold=0.8, debug=False):
    """
    Perform operation using frame list and prediction list.

    Args:
        frame_list (list): List of frame numbers.
        prediction_list (list): List of predictions.
        window (int): Length of window for label smoothening. Default is 5.
        continuous_positives_window (int): Length of window that should have all positive labels. Default is 3.
        threshold (float): Threshold value for smoothening. Default is 0.8.
        debug (bool): Debug flag. Default is False.

    Returns:
        list: List of timestamps.
    """
    continuous_smoothed = smoothen(prediction_list, window,
                                    threshold=threshold,
                                    continuous_positives_window=continuous_positives_window
                                    )
    timestamps = GetTimestamps(frame_list, continuous_smoothed)

    return timestamps
