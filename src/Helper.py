def quantise_beat_duration(beat_duration, grid_size=0.25, max_duration=4.0):
    """
    Quantise a beat duration to the nearest musical note value.

    Args:
        beat_duration (float): The duration of the beat to quantise, in beats.
        grid_size (float, optional): The grid size for quantisation. Defaults to 0.5.
        max_duration (float, optional): The maximum allowed duration. Defaults to 4.0.

    Returns:
        str: The name of the musical note value (e.g., 'crotchet', 'quaver', 'semibreve')
             that best represents the quantised beat duration.
    """
    DURATION_BINS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0]

    quantised = round(beat_duration / grid_size) * grid_size

    if quantised < grid_size:
        quantised = grid_size
    if quantised > max_duration:
        quantised = max_duration

    closest = min(DURATION_BINS, key=lambda x: abs(x-quantised))
    return closest
