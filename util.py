import numpy as np
import time

def record_emg(board_shim, samples_per_point, num_channels, sampling_rate):
    """
    Record an EMG window from the board shim.
    Returns: (rotation, window)
    - rotation: str, e.g., "Left", "Right", or "None" (placeholder logic)
    - window: np.ndarray of shape (samples_per_point, num_channels)
    """
    # Collect data for the window
    data = board_shim.get_board_data(sampling_rate)
    if data is None or data.shape[1] < samples_per_point:
        # Fallback if not enough data
        window = np.zeros((samples_per_point, num_channels), dtype=np.float32)
        rotation = "None"
    else:
        # Take the last samples_per_point samples
        window = data[:, -samples_per_point:].T  # Shape: (samples_per_point, num_channels)
        # Placeholder for rotation detection (e.g., based on EMG patterns)
        # For now, randomly assign or set to "None" - replace with actual logic
        rotation = "None"  # TODO: Implement rotation detection
    
    return rotation, window