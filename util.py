import numpy as np

def record_emg(board_shim, samples_per_point, num_channels, sampling_rate):
    return "None", np.zeros((samples_per_point, num_channels), dtype=np.float32)