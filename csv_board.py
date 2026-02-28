import numpy as np
import pandas as pd
import time

class CSVBoard:
    """Drop-in replacement for board_shim that replays a recorded CSV file."""

    def __init__(self, csv_path, num_channels=8, sampling_rate=500):
        df = pd.read_csv(csv_path, sep='\t')
        # Use raw channels (Channel1..Channel8)
        cols = [f"Channel{i+1}" for i in range(num_channels)]
        self._data = df[cols].values.T  # shape: (num_channels, total_samples)
        self._total = self._data.shape[1]
        self._cursor = 0
        self._chunk_size = 0          # set by get_board_data_count callers
        self._sampling_rate = sampling_rate

    def get_board_data_count(self):
        # Always report enough data available
        return 9999

    def get_board_data(self, num_samples):
        end = self._cursor + num_samples
        if end > self._total:
            # Loop back to start
            self._cursor = 0
            end = num_samples
        chunk = self._data[:, self._cursor:end]
        self._cursor = end
        return chunk  # shape: (num_channels, num_samples)