import numpy as np


class TimeWindow:
    """Stores the most recent n_samples of multichannel data, and corresponding time stamps. Updating with a new
    set of measurements is efficient, since only one row in the data array (self._data) is overwritten."""
    def __init__(self, n_samples=100, n_cols=0, dtype=float):  # , dtypes=np.empty(0), columns=None):
        """

        Args:
            n_samples: Number of samples in the time window.
            n_cols: Number of channels.
            dtype: Data type (float, complex, etc.).
        """

        self.n_samples = n_samples
        self.n_cols = n_cols
        self.n_channels = n_cols

        self._data = np.zeros((self.n_samples, self.n_cols), dtype=dtype)
        self._time = np.zeros(self.n_samples, dtype=float)*np.nan

        self._k = 0  # This is the index of the row to be overwritten on next update.

    def get(self, col_idx=slice(None)):
        """Get time vector and data array. If col_idx is specified, only the corresponding columns are returned. If not
        specified, all columns are returned.

        Args:
            col_idx: Indices of data columns to be returned.

        Returns:
            Time vector and data array.

        """
        return self.get_time(), self.get_col(col_idx)

    def get_time(self):
        """Get time vector.
        Returns:
            Time vector.

        """
        return np.roll(self._time, -self._k)

    def get_col(self, col_idx=slice(None)):
        """
        Get data columns.  If col_idx is specified, only the corresponding columns are returned. If not
        specified, all columns are returned.
        Args:
            col_idx: Indices of data columns to be returned.

        Returns:
            Data array.

        """
        return np.roll(self._data[:, col_idx], -self._k, axis=0)

    def append(self, new_time, new_data):
        """
        Append a new set of measurement data, and corresopnding time stap.
        Args:
            new_time: Time stamp. Single number.
            new_data: Row of measurement data.

        Returns:

        """
        self._time[self._k] = new_time
        self._data[self._k] = new_data
        self._k = (self._k + 1) % self.n_samples


if __name__ == '__main__':
    tw = TimeWindow(n_cols=10)

    tw.append(0, np.arange(10))

    print(tw.get_time())
    print(tw.get_col())