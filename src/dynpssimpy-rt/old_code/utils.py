import threading
import multiprocessing
import numpy as np


class TimeSeriesKeeper:
    def __init__(self, columns, n_samples=100, dtypes=np.empty(0)):

        self.n_samples = n_samples
        self.columns = columns
        # self.data = np.zeros((0, len(columns)))
        self._k = 0

        n_col = len(self.columns)
        if dtypes.shape[0] == 0:
            dtypes = [float]*n_col

        entries = ([(0,)*n_col, ]*self.n_samples)
        self.data = np.array(entries, dtype=[*zip(self.columns, dtypes)])

    def __getitem__(self, key):
        return np.roll(self.data[key], -self._k)

    def append(self, new_data):
        self.data[self._k] = tuple(new_data)
        self._k = (self._k + 1) % self.n_samples