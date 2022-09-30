from .time_window_plot import TimeWindowPlot
from .interfacing import InterfacerQueuesThread
import numpy as np


class RTSimPlot(TimeWindowPlot, InterfacerQueuesThread):
    def __init__(self, rts=None, n_samples=100, update_freq=25, *args, **kwargs):
        self.n_samples = n_samples
        self.update_freq = update_freq
        InterfacerQueuesThread.__init__(self, rts)

    @staticmethod
    def get_init_data(rts):
        return len(rts.sol.x)

    def initialize(self, init_data):
        n_cols = init_data
        TimeWindowPlot.__init__(self, n_samples=self.n_samples, n_cols=n_cols, update_freq=self.update_freq)

    @staticmethod
    def read_input_signal(rts):
        return rts.sol.t, rts.sol.x

    def update(self, input):
        # Update internal states
        t, y = input
        self.append(t, y)


class SyncPlot(RTSimPlot):
    def __init__(self, rts=None, n_samples=100, update_freq=25):
        self.n_samples = n_samples
        self.update_freq = update_freq
        InterfacerQueuesThread.__init__(self, rts, name='SyncPlot')

    @staticmethod
    def get_init_data(rts):
        return []

    def initialize(self, init_data):
        TimeWindowPlot.__init__(self, n_samples=self.n_samples, n_cols=1, update_freq=self.update_freq)

    @staticmethod
    def read_input_signal(rts):
        return rts.sol.t, rts.dt_err