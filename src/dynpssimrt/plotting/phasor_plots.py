from dynpssimrt.interfacing import InterfacerQueuesThread
from .components.phasors import PhasorPlot
import numpy as np


class VoltagePhasorPlot(InterfacerQueuesThread):
    def __init__(self, rts=None, update_freq=25, subtract_mean=True, *args, **kwargs):
        self.update_freq = update_freq
        self.subtract_mean = subtract_mean
        InterfacerQueuesThread.__init__(self, rts, fs=update_freq,)


    @staticmethod
    def get_init_data(rts):
        return rts.ps.n_bus, rts.ps.v_0

    def initialize(self, init_data):
        n_phasors, v_0 = init_data
        self.angles_prev = np.angle(v_0)
        self.angles_prev = np.unwrap(self.angles_prev)

        self.phasor_plot = PhasorPlot(n_phasors, update_freq=self.update_freq)

    @staticmethod
    def read_input_signal(rts):
        return rts.ps.red_to_full.dot(rts.sol.v)

    def update(self, input):
        phasors = input
        angle = np.angle(input)
        angle = np.unwrap(np.vstack([self.angles_prev, angle]), axis=0)[1, :]
        self.angles_prev[:] = angle

        if self.subtract_mean:
            angle -= np.mean(angle)
        phasors_rot = abs(phasors)*np.exp(1j*angle)
        # Update internal states
        self.phasor_plot.update(phasors_rot)


class GenPhasorPlot(VoltagePhasorPlot):
    @staticmethod
    def get_init_data(rts):
        gen_mdl = rts.ps.gen['GEN']
        angle = rts.sol.x[gen_mdl.state_idx_global['angle']]
        return angle

    def initialize(self, init_data):
        angle = init_data

        self.angles_prev = angle
        self.angles_prev = np.unwrap(self.angles_prev)
        n_phasors = len(angle)
        self.phasor_plot = PhasorPlot(n_phasors, update_freq=self.update_freq)

    @staticmethod
    def read_input_signal(rts):
        gen_mdl = rts.ps.gen['GEN']
        angle = rts.sol.x[gen_mdl.state_idx_global['angle']]
        magnitude = gen_mdl.E_f(rts.sol.x, rts.sol.v)
        return angle, magnitude

    def update(self, input):
        # Update internal states
        angle, magnitude = input
        angle -= np.mean(angle)
        phasors = magnitude*np.exp(1j*angle)
        self.phasor_plot.update(phasors)