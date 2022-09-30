import dynpssimpy.real_time_sim.interfacing as rts_if
import dynpssimpy.real_time_sim.utils as rts_utils
import numpy as np



class SimpleTestInterfacer(rts_if.InterfacerQueuesProcess):
    def __init__(self, name='SimpleTestInterfacer', *args, **kwargs):
        super().__init__(name=name, fs=None, *args, **kwargs)
        self.interface_name = name

    @staticmethod
    def read_input_signal(rts):
        # Specify how input signal is read in RealTimeSimulator
        return [rts.sol.t, rts.sol.x]

    @staticmethod
    def apply_ctrl_signal(rts, ctrl_signal):
        # Specify how control signal is applied in RealTimeSimulator
        rts.ps.y_bus_red_mod[0, 0] = ctrl_signal

    def generate_ctrl_signal(self):
        # Generate control signal from internal states
        return np.random.randn(1)*1e-1  # + np.sin(time.time()*2*np.pi)*0.5

    def update(self, input_signal):
        # Update internal states based on input signal
        self.input_signal = input_signal


class Linearizer:
    def __init__(self, rts, name='Linearizer', freq=1):
        self.interface_name = name
        rts.interface_functions[self.interface_name] = self.interface_fun
        self.lin = rts.ps.linearize()
        self.lin.eigenvalue_decomposition()
        self.freq = freq
        self.timer = 0

    def interface_fun(self, rts):
        t = rts.sol.t
        if t > self.timer:
            self.timer += 1/self.freq
            self.lin.linearize(t0=rts.sol.t, x0=rts.sol.x)
            self.lin.eigenvalue_decomposition()
            # self.lin.eigenvalue_decomposition()


class Logger(rts_if.InterfacerQueuesThread):
    log_attributes = ['t', 'dt_loop', 'dt_ideal', 'dt_sim', 'dt_err']
    def __init__(self, name='Logger', n_samples=100, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.ts_keeper = rts_utils.TimeSeriesKeeper(self.log_attributes, n_samples)

    @classmethod
    def read_input_signal(cls, rts):
        # Specify how input signal is read in RealTimeSimulator
        return [rts.sol.t, *[getattr(rts, attr) for attr in cls.log_attributes[1:]]]

    def update(self, input_signal):
        # Update internal states based on input signal
        if not np.isclose(self.ts_keeper['t'][-1], input_signal[0]):
            self.ts_keeper.append(input_signal)
