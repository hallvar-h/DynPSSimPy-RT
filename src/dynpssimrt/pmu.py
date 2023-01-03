import sys
import numpy as np
from synchrophasor.simplePMU import SimplePMU
from .interfacing import InterfacerQueuesThread


class PMUPublisher(InterfacerQueuesThread):
    def __init__(self, rts=None, ip='10.0.0.16', port=1410, name='PMUPublisher', fs=100, set_timestamp=False, publish_frequency=10, *args, **kwargs):
        self.ip = ip
        self.port = port
        fs = publish_frequency
        super().__init__(rts=rts, name=name, fs=fs, *args, **kwargs)
        # ps = rts.ps

    @staticmethod
    def get_init_data(rts):
        station_names = list(rts.ps.buses['name'])
        channel_names = [['Ph']] * len(station_names)
        return [station_names, channel_names]

    def initialize(self, init_data):
        station_names, channel_names = init_data
        self.pmu = SimplePMU(
            self.ip, self.port,
            station_names=station_names,
            channel_names=channel_names,
            set_timestamp=False,
            publish_frequency=self.fs,
        )

    def start(self):
        super().start()
        self.pmu.run()

    @staticmethod
    def read_input_signal(rts):
        # Specify how input signal is read in RealTimeSimulator
        v_full = rts.ps.red_to_full.dot(rts.sol.v)
        # rts.result_stream_pmu.put([rts.sol.t, v_full])
        return [rts.sol.t, v_full]

    def update(self, input_signal):
        if self.pmu.pmu.clients:  # Check if there is any connected PDCs
            t, v = input_signal

            time_stamp = round(t * 1e3) * 1e-3

            # Publish C37.118-snapshot
            pmu_data = [[(mag, ang)] for mag, ang in zip(np.abs(v), np.angle(v))]
            self.pmu.publish(time_stamp, pmu_data)