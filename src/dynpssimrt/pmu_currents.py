import sys
import numpy as np
from synchrophasor.simplePMU import SimplePMU
from .pmu import PMUPublisher


class PMUPublisherCurrents(PMUPublisher):

    @staticmethod
    def get_init_data(rts):
        # station_names = list(rts.ps.buses['name'])
        # channel_names = [['V']] * len(station_names)

        return [rts.ps.buses, rts.ps.lines['Line'].par]

    def initialize(self, init_data):
        bus_data, line_data = init_data
        station_names = list(bus_data['name'])        
        channel_names = []
        channel_types = []
        self.masks_from = []
        self.masks_to = []
        for station_name in station_names:
            bus_idx_from = np.where(line_data['from_bus'] == station_name)[0]
            bus_idx_to = np.where(line_data['to_bus'] == station_name)[0]
            names_from = [f'I[{name}]' for name in line_data['name'][bus_idx_from]]
            names_to = [f'I[{name}]-r' for name in line_data['name'][bus_idx_to]]
            channel_types.append(['v', *['i']*(len(names_from) + len(names_to))])
            channel_names.append(['V', *names_from, *names_to])
            self.masks_from.append(bus_idx_from)
            self.masks_to.append(bus_idx_to)
            
        # station_names, channel_names = init_data
        self.pmu = SimplePMU(
            self.ip, self.port,
            station_names=station_names,
            channel_names=channel_names,
            channel_types=channel_types,
            pdc_id=self.pdc_id,
            set_timestamp=False,
            publish_frequency=self.fs,
        )

    @staticmethod
    def read_input_signal(rts):
        # Specify how input signal is read in RealTimeSimulator
        v_full = rts.ps.red_to_full.dot(rts.sol.v)
        line_currents_from = rts.ps.lines['Line'].i_from(rts.sol.x, rts.sol.v)
        line_currents_to = rts.ps.lines['Line'].i_to(rts.sol.x, rts.sol.v)
        # rts.result_stream_pmu.put([rts.sol.t, v_full])
        return [rts.sol.t, v_full, line_currents_from, line_currents_to]

    @staticmethod
    def complex2pol(vec):
        return [(np.abs(vec_), np.angle(vec_)) for vec_ in vec]

    def update(self, input_signal):
        if self.pmu.pmu.clients:  # Check if there is any connected PDCs
            t, v, line_currents_from, line_currents_to = input_signal

            time_stamp = round(t * 1e3) * 1e-3

            # Remember to make polar!
            phasors = []
            for v_, mask_from, mask_to in zip(v, self.masks_from, self.masks_to):
                v_pol = self.complex2pol([v_])
                i_from_pol = self.complex2pol(line_currents_from[mask_from])
                i_to_pol = self.complex2pol(line_currents_to[mask_to])
                phasors.append([*v_pol, *i_from_pol, *i_to_pol])

            # Publish C37.118-snapshot
            # pmu_data = [[(mag, ang)] for mag, ang in zip(np.abs(v), np.angle(v))]
            self.pmu.publish(time_stamp, phasors)