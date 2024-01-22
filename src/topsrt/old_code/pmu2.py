import sys  # We need sys so that we can pass argv to QApplication
import numpy as np
import time
import threading
import importlib
import tops.dynamic as dps
# import tops.real_time_sim as dps_rts
# from synchrophasor.frame import ConfigFrame2, DataFrame
# from synchrophasor.pmu import Pmu, PmuError
# import time
# import random  # Can be removed?
# import tops.real_time_sim.utils as rts_utils
# import tops.real_time_sim.interfacing as rts_if
# from time import sleep, time
# from src.pypmu_fix.frame import *
# import collections
# import synchrophasor.frame as pypmu_frame

# sys.path.append(pypmu_path)
# sys.path.insert(1, pypmu_path)
# import synchrophasor
# synchrophasor

pypmu_path = r'C:\Users\hallvarh\OneDrive - SINTEF\Projects\NEWEPS\Code\pypmu'
sys.path.insert(1, pypmu_path)
from synchrophasor.simplePMU import SimplePMU

# import logging
# import socket
# from synchrophasor.pmu import Pmu, PmuError
# from select import select
# from threading import Thread
# from multiprocessing import Queue, Event
# from multiprocessing import Process
# from sys import stdout
# from time import sleep, time
# # from src.pypmu_fix.frame import *
# import collections
# import queue


class PMUPublisher(rts_if.InterfacerQueuesThread):
    def __init__(self, rts, ip='10.0.0.16', port=1410, name='PMUPublisher', fs=100, set_timestamp=False, publish_frequency=10, *args, **kwargs):
        fs = publish_frequency
        super().__init__(rts=rts, name=name, fs=fs, *args, **kwargs)
        ps = rts.ps
        station_names = list(ps.buses['name'])
        channel_names = [['Ph']]*len(station_names)

        self.pmu = SimplePMU(
            ip, port,
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


if __name__ == '__main__':

    # Test SimplePMU:
    # pmu = SimplePMU()
    # pmu.run()
    #
    # for i in range(10):
    #     time.sleep(1)
    #     pmu.publish()
    # time.sleep(2)

    importlib.reload(dps)

    # import ps_models.n44 as model_data
    import ps_models.k2a as model_data
    # import ps_models.sm_ib as model_data

    model = model_data.load()

    importlib.reload(dps)
    ps = dps.PowerSystemModel(model=model)
    # ps.use_numba = True

    # Note that running this in interactive mode in PyCharm with Python 3.8 or 3.9
    # does not work. Multiprocessing Pool error? Works with Python 3.6.
    # Also, workaround for 3.8/3.9 is to set __file__ variable to an actual file.
    __file__ = 'tops/pmu_publisher.py'

    ps.power_flow()
    ps.build_y_bus_red(ps.buses['name'])
    ps.init_dyn_sim()
    ps.ode_fun(0, ps.x0)
    # ps.x0[ps.ip][0] += 1e-3
    rts = dps_rts.RealTimeSimulatorThread(ps, dt=5e-3, speed=0.1)
    rts.start()

    pmus = PMUPublisher(rts, publish_frequency=10, phasors=['v_g'])
    pmus.start()

    # print(rts.is_alive())
    # from threading import Thread
    # app, main = main(rts)

    # import matplotlib.pyplot as plt
    # plt.show()