import sys  # We need sys so that we can pass argv to QApplication
import numpy as np
import time
import threading
import importlib
import tops.dynamic as dps
import tops.real_time_sim as dps_rts
from synchrophasor.frame import ConfigFrame2, DataFrame
from synchrophasor.pmu import Pmu, PmuError
import time
import random  # Can be removed?
import tops.real_time_sim.utils as rts_utils
import tops.real_time_sim.interfacing as rts_if
from time import sleep, time
# from src.pypmu_fix.frame import *
import collections
import synchrophasor.frame as pypmu_frame

import logging
import socket
from synchrophasor.pmu import Pmu, PmuError
from select import select
from threading import Thread
from multiprocessing import Queue, Event
from multiprocessing import Process
from sys import stdout
from time import sleep, time
# from src.pypmu_fix.frame import *
import collections
import queue




def angle(phi):
    return (phi + np.pi) % (2*np.pi) - np.pi


def set_time(self, soc=None, frasec=None):
    """
    ### set_time() ###

    Setter for ``soc`` and ``frasec``. If values for ``soc`` or ``frasec`` are
    not provided this method will calculate them.

    **Params:**

    * ``soc`` **(int)** - UNIX timestamp, 32-bit unsigned number. See ``set_soc()``
    method.
    * ``frasec`` **(int)** or **(tuple)** - Fracion of second and Time Quality. See
    ``set_frasec`` method.

    **Raises:**

        FrameError
    When ``soc`` value provided is out of range.

    When ``frasec`` is not valid.

    """

    t = time()  # Get current timestamp

    if soc is not None:
        self.set_soc(soc)
    else:
        self.set_soc(int(t))  # Get current timestamp

    if frasec is not None:
        if isinstance(frasec, collections.Sequence):
            self.set_frasec(*frasec)
        else:
            self.set_frasec(frasec)  # Just set fraction of second and use default values for other arguments.
    else:
        # Calculate fraction of second (after decimal point) using only first 7 digits to avoid
        # overflow (24 bit number).
        self.set_frasec(int((((repr((t % 1))).split("."))[1])[0:6]))

pypmu_frame.DataFrame.set_time = set_time


class PmuFix(Pmu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stopped = False
        self.client_stop_events = []

    def stop(self):
        self._stopped = True

    def send_data(self, phasors=[], analog=[], digital=[], freq=0, dfreq=0,
                  stat=("ok", True, "timestamp", False, False, False, 0, "<10", 0), soc=None, frasec=None):

        # PH_UNIT conversion
        if phasors and self.cfg2.get_num_pmu() > 1:  # Check if multistreaming:
            if not (self.cfg2.get_num_pmu() == len(self.cfg2.get_data_format()) == len(phasors)):
                raise PmuError("Incorrect input. Please provide PHASORS as list of lists with NUM_PMU elements.")

            for i, df in enumerate(self.cfg2.get_data_format()):
                if not df[1]:  # Check if phasor representation is integer
                    phasors[i] = map(lambda x: int(x / (0.00001 * self.cfg2.get_ph_units()[i])), phasors[i])
        elif not self.cfg2.get_data_format()[1]:
            phasors = map(lambda x: int(x / (0.00001 * self.cfg2.get_ph_units())), phasors)

        # AN_UNIT conversion
        if analog and self.cfg2.get_num_pmu() > 1:  # Check if multistreaming:
            if not (self.cfg2.get_num_pmu() == len(self.cfg2.get_data_format()) == len(analog)):
                raise PmuError("Incorrect input. Please provide analog ANALOG as list of lists with NUM_PMU elements.")

            for i, df in enumerate(self.cfg2.get_data_format()):
                if not df[2]:  # Check if analog representation is integer
                    analog[i] = map(lambda x: int(x / self.cfg2.get_analog_units()[i]), analog[i])
        elif not self.cfg2.get_data_format()[2]:
            analog = map(lambda x: int(x / self.cfg2.get_analog_units()), analog)

        data_frame = pypmu_frame.DataFrame(self.cfg2.get_id_code(), stat, phasors, freq, dfreq, analog, digital, self.cfg2, soc, frasec)

        for buffer in self.client_buffers:
            buffer.put(data_frame)

    def acceptor(self):

        while True:
            # print('Acceptor running')
            self.logger.info("[%d] - Waiting for connection on %s:%d", self.cfg2.get_id_code(), self.ip, self.port)

            # Accept a connection on the bound socket and fork a child process to handle it.
            # print('Waiting for socket.accept')
            conn, address = self.socket.accept()
            if self._stopped:
                conn.close()
                break
            # print('Not waiting for socket.accept')

            # Create Queue which will represent buffer for specific client and add it o list of all client buffers
            buffer = Queue()
            self.client_buffers.append(buffer)

            stop_event = Event()
            process = Process(target=self.pdc_handler, args=(conn, address, buffer, self.cfg2.get_id_code(),
                                                             self.cfg2.get_data_rate(), self.cfg1, self.cfg2,
                                                             self.cfg3, self.header, self.buffer_size,
                                                             self.set_timestamp, self.logger.level, stop_event))

            process.daemon = True
            process.start()
            self.clients.append(process)
            self.client_stop_events.append(stop_event)

            # Close the connection fd in the parent, since the child process has its own reference.
            conn.close()

    @staticmethod
    def pdc_handler(connection, address, buffer, pmu_id, data_rate, cfg1, cfg2, cfg3, header,
                    buffer_size, set_timestamp, log_level, stop_event):

        # Recreate Logger (handler implemented as static method due to Windows process spawning issues)
        logger = logging.getLogger(address[0] + str(address[1]))
        logger.setLevel(log_level)
        handler = logging.StreamHandler(stdout)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.info("[%d] - Connection from %s:%d", pmu_id, address[0], address[1])

        # Wait for start command from connected PDC/PMU to start sending
        sending_measurements_enabled = False

        # Calculate delay between recorded_pmu_data_raw frames
        data_rate *= 5
        if data_rate > 0:
            delay = 1.0 / data_rate
        else:
            delay = -data_rate

        try:
            while not stop_event.is_set():
                # print('Looping')

                command = None
                received_data = b""
                readable, writable, exceptional = select([connection], [], [], 0)  # Check for client commands
                # print('Check 1')

                if readable:
                    """
                    Keep receiving until SYNC + FRAMESIZE is received, 4 bytes in total.
                    Should get this in first iteration. FRAMESIZE is needed to determine when one complete message
                    has been received.
                    """
                    # print('Check 1-2')
                    while len(received_data) < 4 and not stop_event.is_set():
                        # print('Check 2')
                        received_data += connection.recv(buffer_size)
                        # print('Check 3')
                    # print('Check 4')
                    bytes_received = len(received_data)
                    total_frame_size = int.from_bytes(received_data[2:4], byteorder="big", signed=False)

                    # Keep receiving until every byte of that message is received
                    while bytes_received < total_frame_size and not stop_event.is_set():
                        message_chunk = connection.recv(min(total_frame_size - bytes_received, buffer_size))
                        if not message_chunk:
                            break
                        received_data += message_chunk
                        bytes_received += len(message_chunk)

                    # If complete message is received try to decode it
                    if len(received_data) == total_frame_size:
                        try:
                            received_message = pypmu_frame.CommonFrame.convert2frame(received_data)  # Try to decode received recorded_pmu_data_raw

                            if isinstance(received_message, pypmu_frame.CommandFrame):
                                command = received_message.get_command()
                                logger.info("[%d] - Received command: [%s] <- (%s:%d)", pmu_id, command,
                                            address[0], address[1])
                            else:
                                logger.info("[%d] - Received [%s] <- (%s:%d)", pmu_id,
                                            type(received_message).__name__, address[0], address[1])
                        except pypmu_frame.FrameError:
                            logger.warning("[%d] - Received unknown message <- (%s:%d)", pmu_id, address[0], address[1])
                    else:
                        logger.warning("[%d] - Message not received completely <- (%s:%d)", pmu_id, address[0],
                                       address[1])

                if command:
                    if command == "start":
                        sending_measurements_enabled = True
                        logger.info("[%d] - Start sending -> (%s:%d)", pmu_id, address[0], address[1])

                    elif command == "stop":
                        logger.info("[%d] - Stop sending -> (%s:%d)", pmu_id, address[0], address[1])
                        sending_measurements_enabled = False

                    elif command == "header":
                        if set_timestamp: header.set_time()
                        connection.sendall(header.convert2bytes())
                        logger.info("[%d] - Requested Header frame sent -> (%s:%d)",
                                    pmu_id, address[0], address[1])

                    elif command == "cfg1":
                        if set_timestamp: cfg1.set_time()
                        connection.sendall(cfg1.convert2bytes())
                        logger.info("[%d] - Requested Configuration frame 1 sent -> (%s:%d)",
                                    pmu_id, address[0], address[1])

                    elif command == "cfg2":
                        if set_timestamp: cfg2.set_time()
                        connection.sendall(cfg2.convert2bytes())
                        logger.info("[%d] - Requested Configuration frame 2 sent -> (%s:%d)",
                                    pmu_id, address[0], address[1])

                    elif command == "cfg3":
                        if set_timestamp: cfg3.set_time()
                        connection.sendall(cfg3.convert2bytes())
                        logger.info("[%d] - Requested Configuration frame 3 sent -> (%s:%d)",
                                    pmu_id, address[0], address[1])

                if sending_measurements_enabled and not buffer.empty():

                    data = buffer.get()
                    if isinstance(data, pypmu_frame.CommonFrame):  # If not raw bytes convert to bytes
                        if set_timestamp: data.set_time()
                        data = data.convert2bytes()

                    sleep(delay)
                    connection.sendall(data)
                    logger.debug("[%d] - Message sent at [%f] -> (%s:%d)",
                                 pmu_id, time(), address[0], address[1])

            # print('Client loop finished')
        except Exception as e:
            print(e)
        finally:
            connection.close()
            logger.info("[%d] - Connection from %s:%d has been closed.", pmu_id, address[0], address[1])


# class SimplePMU:
#     def __init__(self, ip, port, publish_frequency=50, n_phasors=1, n_pmus=1, channel_names=[]):
#         # Initialize PMUs
#         self.pmu = PmuFix(ip=ip, port=port, set_timestamp=False, data_rate=publish_frequency)
#         self.pmu.logger.setLevel("DEBUG")
#
#         self.n_phasors = n_phasors
#         self.n_pmus = n_pmus
#
#         if not channel_names:
#             channel_names = ["Ph{}".format(i) for i in range(n_phasors)]
#
#         conf_kwargs = dict(
#             pmu_id_code=1410,  # PMU_ID
#             time_base=1000000,  # TIME_BASE
#             num_pmu=n_pmus,  # Number of PMUs included in data frame
#             station_name='PMU',  # Station name
#             id_code=1410,  # Data-stream ID(s)
#             data_format=(True, True, True, True),  # Data format - POLAR; PH - REAL; AN - REAL; FREQ - REAL;
#             phasor_num=n_phasors,  # Number of phasors
#             analog_num=1,  # Number of analog values
#             digital_num=1,  # Number of digital status words
#             channel_names=channel_names + [
#                 "ANALOG1", "BREAKER 1 STATUS",
#                 "BREAKER 2 STATUS", "BREAKER 3 STATUS", "BREAKER 4 STATUS", "BREAKER 5 STATUS",
#                 "BREAKER 6 STATUS", "BREAKER 7 STATUS", "BREAKER 8 STATUS", "BREAKER 9 STATUS",
#                 "BREAKER A STATUS", "BREAKER B STATUS", "BREAKER C STATUS", "BREAKER D STATUS",
#                 "BREAKER E STATUS", "BREAKER F STATUS", "BREAKER G STATUS"
#             ],  # Channel Names
#             ph_units=[(0, "v")] * n_phasors,
#             # Conversion factor for phasor channels - (float representation, not important)
#             an_units=[(1, "pow")],  # Conversion factor for analog channels
#             dig_units=[(0x0000, 0xffff)],  # Mask words for digital status words
#             f_nom=50,  # Nominal frequency
#             cfg_count=1,  # Configuration change count
#             data_rate=publish_frequency
#         )
#
#         if n_pmus > 1:
#             conf_kwargs['id_code'] = list(range(conf_kwargs['id_code'], conf_kwargs['id_code'] + self.n_pmus))
#             conf_kwargs['station_name'] = ["PMU {}".format(i) for i in range(self.n_pmus)]
#             for key in ['data_format', 'phasor_num', 'analog_num', 'digital_num',
#                         'channel_names', 'ph_units', 'an_units', 'dig_units', 'f_nom', 'cfg_count']:
#                 conf_kwargs[key] = [conf_kwargs[key]] * self.n_pmus
#
#         cfg = ConfigFrame2(**conf_kwargs)
#
#         self.pmu.set_configuration(cfg)
#         self.pmu.set_header("PMU-Stream from TOPS-RT")
#
#         self.run = self.pmu.run
#
#     def publish(self, time_stamp, phasor_data=[]):
#         soc = int(time_stamp)
#         # frasec = int((((repr((time_stamp % 1))).split("."))[1])[0:6])
#         frasec = int(format(time_stamp % 1, '.6f').split(".")[1])
#
#
#         data_kwargs = dict(
#             soc=soc,
#             frasec=frasec,
#             analog=[9.91],
#             digital=[0x0001],
#             stat=("ok", True, "timestamp", False, False, False, 0, "<10", 0),
#             freq=0,
#             dfreq=0,
#         )
#
#         if not phasor_data:
#             phasor_data = [(random.uniform(215.0, 240.0), random.uniform(-np.pi, np.pi)) for _ in range(self.n_phasors)]
#         if self.n_pmus > 1:
#             data_kwargs['phasors'] = [phasor_data] * self.n_pmus
#         else:
#             data_kwargs['phasors'] = phasor_data
#
#         if self.n_pmus > 1:
#             for key in ['analog', 'digital', 'stat', 'freq', 'dfreq']:
#                 data_kwargs[key] = [data_kwargs[key]]*self.n_pmus
#
#         self.pmu.send_data(**data_kwargs)

class SimplePMU:
    def __init__(self, ip, port, publish_frequency=50, n_phasors=1, n_pmus=1, channel_names=[], set_timestamp=True):
        # Initialize PMUs
        self.ip = ip
        self.port = port
        self.pmu = PmuFix(ip=self.ip, port=self.port, set_timestamp=set_timestamp, data_rate=publish_frequency)
        self.pmu.logger.setLevel("DEBUG")

        self.n_phasors = n_phasors
        self.n_pmus = n_pmus

        if not channel_names:
            channel_names = ["Ph{}".format(i) for i in range(n_phasors)]

        conf_kwargs = dict(
            pmu_id_code=1410,  # PMU_ID
            time_base=1000000,  # TIME_BASE
            num_pmu=n_pmus,  # Number of PMUs included in recorded_pmu_data_raw frame
            station_name='PMU',  # Station name
            id_code=1410,  # Data-stream ID(s)
            data_format=(True, True, True, True),  # Data format - POLAR; PH - REAL; AN - REAL; FREQ - REAL;
            phasor_num=n_phasors,  # Number of phasors
            analog_num=1,  # Number of analog values
            digital_num=1,  # Number of digital status words
            channel_names=channel_names + [
                "ANALOG1", "BREAKER 1 STATUS",
                "BREAKER 2 STATUS", "BREAKER 3 STATUS", "BREAKER 4 STATUS", "BREAKER 5 STATUS",
                "BREAKER 6 STATUS", "BREAKER 7 STATUS", "BREAKER 8 STATUS", "BREAKER 9 STATUS",
                "BREAKER A STATUS", "BREAKER B STATUS", "BREAKER C STATUS", "BREAKER D STATUS",
                "BREAKER E STATUS", "BREAKER F STATUS", "BREAKER G STATUS"
            ],  # Channel Names
            ph_units=[(0, "v")] * n_phasors,
            # Conversion factor for phasor channels - (float representation, not important)
            an_units=[(1, "pow")],  # Conversion factor for analog channels
            dig_units=[(0x0000, 0xffff)],  # Mask words for digital status words
            f_nom=50,  # Nominal frequency
            cfg_count=1,  # Configuration change count
            data_rate=publish_frequency
        )

        if n_pmus > 1:
            conf_kwargs['id_code'] = list(range(conf_kwargs['id_code'], conf_kwargs['id_code'] + self.n_pmus))
            conf_kwargs['station_name'] = ["PMU {}".format(i) for i in range(self.n_pmus)]
            for key in ['data_format', 'phasor_num', 'analog_num', 'digital_num',
                        'channel_names', 'ph_units', 'an_units', 'dig_units', 'f_nom', 'cfg_count']:
                conf_kwargs[key] = [conf_kwargs[key]] * self.n_pmus

        cfg = ConfigFrame2(**conf_kwargs)

        self.pmu.set_configuration(cfg)
        self.pmu.set_header("My PMU-Stream")

        self.run = self.pmu.run

    def cleanup(self):
        self.pmu.stop()
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.ip, self.port))
        self.pmu.socket.close()
        self.pmu.listener.join()
        for stop_event in self.pmu.client_stop_events:
            stop_event.set()

        # Empty queues (remaining items might cause main process to not exit)
        for client_buffer in self.pmu.client_buffers:
            k = 0
            try:
                while True:
                    client_buffer.get(False)
                    k += 1
            except queue.Empty:
                pass

    def publish(self, time_stamp=None, phasor_data=None):

        if time_stamp is None:
            time_stamp = time.time()

        soc = int(time_stamp)
        # frasec = int((((repr((time_stamp % 1))).split("."))[1])[0:6])
        frasec = int(format(time_stamp % 1, '.6f').split(".")[1])

        data_kwargs = dict(
            soc=soc,
            frasec=(frasec, '+'),
            analog=[9.91],
            digital=[0x0001],
            stat=("ok", True, "timestamp", False, False, False, 0, "<10", 0),
            freq=0,
            dfreq=0,
        )

        if phasor_data is None:
            phasor_data = [(random.uniform(215.0, 240.0), random.uniform(-np.pi, np.pi)) for _ in range(self.n_phasors)]
        if self.n_pmus > 1:
            data_kwargs['phasors'] = [phasor_data] * self.n_pmus
        else:
            data_kwargs['phasors'] = phasor_data

        if self.n_pmus > 1:
            for key in ['analog', 'digital', 'stat', 'freq', 'dfreq']:
                data_kwargs[key] = [data_kwargs[key]]*self.n_pmus

        self.pmu.send_data(**data_kwargs)


class PMUPublisher(threading.Thread):
    def __init__(self, rts, publish_frequency=50, phasors=['v_g']):
        threading.Thread.__init__(self)
        self.daemon = True
        self.publish_frequency = publish_frequency
        self.rts = rts
        self.running = True
        self.phasors = phasors

        self.pmus = dict()
        for phasor in self.phasors:
            self.pmus[phasor] = SimplePMU(publish_frequency=self.publish_frequency,
                                          n_phasors=len(getattr(self.rts.ps, phasor)),
                                          n_pmus=1)

    def run(self):
        t_start_sim = time.time()
        # t_adj = 0
        t_prev = time.time()
        t_world = 0
        t_err = 0
        t_target = 0
        for key in self.pmus.keys():
            self.pmus[key].run()

        while self.running:  # and t < self.t_end:
            # print("t_world={:.2f}, t_target={:.2f}, t_wait={:.2f}".format(t_world, t_target, t_err / self.rts.speed))
            if t_world >= t_target:
                t_target += 1 / self.publish_frequency
                # Only publishes new data if simulation solver is finished.
                if self.rts.new_data_ready:
                    # print('PMU-Publisher: Go get data!')
                    with self.rts.new_data_cv:  # condition variable used to both lock and to notify threads
                        # print('PMU-Publisher: Received data!')
                        self.rts.new_data_ready = False
                        self.rts.new_data_cv.notify()  # releases all self.NewPulishData_cv.wait() calls in other threads

                        for phasor in self.phasors:
                            values = getattr(self.rts.ps, phasor)
                            pmu_data = [(mag, ang) for mag, ang in zip(np.abs(values), np.angle(values))]
                            self.pmus[phasor].publish(pmu_data)
            else:
                dt = time.time() - t_prev
                t_prev = time.time()
                t_world += dt * self.rts.speed
                t_err = t_target - t_world
                if t_err > 0:
                    time.sleep(t_err / self.rts.speed)
                elif t_err < 0:
                    print('PMU-Publisher: Overflow! {:.2f} ms.'.format(1000 * t_err))

    def stop(self):
        self.running = False
        if not self.is_alive():
            print('PMU-Publisher: Thread stopped.')


class PMUPublisher2(rts_if.InterfacerQueuesThread):
    def __init__(self, rts, ip='10.0.0.16', port=1410, name='PMUPublisher2', fs=100, set_timestamp=False, publish_frequency=10, *args, **kwargs):
        fs = publish_frequency
        super().__init__(rts=rts, name=name, fs=fs, *args, **kwargs)
        self.pmu = SimplePMU(ip, port, publish_frequency=publish_frequency, set_timestamp=set_timestamp, n_pmus=2, n_phasors=rts.ps.n_bus,
                                     channel_names=list(rts.ps.buses['name']))

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
            pmu_data = [(mag, ang) for mag, ang in zip(np.abs(v), np.angle(v))]
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


