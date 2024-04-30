import sys
import tops.solvers as dps_sol
import threading
import time
import multiprocessing
from .utils import MakeThread, MakeProcess
from tops.simulator import Simulator


class RealTimeSimulator(Simulator):
    def __init__(self, *args, speed=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.speed = speed
        self.dt_sim = 0
        self.t_world = 0
        self.dt_loop = 0
        self.dt_err = 0
        self.dt_ideal = self.dt / self.speed
        self.adjust_time = False
        self.running = True
        self.pause_cv = threading.Condition()
        self.paused = False
        
        self.interface_quitters = dict()  # For sending exit signals to interfaces


    def toggle_pause(self):
        self.paused = not self.paused
        with self.pause_cv:
            self.pause_cv.notify()

    def stopped(self):
        return self._stopped

    def main_loop(self):
        t_prev = time.time()
        sys.stdout.write('Starting Real Time Simulation\n')
        # sys.stdout.write(' '*self.n_markers + '|\n')
        t = 0
        while not self.stopped() and self.sol.t < self.t_end:
            # Allow pausing simulation
            with self.pause_cv:
                while self.paused:
                    self.pause_cv.wait()
                    t_prev = time.time()

            t_sim_0 = time.time()
            if self.speed > 0:
                self.make_simulation_step()

            self.dt_sim = time.time() - t_sim_0
            self.dt_loop = time.time() - t_prev

            t_prev = time.time()
            self.t_world += self.dt_loop * self.speed
            self.dt_err = self.sol.t - self.t_world
            if self.dt_err > 0:
                self.n_markers = 20
                # sys.stdout.write('\r|{:.2f}'.format(1000*self.dt_err) + '-' * (int(self.n_markers * self.dt_loop // self.dt_ideal) - 6) + '|')
                time.sleep(self.dt_err / self.speed)
            elif self.dt_err < 0:
                # print('Overflow! {:.2f} ms.'.format(1000*self.dt_err))
                # sys.stdout.write('\r|---Overflow!{:.2f}'.format(1000*self.dt_err) + '-' * (int(self.n_markers*self.dt_loop//self.dt_ideal) - 14) + '|')
                # sys.stdout.write('\rOverflow! {:.2f} ms.'.format(1000*self.dt_err))
                pass

            self.dt_ideal = self.dt / self.speed

        # with self.interface_functions_lock:
        for key, fun in self.interface_quitters.items():
            # print('Calling {} interface fun.'.format(key))
            fun()

    def set_speed(self, speed):
        self.speed = speed


class RealTimeSimulatorThread(MakeThread, RealTimeSimulator):
    def __init__(self, *args, **kwargs):
        RealTimeSimulator.__init__(self, *args, **kwargs)
        MakeThread.__init__(self, *args, **kwargs)


class RealTimeSimulatorProcess(MakeProcess, RealTimeSimulator):
    def __init__(self, *args, **kwargs):
        RealTimeSimulator.__init__(self, *args, **kwargs)
        MakeProcess.__init__(self, *args, **kwargs)
        self.new_data_cv = multiprocessing.Condition()  # condition variable used to both lock and to notify threads
        self.pause_cv = multiprocessing.Condition()
