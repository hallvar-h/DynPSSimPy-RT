import sys
import dynpssimpy.solvers as dps_sol
import threading
import time
import multiprocessing
from .utils import MakeThread, MakeProcess


class RealTimeSimulator:
    def __init__(self, ps, dt=5e-3, speed=1, solver=dps_sol.ModifiedEulerDAE, log_fun=[], ode_fun=[]):

        self._stopped = False
        self.n_markers = 20
        self.t_end = 100000
        self.dt = dt
        self.adjust_time = False
        self.speed = speed
        self.running = True
        self.log_fun = log_fun
        self.log = callable(self.log_fun)

        self.dt_sim = 0
        self.t_world = 0
        self.dt_loop = 0
        self.dt_err = 0
        self.dt_ideal = self.dt / self.speed

        self.ps = ps

        if callable(ode_fun):
            self.ode_fun = ode_fun
        else:
            self.ode_fun = self.ps.ode_fun

        # self.sol = solver(self.ode_fun, 0, self.ps.x0, self.t_end, max_step=self.dt, first_step=self.dt)
        self.sol = solver(self.ps.state_derivatives, self.ps.solve_algebraic, 0, self.ps.x0, self.t_end, max_step=self.dt, first_step=self.dt)


        self.new_data_cv = threading.Condition()  # condition variable used to both lock and to notify threads
        self.new_data_ready = False
        self.x = self.sol.y
        self.t = self.sol.t

        # self.comm = dict(
        #     # interface_functions_raw=dict(),
        #     input_streams=dict(),
        #     output_streams=dict(),
        #     timers=dict(),
        #     frequency=dict(),
        #     # interface_fun_stream=queue.Queue(),
        # )

        self.interface_functions = dict()
        self.interface_functions_lock = threading.Lock()
        self.interface_quitters = dict()  # For sending exit signals to interfaces
        self.interface_timers = dict()

        self.pause_cv = threading.Condition()
        self.paused = False

    def toggle_pause(self):
        self.paused = not self.paused
        with self.pause_cv:
            self.pause_cv.notify()

    def stopped(self):
        return self._stopped

    def main_loop(self):
        t_start_sim = time.time()
        # t_adj = 0
        t_prev = time.time()
        sys.stdout.write('Starting Real Time Simulation\n')
        # sys.stdout.write(' '*self.n_markers + '|\n')
        while not self.stopped():  # and t < self.t_end:
            # Allow pausing simulation
            with self.pause_cv:
                while self.paused:
                    self.pause_cv.wait()
                    t_prev = time.time()

            # Simulate next step
            t_sim_0 = time.time()
            if self.speed > 0:
                with self.new_data_cv:
                    self.sol.step()

                    with self.interface_functions_lock:
                        for key, fun in self.interface_functions.items():
                            # print('Calling {} interface fun.'.format(key))
                            fun(self)
                    # for key, fun in self.comm['interface_functions'].items():
                    #     fun(self)
                    self.new_data_ready = True
                    self.new_data_cv.notify()

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
                # if self.adjust_time:
                #     t_adj -= t_err

            self.dt_ideal = self.dt / self.speed

            # if self.log:
            #     self.log_fun(self)

        # with self.interface_functions_lock:
        for key, fun in self.interface_quitters.items():
            # print('Calling {} interface fun.'.format(key))
            fun()
        # sys.stdout.write('\n')
        return  # Not sure why this has a return..

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
