from .utils import MakeThread, MakeProcess
from multiprocessing.managers import SyncManager
import queue
import multiprocessing as mp
import marshal
import types


class InterfacerQueues:
    def __init__(self, rts=None, name='InterfacerQueues', fs=None, wait_for_interface=False):
        self._stopped = False
        self.interface_name = name
        self.interface_name_unique = name
        self.fs = fs
        self.wait_for_interface = wait_for_interface

        if rts is not None:
            self.connect(rts)
            print('Added interface {} to RealTimeSimulator.'.format(self.interface_name_unique))

    def connect(self, rts):
        if any([issubclass(cls, mp.Process) for cls in [rts.__class__, self.__class__]]):
            Queue = mp.Queue
        else:
            Queue = queue.Queue

        self.result_stream = Queue()
        self.ctrl_stream = Queue()
        # self.init_stream = Queue()

        interface_name_unique = self.interface_name
        counter = 2
        while interface_name_unique in rts.interface_functions.keys():
            interface_name_unique = self.interface_name + " (" + str(counter) + ")"
            counter += 1

        self.interface_name_unique = interface_name_unique

        interface_fun_no_kwargs = lambda rts: self.interface_fun(
            rts,
            name=interface_name_unique,
            frequency=self.fs,
            read_input_signal=self.read_input_signal,
            apply_ctrl_signal=self.apply_ctrl_signal,
            output_stream=self.result_stream,
            input_stream=self.ctrl_stream,
            wait_for_interface=self.wait_for_interface,
        )
        if interface_name_unique in rts.interface_functions.keys():
            print('Warning: Two or more interfaces with name {} connected. Previous interfaces will be overwritten.')
        rts.interface_functions[interface_name_unique] = interface_fun_no_kwargs
        if self.fs:
            rts.interface_timers[interface_name_unique] = 0

        if hasattr(self, 'get_init_data') and hasattr(self, 'initialize'):
            # self.init_stream.put(self.get_init_data(rts))
            init_data = self.get_init_data(rts)
            self.initialize(init_data)

        quit_fun_no_kwargs = lambda: self.quit_fun(name=interface_name_unique, output_stream=self.result_stream)
        rts.interface_quitters[interface_name_unique] = quit_fun_no_kwargs

    @staticmethod
    def get_init_data(rts):
        pass

    def initialize(self, init_data):
        pass

    @staticmethod
    def read_input_signal(rts):
        pass

    @staticmethod
    def apply_ctrl_signal(rts, ctrl_signal):
        pass

    def generate_ctrl_signal(self):
        # Generate control signal from internal states
        pass

    def update(self, input):
        # Update internal states
        pass

    @staticmethod
    def interface_fun(rts, name, frequency, read_input_signal, apply_ctrl_signal, output_stream, input_stream, wait_for_interface):
        if frequency:
            if rts.sol.t < rts.interface_timers[name]:
                return
            else:
                rts.interface_timers[name] += 1 / frequency
        # Apply control
        try:
            if wait_for_interface:
                ctrl = input_stream.get()
            else:
                ctrl = input_stream.get_nowait()
            apply_ctrl_signal(rts, ctrl)
        except queue.Empty:
            pass

        input = read_input_signal(rts)
        if not wait_for_interface:
            # Not waiting, so remove any previous elements in queue (make sure only newest is left)
            try:
                output_stream.get_nowait()
            except queue.Empty:
                pass

        output_stream.put(input)

    @staticmethod
    def quit_fun(name, output_stream):
        print('Exit message sent to {}.'.format(name))
        output_stream.put(None)

    def start(self):
        self.main_loop()

    def main_loop(self):
        while not self.stopped():
            # This could also be after reading result stream. Which is more realistic?
            ctrl_signal = self.generate_ctrl_signal()
            if ctrl_signal is not None:
                try:
                    self.ctrl_stream.get_nowait()
                except queue.Empty:
                    pass
                self.ctrl_stream.put(ctrl_signal)
            try:
                # res = self.result_stream.get(timeout=1)
                result = self.result_stream.get(timeout=1)
                if result is not None:
                    self.update(result)
                else:
                    print(self.interface_name + ': Received exit message.')
                    break
                # print(res)
            except queue.Empty:
                print(self.interface_name + ': Timeout (waiting for RTS result stream).')

    def stopped(self):
        return self._stopped


class InterfacerQueuesThread(MakeThread, InterfacerQueues):
    def __init__(self, *args, **kwargs):
        InterfacerQueues.__init__(self, *args, **kwargs)
        MakeThread.__init__(self, *args, **kwargs)


class InterfacerQueuesProcess(MakeProcess, InterfacerQueues):
    def __init__(self, *args, **kwargs):
        InterfacerQueues.__init__(self, *args, **kwargs)
        MakeProcess.__init__(self, *args, **kwargs)


class InterfaceListener(MakeThread):
    def __init__(self, init_queue, rts=None, *args, **kwargs):
        super().__init__(name='InterfaceListener', *args, **kwargs)
        self.init_queue = init_queue
        self.kwargs = dict()
        if rts:
            self.connect(rts)

    def connect(self, rts):
        self.rts = rts
        [setattr(self, attr, getattr(rts, attr)) for attr in [
            'interface_functions_lock', 'interface_functions', 'interface_timers', 'interface_quitters'
        ]]

    def main_loop(self):
        while not self.stopped():
            print('Waiting for interfaces messages.')
            message = self.init_queue.get()
            if len(message) == 5:
                read_input_signal_pickled, apply_ctrl_signal_pickled, get_init_data_pickled, interface_init_stream, kwargs = message
                print('Got interface.')

                interface_name_original = kwargs['name']
                interface_name_unique = interface_name_original
                counter = 2
                while interface_name_unique in self.interface_functions.keys():
                    interface_name_unique = interface_name_original + " (" + str(counter) + ")"
                    counter += 1

                kwargs['name'] = interface_name_unique

            code = marshal.loads(read_input_signal_pickled)
            read_input_signal_rebuilt = types.FunctionType(code, globals(), "some_func_name")

            code = marshal.loads(apply_ctrl_signal_pickled)
            apply_ctrl_signal_rebuilt = types.FunctionType(code, globals(), "some_func_name")

            code = marshal.loads(get_init_data_pickled)
            get_init_data_rebuilt = types.FunctionType(code, globals(), "some_func_name")

            init_data = get_init_data_rebuilt(self.rts)
            interface_init_stream.put(init_data)

            def interface_fun_no_kwargs(
                    rts,
                    read_input_signal=read_input_signal_rebuilt,
                    apply_ctrl_signal=apply_ctrl_signal_rebuilt,
                    kwargs=kwargs
            ):
                InterfacerQueues.interface_fun(
                    rts,
                    read_input_signal=read_input_signal,
                    apply_ctrl_signal=apply_ctrl_signal,
                    **kwargs
                )

            # quit_fun_no_kwargs = lambda:
            def quit_fun_no_kwargs(kwargs=kwargs):
                InterfacerQueues.quit_fun(name=kwargs['name'], output_stream=kwargs['output_stream'])

            with self.interface_functions_lock:

                self.interface_functions[kwargs['name']] = interface_fun_no_kwargs
                self.interface_timers[kwargs['name']] = 0
                self.kwargs[kwargs['name']] = kwargs
                self.interface_quitters[kwargs['name']] = quit_fun_no_kwargs

            print('Added interface {} to RealTimeSimulator.'.format(kwargs['name']))

    @staticmethod
    def send_interface_init(manager, interface):
        interface.result_stream = manager.Queue()
        interface.ctrl_stream = manager.Queue()
        interface.init_stream = manager.Queue()

        read_input_signal_pickled = marshal.dumps(interface.read_input_signal.__code__)
        apply_ctrl_signal_pickled = marshal.dumps(interface.apply_ctrl_signal.__code__)
        get_init_data_pickled = marshal.dumps(interface.get_init_data.__code__)

        kwargs = dict(
            name=interface.interface_name,
            frequency=interface.fs,
            output_stream=interface.result_stream,
            input_stream=interface.ctrl_stream,
            wait_for_interface=interface.wait_for_interface,
        )

        init_queue = manager.get_init_queue()
        init_queue.put([
            read_input_signal_pickled,
            apply_ctrl_signal_pickled,
            get_init_data_pickled,
            interface.init_stream,
            kwargs])

        init_data = interface.init_stream.get()
        interface.initialize(init_data)


class QueueManager:
    def __init__(self, server=False, *args, **kwargs):
        class MyQueueManager(SyncManager):
            pass

        if server:
            init_queue = queue.Queue()

            def get_init_queue():
                return init_queue
            MyQueueManager.register('get_init_queue', callable=get_init_queue)
        else:
            MyQueueManager.register('get_init_queue')

        self._manager = MyQueueManager(*args, **kwargs)
        if 'authkey' in kwargs:
            mp.current_process().authkey = b'abracadabra'

        self.connect = self._manager.connect
        self.get_init_queue = self._manager.connect
        [setattr(self, attr, getattr(self._manager, attr)) for attr in [
            'connect', 'get_init_queue', 'Queue'
        ]]

    def start(self):
        s = self._manager.get_server()
        print('Serving forever...')
        s.serve_forever()