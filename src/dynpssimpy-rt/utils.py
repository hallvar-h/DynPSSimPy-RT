import threading
import multiprocessing


class MakeThread(threading.Thread):
    interface_name = 'Thread Template'
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self)
        self.daemon = True
        self._stopped = threading.Event()  # Event that can kill the thread

    def start(self):
        threading.Thread.start(self)

    def run(self):
        self.main_loop()

    def stop(self):
        self._stopped.set()
        if not self.is_alive():
            print(self.interface_name + 'Interfacer thread stopped.')

    def stopped(self):
        return self._stopped.is_set()


class MakeProcess(multiprocessing.Process):
    interface_name = 'Process Template'
    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self)
        self.daemon = True
        self._stopped = multiprocessing.Event()  # Event that can kill the thread

    def start(self):
        multiprocessing.Process.start(self)

    def run(self):
        self.main_loop()

    def stop(self):
        self._stopped.set()
        self.terminate()
        self.join()
        if not self.is_alive():
            print(self.interface_name + ' process stopped.')

    def stopped(self):
        return self._stopped.is_set()