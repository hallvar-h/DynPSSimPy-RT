import multiprocessing as mp
from dynpssimrt.interfacing import QueueManager, InterfaceListener
import sys
import dynpssimpy.dynamic as dps
from dynpssimrt.sim import RealTimeSimulatorThread
# from dynpssimrt.pmu import PMUPublisher
from dynpssimrt.pmu_currents import PMUPublisherCurrents as PMUPublisher
from synchrophasor.test_utils import run_pdc
import time
import socket


def main_pmu(qm_kwargs, ip, port, pdc_id):

    manager = QueueManager(**qm_kwargs)
    manager.connect()


    pmus = PMUPublisher(publish_frequency=25, ip=ip, port=port, pdc_id=pdc_id)

    InterfaceListener.send_interface_init(manager, pmus)
    pmus.start()
    time.sleep(3)


def main(qm_kwargs):

    manager = QueueManager(**qm_kwargs)
    manager.connect()
    init_queue = manager.get_init_queue()
    interface_listener = InterfaceListener(init_queue)

    # import dynpssimpy.ps_models.k2a as model_data
    # import examples.test_systems.n44_ctrl as model_data
    import dynpssimpy.ps_models.ieee39 as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulatorThread(ps, dt=10e-3, speed=1)

    interface_listener.connect(rts)
    interface_listener.start()

    rts.start()
    time.sleep(5)
    rts.stop()


def main_server(qm_kwargs):
    manager = QueueManager(server=True, **qm_kwargs)
    manager.start()


if __name__ == '__main__':

    ip = socket.gethostbyname(socket.gethostname())
    qm_kwargs = dict(address=(ip, 40000), authkey=b'abracadabra')

    pmu_port = 50000
    pdc_id = 1

    p_server = mp.Process(target=main_server, args=(qm_kwargs,))
    p_server.start()

    p_1 = mp.Process(target=main, args=(qm_kwargs,))
    p_1.start()

    p_2 = mp.Process(target=main_pmu, args=(qm_kwargs, ip, pmu_port, pdc_id))
    p_2.start()

    time.sleep(1)
    p_3 = mp.Process(target=run_pdc, args=(3, ip, pmu_port, pdc_id))
    p_3.start()

    p_1.join()
    p_2.join()
    # p_3.join()
    p_server.join()