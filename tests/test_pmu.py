import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
from dynpssimrt.sim import RealTimeSimulatorThread
from dynpssimrt.pmu_currents import PMUPublisherCurrents as PMUPublisher
from synchrophasor.test_utils import run_pdc
import time
import socket
import multiprocessing as mp


def main(ip, port, pdc_id):

    import dynpssimpy.ps_models.k2a as model_data
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulatorThread(ps, dt=10e-3, speed=1, solver=dps_sol.ModifiedEulerDAE)

    pmus = PMUPublisher(rts, publish_frequency=5, phasors=['v_g'], ip=ip, port=port, pdc_id=pdc_id)
    pmus.start()
    rts.start()
    time.sleep(3)


if __name__ == '__main__':

    ip = socket.gethostbyname(socket.gethostname())  # Get local ip automatically
    port = 50000
    pdc_id = 1

    p_main = mp.Process(target=main, args=(ip, port, pdc_id))
    p_pdc = mp.Process(target=run_pdc, args=(2, ip, port, pdc_id))

    p_main.start()
    p_pdc.start()