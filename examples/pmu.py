import tops.dynamic as dps
import tops.solvers as dps_sol
from topsrt.sim import RealTimeSimulator
from topsrt.pmu_v2 import PMUPublisherV2
import threading
import time


if __name__ == '__main__':
    import socket
    ip = socket.gethostbyname(socket.gethostname())  # Get local ip automatically
    port = 50000


    import tops.ps_models.n44 as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.sm_ib as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)

    if not hasattr(ps, 'pll'):
        ps.add_model_data({'pll':{
            'PLL1': [
                ['name',        'T_filter',     'bus'   ],
                *[[f'PLL{i}',    0.1,            bus_name  ] for i, bus_name in enumerate(ps.buses['name'])],
            ],
            'PLL2': [
                ['name',        'K_p',  'K_i',  'bus'   ],
                *[[f'PLL{i}',    10,     1,      bus_name  ] for i, bus_name in enumerate(ps.buses['name'])],
            ]
        }})

    
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulator(ps, dt=5e-3, speed=1, solver=dps_sol.ModifiedEulerDAE)

    pmu = PMUPublisherV2(rts, ip=ip, port=port)
    pmu.start()
    # rts.sol.n_it = 0

    rts_thread = threading.Thread(target=rts.main_loop, daemon=True)
    rts_thread.start()

    while True:
        time.sleep(1)
        print(rts.dt_err)