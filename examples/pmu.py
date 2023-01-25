import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
from dynpssimrt.sim import RealTimeSimulator, RealTimeSimulatorThread
import threading
import time
import sys
from PySide6 import QtWidgets
from dynpssimrt.gui import LineOutageWidget
from dynpssimrt.time_window_plot import TimeWindowPlot
from dynpssimrt.rtsim_plot import RTSimPlot, SyncPlot
from dynpssimrt.pmu import PMUPublisher


def main():

    import socket
    ip = socket.gethostbyname(socket.gethostname())  # Get local ip automatically
    port = 50000

    import dynpssimpy.ps_models.k2a as model_data
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulatorThread(ps, dt=10e-3, speed=1, solver=dps_sol.ModifiedEulerDAE)

    pmus = PMUPublisher(rts, publish_frequency=5, phasors=['v_g'], ip=ip, port=port)
    pmus.start()

    app = QtWidgets.QApplication(sys.argv)

    tw_plot = RTSimPlot(rts=rts, n_samples=1000)
    sync_plot = SyncPlot(rts=rts, n_samples=1000, update_freq=25)

    rts.start()
    tw_plot.start()
    sync_plot.start()

    # Add Control Widgets
    line_outage_ctrl = LineOutageWidget(rts)

    app.exec()

    return app


if __name__ == '__main__':
    main()


