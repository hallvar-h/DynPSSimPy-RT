import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
from dynpssimrt.sim import RealTimeSimulator
import threading
import time
import sys
from PySide6 import QtWidgets
from dynpssimrt.gui import LineOutageWidget, DynamicLoadControlWidget
from dynpssimrt.time_window_plot import TimeWindowPlot
from dynpssimrt.rtsim_plot import SyncPlot
from dynpssimrt.plotting.phasor_plots import VoltagePhasorPlot
from dynpssimrt.plotting.grid_plot import LiveGridPlot3D


def main(rts):
    update_freq = 25
    app = QtWidgets.QApplication(sys.argv)

    sync_plot = SyncPlot(rts=rts, n_samples=1000, update_freq=25)
    voltage_phasor_plot = VoltagePhasorPlot(rts, update_freq=25)
    grid_plot = LiveGridPlot3D(rts=rts, z_ax='angle')

    sync_plot.start()
    voltage_phasor_plot.start()
    grid_plot.start()

    # Add Control Widgets
    line_outage_ctrl = LineOutageWidget(rts)
    load_control = DynamicLoadControlWidget(rts)

    app.exec()

    return app


if __name__ == '__main__':

    import dynpssimpy.ps_models.ieee39 as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.sm_ib as model_data

    model = model_data.load()
    model['loads'] = {'DynamicLoad': model['loads']}

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulator(ps, dt=5e-3, speed=1, solver=dps_sol.ModifiedEulerDAE)

    rts_thread = threading.Thread(target=rts.main_loop, daemon=True)
    rts_thread.start()

    app = main(rts)