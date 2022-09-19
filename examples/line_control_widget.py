import dynpssimpy.dynamic as dps
import dynpssimpy.solvers as dps_sol
from src.dpsrt.sim import RealTimeSimulator
import threading
import time
import sys
from PySide6 import QtWidgets
from src.dpsrt.gui import LineOutageWidget
from src.dpsrt.time_window_plot import TimeWindowPlot


def main(rts):
    update_freq = 25
    app = QtWidgets.QApplication(sys.argv)

    tw_plot = TimeWindowPlot(n_samples=100, n_cols=len(rts.sol.x[0:1]))

    def update_tw():
        while True:
            time.sleep(0.01)
            # tw.append(time.time(), np.random.randn(10))
            tw_plot.append(rts.sol.t, rts.sol.x[0:1])

    update_tw_thread = threading.Thread(target=update_tw, daemon=True)
    update_tw_thread.start()


    # tw_plot = TimeWindowPlot(rts, ['gen', 'GEN', 'state', 'speed'])
    # mdl_type = 'gen'
    # mdl = 'GEN'
    #
    # getattr(rts.ps, mdl_type)[]

    # tw = TimeWindow(100, n_cols)
    # def update_tw():
    #     tw.append(rts.sol.t)
    # time_series_plot = TimeWindowPlot(tw)

    # Add Control Widgets
    line_outage_ctrl = LineOutageWidget(rts)

    app.exec()

    return app


if __name__ == '__main__':

    import dynpssimpy.ps_models.ieee39 as model_data
    # import ps_models.ieee39 as model_data
    # import ps_models.sm_ib as model_data

    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    ps.ode_fun(0, ps.x0)
    rts = RealTimeSimulator(ps, dt=2.5e-3, speed=1, solver=dps_sol.ModifiedEulerDAE)

    rts_thread = threading.Thread(target=rts.run, daemon=True)
    rts_thread.start()

    app = main(rts)