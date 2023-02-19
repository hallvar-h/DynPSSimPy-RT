import pyqtgraph as pg
import numpy as np
from PySide6 import QtWidgets, QtCore


class PhasorPlot(QtWidgets.QWidget):
    def __init__(self, n_phasors, update_freq=50, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_phasors = n_phasors

        # if isinstance(gen_mdls, list):
        #     self.gen_mdls = self.gen_mdls
        # else:
        #     if gen_mdls == 'all':
        #         self.gen_mdls = list(self.ps.gen_mdls.keys())
        #     else:
        #         self.gen_mdls = [gen_mdls]


        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)
        
        # Phasor diagram
        self.graphWidget = pg.GraphicsLayoutWidget(show=True, title="Phasors")
        # self.graphWidget.show()
        # self.setCentralWidget(self.graphWidget)

        self.phasor_0 = np.array([0, 1, 0.9, 1, 0.9, 1]) + 1j * np.array([0, 0, -0.1, 0, 0.1, 0])
        plot_win_ph = self.graphWidget.addPlot(title='Phasors')
        plot_win_ph.setAspectLocked(True)

        # angle = np.concatenate([self.rts.x[gen_mdl.idx][gen_mdl.state_idx['angle']] for gen_mdl in self.ps.gen_mdls.values()])
        # angle -= np.mean(angle)
        # magnitude = np.concatenate([gen_mdl.input['E_f'] for gen_mdl in self.ps.gen_mdls.values()])
        angle = np.zeros(self.n_phasors)
        magnitude = np.ones(self.n_phasors)
        phasors = magnitude*np.exp(1j*angle)
        self.phasors = phasors

        self.pl_ph = []

        for i, phasor in enumerate(phasors[:, None]*self.phasor_0):
            pen = pg.mkPen(color=self.colors(i), width=2)
            pl_ph = pg.PlotCurveItem(phasor.real, phasor.imag, pen=pen)
            plot_win_ph.addItem(pl_ph)
            self.pl_ph.append(pl_ph)

            # self.pl_ph.append(plot_win_ph.plot(phasor.real, phasor.imag, pen=pen))

        plot_win_ph.enableAutoRange('xy', False)

        self.graphWidget.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000//update_freq)
        

    def update(self, phasors):
        # if not np.isclose(self.ts_keeper.time[-1], self.ps.time):
        # Phasors:
        # angle = np.concatenate([self.rts.x[gen_mdl.idx][gen_mdl.state_idx['angle']] for gen_mdl in self.ps.gen_mdls.values()])
        # angle -= np.mean(angle)
        # magnitude = np.concatenate([gen_mdl.input['E_f'] for gen_mdl in self.ps.gen_mdls.values()])
        # phasors = magnitude * np.exp(1j * angle)
        self.phasors = phasors

    def update_plot(self):
        for i, (pl_ph, phasor) in enumerate(zip(self.pl_ph, self.phasors[:, None]*self.phasor_0)):
            pl_ph.setData(phasor.real, phasor.imag)


def main():
    
    n_phasors = 10
    app = QtWidgets.QApplication()
    phasor_plot = PhasorPlot(n_phasors=n_phasors)

    import time
    def update_plot():
        while True:
            new_phasors = np.random.randn(n_phasors) + 1j*np.random.randn(n_phasors)
            phasor_plot.update(phasors=new_phasors)
            time.sleep(0.1)

    import threading
    thr = threading.Thread(target=update_plot, daemon=True)
    thr.start()

    app.exec()




if __name__ == '__main__':
    main()