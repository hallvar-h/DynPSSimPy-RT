from PyQt5 import QtWidgets
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.opengl as gl
import tops.dynamic as dps
import tops.real_time_sim.apps as rts_apps
import tops.utility_functions as dps_uf
import importlib
from pyqtconsole.console import PythonConsole
import tops.real_time_sim.sim as dps_rts
import networkx as nx
import time


def map_idx_fun(x, y):
    if len(x) > 0:
        index = np.argsort(x)
        sorted_x = x[index]
        sorted_index = np.searchsorted(sorted_x, y)

        yindex = np.take(index, sorted_index, mode="clip")
        mask = x[yindex] != y

        yindex[mask] = -1
        return yindex  # np.ma.array(yindex, mask=mask)
    else:
        return np.zeros(0)


class TimeSeriesKeeper:
    def __init__(self):
        pass


class ShortCircuitWidget(QtWidgets.QWidget):
    def __init__(self, rts, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        # Controls
        self.ctrlWidget = QtWidgets.QWidget()
        self.ctrlWidget.setWindowTitle('Short Circuit Buses')

        layout_box = QtWidgets.QVBoxLayout()

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.ctrlWidget)
        slider.setMinimum(1)
        slider.setMaximum(1000)
        # slider.valueChanged.connect(lambda state, args_=(gen_key, i): self.updateExcitation(args_[0], args_[1]))
        # slider.setAccessibleName(gen_key + ' ' + str(i))
        slider.setValue(100)
        self.slider = slider
        layout_box.addWidget(slider)

        for i, bus in enumerate(self.ps.buses):

            button = QtWidgets.QPushButton(bus['name'])
            # button.setCheckable(True)
            # button.setChecked(True)
            layout_box.addWidget(button)  #, k, j + 1)
            button.clicked.connect(lambda state, arg=bus['name']: self.applyShortCircuit(arg))

        self.ctrlWidget.setLayout(layout_box)
        self.ctrlWidget.show()

    def applyShortCircuit(self, bus):
        self.ps.network_event('bus', bus, 'apply_sc')
        # time.sleep(self.slider.value()/1000)
        QtCore.QTimer.singleShot(self.slider.value(), lambda: self.removeShortCircuit(bus=bus))

    def removeShortCircuit(self, bus):
        self.ps.network_event('bus', bus, 'remove_sc')
        print('t={:.2f}s: Bus {} short-circuited for {:.0f} ms.'.format(self.rts.sol.t, bus, self.slider.value()))


class LineOutageWidget(QtWidgets.QWidget):
    def __init__(self, rts, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        # dt = self.dt
        n_samples = 500
        dt = 5e-3

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)

        # Controls
        self.ctrlWidget = QtWidgets.QWidget()
        self.ctrlWidget.setWindowTitle('Lines')

        # self.graphWidget_ctrl = pg.GraphicsLayoutWidget(show=True, title="Controls")
        layout_box = QtWidgets.QVBoxLayout()
        self.check_boxes = []
        for i, line in enumerate(self.ps.lines):
            check_box = QtWidgets.QCheckBox(line['name'])
            check_box.setChecked(True)
            check_box.stateChanged.connect(self.updateLines)
            check_box.setAccessibleName(line['name'])

            layout_box.addWidget(check_box)
            # layout_box.addSpacing(15)
            layout_box.addSpacing(0)


        self.ctrlWidget.setLayout(layout_box)
        self.ctrlWidget.show()

    def updateLines(self):
        if self.sender().isChecked():
            action = 'connect'
        else:
            action = 'disconnect'
        self.ps.network_event('line', self.sender().accessibleName(), action)
        print('t={:.2f}s'.format(self.rts.sol.t) + ': Line ' + self.sender().accessibleName() + ' '+ action + 'ed.')


class GenCtrlWidget(QtWidgets.QWidget):
    def __init__(self, rts, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.setBackgroundRole(QtGui.QPalette.Base)
        # self.setAutoFillBackground(True)

        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        # dt = self.dt
        n_samples = 500
        dt = 20e-3

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)

        # Controls
        self.ctrlWidget = QtWidgets.QWidget()
        self.ctrlWidget.setWindowTitle('Generator Controls')

        layout = QtWidgets.QGridLayout()
        self.sliders = []
        k = 0
        for gen_key, gen_mdl in self.ps.gen_mdls.items():
            for i in range(len(gen_mdl.par)):
                slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.ctrlWidget)
                slider.setMinimum(0)
                slider.setMaximum(300)
                slider.valueChanged.connect(lambda state, args_=(gen_key, i): self.updateExcitation(args_[0], args_[1]))
                slider.setAccessibleName(gen_key + ' ' + str(i))
                slider.setValue(int(100*gen_mdl.input['E_f'][i]))
                self.sliders.append(slider)
                layout.addWidget(slider, k, 0)
                for j, ctrl_type in enumerate(['avr_mdls', 'gov_mdls', 'pss_mdls']):
                    for ctrl_key, ctrl_mdl in getattr(self.ps, ctrl_type).items():
                        if gen_key in ctrl_mdl.gen_idx.keys():
                            if i in ctrl_mdl.gen_idx[gen_key][1]:
                                mask = ctrl_mdl.gen_idx[gen_key][0].copy()
                                mask[mask] = ctrl_mdl.gen_idx[gen_key][1] == i
                                gen_idx = np.argmax(mask)
                                button = QtWidgets.QPushButton(ctrl_key)
                                button.setCheckable(True)
                                button.setChecked(True)
                                layout.addWidget(button, k, j+1)
                                button.clicked.connect(lambda state, args_=(ctrl_type, ctrl_key, gen_idx): self.updateActivation(args_[0], args_[1], args_[2]))
                k += 1

        self.ctrlWidget.setLayout(layout)
        self.ctrlWidget.show()

    def updateExcitation(self, gen_key, gen_idx):
        self.ps.gen_mdls[gen_key].input['E_f'][gen_idx] = self.sender().value()/100

    def updateActivation(self, type, name, idx):
        mdl = getattr(self.ps, type)[name]
        mdl.active[idx] = self.sender().isChecked()
        # print(type, name, idx, self.sender().isChecked())
        activation_string = 'activated' if self.sender().isChecked() else 'deactivated'
        print('t={:.2f}s'.format(self.rts.sol.t) + ': ' + mdl.par['name'][idx] + ' on ' + mdl.par['gen'][idx] + ' ' + activation_string + '.')
        # print(type, name, idx, self.sender().isChecked())


class SimulationControl(QtWidgets.QWidget):
    def __init__(self, rts, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.setBackgroundRole(QtGui.QPalette.Base)
        # self.setAutoFillBackground(True)

        self.rts = rts
        self.ps = rts.ps

        # Controls
        self.ctrlWidget = QtWidgets.QWidget()
        self.ctrlWidget.setWindowTitle('Simulation Controls')
        layout = QtWidgets.QGridLayout()

        # Add speed slider
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.ctrlWidget)
        slider.setMinimum(1)
        slider.setMaximum(200)
        slider.valueChanged.connect(lambda state: self.updateSpeed())
        slider.setAccessibleName('Simulation speed')
        slider.setValue(100)
        layout.addWidget(slider, 0, 0)

        # Pause button
        button = QtWidgets.QPushButton('Pause')
        button.setCheckable(True)
        button.setChecked(False)
        layout.addWidget(button, 1, 0)
        button.clicked.connect(lambda state: self.pauseSimulation())


        # Reset button
        button = QtWidgets.QPushButton('Reset')
        button.setCheckable(False)
        layout.addWidget(button, 2, 0)
        button.clicked.connect(lambda state: self.resetSimulation())

        self.ctrlWidget.setLayout(layout)
        self.ctrlWidget.show()

    def updateSpeed(self):
        self.rts.speed = self.sender().value()/100

    def resetSimulation(self):
        self.rts.toggle_pause()
        self.ps.init_dyn_sim()
        self.rts.sol.x[:] = self.ps.x0
        self.rts.sol.v[:] = self.ps.v0
        self.rts.toggle_pause()

    def pauseSimulation(self):
        self.rts.toggle_pause()


class LivePlotter(QtWidgets.QMainWindow):
    def __init__(self, rts, plots=['angle', 'speed'], *args, **kwargs):
        super(LivePlotter, self).__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        self.graphWidget = pg.GraphicsLayoutWidget(show=True, title="Live plot")
        # self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        dt = self.dt
        n_samples = 250
        dt = 20e-3

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)

        # Phasor diagram
        self.graphWidget_ph = pg.GraphicsLayoutWidget(show=True, title="Phasors")
        self.phasor_0 = np.array([0, 1, 0.9, 1, 0.9, 1]) + 1j * np.array([0, 0, -0.1, 0, 0.1, 0])
        plot_win_ph = self.graphWidget_ph.addPlot(title='Phasors')
        plot_win_ph.setAspectLocked(True)

        phasors = self.ps.e[:, None]*self.phasor_0
        self.pl_ph = []
        for i, phasor in enumerate(phasors):
            pen = pg.mkPen(color=self.colors(i), width=2)
            self.pl_ph.append(plot_win_ph.plot(phasor.real, phasor.imag, pen=pen))
        plot_win_ph.enableAutoRange('xy', False)


        self.plots = plots
        self.ts_keeper = TimeSeriesKeeper()
        self.ts_keeper.time = np.arange(-n_samples*dt, 0, dt)
        # self.y = np.zeros_like(self.ts_keeper.time)
        self.pl = {}


        for plot in self.plots:
            graphWidget = self.graphWidget.addPlot(title=plot)
            # p_1 = self.addPlot(title="Updating plot 1")
            n_series = len(getattr(self.ps, plot))
            setattr(self.ts_keeper, plot, np.zeros((n_samples, n_series)))

            pl_tmp = []
            for i in range(n_series):
                pen = pg.mkPen(color=self.colors(i), width=2)
                pl_tmp.append(graphWidget.plot(self.ts_keeper.time, np.zeros(n_samples), pen=pen))

            self.pl[plot] = pl_tmp
            self.graphWidget.nextRow()


        # self.pl_1 = self.graphWidget.plot(self.ts_keeper.time, self.y)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(25)

    def update(self):
        if not np.isclose(self.ts_keeper.time[-1], self.ps.time):
            # print(np.isclose(self.ts_keeper.time[-1], self.ps.time))
            # print(self.ts_keeper.time[-1]-  self.ps.time)
            self.ts_keeper.time = np.append(self.ts_keeper.time[1:], self.ps.time)

            # Phasors:
            phasors = self.ps.e[:, None] * self.phasor_0
            for i, (pl_ph, phasor) in enumerate(zip(self.pl_ph, phasors)):
                pl_ph.setData(phasor.real, phasor.imag)

            for plot in self.plots:
                old_data = getattr(self.ts_keeper, plot)[1:, :]
                new_data = getattr(self.ps, plot)
                setattr(self.ts_keeper, plot, np.vstack([old_data, new_data]))
                plot_data = getattr(self.ts_keeper, plot)
                for i, pl in enumerate(self.pl[plot]):
                    pl.setData(self.ts_keeper.time, plot_data[:, i])


class PhasorPlot(QtWidgets.QWidget):
    def __init__(self, rts, update_freq=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

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
        # self.setCentralWidget(self.graphWidget)

        self.phasor_0 = np.array([0, 1, 0.9, 1, 0.9, 1]) + 1j * np.array([0, 0, -0.1, 0, 0.1, 0])
        plot_win_ph = self.graphWidget.addPlot(title='Phasors')
        plot_win_ph.setAspectLocked(True)

        angle = np.concatenate([self.rts.x[gen_mdl.idx][gen_mdl.state_idx['angle']] for gen_mdl in self.ps.gen_mdls.values()])
        angle -= np.mean(angle)
        magnitude = np.concatenate([gen_mdl.input['E_f'] for gen_mdl in self.ps.gen_mdls.values()])
        phasors = magnitude*np.exp(1j*angle)

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
        self.timer.timeout.connect(self.update)
        self.timer.start(1000//update_freq)

    def update(self):
        # if not np.isclose(self.ts_keeper.time[-1], self.ps.time):
        # Phasors:
        angle = np.concatenate([self.rts.x[gen_mdl.idx][gen_mdl.state_idx['angle']] for gen_mdl in self.ps.gen_mdls.values()])
        angle -= np.mean(angle)
        magnitude = np.concatenate([gen_mdl.input['E_f'] for gen_mdl in self.ps.gen_mdls.values()])
        phasors = magnitude * np.exp(1j * angle)
        for i, (pl_ph, phasor) in enumerate(zip(self.pl_ph, phasors[:, None]*self.phasor_0)):
            pl_ph.setData(phasor.real, phasor.imag)


class PhasorPlotFast(QtWidgets.QWidget):
    def __init__(self, rts, update_freq=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        self.dt

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)
        # Phasor diagram
        self.graphWidget = pg.GraphicsLayoutWidget(show=True, title="Phasors")
        # self.setCentralWidget(self.graphWidget)

        self.phasor_0 = np.array([0, 1, 0.9, 1, 0.9, 1]) + 1j * np.array([0, 0, -0.1, 0, 0.1, 0])
        plot_win_ph = self.graphWidget.addPlot(title='Phasors')
        plot_win_ph.setAspectLocked(True)

        # angle = self.rts.x[self.ps.gen_mdls['GEN'].state_idx['angle']]
        # magnitude = self.ps.gen_mdls['GEN'].input['E_f']
        angle = np.concatenate([self.rts.x[gen_mdl.idx][gen_mdl.state_idx['angle']] for gen_mdl in self.ps.gen_mdls.values()])
        angle -= np.mean(angle)
        magnitude = np.concatenate([gen_mdl.input['E_f'] for gen_mdl in self.ps.gen_mdls.values()])
        phasors = magnitude*np.exp(1j*angle)

        self.pl_ph = []

        # for i, phasor in enumerate(phasors[:, None]*self.phasor_0):
        phasors_points = np.kron(phasors, self.phasor_0)
        connect = np.tile(np.append(np.ones(len(phasors)-1, dtype=bool), 0), len(self.phasor_0))

        self.pl_ph = pg.PlotCurveItem(phasors_points.real, phasors_points.imag, connect=connect)
        plot_win_ph.addItem(self.pl_ph)

            # self.pl_ph.append(plot_win_ph.plot(phasor.real, phasor.imag, pen=pen))

        plot_win_ph.enableAutoRange('xy', False)

        self.graphWidget.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000//update_freq)

    def update(self):
        # if not np.isclose(self.ts_keeper.time[-1], self.ps.time):
        # Phasors:
        # angle = self.rts.x[self.ps.gen_mdls['GEN'].state_idx['angle']]
        # magnitude = self.ps.gen_mdls['GEN'].input['E_f']
        angle = np.concatenate([self.rts.x[gen_mdl.idx][gen_mdl.state_idx['angle']] for gen_mdl in self.ps.gen_mdls.values()])
        angle -= np.mean(angle)
        magnitude = np.concatenate([gen_mdl.input['E_f'] for gen_mdl in self.ps.gen_mdls.values()])
        phasors = magnitude * np.exp(1j * angle)
        # for i, (pl_ph, phasor) in enumerate(zip(self.pl_ph, phasors[:, None]*self.phasor_0)):
        phasors_points = np.kron(phasors, self.phasor_0)
        self.pl_ph.setData(phasors_points.real, phasors_points.imag)


class TimeSeriesPlot(QtWidgets.QWidget):
    def __init__(self, rts, plots=['angle', 'speed'], update_freq=50, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        self.graphWidget = pg.GraphicsLayoutWidget(show=True, title="Live plot")

        n_samples = 500

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0,
                                            sat=255, alpha=255)
        self.plots = plots
        self.ts_keeper = TimeSeriesKeeper()
        self.ts_keeper.time = np.arange(-n_samples * self.dt, 0, self.dt)
        # self.y = np.zeros_like(self.ts_keeper.time)
        self.pl = {}

        for plot in self.plots:
            graphWidget = self.graphWidget.addPlot(title=plot)
            # p_1 = self.addPlot(title="Updating plot 1")
            # n_series = len(rts.ps.gen_mdls['GEN'].state_idx[plot])
            n_series = sum([len(gen) for gen in self.ps.gen.values()])
            setattr(self.ts_keeper, plot, np.zeros((n_samples, n_series)))

            pl_tmp = []
            for i in range(n_series):
                pen = pg.mkPen(color=self.colors(i), width=2)
                # pl_tmp.append(graphWidget.plot(self.ts_keeper.time, np.zeros(n_samples), pen=pen))
                pl_ph = pg.PlotCurveItem(self.ts_keeper.time, np.zeros(n_samples), pen=pen)
                graphWidget.addItem(pl_ph)
                pl_tmp.append(pl_ph)

            self.pl[plot] = pl_tmp
            self.graphWidget.nextRow()

        self.graphWidget.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000//update_freq)

    def update(self):
        rts = self.rts
        if not np.isclose(self.ts_keeper.time[-1], self.ps.time):
            self.ts_keeper.time = np.append(self.ts_keeper.time[1:], self.ps.time)

            for plot in self.plots:
                old_data = getattr(self.ts_keeper, plot)[1:, :]
                # new_data = getattr(self.ps, plot)
                # new_data = rts.x[rts.ps.gen_mdls['GEN'].state_idx[plot]]
                new_data = np.concatenate([self.rts.x[gen_mdl.idx][gen_mdl.state_idx[plot]] for gen_mdl in self.ps.gen_mdls.values()])
                setattr(self.ts_keeper, plot, np.vstack([old_data, new_data]))
                plot_data = getattr(self.ts_keeper, plot)
                for i, pl in enumerate(self.pl[plot]):
                    pl.setData(self.ts_keeper.time, plot_data[:, i])


class TimeSeriesPlot2(QtWidgets.QWidget):
    def __init__(self, ts_keeper, update_freq=50, title='', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ts_keeper = ts_keeper

        self.graphWidget = pg.GraphicsLayoutWidget(show=True, title=title)

        # n_samples = 50

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0,
                                            sat=255, alpha=255)


        self.plots = self.ts_keeper.columns[1:]  # Assuming time on first column
        self.plot_ax = self.ts_keeper.plt_axis if hasattr(self.ts_keeper, 'plt_axis') else range(
            len(ts_keeper.columns) - 1)
        graphWidgets = []
        for i in range(max(self.plot_ax) + 1):
            graphWidget = self.graphWidget.addPlot()
            graphWidget.addLegend()
            graphWidgets.append(graphWidget)
            self.graphWidget.nextRow()

        self.pl = dict()
        for i, (plot, ax) in enumerate(zip(self.plots, self.plot_ax)):
            pen = pg.mkPen(color=self.colors(i), width=2)
            self.pl[plot] = graphWidgets[ax].plot(self.ts_keeper['t'], np.zeros(self.ts_keeper.n_samples), pen=pen, name=plot)

        self.graphWidget.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 // update_freq)

    def update(self):
        # pass
        for plot in self.plots:
            self.pl[plot].setData(self.ts_keeper['t'], self.ts_keeper[plot])


class TimeSeriesPlot3(QtWidgets.QWidget):
    def __init__(self, ts_keeper=None, plots=None, update_freq=50, title='', *args, **kwargs):
        super().__init__(*args, **kwargs)

        if ts_keeper is not None:
            self.ts_keeper = ts_keeper
        else:
            self.ts_keeper = dps_rts.TimeSeriesKeeper(['t'] + list(plots))

        n_plots = len(plots)

        self.graphWidget = pg.GraphicsLayoutWidget(show=True, title=title)

        # n_samples = 50

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0,
                                            sat=255, alpha=255)
        self.plots = plots
        self.plot_ax = self.ts_keeper.plt_axis if hasattr(self.ts_keeper, 'plt_axis') else [0]*(n_plots)
            # else range(len(columns) - 1)
        graphWidgets = []
        for i in range(max(self.plot_ax) + 1):
            graphWidget = self.graphWidget.addPlot()
            graphWidget.addLegend()
            graphWidgets.append(graphWidget)
            self.graphWidget.nextRow()

        self.pl = dict()
        for i, (plot, ax) in enumerate(zip(self.plots, self.plot_ax)):
            pen = pg.mkPen(color=self.colors(i), width=2)
            self.pl[plot] = graphWidgets[ax].plot(self.ts_keeper['t'], np.zeros(self.ts_keeper.n_samples), pen=pen, name=plot)

        self.graphWidget.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000 // update_freq)

    def get_data(self):
        return np.random.randn(len(self.plots))

    def get_time(self):
        return self.ts_keeper['t'][-1] + 1

    def update_data(self):
        t = self.get_time()
        if not np.isclose(self.ts_keeper['t'][-1], t):
            self.ts_keeper.append([t, *list(self.get_data())])

            for plot in self.plots:
                self.pl[plot].setData(self.ts_keeper['t'], self.ts_keeper[plot])


class LineCurrentPlot(TimeSeriesPlot3):
    def __init__(self, rts, title='Line Current plot', *args, **kwargs):
        self.rts = rts
        self.ps = self.rts.ps
        super().__init__(plots=self.ps.lines['name'], title=title, *args, **kwargs)

    def get_time(self):
        return self.rts.sol.t

    def get_data(self):
        rts = self.rts
        ps = self.rts.ps
        v_full = self.rts.sol.v  #ps.red_to_full.dot(rts.sol.v)
        i_from = ps.v_to_i_lines.dot(v_full)
        i_to = ps.v_to_i_lines.dot(v_full)
        return abs(i_from)


class TimeSeriesPlotFast(QtWidgets.QWidget):
    def __init__(self, rts, plots=['angle', 'speed'], update_freq=50, n_samples=500, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        self.graphWidget = pg.GraphicsLayoutWidget(show=True, title="Live plot")
        # self.graphWidget = pg.PlotWidget()
        # self.setCentralWidget(self.graphWidget)

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0,
                                            sat=255, alpha=255)
        self.plots = plots
        self.ts_keeper = TimeSeriesKeeper()
        self.ts_keeper.time = np.arange(-n_samples * self.dt, 0, self.dt)
        # self.y = np.zeros_like(self.ts_keeper.time)
        self.pl = {}

        for plot in self.plots:
            graphWidget = self.graphWidget.addPlot(title=plot)
            # p_1 = self.addPlot(title="Updating plot 1")
            # n_series = len(rts.ps.gen_mdls['GEN'].state_idx[plot])
            n_series = sum([len(gen) for gen in self.ps.gen.values()])

            setattr(self.ts_keeper, plot, np.zeros((n_samples, n_series)))
            connect = np.ones(n_samples*n_series, dtype=bool)
            connect[n_samples - 1:(n_samples * n_series):n_samples] = False
            self.pl[plot] = graphWidget.plot(np.tile(self.ts_keeper.time, n_series), np.zeros(n_samples*n_series), connect=connect)
            self.graphWidget.nextRow()

        self.graphWidget.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000//update_freq)

    def update(self):
        rts = self.rts
        if not np.isclose(self.ts_keeper.time[-1], self.ps.time):
            self.ts_keeper.time = np.append(self.ts_keeper.time[1:], self.ps.time)

            for plot in self.plots:
                # old_data =
                # new_data = getattr(self.ps, plot)
                # new_data = rts.x[rts.ps.gen_mdls['GEN'].state_idx[plot]]
                new_data = np.concatenate([self.rts.x[gen_mdl.idx][gen_mdl.state_idx[plot]] for gen_mdl in self.ps.gen_mdls.values()])

                setattr(self.ts_keeper, plot, np.vstack([getattr(self.ts_keeper, plot)[1:, :], new_data]))
                plot_data = getattr(self.ts_keeper, plot)

                n_series = len(rts.ps.gen_mdls['GEN'].state_idx[plot])

                x, y = np.tile(self.ts_keeper.time, n_series), plot_data.T.flatten()
                # for i, pl in enumerate(self.pl[plot]):
                self.pl[plot].setData(x, y)


class GridPlot3D(QtWidgets.QWidget):
    def __init__(self, rts, update_freq=50, z_ax='abs_pu', use_colors=False, rotating=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt
        ps = self.ps
        self.z_ax = z_ax

        # nx.draw(G)
        self.scale = 10

        if self.z_ax == 'angle':
            self.scale_z = 10*np.ones(ps.n_bus)
            self.offset_z = 3  # *np.ones(ps.n_bus)
        elif self.z_ax == 'abs':
            self.scale_z = 10 * ps.v_n / max(ps.v_n) * 0.3
            self.offset_z = 0  # np.zeros(ps.n_bus)

        elif self.z_ax == 'abs_pu':
            self.scale_z = 10 * 0.3*(ps.v_n**0)
            self.offset_z = 0  # np.zeros(ps.n_bus)
        # elif self.z_ax == 'both':
        #     self.scale_z = 10*np.ones(ps.n_bus)
        #     self.offset_z = 12*np.ones(ps.n_bus)


        line_admittances = np.zeros(len(ps.lines), dtype=[('Y', float)])
        for i, line in enumerate(ps.lines):
            line_admittances[i] = abs(ps.read_admittance_data('line', line)[2])

        trafo_admittances = np.zeros(len(ps.transformers), dtype=[('Y', float)])
        for i, trafo in enumerate(ps.transformers):
            trafo_admittances[i] = abs(ps.read_admittance_data('transformer', trafo)[2])

        self.G = nx.MultiGraph()
        self.G.add_nodes_from(ps.buses['name'])
        # G.add_edges_from(ps.lines[['from_bus', 'to_bus']])
        # G.add_edges_from(ps.transformers[['from_bus', 'to_bus']])
        self.G.add_weighted_edges_from(
            dps_uf.combine_recarrays(ps.lines, line_admittances)[['from_bus', 'to_bus', 'Y']])
        self.G.add_weighted_edges_from(
            dps_uf.combine_recarrays(ps.transformers, trafo_admittances)[['from_bus', 'to_bus', 'Y']])

        self.grid_layout()

        self.n_edges = len(ps.lines) + len(ps.transformers)
        self.edge_from_bus = np.concatenate([dps_uf.lookup_strings(type_['from_bus'], ps.buses['name']) for type_ in [ps.lines, ps.transformers]])
        self.edge_to_bus = np.concatenate([dps_uf.lookup_strings(type_['to_bus'], ps.buses['name']) for type_ in [ps.lines, ps.transformers]])

        self.edge_x = np.vstack([self.x[self.edge_from_bus], self.x[self.edge_to_bus]]).T
        self.edge_y = np.vstack([self.y[self.edge_from_bus], self.y[self.edge_to_bus]]).T
        self.edge_z = np.vstack([self.z[self.edge_from_bus], self.z[self.edge_to_bus]]).T

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)

        self.window = gl.GLViewWidget()
        # self.window.setBackgroundColor('w')
        self.window.setWindowTitle('Grid')
        self.window.setGeometry(0, 110, 1920, 1080)
        self.window.setCameraPosition(distance=30, elevation=12)
        self.window.show()

        self.rotating = rotating

        self.gz = gl.GLGridItem()
        self.gz.translate(dx=0, dy=0, dz=-self.offset_z)
        self.window.addItem(self.gz)

        color = np.ones((ps.n_bus, 4))
        color[:, -1] = 0.5
        if use_colors:
            color[ps.gen_bus_idx, :] = np.array([self.colors(i).getRgb() for i in range(self.ps.n_gen_bus)]) / 255
        else:
            color[ps.gen_bus_idx, :] = np.array([1 / 3, 2 / 3, 1, 0.5])[None, :]

        self.points = gl.GLScatterPlotItem(
            pos=np.vstack([self.x, self.y, self.z]).T,
            color=color,
            size=15
        )

        self.edge_x_mod = np.append(self.edge_x, np.nan*np.ones((self.n_edges, 1)), axis=1)
        self.edge_y_mod = np.append(self.edge_y, np.nan*np.ones((self.n_edges, 1)), axis=1)
        self.edge_z_mod = np.append(self.edge_z, np.nan*np.ones((self.n_edges, 1)), axis=1)
        # line_x_mod = line_x
        # line_y_mod = line_y
        edge_pos = np.vstack([self.edge_x_mod.flatten(), self.edge_y_mod.flatten(), self.edge_z_mod.flatten()]).T
        line_color = np.ones((self.n_edges, 4))
        line_color[:, -1] = 0.5
        line_color[len(ps.lines):, :] = np.array([1 / 3, 2 / 3, 1, 0.5])[None, :]
        # self.scale_branch_flows = 4
        line_widths = np.ones((self.n_edges, 4))
        self.lines = gl.GLLinePlotItem(pos=edge_pos, color=np.repeat(line_color, 3, axis=0), antialias=True, width=2)

        self.window.addItem(self.points)
        self.window.addItem(self.lines)

        # self.axis = gl.GLAxisItem()

        # self.graphWidget.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000//update_freq)
        self.t_prev = time.time()

    def grid_layout(self, layout='spring_layout'):
        ps = self.ps

        if layout == 'spring_layout':
            pos = getattr(nx, layout)(self.G, seed=0)
        else:
            pos = getattr(nx, layout)(self.G)

        x = np.zeros(ps.n_bus)
        y = np.zeros(ps.n_bus)
        for i, key in enumerate(pos.keys()):
            x[i], y[i] = self.scale * pos[key]

        z = self.scale_z * 1

        self.x = x
        self.y = y
        self.z = z
        self.x0 = x.copy()
        self.y0 = y.copy()
        self.z0 = z.copy()

    def update(self):
        if self.rotating:
            x, y, z = self.window.cameraPosition()
            t_now = time.time()
            dt = t_now - self.t_prev
            self.t_prev = t_now
            new_angle = np.arctan2(y, x) * 180 / np.pi + 0.01*360*dt
            self.window.setCameraPosition(azimuth=new_angle)

        ps = self.rts.ps

        v = ps.red_to_full.dot(self.rts.sol.v)
        if self.z_ax == 'angle':
            gen_mean_angle = np.mean(np.unwrap(self.rts.x[ps.gen_mdls['GEN'].state_idx_global['angle']]))
            v_angle = np.angle(v) - gen_mean_angle
            v_angle = (v_angle + np.pi) % (2*np.pi) - np.pi
            v_angle -= np.mean(v_angle)
            k = v_angle
            self.z = self.scale_z*k + self.offset_z
        elif self.z_ax in ['abs', 'abs_pu']:
            k = abs(v)
            self.z = self.scale_z*k + self.offset_z

        # Branch currents
        # flows = abs(np.concatenate([ps.v_to_i_lines.dot(v), ps.v_to_i_trafos.dot(v)]))

        # elif self.z_ax == 'both':
        #     gen_mean_angle = np.mean(np.unwrap(self.rts.x[ps.gen_mdls['GEN'].state_idx_global['angle']]))
        #     v_angle = np.angle(v) - gen_mean_angle
        #     v_angle = (v_angle + np.pi) % (2*np.pi) - np.pi
        #     k = v_angle
        #     self.z = self.scale_z * v_angle + self.offset_z
        #     self.x = (abs(v) - 1)*self.scale_x


        # amp = self.scale*0.1
        # dx = amp*np.cos(v_angle)
        # dy = amp*np.sin(v_angle)
        # self.x = self.x0 + dx
        # self.y = self.y0 + dy

        self.edge_x = np.vstack([self.x[self.edge_from_bus], self.x[self.edge_to_bus]]).T
        self.edge_y = np.vstack([self.y[self.edge_from_bus], self.y[self.edge_to_bus]]).T
        self.edge_z = np.vstack([self.z[self.edge_from_bus], self.z[self.edge_to_bus]]).T

        self.edge_x_mod = np.append(self.edge_x, np.nan * np.ones((self.n_edges, 1)), axis=1)
        self.edge_y_mod = np.append(self.edge_y, np.nan * np.ones((self.n_edges, 1)), axis=1)
        self.edge_z_mod = np.append(self.edge_z, np.nan * np.ones((self.n_edges, 1)), axis=1)
        edge_pos = np.vstack([self.edge_x_mod.flatten(), self.edge_y_mod.flatten(), self.edge_z_mod.flatten()]).T

        self.points.setData(pos=np.vstack([self.x, self.y, self.z]).T)
        self.lines.setData(pos=edge_pos)


class SimulationStatsPlot(QtWidgets.QWidget):
    def __init__(self, rts, update_freq=50, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        self.graphWidget = pg.GraphicsLayoutWidget(show=True, title="Simulation Stats")

        n_samples = 50

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0,
                                            sat=255, alpha=255)

        self.ts_keeper = TimeSeriesKeeper()
        self.ts_keeper.time = np.arange(-n_samples * self.dt, 0, self.dt)
        # self.y = np.zeros_like(self.ts_keeper.time)
        self.pl = {}
        # self.axes = [0, 1, 1]
        self.plots = ['dt_loop', 'dt_ideal', 'dt_sim', 'dt_err']
        self.plot_ax = [0, 0, 0, 1]
        graphWidgets = [self.graphWidget.addPlot(title='Synchronization') for i in range(max(self.plot_ax)+1)]

        for i, (plot, ax) in enumerate(zip(self.plots, self.plot_ax)):
            setattr(self.ts_keeper, plot, np.zeros(n_samples))
            pen = pg.mkPen(color=self.colors(i), width=2)
            self.pl[plot] = graphWidgets[ax].plot(self.ts_keeper.time, np.zeros(n_samples), pen=pen)
            # self.graphWidget.nextRow()

        self.graphWidget.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000//update_freq)

    def update(self):
        if not np.isclose(self.ts_keeper.time[-1], self.ps.time):
            self.ts_keeper.time = np.append(self.ts_keeper.time[1:], self.ps.time)

            for plot in self.plots:
                old_data = getattr(self.ts_keeper, plot)[1:]
                new_data = getattr(self.rts, plot)
                setattr(self.ts_keeper, plot, np.append(old_data, new_data))
                plot_data = getattr(self.ts_keeper, plot)
                # for i, pl in enumerate(self.pl[plot]):
                self.pl[plot].setData(self.ts_keeper.time, plot_data)


class SimulationStatsPlot2(QtWidgets.QWidget):
    def __init__(self, rts, update_freq=50, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = rts_apps.Logger(rts=rts, n_samples=500)
        self.logger.start()
        # self.rts = rts
        # self.ps = rts.ps
        # self.dt = self.rts.dt

        self.graphWidget = pg.GraphicsLayoutWidget(show=True, title="Simulation Stats")

        # n_samples = 50

        # self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)

        # self.ts_keeper = TimeSeriesKeeper()
        # self.ts_keeper.time = np.arange(-n_samples * self.dt, 0, self.dt)
        # # self.y = np.zeros_like(self.ts_keeper.time)
        # self.pl = {}
        # # self.axes = [0, 1, 1]
        # self.plots = self.logger.log_attributes[1:]
        self.plots = self.logger.log_attributes[-1:]
        self.plot_ax = [0]
        graphWidgets = [self.graphWidget.addPlot(title='Synchronization') for i in range(max(self.plot_ax)+1)]

        self.pl = dict()
        for i, (plot, ax) in enumerate(zip(self.plots, self.plot_ax)):
            # pen = pg.mkPen(color=self.colors(i), width=2)
            self.pl[plot] = graphWidgets[ax].plot(self.logger.ts_keeper['t'], np.zeros(self.logger.ts_keeper.n_samples))  # , pen=pen)

        self.graphWidget.show()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000//update_freq)

    def update(self):
        # pass
        for plot in self.plots:
            self.pl[plot].setData(self.logger.ts_keeper['t'], self.logger.ts_keeper[plot])


def gui_1(rts):
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)

    phasor_plot = PhasorPlotFast(rts, update_freq=30)
    ts_plot = TimeSeriesPlotFast(rts, ['speed'], update_freq=30)  # , 'speed', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st'])
    stats_plot = SimulationStatsPlot2(rts, update_freq=30)

    # Add Control Widgets
    line_outage_ctrl = LineOutageWidget(rts)
    sc_ctrl = ShortCircuitWidget(rts)
    gen_ctrl = GenCtrlWidget(rts)
    sim_ctrl = SimulationControl(rts)

    console = PythonConsole()
    console.push_local_ns('rts', rts)
    console.show()
    console.eval_in_thread()

    app.exec_()

    return app


def main(rts):
    update_freq = 25
    app = QtWidgets.QApplication(sys.argv)
    phasor_plot = PhasorPlot(rts, update_freq=update_freq)
    grid_plot = GridPlot3D(rts, update_freq=update_freq, z_ax='abs')
    grid_plot_angle = GridPlot3D(rts, update_freq=update_freq, z_ax='angle')
    ts_plot = TimeSeriesPlotFast(rts, ['angle', 'speed'], update_freq=update_freq)  # , 'speed', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st'])
    # stats_plot = SimulationStatsPlot(rts, update_freq=update_freq)  # , 'speed', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st'])
    stats_plot = SimulationStatsPlot2(rts, update_freq=update_freq)  # , 'speed', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st'])

    # line_current_keeper = TimeSeriesKeeperWidget(rts)
    # line_current_plot = TimeSeriesPlot2(line_current_keeper.ts_keeper, title='TCSC control')

    # simple_pod_control_test = SimplePODTestControl(rts)

    # Add Control Widgets
    line_outage_ctrl = LineOutageWidget(rts)
    sc_ctrl = ShortCircuitWidget(rts)
    excitation_ctrl = GenCtrlWidget(rts)

    # console = PythonConsole()
    console = PythonConsole()
    console.push_local_ns('rts', rts)
    # console.push_local_ns('ts_plot', ts_plot)
    console.push_local_ns('phasor_plot', phasor_plot)
    console.push_local_ns('line_outage_ctrl', line_outage_ctrl)
    console.push_local_ns('excitation_ctrl', excitation_ctrl)
    console.show()
    console.eval_in_thread()
    app.exec_()

    return app
    # sys.exit(app.exec_())


if __name__ == '__main__':

    import ps_models.ieee39 as model_data
    model = model_data.load()

    [importlib.reload(mdl) for mdl in [dps_rts, dps]]
    ps = dps.PowerSystemModel(model=model)
    ps.use_numba = False

    ps.power_flow()
    ps.build_y_bus_red(ps.buses['name'])
    ps.init_dyn_sim()
    ps.ode_fun(0, ps.x0)
    rts = dps_rts.RealTimeSimulatorThread(ps, dt=10e-3, speed=1)
    # ShortCircuitWidget(rts)
    # grid_plot = GridPlot3D(rts, update_freq=25, z_ax='abs')

    rts.start()

    from threading import Thread
    app = main(rts)
    rts.stop()
