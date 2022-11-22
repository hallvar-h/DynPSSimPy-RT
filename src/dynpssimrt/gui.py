from PySide6 import QtWidgets, QtCore
from pyqtgraph.console import ConsoleWidget
import numpy as np


class LineOutageWidget(QtWidgets.QWidget):
    def __init__(self, rts, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps
        self.dt = self.rts.dt

        # Controls
        self.ctrlWidget = QtWidgets.QWidget()
        self.ctrlWidget.setWindowTitle('Lines')

        lines_per_col = 10
        layout_box = QtWidgets.QGridLayout()

        self.check_boxes = []
        n_lines = self.ps.lines['Line'].n_units
        n_cols = np.ceil(n_lines/lines_per_col)
        for i, line in enumerate(self.ps.lines['Line'].par):
            check_box = QtWidgets.QCheckBox(line['name'])
            check_box.setChecked(True)
            check_box.stateChanged.connect(self.updateLines)
            check_box.setAccessibleName(line['name'])

            row = i%lines_per_col
            col = int((i - row)/lines_per_col)
            
            layout_box.addWidget(check_box, row, col)
            # layout_box.addSpacing(15)
            # layout_box.addSpacing(0)
        

        self.ctrlWidget.setLayout(layout_box)
        self.ctrlWidget.show()

    def updateLines(self):
        if self.sender().isChecked():
            action = 'connect'
        else:
            action = 'disconnect'
        self.ps.lines['Line'].event(self.ps, self.sender().accessibleName(), action)
        # self.ps.network_event('line', self.sender().accessibleName(), action)
        print('t={:.2f}s'.format(self.rts.sol.t) + ': Line ' + self.sender().accessibleName() + ' '+ action + 'ed.')


class DynamicLoadControlWidget(QtWidgets.QWidget):
    def __init__(self, rts, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rts = rts
        self.ps = rts.ps

        # Controls
        self.ctrlWidget = QtWidgets.QWidget()
        self.ctrlWidget.setWindowTitle('DynamicLoadControlWidget')

        layout_box = QtWidgets.QGridLayout()
        self.check_boxes = []
        self.load_mdl = self.ps.loads['DynamicLoad']
        self.sliders_G = []
        self.sliders_B = []
        for i, dyn_load in enumerate(self.load_mdl.par):
            y_load_0 = self.load_mdl.y_load[i]
            G_0 = y_load_0.real
            B_0 = y_load_0.imag
            
             # Add slider
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.ctrlWidget)
            self.speed_slider = slider
            slider.setMinimum(1)
            slider.setMaximum(300)
            slider.valueChanged.connect(lambda state, i=i, target='G': self.updateLoad(i, target))
            slider.setAccessibleName(dyn_load['name'])
            slider.setValue(round(G_0*100))
            layout_box.addWidget(slider, i, 0)
            self.sliders_G.append(slider)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.ctrlWidget)
            self.speed_slider = slider
            slider.setMinimum(1)
            slider.setMaximum(300)
            slider.valueChanged.connect(lambda state, i=i, target='B': self.updateLoad(i, target))
            slider.setAccessibleName(dyn_load['name'])
            slider.setValue(round(B_0*100))
            layout_box.addWidget(slider, i, 1)
            self.sliders_B.append(slider)

        self.ctrlWidget.setLayout(layout_box)
        self.ctrlWidget.show()

    def updateLoad(self, load_idx, target):
        print(load_idx, target)
        y_prev = self.load_mdl.y_load[load_idx]
        self.load_mdl.y_load[load_idx] = self.sliders_G[load_idx].value()/100 + 1j*self.sliders_B[load_idx].value()/100
        #print(self.load_mdl.p(x, v)*self.rts.ps.s_n)
        # self.ps.network_event('line', self.sender().accessibleName(), action)
        #print('t={:.2f}s'.format(self.rts.sol.t) + ': Line ' + self.sender().accessibleName() + ' '+ action + 'ed.')


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
        self.speed_slider = slider
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
        self.rts.speed = self.speed_slider.value()/100

    def resetSimulation(self):
        self.rts.toggle_pause()
        # self.ps.init_dyn_sim()
        self.rts.sol.x[:] = self.ps.x_0
        self.rts.sol.v[:] = self.ps.v_0
        self.rts.toggle_pause()

    def pauseSimulation(self):
        self.rts.toggle_pause()


