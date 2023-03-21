import numpy as np
from .time_window import TimeWindow
import PySide6.QtCore as QtCore
import pyqtgraph as pg


class TimeWindowPlot(TimeWindow):
    def __init__(self, update_freq=25, title='', max_plots=10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.graphWidget = pg.GraphicsLayoutWidget(show=True, title=title)

        self.colors = lambda i: pg.intColor(
            i, hues=9, values=1, maxValue=255, minValue=150,
            maxHue=360, minHue=0, sat=255, alpha=255
        )

        self.plotWidget = self.graphWidget.addPlot()
        self.pl = []
        for i in range(min(max_plots, self.n_cols)):
            pen = pg.mkPen(color=self.colors(i), width=2)
            pl = self.plotWidget.plot(self.get_time(), np.zeros(self.n_samples), pen=pen)
            self.pl.append(pl)

        self.graphWidget.show()

        if update_freq is not None:
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_plot)
            self.timer.start(1000 // update_freq)

    def update_plot(self):
        time_stamps = self.get_time()
        # time_stamp = np.arange(self.ts_keeper.n_samples)
        y_data = self.get_col()

        for i, pl in enumerate(self.pl):
            pl.setData(time_stamps, y_data[:, i])