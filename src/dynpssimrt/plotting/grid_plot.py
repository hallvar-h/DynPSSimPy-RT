from PySide6 import QtWidgets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
# import dynpssimpy.dynamic as dps
# import dynpssimpy.real_time_sim.apps as rts_apps
import dynpssimpy.utility_functions as dps_uf
# import importlib
# from pyqtconsole.console import PythonConsole
# import dynpssimpy.real_time_sim.sim as dps_rts
import networkx as nx
import time


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


        line_admittances = np.zeros(ps.lines['Line'].n_units, dtype=[('Y', float)])
        line_admittances[:] = abs(ps.lines['Line'].admittance)
        # for i, line in enumerate(ps.lines):
            # line_admittances[i] = abs(ps.read_admittance_data('line', line)[2])

        trafo_admittances = np.zeros(ps.trafos['Trafo'].n_units, dtype=[('Y', float)])
        trafo_admittances[:] = abs(ps.trafos['Trafo'].admittance)
        # for i, trafo in enumerate(ps.transformers):
            # trafo_admittances[i] = abs(ps.read_admittance_data('transformer', trafo)[2])

        self.G = nx.MultiGraph()
        self.G.add_nodes_from(ps.buses['name'])
        # G.add_edges_from(ps.lines[['from_bus', 'to_bus']])
        # G.add_edges_from(ps.transformers[['from_bus', 'to_bus']])
        self.G.add_weighted_edges_from(
            dps_uf.combine_recarrays(ps.lines['Line'].par, line_admittances)[['from_bus', 'to_bus', 'Y']])
        self.G.add_weighted_edges_from(
            dps_uf.combine_recarrays(ps.trafos['Trafo'].par, trafo_admittances)[['from_bus', 'to_bus', 'Y']])

        self.grid_layout()

        self.n_edges = ps.lines['Line'].n_units + ps.trafos['Trafo'].n_units
        self.edge_from_bus = np.concatenate([dps_uf.lookup_strings(type_.par['from_bus'], ps.buses['name']) for type_ in [ps.lines['Line'], ps.trafos['Trafo']]])
        self.edge_to_bus = np.concatenate([dps_uf.lookup_strings(type_.par['to_bus'], ps.buses['name']) for type_ in [ps.lines['Line'], ps.trafos['Trafo']]])

        self.edge_x = np.vstack([self.x[self.edge_from_bus], self.x[self.edge_to_bus]]).T
        self.edge_y = np.vstack([self.y[self.edge_from_bus], self.y[self.edge_to_bus]]).T
        self.edge_z = np.vstack([self.z[self.edge_from_bus], self.z[self.edge_to_bus]]).T

        self.colors = lambda i: pg.intColor(i, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255)

        self.window = gl.GLViewWidget()
        # self.window.setBackgroundColor('w')
        self.window.setWindowTitle('Grid')
        self.window.setGeometry(0, 110, 1000, 500)
        self.window.setCameraPosition(distance=30, elevation=12)
        self.window.show()

        self.rotating = rotating

        self.gz = gl.GLGridItem()
        self.gz.translate(dx=0, dy=0, dz=-self.offset_z)
        self.window.addItem(self.gz)

        color = np.ones((ps.n_bus, 4))
        color[:, -1] = 0.5
        # if use_colors:
            # color[ps.gen_bus_idx, :] = np.array([self.colors(i).getRgb() for i in range(self.ps.n_gen_bus)]) / 255
        # else:
            # color[ps.gen_bus_idx, :] = np.array([1 / 3, 2 / 3, 1, 0.5])[None, :]

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
        # line_color[len(ps.lines):, :] = np.array([1 / 3, 2 / 3, 1, 0.5])[None, :]
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
            gen_mean_angle = np.mean(np.unwrap(self.rts.x[ps.gen['GEN'].state_idx_global['angle']]))
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
