from PySide6 import QtWidgets
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
import dynpssimpy.utility_functions as dps_uf
import networkx as nx
import time
from dynpssimrt.interfacing import InterfacerQueuesThread


class GridPlot3D(QtWidgets.QWidget):
    def __init__(
        self,
        bus_names,
        n_lines,
        n_trafos,
        lines_from_to_y,
        trafos_from_to_y,
        from_buses_all,
        to_buses_all,
        offset_z,
        scale=10,
        update_freq=50,
        z_ax='angle',
        use_colors=False
    ):
        super().__init__()
        self.z_ax = z_ax
        self.scale = scale

        # nx.draw(G)
        self.n_bus = n_bus = len(bus_names)
        


        self.G = nx.MultiGraph()
        self.G.add_nodes_from(bus_names)
        # G.add_edges_from(ps.lines[['from_bus', 'to_bus']])
        # G.add_edges_from(ps.transformers[['from_bus', 'to_bus']])

        self.G.add_weighted_edges_from(lines_from_to_y)
        self.G.add_weighted_edges_from(trafos_from_to_y)
        
        self.grid_layout()

        self.n_edges = n_lines + n_trafos
        self.edge_from_bus = from_buses_all
        self.edge_to_bus = to_buses_all

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

        # self.rotating = rotating

        self.gz = gl.GLGridItem()
        self.gz.translate(dx=0, dy=0, dz=-offset_z)
        self.window.addItem(self.gz)

        color = np.ones((self.n_bus, 4))
        color[:, -1] = 0.5
        if use_colors:
            color = np.array([self.colors(i).getRgb() for i in range(self.n_bus)]) / 255
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

        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(self.update)
        # self.timer.start(1000//self.update_freq)
        # self.t_prev = time.time()

    def grid_layout(self, layout='spring_layout'):
        # ps = self.ps

        if layout == 'spring_layout':
            pos = getattr(nx, layout)(self.G, seed=0)
        else:
            pos = getattr(nx, layout)(self.G)

        x = np.zeros(self.n_bus)
        y = np.zeros(self.n_bus)
        for i, key in enumerate(pos.keys()):
            x[i], y[i] = self.scale * pos[key]

        z = np.ones(self.n_bus) * 1

        self.x = x
        self.y = y
        self.z = z
        self.x0 = x.copy()
        self.y0 = y.copy()
        self.z0 = z.copy()

    def update(self, z):

        # if self.rotating:
        #     x, y, z = self.window.cameraPosition()
        #     t_now = time.time()
        #     dt = t_now - self.t_prev
        #     self.t_prev = t_now
        #     new_angle = np.arctan2(y, x) * 180 / np.pi + 0.01*360*dt
        #     self.window.setCameraPosition(azimuth=new_angle)


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
        # self.x = x
        # self.y = y
        self.z = z

        self.edge_x = np.vstack([self.x[self.edge_from_bus], self.x[self.edge_to_bus]]).T
        self.edge_y = np.vstack([self.y[self.edge_from_bus], self.y[self.edge_to_bus]]).T
        self.edge_z = np.vstack([self.z[self.edge_from_bus], self.z[self.edge_to_bus]]).T

        self.edge_x_mod = np.append(self.edge_x, np.nan * np.ones((self.n_edges, 1)), axis=1)
        self.edge_y_mod = np.append(self.edge_y, np.nan * np.ones((self.n_edges, 1)), axis=1)
        self.edge_z_mod = np.append(self.edge_z, np.nan * np.ones((self.n_edges, 1)), axis=1)
        edge_pos = np.vstack([self.edge_x_mod.flatten(), self.edge_y_mod.flatten(), self.edge_z_mod.flatten()]).T

        self.points.setData(pos=np.vstack([self.x, self.y, self.z]).T)
        self.lines.setData(pos=edge_pos)


class LiveGridPlot3D(InterfacerQueuesThread):
    def __init__(self, rts=None, update_freq=50, z_ax='angle', use_colors=False, *args, **kwargs):
        self.z_ax = z_ax
        InterfacerQueuesThread.__init__(self, rts, fs=update_freq)
        
        
    @staticmethod
    def get_init_data(rts):
        ps = rts.ps
        bus_names = ps.buses['name']
        line_par = ps.lines['Line'].par
        trafo_par = ps.trafos['Trafo'].par
        line_admittances_ = ps.lines['Line'].admittance
        trafo_admittances_ = ps.trafos['Trafo'].admittance
        # v_n = ps.
        return bus_names, line_par, trafo_par, line_admittances_, trafo_admittances_ 

    def initialize(self, init_data):
        bus_names, line_par, trafo_par, line_admittances_, trafo_admittances_  = init_data
        self.n_bus = len(bus_names)
        # v_n = 
        n_lines = len(line_par)
        n_trafos = len(trafo_par)

        if self.z_ax == 'angle':
            self.scale_z = 10*np.ones(self.n_bus)
            self.offset_z = 3  # *np.ones(ps.n_bus)
        # elif self.z_ax == 'abs':
            # self.scale_z = 10 * v_n / max(v_n) * 0.3
            # self.offset_z = 0  # np.zeros(ps.n_bus)

        elif self.z_ax == 'abs_pu':
            self.scale_z = 10 * 0.3*np.ones(self.n_bus)
            self.offset_z = 0  # np.zeros(ps.n_bus)
        # elif self.z_ax == 'both':
        #     self.scale_z = 10*np.ones(ps.n_bus)
        #     self.offset_z = 12*np.ones(ps.n_bus)
        
        line_admittances = np.zeros(n_lines, dtype=[('Y', float)])
        line_admittances[:] = abs(line_admittances_)
        
        trafo_admittances = np.zeros(n_trafos, dtype=[('Y', float)])
        trafo_admittances[:] = abs(trafo_admittances_)
        
        lines_from_to_y = dps_uf.combine_recarrays(
            line_par,
            line_admittances
        )[['from_bus', 'to_bus', 'Y']]

        trafos_from_to_y = dps_uf.combine_recarrays(
            trafo_par,
            trafo_admittances
        )[['from_bus', 'to_bus', 'Y']]

        from_buses_all = np.concatenate(
            [dps_uf.lookup_strings(type_par['from_bus'], bus_names) for type_par in [line_par, trafo_par]])
        to_buses_all = np.concatenate(
            [dps_uf.lookup_strings(type_par['to_bus'], bus_names) for type_par in [line_par, trafo_par]])

        self.grid_plot = GridPlot3D(
            bus_names,
            n_lines,
            n_trafos,
            lines_from_to_y,
            trafos_from_to_y,
            from_buses_all,
            to_buses_all,
            self.offset_z,
        )

    @staticmethod
    def read_input_signal(rts):
        voltages = rts.ps.red_to_full.dot(rts.sol.v)
        gen_angles = rts.sol.x[rts.ps.gen['GEN'].state_idx_global['speed']]
        return voltages, gen_angles
    
    def update(self, input):
        voltages, gen_angles = input
        
        v = voltages
        if self.z_ax == 'angle':
            gen_mean_angle = np.mean(np.unwrap(gen_angles))
            v_angle = np.angle(v) - gen_mean_angle
            v_angle = (v_angle + np.pi) % (2*np.pi) - np.pi
            v_angle -= np.mean(v_angle)
            k = v_angle
            self.z = self.scale_z*k + self.offset_z
        elif self.z_ax in ['abs', 'abs_pu']:
            k = abs(v)
            self.z = self.scale_z*k + self.offset_z

        self.grid_plot.update(self.z)
