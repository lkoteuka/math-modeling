import numpy as np
import math
import json
import csv
import sys
import time
import ctypes
import OpenGL
import OpenGL.GL as gl
import matplotlib.pyplot as plt

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtOpenGL
from PyQt5.QtCore import QThread

from OpenGL import GLU

from scipy.integrate import odeint
from scipy.optimize import fsolve


class TimeThread(QThread):
    def __init__(self, mainwindow, parent=None):
        super().__init__()
        self.mainwindow = mainwindow

    def run(self):
        f = open(temperature_file, "w+")
        f.close()

        self.mainwindow.glWidget.T0 = []

        for t_i in range(time_steps):
            self.mainwindow.glWidget.t = np.linspace(t_i, t_i+1, 2)
            self.mainwindow.glWidget.Parsing(model_name, parameters_file, temperature_file, temp_max, temp_min)
            
            if t_i == 0:
                self.mainwindow.Time_.setText(f"Steps:\t\t\t {t_i} from {time_steps}")
                self.mainwindow.slider.setValue(0)
            
            self.mainwindow.Time_.setText(f"Steps:\t\t\t {t_i + 1} from {time_steps}")
            self.mainwindow.slider.setValue(t_i + 1)
            time.sleep(0.05)
            
           

class GLWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)
            
    def initializeGL(self):
        self.qglClearColor(QtGui.QColor(255, 255, 255))
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.Parts = []
        self.T = []
        self.T0 = []
        self.t = []

        f = open(temperature_file, "w+")
        f.close()

        self.coef_b = 20
        self.coef_c = 3
        self.d = 4

        self.qe = 100
        self.deg = 4

        self.rotX = 0.0
        self.rotY = 0.0
        self.rotZ = 0.0
         
    def resizeGL(self, width, height):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)

        GLU.gluPerspective(45.0, aspect, 1.0, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        for i in range(16):
            gl.glBegin(gl.GL_QUADS)
            gl.glColor3f (1.0 - i*16/256, 0.0, 0.0 + i*16/256)
            gl.glVertex3fv([-5.25, 4.0 - i*0.5, -10.0])
            gl.glVertex3fv([-4.75, 4.0 - i*0.5, -10.0])
            gl.glVertex3fv([-4.75, 3.5-  i*0.5, -10.0])
            gl.glVertex3fv([-5.25, 3.5 - i*0.5, -10.0])
            gl.glEnd()

        gl.glPushMatrix()

        gl.glTranslate(0.0, 5.0, -40.0)
        gl.glRotate(self.rotX, 1.0, 0.0, 0.0)
        gl.glRotate(self.rotY, 0.0, 1.0, 0.0)
        gl.glRotate(self.rotZ, 0.0, 0.0, 1.0)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        
        if len(self.Parts) != 0:
            buffer_offset = ctypes.c_void_p
            float_size = ctypes.sizeof(ctypes.c_float)

            gl.glVertexPointer(3, gl.GL_FLOAT, 24, buffer_offset(self.VtxArray.ctypes.data))
            gl.glColorPointer(3, gl.GL_FLOAT, 24, buffer_offset(self.VtxArray.ctypes.data + float_size * 3))
            
            for i in range(self.count):
                gl.glDrawElements(gl.GL_TRIANGLES, len(self.IdxArray[i]), gl.GL_UNSIGNED_INT, self.IdxArray[i])

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        gl.glPopMatrix()

    def Parsing(self, Mname, Pname, Tname, Tmax, Tmin):
        if len(self.Parts) == 0:
            self.count = 5
            Part_i = np.zeros(self.count)

            self.Parts = []
            for i in range(self.count):
                self.Parts.append([])
                for vf in range(2):
                    self.Parts[i].append([])
            
            for line in open(Mname, "r"):
                values = line.split()
                if not values: continue

                if len(values) == 3 and values[2].startswith("Part"):
                    print(values)
                    for i in range(self.count):
                        Part_i[i] = 0
                    part_ind = int(values[2][4])
                    # switch parts 3 and 4
                    if part_ind == 3:
                        part_ind = 4
                    elif part_ind == 4:
                        part_ind = 3
                    Part_i[part_ind - 1] = 1

                if values[0] == 'v':
                    index = 0
                    for i in range(self.count):
                        if Part_i[i] == 1:
                            index = i
                    self.Parts[index][0].append(list(map(float, values[1:4])))

                if values[0] == 'f':
                    index = 0
                    for i in range(self.count):
                        if Part_i[i] == 1:
                            index = i
                
                    if len(self.Parts[index][1]) == 0:
                        err = 1
                    for i in range(len(values) - 1):
                        values[i + 1] = int(values[i + 1]) - err
                    self.Parts[index][1].append(list(map(int, values[1:4])))
            
            self.S_ij = np.genfromtxt('S_ij.csv', delimiter=';')
            print(self.S_ij)

        with open(Pname, 'r') as f:
            param = json.load(f)
        self.C = param['c']
        self.Lamb = param['lamb']
        self.E = param['e']
        self.Q = param['Q']

        self.T = self.Solve(Tname)

    def Solve(self, Tname):
        if len(self.T0) == 0:
            # self.T0 = fsolve(self.Stationary_solution, np.zeros(self.count), args=(self.t[0]))
            # print(self.t)
            self.T0 = fsolve(self.ode, np.zeros(self.count), args=(self.t[0]))

            # self.T0 = [20.0] * 5
            self.T0 = [20, 0, 20, 20, 20]
            # self.T = odeint(self.ODE, self.T0, self.t)
            print(self.T0)
            self.T = odeint(self.ode, self.T0, self.t)

            temp_data = []
            temp_data.append([self.t[0], self.T[0][0], self.T[0][1], self.T[0][2], self.T[0][3], self.T[0][4]])
            myFile = open(Tname, 'a', newline="")
            with myFile:
                writer = csv.writer(myFile, delimiter=';')
                writer.writerows(temp_data)
        else:
            # self.T = odeint(self.ODE, self.T[1], self.t)
            self.T = odeint(self.ode, self.T[1], self.t)

            temp_data = []
            for i in range(len(self.t) - 1):
                temp_data.append([self.t[i], self.T[i][0], self.T[i][1], self.T[i][2], self.T[i][3], self.T[i][4]])
            myFile = open(Tname, 'a', newline="")
            with myFile:
                writer = csv.writer(myFile, delimiter=';')
                writer.writerows(temp_data)

        return self.T

    def Plot_sol(self):
        with open(temperature_file) as File:
            reader = csv.reader(File, delimiter=';', quotechar=',', quoting=csv.QUOTE_MINIMAL)
            T_plot = []
            for row in reader:
                T_plot.append([float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
            
            t_plot = np.linspace(0, time_steps, time_steps)
            T_plot = np.array(T_plot)

            plt.plot(t_plot, T_plot[:, 0], 'b', label='Part1')
            plt.plot(t_plot, T_plot[:, 1], 'g', label='Part2')
            plt.plot(t_plot, T_plot[:, 2], 'r', label='Part3')
            plt.plot(t_plot, T_plot[:, 3], 'c', label='Part4')
            plt.plot(t_plot, T_plot[:, 4], 'm', label='Part5')
            plt.legend(loc='best')
            plt.xlabel('t')
            plt.grid()
            plt.show()

    def ode(self, T, t):
        T1, T2, T3, T4, T5 = T
        C0 = 5.67
        dTdt = [(self.Lamb[0] * self.S_ij[0][1] * (T2 - T1) \
                    - self.E[0] * self.S_ij[0][0] * C0 * (T1/self.qe)**self.deg) / self.C[0],
                
                (self.Lamb[1] * self.S_ij[1][2] * (T3 - T2) \
                    + self.Lamb[0] * self.S_ij[1][0] * (T1 - T2) \
                        - self.E[1] * self.S_ij[1][1] * C0 * (T2/self.qe)**self.deg) / self.C[1],
                
                (self.Lamb[2] * self.S_ij[2][3] * (T4 - T3) \
                    + self.Lamb[1] * self.S_ij[2][1] * (T2 - T3) \
                        - self.E[2] * self.S_ij[2][2] * C0 * (T3/self.qe)**self.deg) / self.C[2],
                
                (self.Lamb[3] * self.S_ij[3][4] * (T5 - T4) \
                    + self.Lamb[2] * self.S_ij[3][2] * (T3 - T4) \
                        - self.E[3] * self.S_ij[3][3] * C0 * (T4/self.qe)**self.deg) / self.C[3],
                
                (self.Lamb[3] * self.S_ij[4][3] * (T4 - T5) \
                    - self.E[4] * self.S_ij[4][4] * C0 * (T5/self.qe)**self.deg \
                        + self.Q[4] * (self.coef_b + self.coef_c * math.sin(t/self.d))) / self.C[4]]
        return dTdt


    def initGeometry(self, Tmax, Tmin):
        if Tmax < Tmin:
            Tmax, Tmin = Tmin, Tmax
        
        if len(self.Parts) != 0:
            if self.t[0] == 0:
                for t_i in range(2):
                    Vtx = []
                    for i in range(self.count):
                        for j in range(len(self.Parts[i][0])):
                            Vtx += self.Parts[i][0][j]
                            Vtx += self.color_red(self.T[t_i][i] - Tmin, Tmax - Tmin)
                    self.VtxArray = np.array(Vtx, dtype = np.float32)

                    self.IdxArray = []
                    for i in range(self.count):
                        self.IdxArray.append(np.array(sum(self.Parts[i][1], [])))
                    
                    self.glDraw()
            else:
                Vtx = []
                for i in range(self.count):
                    for j in range(len(self.Parts[i][0])):
                        Vtx += self.Parts[i][0][j]
                        Vtx += self.color_red(self.T[1][i] - Tmin, Tmax - Tmin)
                self.VtxArray = np.array(Vtx, dtype = np.float32)

                self.IdxArray = []
                for i in range(self.count):
                    self.IdxArray.append(np.array(sum(self.Parts[i][1], [])))
                
                self.glDraw()

    def color_red(self, val, delta):
        val = val / delta
        return [0.0 + val, 0.0, 1.0 - val]

    def setRotY(self, val):
        self.rotY = np.pi * val
        self.glDraw()

        
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        
        self.resize(800, 900)
        self.setWindowTitle('Task1')

        self.glWidget = GLWidget(self)

        self.initGUI()
        
    def initGUI(self):
        central_widget = QtWidgets.QWidget()
        gui_layout = QtWidgets.QGridLayout()
        central_widget.setLayout(gui_layout)

        self.setCentralWidget(central_widget)

        gui_layout.addWidget(self.glWidget, 0, 0, 1, 0)

        self.Max = QtWidgets.QLabel(f'Maximum temperature:\t {temp_max}')
        self.Max.setFixedSize(450, 20)
        gui_layout.addWidget(self.Max, 2, 0)

        self.Min = QtWidgets.QLabel(f'Minimum temperature:\t {temp_min}')
        self.Min.setFixedSize(450, 20)
        gui_layout.addWidget(self.Min, 3, 0)

        self.Time_name = QtWidgets.QLabel(f'Time steps:\t\t {time_steps}')
        self.Time_name.setFixedSize(450, 20)
        gui_layout.addWidget(self.Time_name, 4, 0)

        self.Mbtn = QtWidgets.QLabel(f"Model file:\t\t {model_name}")
        self.Mbtn.setFixedSize(450, 20)
        gui_layout.addWidget(self.Mbtn, 5, 0)

        self.Pbtn = QtWidgets.QLabel(f"Parameter file:\t\t {parameters_file}")
        self.Pbtn.setFixedSize(450, 20)
        gui_layout.addWidget(self.Pbtn, 6, 0)

        self.Tbtn = QtWidgets.QLabel(f"Temperature file:\t\t {temperature_file}")
        self.Tbtn.setFixedSize(450, 20)
        gui_layout.addWidget(self.Tbtn, 7, 0)

        sliderY = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sliderY.valueChanged.connect(lambda val: self.glWidget.setRotY(val))
        gui_layout.addWidget(sliderY, 1, 0, 1, 0)

        self.Time_ = QtWidgets.QLabel(f"Steps:\t\t\t 0 from {time_steps}")
        self.Time_.setFixedSize(450, 20)
        gui_layout.addWidget(self.Time_, 8, 0)

        self.Start_btn = QtWidgets.QPushButton("Let's go!")
        self.Start_btn.clicked.connect(self.start)
        gui_layout.addWidget(self.Start_btn, 9, 0)

        self.Start_btn = QtWidgets.QPushButton("Plot")
        self.Start_btn.clicked.connect(self.plot)
        gui_layout.addWidget(self.Start_btn, 10, 0)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.start_thread)

    
    def start_thread(self):
        self.glWidget.initGeometry(temp_max, temp_min)

    def plot(self):
        self.glWidget.Plot_sol()

    def start(self):
        self.slider.setRange(0, time_steps - 1)
        self.time_th = TimeThread(mainwindow=self)
        self.time_th.start()


model_name = 'model2.obj'
parameters_file = 'param2.json'
temperature_file = 'temperature.csv'
temp_max = 60
temp_min = 20
time_steps = 100

app = QtWidgets.QApplication(sys.argv)
win = MainWindow()
win.show()
sys.exit(app.exec_())
