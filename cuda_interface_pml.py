import os
import numpy as np
import ctypes
from ctypes import c_float, c_int, CDLL


lib2_path = 'simulator/bin/cuda_regression_PML.so'
try:
    clib = CDLL(os.path.abspath(lib2_path))
except:
    print('Error importing library')

cuda_init_sim = clib.init_memory_sim
cuda_init_sim.restype = None

cuda_set_cquad = clib.setCquad
cuda_set_cquad.restype = None

cuda_set_source = clib.setSource
cuda_set_source.restype = None

cuda_simulate_c = clib.cuda_simulate
cuda_simulate_c.restype = None

class Cuda_interface_PML:
    def __init__(self, X, Z, T, PML_size, c, n_source, pos_source, n_sensor, pos_sensor, source,
                 dx, dz, dt, pixel_per_element=1, delta=0,
                 initial=None, multishot=False):
        self.c_float_p = ctypes.POINTER(ctypes.c_float)
        self.c_int_p = ctypes.POINTER(ctypes.c_int)

        # TODO: typecheck e sizecheck dos parametros

        # self.pos_sensor = np.where(pos_sensor)
        # self.pos_source = np.where(pos_source)
        self.pos_sensor = pos_sensor
        self.pos_source = pos_source

        self.X = X
        self.Z = Z
        self.T = T
        self.PML_size = PML_size
        self.cquad = (c**2).astype(np.single)
        self.n_source = n_source
        self.pos_source_x = self.pos_source[1]
        self.pos_source_z = self.pos_source[0]
        self.n_sensor = n_sensor
        self.pos_sensor_x = self.pos_sensor[1]
        self.pos_sensor_z = self.pos_sensor[0]
        self.source = source.astype(np.single)
        self.dx = dx
        self.dz = dz
        self.dt = dt
        self.pixel_per_element = pixel_per_element
        self.delta = delta

        self.d_x, self.d_z = self._calcPML()
        self.d_x += np.ones((Z, X)) * self.delta
        self.d_z += np.ones((Z, X)) * self.delta

        self.constvec = np.asarray([self.dx, self.dz, self.dt]).astype(np.single)
        self._pos_revert = None
        self._pos_revert_x = None
        self._pos_revert_z = None
        self._tammaskrev = 0

        self._multishot = multishot
        if multishot:
            self.recording = np.zeros((n_source, n_sensor, T)).astype(np.single)
        else:
            self.recording = np.zeros((n_sensor, T)).astype(np.single)

        self.idx_source = -1

        self.cquadptr = self.cquad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if initial is None:
            self.initial = np.zeros((Z, X, 2))
        else:
            self.initial = initial

        if self.pixel_per_element > 1:
            self.n_source_real = self.n_source // self.pixel_per_element
            self.n_sensor_real = self.n_sensor // self.pixel_per_element

        self.observed = np.zeros((self.n_source_real, self.n_sensor_real, self.T))

        cq = self.cquad.flatten()
        init = self.initial.flatten()
        src = self.source.flatten()
        ddx = self.d_x.flatten()
        ddz = self.d_z.flatten()

        X_int = c_int(X)
        Z_int = (c_int)(Z)
        T_int = (c_int)(T)
        cquad_arr = (c_float * len(cq))(*cq)
        initial_arr = (c_float * len(init))(*init)
        d_x_arr = (c_float * len(ddx))(*ddx)
        d_z_arr = (c_float * len(ddx))(*ddz)
        constvec_arr = (c_float * len(self.constvec))(*self.constvec)

        self.observed = None
        self.gradptr = None

        n_source_int = c_int(self.n_source)
        source_x_arr = (c_int * len(self.pos_source_x))(*self.pos_source_x)
        source_z_arr = (c_int * len(self.pos_source_z))(*self.pos_source_z)
        source_arr = (c_float * len(src))(*src)

        n_sensor_int = c_int(self.n_sensor)
        sensor_x_arr = (c_int * len(self.pos_sensor_x))(*self.pos_sensor_x)
        sensor_z_arr = (c_int * len(self.pos_sensor_z))(*self.pos_sensor_z)
        pixel_per_element_int = c_int(self.pixel_per_element)

        self.recptr = np.asarray((0)).ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
        cuda_init_sim(X_int, Z_int, T_int, cquad_arr, constvec_arr, d_x_arr, d_z_arr,
                      n_source_int, source_x_arr, source_z_arr, n_sensor_int, sensor_x_arr, sensor_z_arr, pixel_per_element_int,
                      source_arr, initial_arr, self.recptr)


    def _calcPML(self):
        R = 1e-9
        Vmax = np.sqrt(self.cquad[self.PML_size+1, self.X // 2])
        Lp = self.PML_size * self.dx

        d0 = 3 * Vmax * np.log10(1 / R) / (2 * Lp ** 3)
        x = np.linspace(0, Lp, self.PML_size)
        damp_profile = d0 * x[:, np.newaxis]**2

        d_x = np.zeros((self.Z, self.X))
        d_z = np.zeros((self.Z, self.X))
        d_z[:self.PML_size, :] = damp_profile[-1::-1]
        d_z[-self.PML_size:, :] = damp_profile
        d_x[:, -self.PML_size:] = damp_profile.T
        d_x[:, :self.PML_size] = damp_profile[-1::-1].T

        return d_x * self.dt, d_z * self.dt


    def _set_c_quad(self, cquad):
        self.cquad = cquad.astype(np.single)
        cquadptr = self.cquad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cuda_set_cquad(cquadptr)


    def _set_source(self, n_source, pos_s):
        pos_source = pos_s

        n_source = n_source
        pos_source_x = pos_source[1]
        pos_source_z = pos_source[0]

        n_source_int = c_int(n_source)
        source_x_arr = (c_int * len(pos_source_x))(*pos_source_x)
        source_z_arr = (c_int * len(pos_source_z))(*pos_source_z)

        cuda_set_source(n_source_int, source_x_arr, source_z_arr)


    def simulate(self, output=False, c=None, idx_source=None):
        out_int = 0
        if output:
            out_int = 1
        if c is not None:
            self._set_c_quad(c ** 2)
        if idx_source is not None:
            self.idx_source = idx_source
        else:
            self.idx_source = -1

        if not self._multishot and self.pixel_per_element > 1:
            cuda_simulate_c(out_int, c_int(self.idx_source))
            ctypes.memmove(self.recording.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.recptr[0],
                           self.n_sensor * self.T * ctypes.sizeof(ctypes.c_float))
            self._multishot = False
            self.rec = np.zeros((self.n_sensor_real, self.T))
            for i in range(self.n_sensor_real):
                self.rec[i, :] = np.sum(self.recording[i * self.pixel_per_element:(i + 1) * self.pixel_per_element, :],
                                        axis=0)
        elif not self._multishot:
            cuda_simulate_c(out_int, c_int(self.idx_source))
            ctypes.memmove(self.recording.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.recptr[0],
                           self.n_sensor * self.T * ctypes.sizeof(ctypes.c_float))
            self._multishot = False
        elif self.pixel_per_element > 1:
            self.recording = np.zeros((self.n_sensor, self.T)).astype(np.single)
            buffer = np.zeros((self.n_sensor, self.T)).astype(np.single)
            tru_buffer = np.zeros((self.n_sensor_real, self.T)).astype(np.single)
            self.rec = np.zeros((self.n_source_real, self.n_sensor_real, self.T))

            for s in range(self.n_source_real):
                self._set_source(self.pixel_per_element,
                                 self.pos_source[:, s*self.pixel_per_element:(s+1)*self.pixel_per_element])
                cuda_simulate_c(out_int, c_int(s))
                ctypes.memmove(buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.recptr[0],
                               self.n_sensor * self.T * ctypes.sizeof(ctypes.c_float))
                for i in range(self.n_sensor_real):
                    tru_buffer[i, :] = np.sum(
                        buffer[i * self.pixel_per_element:(i + 1) * self.pixel_per_element, :], axis=0)
                self.rec[s, :, :] = tru_buffer

            # crosstalk modeling
            # crosst = np.sum(self.rec, axis=1)
            # cross_all = np.repeat(crosst[:, np.newaxis, :], 128, axis=1)
            # cross_factor = 0.0075
            # self.rec += cross_factor * cross_all

            # time_gate = 1000
            # gain_gate = 0.01
            # gated_sim = np.zeros_like(self.rec)
            # gated_sim[:, :time_gate] = self.rec[:, :time_gate] * gain_gate
            # gated_sim[:, time_gate:] = self.rec[:, time_gate:]

            # normalize
            # self.rec /= np.max(np.abs(self.rec))

            self._multishot = True
        else:
            buffer = np.zeros((self.n_sensor, self.T)).astype(np.single)
            for s in range(self.n_source):
                cuda_simulate_c(out_int, c_int(s))
                ctypes.memmove(buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.recptr[0],
                               self.n_sensor * self.T * ctypes.sizeof(ctypes.c_float))
                self.recording[s, :, :] = buffer
            self._multishot = True
