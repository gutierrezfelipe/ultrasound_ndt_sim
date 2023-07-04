import os
import numpy as np
import ctypes
from ctypes import c_float, c_int, CDLL, CFUNCTYPE, POINTER


lib2_path = 'simulator/bin/cuda_regression.so'
try:
    clib = CDLL(os.path.abspath(lib2_path))
except:
    print('Error importing library')

cuda_init_sim = clib.init_memory_sim
cuda_init_sim.restype = None

cuda_set_cquad = clib.setCquad
cuda_set_cquad.restype = None

cuda_simulate_c = clib.cuda_simulate2
cuda_simulate_c.restype = None


class Cuda_interface:
    def __init__(self, X, Z, T, c, n_source, pos_source, n_sensor, pos_sensor, source, initial=None, multishot=False):
        self.c_float_p = ctypes.POINTER(ctypes.c_float)
        self.c_int_p = ctypes.POINTER(ctypes.c_int)

        # TODO: typecheck e sizecheck dos parametros

        self.pos_sensor = np.where(pos_sensor)
        self.pos_source = np.where(pos_source)

        self.X = X
        self.Z = Z
        self.T = T
        self.cquad = (c**2).astype(np.single)
        self.n_source = n_source
        self.pos_source_x = pos_source[1]
        self.pos_source_z = pos_source[0]
        self.n_sensor = n_sensor
        self.pos_sensor_x = pos_sensor[1]
        self.pos_sensor_z = pos_sensor[0]
        self.source = source.astype(np.single)

        self.grad = np.zeros((Z, X)).astype(np.single)
        self.mse = np.zeros(1)

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

        cq = self.cquad.flatten()
        init = self.initial.flatten()
        src = self.source.flatten()

        X_int = c_int(X)
        Z_int = (c_int)(Z)
        T_int = (c_int)(T)
        cquad_arr = (c_float * len(cq))(*cq)
        initial_arr = (c_float * len(init))(*init)

        n_source_int = c_int(self.n_source)
        source_x_arr = (c_int * len(self.pos_source_x))(*self.pos_source_x)
        source_z_arr = (c_int * len(self.pos_source_z))(*self.pos_source_z)
        source_arr = (c_float * len(src))(*src)

        n_sensor_int = c_int(self.n_sensor)
        sensor_x_arr = (c_int * len(self.pos_sensor_x))(*self.pos_sensor_x)
        sensor_z_arr = (c_int * len(self.pos_sensor_z))(*self.pos_sensor_z)

        self.recptr = np.asarray((0)).ctypes.data_as(ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))
        cuda_init_sim(X_int, Z_int, T_int, cquad_arr, n_source_int, source_x_arr, source_z_arr, n_sensor_int, sensor_x_arr, sensor_z_arr, source_arr, initial_arr, self.recptr)


    def _set_c_quad(self, cquad):
        self.cquad = cquad.astype(np.single)
        cquadptr = self.cquad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        cuda_set_cquad(cquadptr)


    def _set_source(self, n_source, pos_source, source):
        self.pos_source = np.where(pos_source)

        self.source = source.astype(np.single)
        self.n_source = n_source
        self.pos_source_x = pos_source[1]
        self.pos_source_z = pos_source[0]

        self.sourceptr = self.source.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        src = self.source.flatten()

        n_source_int = c_int(self.n_source)
        source_x_arr = (c_int * len(self.pos_source_x))(*self.pos_source_x)
        source_z_arr = (c_int * len(self.pos_source_z))(*self.pos_source_z)
        source_arr = (c_float * len(src))(*src)

        clib.set_source(n_source_int, source_x_arr, source_z_arr, source_arr)


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

        if not self._multishot:
            cuda_simulate_c(out_int, c_int(self.idx_source))
            ctypes.memmove(self.recording.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.recptr[0], self.n_sensor * self.T * ctypes.sizeof(ctypes.c_float))
            self._multishot = False
        else:
            buffer = np.zeros((self.n_sensor, self.T)).astype(np.single)
            for s in range(self.n_source):
                cuda_simulate_c(out_int, c_int(s))
                ctypes.memmove(buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.recptr[0], self.n_sensor * self.T * ctypes.sizeof(ctypes.c_float))
                self.recording[s, :, :] = buffer
            self._multishot = True
