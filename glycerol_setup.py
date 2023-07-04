import matplotlib
import matplotlib.pyplot as plt
from time import perf_counter
import math
import numpy as np
from scipy import signal
from cuda_interface_pml import Cuda_interface_PML


def add_noise(u, db):
    s = np.sqrt(np.mean(u**2)*10**(-db/10))
    return u + s*np.random.randn(*u.shape)


matplotlib.use('TkAgg')
colormap = 'jet'

# Parameters
# T = 25e-6  # [s] small for debugging GPU transfers
# T = 80e-6  # [s] empty sandwich test
T = 100e-6  # [s] glycerin circle
Lx = 45e-3  # [m]
Lz = 45e-3  # [m]
dt = 8e-9  # [s/iteration]
dx = 6.175e-5  # [m/pixel]
dz = dx  # [m/pixel]
dh = dx
Nt = round(T / dt)  # number of [iterations]
Nx = round(Lx / dx)  # number of x [pixel]
Nz = round(Lz / dz)  # number of z [pixel]

deriv_accuracy = 4

# Source setup
distance_from_bounds = 2e-3
case_active_offset = 1e-3
# Transducer Characteristics
element_width = 4.2e-4  # [m]
gap_between_elements = 8e-5  # [m]
pitch = 5e-4  # distance between elements center [m]

pixel_per_gap = round(gap_between_elements / dx)

pixel_per_element = round(element_width / dx)
n_es = 6  # number of elements in the source transducer
Ns = n_es * pixel_per_element

#active_width = n_es * element_width + (n_es - 1) * gap_between_elements

#transducer_corner_x = round(Nx/2) - round((active_width/2)/dx)  # For Phased Array Transducer
#transducer_corner_x = Nx//6
transducer_corner_z = round(distance_from_bounds/dz)

pulse = np.zeros((Ns, Nt))
t = np.linspace(0, T-dt, Nt)
frequency = 5e6  # [Hz]
delay = 1e-6  # [s]
bandwidth = 0.82  # dado do manual olympus
# f = signal.gausspulse(t - delay, frequency, bandwidth)
#f = np.load("olympus_source_setup_41mm.npy")
s_olympus = np.load("olympus_source_setup_41mm.npy")
s_oly = np.roll(s_olympus, -95)  # smaller time without signal
f_olympus = signal.resample(s_oly, Nt)
fs = f_olympus / np.max(np.abs(f_olympus))

# Signal for each element
for i in range(Ns):
    #pulse[i] = signal.gausspulse(t-delay, frequency, bandwidth)
    pulse[i] = fs

## Receptor setup
# Transducer Characteristics
pixel_per_element = round(element_width / dx)
n_e = 128
Nm = n_e * pixel_per_element

active_width = n_e//2 * element_width + (n_e//2 - 1) * gap_between_elements

sensor_corner_x = round(Nx/2) - round((active_width/2)/dx)
sensor_corner_z_1 = round(distance_from_bounds / dz)
sensor_corner_z_2 = round(Nz - distance_from_bounds / dx)
#sensor_corner_z_2 = transducer_corner_z

# Reflection Sensor Location
pos_sensor = np.zeros((2, Nm), dtype='int')
xzm = np.full((Nz, Nx), False)
z = sensor_corner_z_1
x = sensor_corner_x
pos_sensor[0, :Nm//2] = z
element_corner_x = sensor_corner_x
for i in range(n_e//2):
    for j in range(pixel_per_element):
        # x = int(Nx // 5 + i * ((Nx - 2 * Nx // 5) / Ns))
        x = element_corner_x + 1 * j
        pos_sensor[1, i * pixel_per_element + j] = x
        xzm[z, x] = True
    element_corner_x += pixel_per_element + pixel_per_gap

# # Transmission Sensors Location
z = sensor_corner_z_2
x = sensor_corner_x
pos_sensor[0, Nm//2:] = z
element_corner_x = sensor_corner_x
for i in range(n_e//2, n_e, 1):
    for j in range(pixel_per_element):
        # x = int(Nx // 5 + i * ((Nx - 2 * Nx // 5) / Ns))
        x = element_corner_x + 1 * j
        pos_sensor[1, i * pixel_per_element + j] = x
        xzm[z, x] = True
    element_corner_x += pixel_per_element + pixel_per_gap


## SOURCE elements
between_transducers = (sensor_corner_z_2 - transducer_corner_z) * dz
transducer_case_width = active_width + 2 * case_active_offset
case_offset = round(case_active_offset / dx)
z = transducer_corner_z
xzs = np.full((Nz, Nx), False)
pos_source = np.zeros((2, Ns), dtype='int')
pos_source[0, :Ns//2] = z
pos_source[0, Ns//2:] = sensor_corner_z_2
# emissor elements based on sensor/receiver positions
element_s = [0, 31, 63, 64, 95, 127]
# element_s = [31]
for el in range(n_es):
    pos_source[1, (el*pixel_per_element):((el+1)*pixel_per_element)] = pos_sensor[1, (element_s[el] * pixel_per_element):((element_s[el]+1)*pixel_per_element)]
    xzs[pos_sensor[0, (element_s[el] * pixel_per_element):((element_s[el]+1)*pixel_per_element)], pos_sensor[1, (element_s[el] * pixel_per_element):((element_s[el]+1)*pixel_per_element)]] = True


## Media setup
ad = math.sqrt((dx * dz) / (dt ** 2))  # Adimensionality constant
cMat = {}
# cMat['silicone'] = 1235/ad  # 5490
cMat['water'] = 1480  # /ad  # 5490
cMat['transducer'] = 3670  # /ad  # 3334/ad  # 2960/ad
# cMat['polypropylene'] = 2660/ad
# cMat['acrylic'] = 2777/ad
cMat['glycerol'] = 1920  # /ad  # 5490
# cMat['copper'] = 4000/ad

media = cMat['water'] * np.ones((Nz, Nx))  # * ad
media[0:sensor_corner_z_1+1, pos_sensor[1, 0]-case_offset:pos_sensor[1, -1]+case_offset] = cMat['transducer']  # * ad  # transdutor de cima - com tamanho correto do case
media[sensor_corner_z_2-1:, pos_sensor[1, 0]-case_offset:pos_sensor[1, -1]+case_offset] = cMat['transducer']  # * ad  # transdutor de baixo - com tamanho correto do case

# Circle 2 - preencher com glicerina
radius = 10e-3/2/dx  # 10mm diameter canudo
# z_center = transducer_corner_z + round(0.0245/dx) + radius
z_center = round(Nz * 0.50)
x_center = round(Nx * 0.55)
for j in range(Nz):
    for i in range(Nx):
        if (i - x_center)**2 + (j - z_center)**2 < radius**2: #and j <= z_center:
            media[j, i] = cMat['glycerol']  # * ad

# Initial guess (w/ transducers material)
guess = cMat['water'] * np.ones((Nz, Nx))  # * ad
guess[0:sensor_corner_z_1+1, pos_sensor[1, 0]-case_offset:pos_sensor[1, -1]+case_offset] = cMat['transducer']  # * ad  # transdutor de cima - com tamanho correto do case
guess[sensor_corner_z_2-1:, pos_sensor[1, 0]-case_offset:pos_sensor[1, -1]+case_offset] = cMat['transducer']   # * ad  # transdutor de baixo - com tamanho correto do case

# Perfect initial guess
# guess = media.copy()

# Gradient mask
gmask = np.full((Nz, Nx), 0)
p1a = int(Nz//5)  # //3
p2a = int(Nx//5)  # //3
gmask[p1a:-p1a, p2a:-p2a] = 1

plt.figure()
plt.imshow(gmask, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
plt.imshow(media, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
plt.imshow(xzs, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
plt.imshow(xzm, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
# plt.show()

# Courant check
courant = max(cMat.values()) * dt / dx
if courant > 1:
    raise ValueError("Courant error\n")
else:
    print(f"Courant OK: {courant}\n")

## S
S = min(cMat.values()) / (2 * frequency * dx)  # * ad
print(f"S = {S} [points / wavelenght]")
if S < 4:
    print(f"Suggested dx = {min(cMat.values()) / (2 * frequency * 4)} [m / pixel] for S = 4\n")  # *ad

# DADOS REAIS
recFWI = np.load(
    '/home/felipegutierrez/PycharmProjects/fwi/cuda_regression/wrapper/glycerol_canudo_1_32_64_65_96_128.npy')

# recFWI = np.load(
#     '/home/felipegutierrez/PycharmProjects/fwi/cuda_regression/wrapper/frente_vazio_1_32_64_65_96_128.npy')

time_gate = 1000
gain_gate = 0.01
gated_obs = np.zeros_like(recFWI)
gated_obs[:, :, :time_gate] = recFWI[:, :, :time_gate] * gain_gate
gated_obs[:, :, time_gate:] = recFWI[:, :, time_gate:]

# normalization
obs = gated_obs / np.max(np.abs(gated_obs))


# Simulation PML multishot
pml_size = 20
# delta = 1.5e4 * dt
delta = 0
Sim_Class = Cuda_interface_PML(Nx, Nz, Nt, pml_size, media, Ns, pos_source, Nm, pos_sensor, pulse,
                               dx, dz, dt, pixel_per_element, delta, multishot=True)

print("Acoustic wave simulation")
print(f"{Nz}x{Nx} Grid x {Nt} iterations - dx = {dx} m x dz = {dz} m x dt = {dt} s")
start = perf_counter()
Sim_Class.simulate(output=False)
end = perf_counter()
time = end - start
print(f"Simulation Time: {time} s")
print(f"Simulation Time: {time/60} min")

# Sim_Class_guess = Cuda_interface_PML(Nx, Nz, Nt, pml_size, guess, Ns, pos_source, Nm, pos_sensor, pulse,
#                                dx, dz, dt, pixel_per_element, delta, multishot=True)
#
# print("Acoustic wave simulation")
# print(f"{Nz}x{Nx} Grid x {Nt} iterations - dx = {dx} m x dz = {dz} m x dt = {dt} s")
# start = perf_counter()
# Sim_Class_guess.simulate(output=False)
# end = perf_counter()
# time = end - start
# print(f"Simulation Time: {time} s")
# print(f"Simulation Time: {time/60} min")


# plt.figure()
# plt.imshow(np.log10(np.abs(signal.hilbert(Sim_Class.rec[1, :, :]))+0.01).T,
#            aspect='auto', extent=[0, n_e, t[-1] * 1e6, 0])
#
# plt.figure()
# plt.imshow(np.log10(np.abs(signal.hilbert(recFWI[1, :, :]))+0.01).T,
#            aspect='auto', extent=[0, n_e, t[-1] * 1e6, 0])
#
# plt.show()


############################################################
#                                                          #
#               DEBUG                                      #
#                                                          #
############################################################

gated_sim = np.zeros_like(Sim_Class.rec)
gated_sim[:, :, :time_gate] = Sim_Class.rec[:, :, :time_gate] * gain_gate
gated_sim[:, :, time_gate:] = Sim_Class.rec[:, :, time_gate:]
gated_sim[:, :] /= np.max(np.abs(gated_sim[:, :]))

# gated_sim_guess = np.zeros_like(Sim_Class.rec)
# gated_sim_guess[:, :, :time_gate] = Sim_Class_guess.rec[:, :, :time_gate] * gain_gate
# gated_sim_guess[:, :, time_gate:] = Sim_Class_guess.rec[:, :, time_gate:]
# gated_sim_guess[:, :] /= np.max(np.abs(gated_sim_guess[:, :]))



