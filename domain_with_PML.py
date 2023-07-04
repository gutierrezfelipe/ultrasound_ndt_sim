import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from time import perf_counter
import math
import numpy as np
from scipy import signal
from cuda_interface_pml import Cuda_interface_PML


matplotlib.use('TkAgg')
colormap = 'jet'

# Parameters
# T = 30e-6  # [s]
# T = 0.00012  # [s]
T = 8e-5  # [s]
# T = 1e-4  # [s]
Lx = 40e-3  # [m]
Lz = 40e-3  # [m]
dt = 8e-9  # [s/iteration]
dx = 3e-5  # [m/pixel]
dz = dx  # [m/pixel]
dh = dx
Nt = round(T / dt)  # number of [iterations]
Nx = round(Lx / dx)  # number of x [pixel]
Nz = round(Lz / dz)  # number of z [pixel]

deriv_accuracy = 4

# Source setup
# Transducer Characteristics
element_width = 4.2e-4  # [m]
gap_between_elements = 8e-5  # [m]
pitch = 5e-4  # distance between elements center [m]

pixel_per_gap = round(gap_between_elements / dx)

pixel_per_element = round(element_width / dx)
n_es = 1  # number of elements in the source transducer
Ns = n_es * pixel_per_element

active_width = n_es * element_width + (n_es - 1) * gap_between_elements

transducer_corner_x = round(Nx/2) - round((active_width/2)/dx)  # For Phased Array Transducer
#transducer_corner_x = Nx//6
transducer_corner_z = round(Nz*0.5)

# Source location
# z = transducer_corner_z
# xzs = np.full((Nz, Nx), False)
# pos_source = np.zeros((2, Ns), dtype='int')
# pos_source[0, :] = z
# element_corner_x = transducer_corner_x
# for i in range(n_es):
#     for j in range(pixel_per_element):
#         #x = int(Nx // 5 + i * ((Nx - 2 * Nx // 5) / Ns))
#         x = element_corner_x + j
#         pos_source[1, i * pixel_per_element + j] = x
#         xzs[z, x] = True
#     element_corner_x += pixel_per_element + pixel_per_gap  # For Phased Array Transducer
#     #element_corner_x += int(Nx // 5 + i * ((Nx - 2 * Nx // 5) / Ns))

pulse = np.zeros((Ns, Nt))

t = np.linspace(0, T-dt, Nt)
frequency = 5e6  # [Hz]
delay = 1e-6  # [s]
bandwidth = 0.82  # dado do manual olympus
f = signal.gausspulse(t - delay, frequency, bandwidth)
# f = np.load("data/olympus_source_setup_41mm.npy")


# Signal for each element
for i in range(Ns):
    pulse[i] = f

## Receptor setup
# Transducer Characteristics
pixel_per_element = round(element_width / dx)
n_e = 128
Nm = n_e * pixel_per_element

active_width = n_e//2 * element_width + (n_e//2 - 1) * gap_between_elements

sensor_corner_x = round(Nx/2) - round((active_width/2)/dx)
sensor_corner_z_1 = round(Nz*0.05)
sensor_corner_z_2 = round(Nz*0.95)
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

# Middle Source Sensor Location
#pos_sensor = np.zeros((2, Nm), dtype='int')
#xzm = np.full((Nz, Nx), False)
# z = round(Nz * 0.5)
# x = sensor_corner_x
# pos_sensor[0, Nm//2:] = z
# element_corner_x = sensor_corner_x
# for i in range(n_e//2, n_e, 1):
#     for j in range(pixel_per_element):
#         # x = int(Nx // 5 + i * ((Nx - 2 * Nx // 5) / Ns))
#         x = element_corner_x + 1 * j
#         pos_sensor[1, i * pixel_per_element + j] = x
#         xzm[z, x] = True
#     element_corner_x += pixel_per_element + pixel_per_gap

## Central single element source (31st element of center receiver)
# z = sensor_corner_z_1
z = transducer_corner_z
xzs = np.full((Nz, Nx), False)
pos_source = np.zeros((2, Ns), dtype='int')
pos_source[0, :] = z
element_start = pixel_per_element * 31
element_corner_x = pos_sensor[1, element_start]
for i in range(n_es):
    for j in range(pixel_per_element):
        #x = int(Nx // 5 + i * ((Nx - 2 * Nx // 5) / Ns))
        x = element_corner_x + j
        pos_source[1, i * pixel_per_element + j] = x
        xzs[z, x] = True
    element_corner_x += pixel_per_element + pixel_per_gap  # For Phased Array Transducer
    #element_corner_x += int(Nx // 5 + i * ((Nx - 2 * Nx // 5) / Ns))

## Media setup
ad = math.sqrt((dx * dz) / (dt ** 2))  # Adimensionality constant
cMat = {}
cMat['silicone'] = 1235/ad  # 5490
cMat['water'] = 1480/ad  # 5490
#cMat['aluminium'] = 6000/ad
#cMat['copper'] = 4000/ad
media = cMat['water'] * np.ones((Nz, Nx)) * ad

media[:deriv_accuracy, :] = 0
media[Nz-deriv_accuracy, :] = 0
media[:deriv_accuracy, 0] = 0
media[:, Nx-deriv_accuracy] = 0

# Square
z_center = round(Nz * 0.33)
x_center = round(Nx * 0.20)

square_side_half = round(5e-3/dx/2)
media[z_center-square_side_half:z_center+square_side_half, x_center-square_side_half:x_center+square_side_half] = cMat['silicone'] * ad


# Circle 1
radius = 2e-3/2/dx  # 0.5mm diameter circle hole
#z_center = transducer_corner_z + round(0.0245/dx) + radius
z_center = round(Nz * 0.20)
x_center = round(Nx * 0.70)
for j in range(Nz):
    for i in range(Nx):
        if (i - x_center)**2 + (j - z_center)**2 < radius**2: #and j <= z_center:
            media[j, i] = cMat['silicone'] * ad


# Circle 2
radius = 5e-4/2/dx  # 0.5mm diameter circle hole
#z_center = transducer_corner_z + round(0.0245/dx) + radius
z_center = round(Nz * 0.15)
x_center = round(Nx * 0.50)
for j in range(Nz):
    for i in range(Nx):
        if (i - x_center)**2 + (j - z_center)**2 < radius**2: #and j <= z_center:
            media[j, i] = cMat['silicone'] * ad


plt.figure()
plt.imshow(media, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
plt.imshow(xzs, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
plt.imshow(xzm, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
#plt.show()

## Courant check
courant = max(cMat.values())
if courant > 1:
    raise ValueError("Courant error\n")
else:
    print(f"Courant OK: {courant}\n")

## S
S = min(cMat.values())*ad / (2 * frequency * dx)
print(f"S = {S} [points / wavelenght]")
print(f"Suggested dx = {min(cMat.values())*ad / (2 * frequency * 4)} [m / pixel] for S = 4\n")


# Simulation PML singleshot
pml_size = 20
# delta = 1e4 * dt
delta = 0
Sim_Class = Cuda_interface_PML(Nx, Nz, Nt, pml_size, media, Ns, pos_source, Nm, pos_sensor, pulse,
                               dx, dz, dt, pixel_per_element, delta)

print("Acoustic wave simulation")
print(f"{Nz}x{Nx} Grid x {Nt} iterations - dx = {dx} m x dz = {dz} m x dt = {dt} s")
start = perf_counter()
Sim_Class.simulate(output=True)
end = perf_counter()
time = end - start
print(f"Simulation Time: {time} s")
print(f"Simulation Time: {time/60} min")

# B-Scan
plt.figure()
plt.imshow(np.log10(np.abs(signal.hilbert(Sim_Class.recording))+1).T, aspect='auto')


# sum = 0
# start_total = perf_counter()
# for k in range(25):
#     start = perf_counter()
#     Sim_Class.simulate(output=False)
#     end = perf_counter()
#     time = end - start
#     print(f"Simulation {k} Time: {time} s")
#     sum += time
#     #write row
#     #writer.writerow(f"{time}")
# print(f"Simulation Mean Time: {sum/(k+1)} s")
# print(f"Simulation TOTAL Time: {end - start_total} s")
# print(f"Simulation TOTAL Time: {(end - start_total)/60} min")

