import matplotlib
import matplotlib.pyplot as plt
from time import perf_counter
import math
import numpy as np
from scipy import signal
from cuda_interface import Cuda_interface


matplotlib.use('TkAgg')
colormap = 'jet'

# Parameters
T = 30e-6  # [s]
#T = 0.00012  # [s]
Lx = 40e-3  # [m]
Lz = 40e-3  # [m]
dt = 4e-9  # [s/iteration]
dx = 2e-5  # [m/pixel]
dz = dx  # [m/pixel]
dh = dx
Nt = round(T / dt)  # number of [iterations]
Nx = round(Lx / dx)  # number of x [pixel]
Nz = round(Lz / dz)  # number of z [pixel]

deriv_accuracy = 4

# Source setup
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

# # Source location
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
bandwidth = 0.9
f = signal.gausspulse(t - delay, frequency, bandwidth)

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

# Transmission Sensors Location
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
media = cMat['water'] * np.ones((Nz, Nx))

# Square
z_center = round(Nz * 0.33)
x_center = round(Nx * 0.20)

square_side_half = round(5e-3/dx/2)
media[z_center-square_side_half:z_center+square_side_half, x_center-square_side_half:x_center+square_side_half] = cMat['silicone']


# Circle 1
radius = 2e-3/2/dx  # 2mm diameter circle hole
#z_center = transducer_corner_z + round(0.0245/dx) + radius
z_center = round(Nz * 0.20)
x_center = round(Nx * 0.70)
for j in range(Nz):
    for i in range(Nx):
        if (i - x_center)**2 + (j - z_center)**2 < radius**2: #and j <= z_center:
            media[j, i] = cMat['silicone']


# Planar Reflector

# Circle 2
radius = 5e-4/2/dx  # 0.5mm diameter circle hole
#z_center = transducer_corner_z + round(0.0245/dx) + radius
z_center = round(Nz * 0.15)
x_center = round(Nx * 0.50)
for j in range(Nz):
    for i in range(Nx):
        if (i - x_center)**2 + (j - z_center)**2 < radius**2: #and j <= z_center:
            media[j, i] = cMat['silicone']


plt.figure()
plt.imshow(media, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
plt.imshow(xzs, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
plt.imshow(xzm, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
#plt.show()


# PADDING
needed_Lx = 54e-3
needed_Lz = needed_Lx
Nx_borda = round(needed_Lx / dx)
Nz_borda = round(needed_Lz / dz)

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


difference = Nx_borda - Nx
new_media = np.pad(media, round(difference / 2), pad_with, padder=media[0][0])
new_xzm = np.pad(xzm, round(difference / 2), pad_with, padder=False)
new_xzs = np.pad(xzs, round(difference / 2), pad_with, padder=False)

plt.figure()
plt.imshow(new_media, alpha=0.5, extent=[0, 1000 * needed_Lx, 1000 * needed_Lz, 0])
plt.imshow(new_xzs, alpha=0.5, extent=[0, 1000 * needed_Lx, needed_Lz * Lz, 0])
plt.imshow(new_xzm, alpha=0.5, extent=[0, 1000 * needed_Lx, 1000 * needed_Lz, 0])

new_pos_source = np.where(new_xzs)
new_pos_sensor = np.where(new_xzm)

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

# Simulation singleshot
#Sim_Class = Cuda_interface(Nx, Nz, Nt, media, Ns, pos_source, Nm, pos_sensor, pulse, multishot=False)
Sim_Class = Cuda_interface(Nx_borda, Nz_borda, Nt, new_media, Ns, new_pos_source, Nm, new_pos_sensor, pulse, multishot=False)

print("Acoustic wave simulation")
print(f"{Nz}x{Nx} Grid x {Nt} iterations - dx = {dx} m x dz = {dz} m x dt = {dt} s")
start = perf_counter()
#Sim_Class.simulate(output=True)
Sim_Class.simulate(output=False)
end = perf_counter()
time = end - start
print(f"Simulation Time: {time} s")

# B-Scan
plt.figure()
plt.imshow(np.log10(np.abs(signal.hilbert(Sim_Class.recording))+0.01).T, aspect='auto')
plt.show()

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
