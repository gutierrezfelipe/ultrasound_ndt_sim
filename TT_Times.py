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
T = 16e-6  # [s]
Lx = 100e-3  # [m]
Lz = 100e-3  # [m]
dt = 8e-9  # [s/iteration]
dx = 2.5e-5  # [m/pixel]
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
transducer_corner_z = round(Nz*0.05)

# Source location
z = transducer_corner_z
xzs = np.full((Nz, Nx), False)
pos_source = np.zeros((2, Ns), dtype='int')
pos_source[0, :] = z
element_corner_x = transducer_corner_x
for i in range(n_es):
    for j in range(pixel_per_element):
        x = element_corner_x + j
        pos_source[1, i * pixel_per_element + j] = x
        xzs[z, x] = True
    element_corner_x += pixel_per_element + pixel_per_gap  # For Phased Array Transducer

# Signal setup
pulse = np.zeros((Ns, Nt))

t = np.linspace(0, T-dt, Nt)
frequency = 5e6  # [Hz]
delay = 1e-6  # [s]
bandwidth = 0.82  # data from olympus manual
f = signal.gausspulse(t - delay, frequency, bandwidth)
fs = 100 * f / np.max(f)

# Signal for each element
for i in range(Ns):
    pulse[i] = fs

# Receptor setup
# Transducer Characteristics
pixel_per_element = round(element_width / dx)
n_e = 2
Nm = n_e * pixel_per_element

active_width = n_e//2 * element_width + (n_e//2 - 1) * gap_between_elements

sensor_corner_x = round(Nx/2) - round((active_width/2)/dx)
sensor_corner_z_1 = round(Nz*0.05)
sensor_corner_z_2 = round(Nz*0.95)

# Reflection Sensor Location
pos_sensor = np.zeros((2, Nm), dtype='int')
xzm = np.full((Nz, Nx), False)
z = sensor_corner_z_1
x = sensor_corner_x
pos_sensor[0, :Nm//2] = z
element_corner_x = sensor_corner_x
for i in range(n_e//2):
    for j in range(pixel_per_element):
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
        x = element_corner_x + 1 * j
        pos_sensor[1, i * pixel_per_element + j] = x
        xzm[z, x] = True
    element_corner_x += pixel_per_element + pixel_per_gap


## Media setup
ad = math.sqrt((dx * dz) / (dt ** 2))  # Adimensionality constant
cMat = {}
cMat['water'] = 1480/ad  # 5490
media = cMat['water'] * ad * np.ones((Nz, Nx))

plt.figure()
plt.imshow(media, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
plt.imshow(xzs, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
plt.imshow(xzm, alpha=0.5, extent=[0, 1000 * Lx, 1000 * Lz, 0])
# plt.show()

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
# delta = 1.5e4 * dt
delta = 0
Sim_Class = Cuda_interface_PML(Nx, Nz, Nt, pml_size, media, Ns, pos_source, Nm, pos_sensor, pulse,
                               dx, dz, dt, pixel_per_element, delta)

print("Acoustic wave simulation")
print(f"{Nz}x{Nx} Grid x {Nt} iterations - dx = {dx} m x dz = {dz} m x dt = {dt} s")
start = perf_counter()
Sim_Class.simulate(output=False)
end = perf_counter()
time = end - start
print(f"Simulation Time: {time} s")

# B-Scan
plt.figure()
plt.imshow(np.log10(np.abs(signal.hilbert(Sim_Class.recording))+10).T, aspect='auto')

# Plot of the recordings
# Signal Analyses - Single Emissor
fig_sig = plt.figure()
ax_s = fig_sig.add_subplot(111)
fig_sig.subplots_adjust(left=0.25, bottom=0.25)
plt.title('PV_PML')
element0 = 0
emissor0 = 0
delta_e = 1
im1 = ax_s.plot(t, Sim_Class.rec[element0]/np.max(Sim_Class.rec[element0]), label=f'sim_e_r{element0}')
ax_s.legend()

axcolor = 'lightgoldenrodyellow'
axelement = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
selement_sim = Slider(axelement, 'Receptor_sim', 0, n_e-1, valinit=element0, valstep=delta_e)


def update_rec_vs_data(val):
    element_sim = selement_sim.val
    ax_s.cla()
    ax_s.plot(t, Sim_Class.rec[element_sim]/np.max(Sim_Class.rec[element_sim]),
              label=f'sim_e_r{element_sim}')
    ax_s.legend()
    fig_sig.canvas.draw_idle()


selement_sim.on_changed(update_rec_vs_data)

plt.show()

# Runtime test
sum = 0
times = []
start_total = perf_counter()
for k in range(25):
    start = perf_counter()
    Sim_Class.simulate(output=False)
    end = perf_counter()
    time = end - start
    print(f"Simulation {k} Time: {time} s")
    times.append(time)
    sum += time
print(f"\n\nSimulation MEAN Time: {sum/(k+1)} s")
print(f"\nSimulation TOTAL Time: {end - start_total} s")
print(f"Simulation TOTAL Time: {(end - start_total)/60} min")

runtime = np.array(times)
np.save(f'times/{Nz}_{Nx}_{Nt}.npy', runtime)
#np.save(f'{Nz}_{Nx}_{Nt}_boundary_reflection.npy', Sim_Class.rec)

print('end')
