import numpy as np
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


matplotlib.use('TkAgg')

filename1 = 'data/glycerol_canudo_1_32_64_65_96_128.npy'
filename2 = 'data/sim_glycerol_canudo_1_32_64_65_96_128.npy'

rec1 = np.load(filename1)
rec2 = np.load(filename2)

## Plot of the recordings
## Signal Analyses - Single Emissor
fig_sig = plt.figure()
ax_s = fig_sig.add_subplot(111)
fig_sig.subplots_adjust(left=0.25, bottom=0.25)
plt.title('PV_PML')
element0 = 0
emissor0 = 0
delta_e = 1
im1 = ax_s.plot(rec1[element0]/np.linalg.norm(rec1[element0]), label=f'{filename1}_r{element0}')
ax_s.plot(rec2[element0]/np.linalg.norm(rec2[element0]), label=f'{filename2}_r{element0}')
ax_s.grid()
ax_s.legend()
#ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
axelement = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
selement_sim = Slider(axelement, 'Receptor_sim', 0, rec1.shape[1]-1, valinit=element0, valstep=delta_e)
axelement2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
selement_sim2 = Slider(axelement2, 'Receptor_sim_PML', 0, rec2.shape[1]-1, valinit=element0, valstep=delta_e)

def update_rec_vs_data(val):
    element_sim = selement_sim.val
    element_sim2 = selement_sim2.val
    ax_s.cla()
    ax_s.plot(rec1[element_sim]/np.linalg.norm(rec1[element_sim]),
              label=f'{filename1}_r{element_sim}')
    ax_s.plot(rec2[element_sim2]/np.linalg.norm(rec2[element_sim2]),
              label=f'{filename2}_r{element_sim2}')
    ax_s.legend()
    ax_s.grid()
    fig_sig.canvas.draw_idle()


selement_sim.on_changed(update_rec_vs_data)
selement_sim2.on_changed(update_rec_vs_data)


## B-scans
# plt.figure()
# plt.imshow(np.log10(np.abs(signal.hilbert(rec1[1, :, :]))+0.01).T,
#            aspect='auto', extent=[0, rec1.shape[1]-1, t[-1] * 1e6, 0])
#
# plt.figure()
# plt.imshow(np.log10(np.abs(signal.hilbert(rec2[1, :, :]))+0.01).T,
#            aspect='auto', extent=[0, rec2.shape[1]-1, t[-1] * 1e6, 0])


plt.show()


print('end')
