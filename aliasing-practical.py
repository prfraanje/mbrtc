# Author:  Rufus Fraanje
# Email:   p.r.fraanje@hhs.nl
# Licence: GNU General Public License (GNU GPLv3)
# Creation date: 2019-03-07
# Last modified: 2019-03-08


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

def calc_plot_data(duration,f,phase,fsample,number):

    def calc_alias(f,phase,fsample,number,t):
        x_alias = 0.*t
        falias = 0.
        return x_alias,falias

    tc = np.arange(0,duration,2/f*1/1000.)  # continuous time, 2 periods 1000 samples
    xc = np.sin(2*np.pi*f*tc + phase) # (approximated) continuous time signal
    td = np.arange(0,duration,1/fsample)    # time axis of samples
    xd = np.sin(2*np.pi*f*td + phase) # sampled signal
    xalias,falias = calc_alias(f,phase,fsample,number,tc)

    return tc,xc,td,xd,xalias,falias


f = 2.  # frequency of signal in Hz
phase = 0*np.pi/4 # phase of signal in radians
fsample = 6.   # sampling rate in Hz
number = 1 # number of alias
duration = 1. # simulation time

tc,xc,td,xd,xalias,falias = calc_plot_data(duration,f,phase,fsample,number)


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1,bottom=0.25)
ax.plot(tc,xc,'b',label=f'Signal')
ax.stem(td,xd,linefmt='g-',basefmt='g-',label='Sampled')
ax.plot(tc,xalias,'r-.',label=f'Alias')
ax.set_title(f'Sine wave with sampled and alias version')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Signal value [V]')
ax.legend(loc='upper right')
ax.axis([0,duration,-1.05,1.05])


axcolor='lightgoldenrodyellow'
axf       = plt.axes([0.2,0.15,0.65,0.03],facecolor=axcolor)
axfsample = plt.axes([0.2,0.11,0.65,0.03],facecolor=axcolor)
axphase = plt.axes([0.2,0.07,0.65,0.03],facecolor=axcolor)
axnumber   = plt.axes([0.2,0.03,0.05,0.03],facecolor=axcolor)
axfalias = plt.axes([0.35,0.03,0.07,0.03],facecolor=axcolor)
axduration = plt.axes([0.5,0.03,0.05,0.03],facecolor=axcolor)

sf       = Slider(axf,'Freq.',0.1,40,valinit=f,valstep=0.1)
sfsample = Slider(axfsample,'Fsample',0.2,40,valinit=fsample,valstep=0.2)
sphase = Slider(axphase,'phase',-3,3,valinit=phase,valstep=0.1)
tnumber = TextBox(axnumber,'number',initial=str(number),color=axcolor,hovercolor='0.975')
tduration = TextBox(axduration,'duration',initial=str(duration),color=axcolor,hovercolor='0.975')
tfalias = TextBox(axfalias,'falias',initial=f'{falias:.2f}',color='lightgray',hovercolor='lightgray')

def update_plot(val):
    f = sf.val
    fsample = sfsample.val
    phase = sphase.val
    number = int(tnumber.text)
    duration = float(tduration.text)
    tc,xc,td,xd,xalias,falias = calc_plot_data(duration,f,phase,fsample,number)
    tfalias.set_val(f'{falias:.2f}')
    tfalias.stop_typing() # remove cursor

    ax.cla()
    ax.plot(tc,xc,'b',label=f'Signal')
    ax.stem(td,xd,linefmt='g-',basefmt='g-',label='Sampled')
    ax.plot(tc,xalias,'r-.',label=f'Alias')
    ax.set_title(f'Sine wave with sampled and alias version')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal value [V]')
    ax.legend(loc='upper right')
    ax.axis([0,duration,-1.05,1.05])

    fig.canvas.draw_idle()

    return True

sf.on_changed(update_plot)
sfsample.on_changed(update_plot)
sphase.on_changed(update_plot)
tnumber.on_submit(update_plot)
tduration.on_submit(update_plot)
tfalias.stop_typing() # remove cursor

axreset = plt.axes([0.75, 0.03, 0.1, 0.03])
button = Button(axreset, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    sf.reset()
    sfsample.reset()
    sphase.reset()
    tnumber.set_val(str(number))
    tduration.set_val(str(duration))
    return True

button.on_clicked(reset)

plt.show()
