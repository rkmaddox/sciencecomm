# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:13:37 2014

@author: rkmaddox
"""

import pyglet
import pygame
import numpy as np
from pyglet.window import key
import scipy.signal as sig
import matplotlib.pyplot as plt
plt.ion()
from expyfun import stimuli as stim
import pyfftw.interfaces.numpy_fft as fft
from expyfun import ExperimentController

base_vol = 0.01
isi = 0.33  # half period (seconds)

# Set up the left an right sounds to play
sound_files = None  # ['left.wav', 'right.wav']
if sound_files is None:
    fs = 44100
    fc = 2e3
    sound_len = int(np.round(isi * 0.8 * fs))
    sound_dur = float(sound_len) / fs
#    sounds = stim.window_edges(np.random.randn(1, sound_len), fs,
#                               sound_dur / 2.5)
#    b, a = sig.butter(2, fc / (fs / 2))
#    sounds = sig.lfilter(b, a, sounds)
    t = np.arange(sound_len, dtype=float) / fs
    sounds = np.zeros(sound_len)
    f0 = 200
    for f in range(f0, 1301, f0):
        sounds += np.sin(2 * np.pi * t * f)
    sounds *= base_vol / stim.rms(sounds, keepdims=True)
    sounds *= np.exp(-t / sound_dur * 4)
    sounds = stim.window_edges(sounds, fs, 0.01)
else:
    assert(len(sound_files) == 1)
    temp = []
    for wav in sound_files:
        temp += [stim.read_wav(wav)[0]]
    fs = stim.read_wav(sound_files[0])[1]
    lens = [w.shape[1] for w in temp]
    sounds = np.zeros((2, np.max(lens)))
    for si, l in enumerate(lens):
        sounds[si, :l] = temp[si]
    sounds = sig.resample(sounds, 44100 * sounds.shape[1] / fs, axis=1)
    fs = 44100
    sound_len = sounds.shape[1]



# Make the ITD function
def delay(x, time, fs, axis=-1, keeplength=False, pad=1):
    extra_pad = 200  # add 200 samples to prevent wrapping
    samps = int(np.floor(time * fs))
    s = list(x.shape)
    sz_pre = np.copy(s)
    sz_post = np.copy(s)
    sz_fft = np.copy(s)
    sz_pre[axis] = samps
    sz_post[axis] = pad + extra_pad
    x = np.concatenate((np.zeros(sz_pre), x, np.zeros(sz_post)), axis)
    sz_fft[axis] = int(np.round(2 ** np.ceil(np.log2(x.shape[axis])) -
                                x.shape[axis]))
    x = np.concatenate((x, np.zeros(sz_fft)), axis)
    new_len = sz_pre[axis] + s[axis] + sz_post[axis] + sz_fft[axis]

    # x[n-k] <--> X(jw)e^(-jwk) where w in [0, 2pi)
    if type(time) is not int:
        theta = (-np.arange(new_len).astype(float) * fs * 2 * np.pi / new_len *
                 (time - np.float(samps) / fs))
        theta[-(new_len // 2) + 1:] = -theta[(new_len // 2):1:-1]
        st = [1 for _ in range(x.ndim)]
        st[axis] = new_len
        x = np.real(fft.ifft(fft.fft(x, axis=axis) *
                             np.exp(1j * theta.reshape(st))))
    if keeplength:
        x = np.take(x, range(s[axis]), axis)
    else:
        x = np.take(x, range(s[axis] + samps + pad), axis)
    inds = tuple([slice(si) for si in sz_pre])
    x[inds] = 0
    return x

lr = 'LR'
n_delay = 1024 / 2
itds = np.exp(np.linspace(np.log(1e-6), np.log(750e-6), n_delay))
itd_max = 1000e-6
itds = np.linspace(0, itd_max, n_delay)
x = np.zeros((n_delay, sound_len))

for ii, itd in enumerate(itds):
    print(int(np.round(itd * 1e6)))
    x[ii] = delay(sounds, itd, fs, keeplength=True)


info_string = \
    '''
<pre>   Breath:  5 seconds
Heartbeat:  0.8 seconds
    Blink:  0.2 seconds
      ITD: %+1.6f seconds
      '''
info_string = '<pre>Interaural Time Difference: %+1.6f seconds'

pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()


def get_sound_ind():
    pygame.event.pump()
    if not joystick.get_button(0):
        axis = (joystick.get_axis(0) + joystick.get_axis(1)) / 2.
        ind = int(np.round(axis * (n_delay - 1)))
        itd = ind * itd_max / (n_delay - 1)
    else:
        ind = itd = 0
    return (itd, np.abs(ind))
    

with ExperimentController('ITD', stim_rms=base_vol,
                          output_dir=None, check_rms=None, participant='PSC',
                          session='', verbose=0) as ec:
    t0 = -np.inf
    while(1):
        ec.check_force_quit()
        itd, ind = get_sound_ind()
        sign = np.sign(itd)
        oc = 1 - np.abs(itd / itds[-1])
        if np.sign(itd) > 0:
            color = [1, oc, oc]
        else:
            color = [oc, oc, 1]
        ec.screen_text(info_string % itd,
                       font_name='Courier', font_size=40, color=color)
        ec.flip()
        y = np.concatenate((x[0][np.newaxis, :],
                            x[ind][np.newaxis, :]), 0)
        if np.sign(itd) > 0:
            y = y[::-1]
        ec.load_buffer(y)
        ec.wait_until(t0 + isi)
        t0 = ec.play()
        ec.wait_secs(sound_dur + .01)
        ec.stop()
