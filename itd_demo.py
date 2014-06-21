# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:13:37 2014

@author: rkmaddox
"""

import pyglet
import numpy as np
from pyglet.window import key
import scipy.signal as sig
import matplotlib.pyplot as plt
from expyfun import stimuli as stim
import pyfftw.interfaces.numpy_fft as fft
from expyfun import ExperimentController

base_vol = 0.1
isi = 0.8  # half period (seconds)

# Set up the left an right sounds to play
sound_files = None
if sound_files is None:
    fs = 44100
    fc = 2e3
    sound_len = int(np.round(isi * 0.8 * fs))
    sounds = np.random.randn(2, sound_len)
    b, a = sig.butter(2, fc / (fs / 2))
    sounds = sig.lfilter(b, a, sounds)
    sounds *= base_vol / np.sqrt(np.mean(sounds ** 2, axis=-1, keepdims=True))
    sounds[0] *= 0.5 * (1 - np.cos(2 * np.pi * 5 / (isi * 0.8) *
                                   np.arange(sounds.shape[-1]) / fs))
    sounds[1] *= 0.5 * (1 - np.cos(2 * np.pi * 6 / (isi * 0.8) *
                                   np.arange(sounds.shape[-1]) / fs))
else:
    pass  # load the sounds from wave files and fix their RMS


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
n_delay = 16 / 2
itds = np.exp(np.linspace(np.log(1e-6), np.log(750e-6), n_delay))
itds = np.linspace(0, 750e-6, n_delay)
x = np.zeros((2, n_delay, sound_len))

for ii, itd in enumerate(itds):
    print(int(np.round(itd * 1e6)))
    for si in range(2):
        x[si, ii] = delay(sounds[si], itd, fs, keeplength=True)

info_string = [
    '''
   Breath:  5 seconds\n
Heartbeat:  0.8 seconds\n
    Blink:  0.2 seconds\n
      ITD: -%1.6f seconds
    ''',
    '''
   Breath:  5 seconds\n
Heartbeat:  0.8 seconds\n
    Blink:  0.2 seconds\n
      ITD:   %1.6f seconds
    ''']
with ExperimentController('ITD', stim_rms=base_vol,
                          output_dir=None, check_rms=None, participant='PSC',
                          session='', verbose=0) as ec:
    for ii, itd in enumerate(itds):
        for si in range(2):
            itd *= -1
            sign = np.sign(itd)
            ec.screen_text(info_string[sign > 0] % (np.abs(itd)),
                           font_name='Courier')
            ec.flip()
            y = np.concatenate((x[si, 0][np.newaxis, :],
                                x[si, ii][np.newaxis, :]), 0)
            if np.sign(itd) > 0:
                y = y[::-1]
            ec.load_buffer(y)
            ec.play()
            ec.wait_secs(isi)
            ec.stop()
