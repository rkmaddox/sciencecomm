# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:13:37 2014

@author: rkmaddox
"""

import pyglet
import numpy as np
from pyglet.window import key
import scipy.signal as sig

base_vol = 0.1
isi = 0.8  # half period (seconds)

# Set up the left an right sounds to play
sound_files = None
if sound_files is None:
    fs = 22050
    fc = 2e3
    sound_len = int(np.round(isi * 0.8 * fs))
    sounds = np.random.randn(2, sound_len)
    b, a = sig.butter(2, fc / (fs / 2))
    sounds = sig.lfilter(b, a, sounds)
    sounds *= base_vol / np.sqrt(np.mean(sounds ** 2, axis=-1, keepdims=True))
else:
    pass  # load the sounds from wave files and fix their RMS


# Make the ITD function
def delay(x, time, fs, axis=-1, keeplength=False, pad=1):
    extra_pad = 200  # add 200 samples to prevent wrapping
    samps = int(np.floor(time * fs))
    s = list(x.shape)
    sz_pre = np.copy(s)
    sz_post = np.copy(s)
    sz_pre[axis] = samps
    sz_post[axis] = pad + extra_pad
    x = np.concatenate((np.zeros(sz_pre), x, np.zeros(sz_post)), axis)
    new_len = sz_pre[axis] + sz_post[axis] + s[axis]

    # x[n-k] <--> X(jw)e^(-jwk) where w in [0, 2pi)
    if type(time) is not int:
        theta = (-np.arange(new_len).astype(float) * fs * 2 * np.pi / new_len *
                 (time - np.float(samps) / fs))
        theta[-(new_len // 2) + 1:] = -theta[(new_len // 2):1:-1]
        st = [1 for _ in range(x.ndim)]
        st[axis] = new_len
        x = np.real(np.fft.ifft(np.fft.fft(x, axis=axis) *
                                np.exp(1j * theta.reshape(st))))

    if keeplength:
        x = np.take(x, range(s[axis]), axis)
    else:
        x = np.take(x, range(s[axis] + samps + pad), axis)
    inds = tuple([slice(si) for si in sz_pre])
    x[inds] = 0
    return x
