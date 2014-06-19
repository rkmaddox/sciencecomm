# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:53:00 2014

@author: rkmaddox
"""

import expyfun.stimuli as stim
import numpy as np
from glob import glob
import os
from os.path import join
import matplotlib.pyplot as plt

opath = 'strings'
spath = 'maskers'

genders = ['M', 'F']  # male in left channel, female in right
regions = ['NW', 'CH']
talkers = [str(i) for i in range(5)]
loc = ['C', 'S']
az = [0, -60]

n_talkers = 3
wavs = [[np.zeros((2, 0)) for _ in range(2)] for _ in range(2)]
for ti, t in enumerate(talkers[:n_talkers]):
    #for ri, r in enumerate(regions):
    ri = 0
    r = regions[ri]
    wav, fs = stim.read_wav(join(opath, r + t + '.wav'))
    for gi, g in enumerate(genders):
        for li, (l, a) in enumerate(zip(loc, az)):
            wav_loc = stim.convolve_hrtf(wav[gi], fs, a)
            lens = [wavs[gi][li].shape[-1], wav_loc.shape[-1]]
            dl = lens[0] - lens[1]
            if dl < 0:
                wavs[gi][li] = np.concatenate((wavs[gi][li],
                                               np.zeros((2, -dl))), -1)
            if dl > 0:
                wav_loc = np.concatenate((wav_loc, np.zeros((2, dl))), -1)
            wavs[gi][li] += wav_loc

for gi, g in enumerate(genders):
    for li, (l, a) in enumerate(zip(loc, az)):
        fn = join(spath, '%s_%s_%s.wav' % (r, g, l))
        data = wavs[gi][li] / np.sqrt(n_talkers)
        data = np.minimum(1, np.maximum(-1, data))
        stim.write_wav(fn, data, fs, overwrite=True)
