# -*- coding utf-8 -*-
"""
Created on Thu May 29 113534 2014

@author rkmaddox
"""

import expyfun.stimuli as stim
import numpy as np
import os
from glob import glob

rand = np.random.RandomState(0)

audio_path = os.path.expanduser('~/Data2/pnnc/audio/')
text_path = os.path.expanduser('~/Data2/pnnc/transcripts/')
save_path = 'strings'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
file_str = os.path.join(audio_path, '%s%s%s_%s.wav')

sentences = [s[-9:-4] for s in glob(os.path.join(text_path, '*.txt'))]
regions = ['NW', 'CH']
genders = ['M', 'F']

n_sent = len(sentences)
n_regions = len(regions)
n_genders = len(genders)
talkers = [[[t[-12:-10]
             for t in np.sort(glob(file_str % (r, g, '*',
                                               sentences[0]))).tolist()]
            for g in genders]
           for r in regions]
n_talkers = len(talkers[0][0])

# For each talker make a string
for ri, r in enumerate(regions):  # Make pairs out of each region
    for ti in range(n_talkers):  # pairs should be gender-paired
        order = rand.permutation(n_sent)
        string = np.zeros((2, 0))
        save_fn = ''
        for si in order:
            s = sentences[si]
            wavs = []
            for gi, g in enumerate(genders):
                t = talkers[ri][gi][ti]
                fn = file_str % (r, g, t, s)
                w, fs = stim.read_wav(fn, verbose=False)
                wavs += [w]
                if si == order[0]:
                    save_fn += '%s%s%s_' % (r, g, t)
            lens = np.array([w.shape[-1] for w in wavs])
            len_max = lens.max()
            shifts = ((len_max - lens) / 2.).astype(int)
            string = np.concatenate((string, np.zeros((2, len_max))), -1)
            string_len = string.shape[-1]
            for gi, start_ind in enumerate(shifts - len_max + string_len):
                string[gi, start_ind:start_ind + lens[gi]] = wavs[gi]
        save_fn = save_fn[:-1] + '.wav'
        save_fn = r + str(ti) + '.wav'
        stim.write_wav(os.path.join(save_path, save_fn),
                       string, fs, overwrite=True)
        print('Finished talker %i / %i.' % (ti + 1, n_talkers))
