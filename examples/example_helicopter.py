from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal as sp
import numpy as np

import simpleaudio

import hrtf

root_path = Path('/home/olivier/Documents/NeuroAudio')
path_transfer = root_path.joinpath('KEMAR')

tfs = hrtf.HRTF(path_transfer)

file_helico = "/home/olivier/Documents/NeuroAudio/194250__deleted-user-3544904__helicopter.wav"

nsec_audio = 10
fs, wav = wavfile.read(file_helico)
wav = wav[:nsec_audio * fs, :]


npos = 2000
hxyz = {'x': np.linspace(-50, 50, npos) + 50,
        'y': np.linspace(-200, 200, npos),
        'z': np.linspace(45, 45, npos),
        't': np.linspace(0, nsec_audio, npos)}

# loop over t_step windows, normally there will be a difference between
t_step = 0.1
ns_win = int(t_step * fs) + 1
ns_wav = wav.shape[0]
out = np.zeros_like(wav, dtype=np.single)
win = sp.windows.hann(ns_win)
first = 0
while True:
    last = int(min(first + ns_win, ns_wav))
    if last == ns_wav:
        break
    _twin = (first + last) / fs / 2
    x, y, z = [np.interp(_twin, hxyz['t'], hxyz['x']), np.interp(_twin, hxyz['t'], hxyz['y']),
               np.interp(_twin, hxyz['t'], hxyz['z'])]
    tf = tfs.get_hrtf_cart(x, y, z)
    out[first: last, 0] += np.convolve(wav[first: last, 0], tf[0, :], mode='same')
    out[first: last, 1] += np.convolve(wav[first: last, 1], tf[1, :], mode='same')
    first += int(ns_win / 2)

plt.plot(out)


import simpleaudio
simpleaudio.play_buffer(out.astype(np.int16), 2, 2, fs).wait_done()

# simpleaudio.play_buffer(wav.astype(np.int16), 2, 2, fs).wait_done()
