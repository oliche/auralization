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


file_heels = "/home/olivier/Documents/NeuroAudio/172358__avakas__high-heels-on-marble-floor.wav"
file_voice = "/home/olivier/Documents/NeuroAudio/104720__tim-kahn__can-you-keep-my-secret.wav"

fs, wav_voice = wavfile.read(file_voice)
fs, wav_heels = wavfile.read(file_heels)
wav_heels = np.ascontiguousarray(np.tile(wav_heels, (2, 1)).T)


ns_wav = wav_heels.shape[0]
nsec_audio = wav_heels.shape[0] / fs

npos = 2000
hxyz = {'x': np.linspace(5, -1, npos),
        'y': np.linspace(3, 0, npos),
        'z': np.linspace(0, 0, npos),
        't': np.linspace(0, nsec_audio, npos)}

# loop over t_step windows, normally there will be a difference between
t_step = 0.1
ns_win = int(t_step * fs) + 1
out = np.zeros_like(wav_heels, dtype=np.single)
win = sp.windows.hann(ns_win)
first = 0
while True:
    last = int(min(first + ns_win, ns_wav))
    if last == ns_wav:
        break
    _twin = (first + last) / fs / 2
    # get location at current time, get transfer functions
    x, y, z = [np.interp(_twin, hxyz['t'], hxyz['x']),
               np.interp(_twin, hxyz['t'], hxyz['y']),
               np.interp(_twin, hxyz['t'], hxyz['z'])]
    tf = tfs.get_hrtf_cart(x, y, z)
    # apply TFs to each channel
    out[first: last, 0] += np.convolve(wav_heels[first: last, 0], tf[0, :], mode='same')
    out[first: last, 1] += np.convolve(wav_heels[first: last, 1], tf[1, :], mode='same')
    first += int(ns_win / 2)


tf = tfs.get_hrtf(r=1, elevation=15, azimuth=270)
# simpleaudio.play_buffer(wav, wav.shape[1], 2, fs).wait_done()
ss = np.zeros_like(wav_voice)
ss[:, 0] = np.convolve(wav_voice[:, 0], tf[0, :], mode='same')
ss[:, 1] = np.convolve(wav_voice[:, 1], tf[1, :], mode='same')


simpleaudio.play_buffer(out.astype(np.int16), 2, 2, fs).wait_done()
simpleaudio.play_buffer(ss.astype(np.int16), 2, 2, fs).wait_done()
