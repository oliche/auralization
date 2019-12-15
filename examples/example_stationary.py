from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

import simpleaudio

from hrtf import HRTF

root_path = Path('/home/olivier/Documents/NeuroAudio')
path_transfer = root_path.joinpath('KEMAR')

tfs = HRTF(path_transfer)

fix, axes = plt.subplots(nrows=2, sharex='all')
# axes[0].plot(t, pos), axes[0].title.set_text('Extracted')
axes[0].imshow(tfs.waveform[:, 0, :].transpose(), aspect='auto')
axes[1].plot(tfs.elevation)
axes[1].plot(tfs.azimuth)


file_heels = "/home/olivier/Documents/NeuroAudio/172358__avakas__high-heels-on-marble-floor.wav"
file_voice = "/home/olivier/Documents/NeuroAudio/104720__tim-kahn__can-you-keep-my-secret.wav"
file_helico = "/home/olivier/Documents/NeuroAudio/194250__deleted-user-3544904__helicopter.wav"
file_pop = '/home/olivier/Documents/NeuroAudio/392624__kenrt__champagne-cork.wav'

fs, wav = wavfile.read(file_voice)
# wav = wav[:200000, :]



##

tf = tfs.get_hrtf(r=1.4, elevation=0, azimuth=0)
# simpleaudio.play_buffer(wav, wav.shape[1], 2, fs).wait_done()
ss = np.zeros_like(wav)
ss[:, 0] = np.convolve(wav[:, 0], tf[0, :], mode='same')
ss[:, 1] = np.convolve(wav[:, 1], tf[1, :], mode='same')

# plt.plot(rms)
simpleaudio.play_buffer(ss.astype(np.int16), 2, 2, fs).wait_done()
# plt.figure(), plt.plot(wav), plt.plot(ss)


