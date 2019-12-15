from pathlib import Path

from scipy.io import wavfile
import numpy as np

"""
KEMAR is 1.4 meters away from the microphones
elevation [-90, 90], 90 vertical up
azimuth [0 - 360], 90 right
x: medio-lateral right positive
y: antero-posterior front positive
z: dorso-ventral up positive
"""
RADIUS = 1.4


class HRTF:

    def __init__(self, path_transfer):
        """
        HRTF(path_transfer_functions)
        self.azimuth
        self.elevation
        self.waveform: left and right transfer functions
        """
        self.path = Path(path_transfer)
        self.files = sorted([f for f in self.path.rglob('*.wav')
                             if f.parts[-2].startswith('elev') and f.name.startswith('L')])
        self.azimuth = np.zeros(len(self.files))
        self.elevation = np.zeros(len(self.files))
        self.radius = np.zeros(len(self.files)) + RADIUS
        self.waveform = np.zeros((len(self.files), 2, 512), np.int16)

        for i, file_left in enumerate(self.files):
            n = file_left.name
            file_right = file_left.parent.joinpath(f'R{file_left.name[1:]}')
            self.elevation[i] = np.float(n[1:n.find('e')])
            self.azimuth[i] = np.float(n[n.find('e') + 1:n.find('a')])
            _, self.waveform[i, 0, :] = wavfile.read(file_left)
            _, self.waveform[i, 1, :] = wavfile.read(file_right)

        self.x, self.y, self.z = sph2cart(self.radius, self.elevation, self.azimuth)
        # normalize waveforms by their rms - need to cast to float for doing so
        rms = np.sqrt(np.sum(np.single(self.waveform) ** 2, axis=2))
        self.waveform = (self.waveform.T / rms.T).T

    def get_hrtf(self, r, elevation, azimuth):
        """
        Finds the closest transfer function according to the direction
        Scales the amplitude using spherical divergence
        """
        x, y, z = sph2cart(RADIUS, elevation, azimuth)
        return self.get_hrtf_cart(x, y, z)

    def get_hrtf_cart(self, x, y, z):
        """
        Finds the closest transfer function according to the direction
        Scales the amplitude using spherical divergence
        """
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        wind = np.argmin(np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2))
        return self.waveform[wind, :, :] / np.sqrt(r)


def sph2cart(r, elevation, azimuth):
    y = r * np.cos(elevation / 180 * np.pi) * np.cos(azimuth / 180 * np.pi)
    x = r * np.cos(elevation / 180 * np.pi) * np.sin(azimuth / 180 * np.pi)
    z = r * np.sin(elevation / 180 * np.pi)
    return x, y, z


def cart2sph(x, y, z):
    azimuth = np.arctan2(x, y) * 180 / np.pi
    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return r, elevation, azimuth
