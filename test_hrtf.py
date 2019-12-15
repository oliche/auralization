import numpy as np
from pathlib import Path

import hrtf


def test_coordinate_transform():
    x = np.array([0, 1., -1., 0, 0, 0])
    y = np.array([0, 0., 0., 1, -1, 0])
    z = np.array([0, 0., 0., 0, 0, 1])
    r = np.array([0, 1., 1., 1, 1, 1])
    a = np.array([0, 90., -90., 0, 180, 0])
    e = np.array([0, 0., 0., 0, 0, 90])

    r_, e_, a_ = hrtf.cart2sph(x, y, z)
    x_, y_, z_ = hrtf.sph2cart(r, e, a)

    assert np.all(np.isclose(r, r_))
    assert np.all(np.isclose(a, a_))
    assert np.all(np.isclose(e, e_))
    assert np.all(np.isclose(x, x_))
    assert np.all(np.isclose(y, y_))
    assert np.all(np.isclose(z, z_))


def test_hrtf():
    root_path = Path('/home/olivier/Documents/NeuroAudio')
    path_transfer = root_path.joinpath('KEMAR')
    tfs = hrtf.HRTF(path_transfer)
    tfs.get_hrtf(r=1, elevation=0, azimuth=0)
