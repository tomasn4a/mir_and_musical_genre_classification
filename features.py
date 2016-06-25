# -*- coding: utf8 -*-

from helpers import *


def zero_crossing(freqs,window_size):
    # First we split our audio into windows of window_size
    num_windows = int(np.ceil(len(freqs)/float(window_size)))
    time = (np.arange(0,num_windows - 1) * (window_size / 44100.))

    zc = []

    # Now calculate the zero crossing. For that we first take the sign of
    # every point in our file: -1, 0, or 1 and then take the absolute value
    # of the pairwise difference: 2 if it crosses 0, 1 if it goes to zero
    # and 0 if it doesn't. Finally we take the mean across the window

    for i in range(0,num_windows-1):
        start = i * window_size
        end = np.min([start+window_size-1, len(freqs)])

        abs_value_diff = np.abs(np.diff(np.sign(freqs[start:end])))
        zc.append(0.5*np.mean(abs_value_diff))

    return zc, time


def root_mean_square(freqs,window_size):
    # First we split our audio into windows of window_size
    num_windows = int(np.ceil(len(freqs)/float(window_size)))
    time = (np.arange(0,num_windows - 1) * (window_size / 44100.))

    rms = []
    squares = [x**2 for x in freqs]
    for i in range(0,num_windows-1):
        start = i * window_size
        end = np.min([start+window_size-1, len(freqs)])

        rms.append(np.sqrt(np.mean(squares[start:end])))

    return rms, time


def spectral_centroid(wavedata, window_size):
    magnitude_spectrum, hop = np.abs(stft(wavedata, window_size))
    tbins, freqs = np.shape(magnitude_spectrum)

    times = [x*hop/44100 for x in np.arange(tbins-1)]

    sc = []

    for bin in range(tbins - 1):
        power = np.abs([x**2 for x in magnitude_spectrum[bin]])
        f_times_p = [x * y for x, y in zip(power, np.arange(1,tbins+1))]
        sc.append(np.sum(f_times_p)/np.sum(power))

    sc = np.nan_to_num(sc)

    return sc, times

