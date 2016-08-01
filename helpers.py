import wave
import pylab
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
import pandas as pd
import librosa
import sklearn

# CONSTANTS USED
# The directories should be modified to follow your own organization
MAIN_DIR = '/Users/tomas/Music/Test'
PROJ_DIR = '/Users/tomas/Documents/MOOCS/udacity/mlnd/p5'
FFT_WIN = 1024
OVERLAP = 0.4
HOP = int(np.ceil((1-OVERLAP)*FFT_WIN))
SAMP_RATE = 44100
NUM_CEPS = 13

def open_tune(path_to_tune, duration=-1):
    tune = wave.open(path_to_tune, 'r')
    if duration == -1:
        frames = tune.readframes(-1)
    else:
        frames = tune.readframes(duration*SAMP_RATE)
    freqs = pylab.fromstring(frames, 'Int16')
    tune.close()

    return freqs


def get_names(path_to_tune):
    tail, track = os.path.split(path_to_tune)
    tail, album = os.path.split(tail)
    _, artist = os.path.split(tail)
    return track, album, artist


def names_to_ascii(track, album, artist):
    ascii_title = ''.join(i for i in track[:-4] if ord(i) < 128)
    ascii_album = ''.join(i for i in album if ord(i) < 128)
    ascii_artist = ''.join(i for i in artist if ord(i) < 128)
    return ascii_title, ascii_album, ascii_artist

def get_genre(path_to_tune):
    tail, track = os.path.split(path_to_tune)
    tail, dir1 = os.path.split(tail)
    tail, dir2 = os.path.split(tail)
    tail, dir3 = os.path.split(tail)
    return dir3


def get_feature_names():
    # First we create the column names of the data frame
    colnames = ['genre', 'ZCR', 'RMS', 'SC']
    #colnames = ['genre', 'ZCR mean', 'RMS mean', 'SC mean']

    for i in range(NUM_CEPS):
        colnames.append('MFCC' + str(i))
    # for i in range(NUM_CEPS):
    #     colnames.append('MFCC' + str(i) + ' std')
    for i in range(NUM_CEPS):
        colnames.append('MFCCD' + str(i))
    # for i in range(NUM_CEPS):
    #     colnames.append('MFCCD' + str(i) + ' std')

    return colnames


def populate_df(main_directory, duration = 60):
    # Initialize dataframe
    colnames = get_feature_names()
    df = pd.DataFrame(columns=colnames)

    os.chdir(main_directory)
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file.endswith('.wav'):
                feats = []
                print "opening " + file
                path_to_tune = root + '/' + file
                genre = get_genre(path_to_tune)
                feats.append(genre)

                data = open_tune(path_to_tune, duration=duration)

                print "extracting zc"
                zc = librosa.feature.zero_crossing_rate(data)
                feats.append(zc[0])
                # feats.append(np.std(zc[0]))

                print "extracting rms"
                rms = librosa.feature.rmse(data)
                feats.append(rms[0])
                # feats.append(np.std(rms[0]))

                print "extracting sc"
                sc = librosa.feature.spectral_centroid(data, sr=SAMP_RATE)
                feats.append(sc[0])
                # feats.append(np.std(sc[0]))

                print "extracting MFCCs \n"
                mfcc = librosa.feature.mfcc(data, sr=SAMP_RATE, n_mfcc=NUM_CEPS, n_fft=FFT_WIN, hop_length=HOP)
                delta_mfcc = librosa.feature.delta(mfcc)

                for i in range(NUM_CEPS):
                    feats.append(mfcc[i])
                # for i in range(NUM_CEPS):
                #     feats.append(np.std(mfcc[i]))
                for i in range(NUM_CEPS):
                    feats.append(delta_mfcc[i])
                # for i in range(NUM_CEPS):
                #     feats.append(np.std(delta_mfcc[i]))

                df.loc[len(df)] = feats

    return df

def stft(samples, frame_size, overlap=0.4, window=np.hanning):
    # First calculate our Hann (or other) weights
    win_weights = window(frame_size)

    # Calculate hop size
    hop = int(np.ceil((1-overlap)*frame_size))

    # Add zeros at the end to make sure we read the entire file
    samples = np.append(samples, np.zeros(frame_size))

    # Now we reshape our data using strides. If we had no overlap between our windows
    # we could simply make sure the length of our data is a multiple of frame_size
    # and then simply reshape as a (len(data)/frame_size, frame_size) matrix. But with
    # overlap it is a bit more complicated. Strides give us the number of bytes we have
    # to step to go the next item in an array. We can play with these numbers to get what
    # we want. As an example imagine an array of 1byte ints, a frame size of 10 and
    # overlap of 0.5. We want to reshape our array so that it takes 1 byte to move to the
    # next item in a row, but only 5 bytes to move to the next row.

    # First let's figure out the dimensions of our new data. The number of columns is simply
    # the frame size and the number of rows is:
    rows = np.ceil((len(samples) - frame_size) / float(hop)) + 1

    # Now let's reshape the data
    frames = as_strided(samples, shape=(rows, frame_size),
                        strides=(samples.strides[0] * hop, samples.strides[0]))

    # Finally let's scale each row by the window weights
    windowed_frames = []
    for row in frames:
        windowed_frames.append([x*y for x,y in zip(row, win_weights)])

    # And take the Fourier Transform
    return np.fft.rfft(windowed_frames), hop


def power_spectrum(magnitudes):
    return [x**2 for x in magnitudes]

