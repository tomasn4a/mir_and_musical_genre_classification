# -*- coding: utf8 -*-

import numpy as np
import wave
import pylab
import os
import matplotlib.pyplot as plt
import pandas as pd
import librosa
from helpers import *
import matplotlib.cm as cm

HOP = 512
WIN_LEN = 2048
SAMP_RATE = 44100

###############################################################################
# Plot the waveform of a few tunes
def plot_full_wave(path_to_tune, duration=-1):
    fig, ax = plt.subplots()

    freqs = open_tune(path_to_tune, duration=duration)
    track, album, artist = get_names(path_to_tune)

    # Normalize frequencies
    freqs = freqs / float(2 ** 15)


    # Matplotlib does not handle non-ASCII characters too well so:
    ascii_title, ascii_album, ascii_artist = names_to_ascii(track, album, artist)

    time = [x/float(sample_rate) for x in range(len(freqs))]

    ax.plot(time, freqs)
    ax.set_title(ascii_artist + " - " + ascii_album + ":\n" + ascii_title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Amplitde (AU)", fontsize=8)
    ax.set_xlim([0, time[-1]])
    ax.set_ylim([-1, 1])

    fig.tight_layout()
    fig.savefig(track+'_wave.png')


###############################################################################
# Plot a spectrogram of a tune
def plot_full_spec(path_to_tune, duration=-1):
    fig, ax = plt.subplots()

    freqs = open_tune(path_to_tune, duration=duration)
    track, album, artist = get_names(path_to_tune)

    # Matplotlib does not handle non-ASCII characters too well so:
    ascii_title, ascii_album, ascii_artist = names_to_ascii(track, album, artist)

    ax.specgram(freqs, Fs=SAMP_RATE)
    ax.set_title(ascii_artist+" - "+ascii_album+":\n"+ascii_title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_xlim([0,len(freqs)/float(sample_rate)])
    ax.set_ylim([0,sample_rate/2])
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Frequency (Hz)", fontsize=8)

    fig.tight_layout()
    fig.savefig(track+'_spec.png')


###############################################################################
# Plot the waveform and spectrogram
def plot_full_comp(path_to_tune, duration=-1):
    fig, ax = plt.subplots(2, sharex=True, figsize = (14,6))

    freqs = open_tune(path_to_tune, duration=duration)
    track, album, artist = get_names(path_to_tune)

    # Normalize frequencies
    freqs_norm = freqs / float(2 ** 15)

    # Matplotlib does not handle non-ASCII characters too well so:
    ascii_title, ascii_album, ascii_artist = names_to_ascii(track, album, artist)

    time = [x/float(sample_rate) for x in range(len(freqs))]

    ax[0].plot(time, freqs_norm)
    ax[0].set_title(ascii_artist + " - " + ascii_album + ":\n" + ascii_title, fontsize=10)
    ax[0].tick_params(labelsize=8)
    ax[0].set_ylabel("Amplitde (AU)", fontsize=8)
    ax[0].set_ylim([-1, 1])
    ax[1].specgram(freqs, Fs=sample_rate)
    ax[1].tick_params(labelsize=8)
    ax[1].set_xlim([0, time[-1]])
    ax[1].set_ylim([0,sample_rate/2])
    ax[1].set_xlabel("Time (s)", fontsize=8)
    ax[1].set_ylabel("Frequency (Hz)", fontsize=8)


    fig.tight_layout()
    fig.savefig(track[:-4]+'_comp.png')


###############################################################################
# Plot the waveform of a few tunes
def plot_waves(paths_to_tunes, begin=0, n_seconds=4):
    fig, axs = plt.subplots(len(paths_to_tunes), 1)

    for ax, tune in zip(axs.flat, paths_to_tunes):
        tail, track = os.path.split(tune)
        tail, dir1 = os.path.split(tail)
        _, dir2 = os.path.split(tail)

        tune = wave.open(tune, 'r')

        frames = tune.readframes(-1)
        sample_rate = tune.getframerate()

        max_frame = int(begin * sample_rate) + int(n_seconds * sample_rate)
        if max_frame > len(frames):
            print "Please select a shorter window"
            pass

        freqs = pylab.fromstring(frames, 'Int16')
        # Normalize frequencies
        freqs = freqs / float(2 ** 15)

        tune.close()

        # Matplotlib does not handle non-ASCII characters too well so:
        ascii_title = ''.join(i for i in track[:-4] if ord(i) < 128)
        ascii_album = ''.join(i for i in dir1 if ord(i) < 128)
        ascii_artist = ''.join(i for i in dir2 if ord(i) < 128)

        time = np.arange(begin, begin + n_seconds, 1.0 / sample_rate)

        ax.plot(time, freqs[int(begin * sample_rate):max_frame])
        ax.set_title(ascii_artist + " - " + ascii_album + ":\n" + ascii_title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitde (AU)", fontsize=8)
        ax.set_xlim([begin, begin + n_seconds])
        ax.set_ylim([-1, 1])

    fig.tight_layout()
    fig.savefig('waveforms.png')


###############################################################################
# Plot a spectrogram of a few tunes
def plot_specs(paths_to_tunes, begin=0, n_seconds=0):
    fig, axs = plt.subplots(len(paths_to_tunes), 1)
    n_secs_dup = n_seconds

    for ax, tune in zip(axs.flat, paths_to_tunes):
        n_seconds = n_secs_dup

        freqs = open_tune(path_to_tune, duration=duration)
        track, album, artist = get_names(path_to_tune)

        if n_seconds == 0:
            n_seconds = len(freqs)/sample_rate

        # Matplotlib does not handle non-ASCII characters too well so:
        ascii_title, ascii_album, ascii_artist = names_to_ascii(track, album, artist)

        ax.specgram(freqs, Fs=sample_rate)
        ax.set_title(ascii_artist+" - "+ascii_album+":\n"+ascii_title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlim([begin,begin+n_seconds])
        ax.set_ylim([0,sample_rate/2])
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Frequency (Hz)", fontsize=8)

    fig.tight_layout()
    fig.savefig('spectrograms.png')


###############################################################################
# Superimpose feature on waveform
def sup_wave(path_to_tune, feature, duration=-1):
    fig, ax = plt.subplots(figsize = (14,8))

    freqs = open_tune(path_to_tune, duration=duration)
    track, album, artist = get_names(path_to_tune)

    freqs = freqs / float(2 ** 15)

    time = [x/float(sample_rate) for x in range(len(freqs))]

    if feature == 'ZCR':
        feat = librosa.feature.zero_crossing_rate(freqs)[0]
        times = [x*float(HOP)/SAMP_RATE for x in range(len(feat))]

    if feature == 'RMS':
        feat = librosa.feature.rmse(freqs)[0]
        times = [x*float(HOP)/SAMP_RATE for x in range(len(feat))]

    if feature == 'SC':
        feat = librosa.feature.spectral_centroid(freqs, sr = 44100)[0]
        times = [x * float(HOP) / SAMP_RATE for x in range(len(feat))]

    max_feat = max(feat)
    feat_norm = [x/max_feat for x in feat]

    # Matplotlib does not handle non-ASCII characters too well so:
    ascii_title, ascii_album, ascii_artist = names_to_ascii(track, album, artist)

    ax.plot(time, freqs, color='g', alpha=0.2)
    ax.set_title(feature + ": " + ascii_artist + " - " + ascii_album + ":\n" + ascii_title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Amplitde (AU)", fontsize=8)
    ax.set_xlim([0, time[-1]])
    ax.set_ylim([-1, 1])

    ax2 = ax.twinx()
    ax2.plot(times, feat_norm, color='r')
    ax2.tick_params(labelsize=8)
    ax2.set_ylabel(feature + ' (AU)', fontsize=8)
    ax2.set_xlim([0, time[-1]])
    ax2.set_ylim([-1,1])


    fig.tight_layout()
    fig.savefig(feature+' - '+track[:-4]+'_wave.png')


###############################################################################
# Superimpose feature on spectrum
def sup_spec(path_to_tune, feature, duration=-1):
    fig, ax = plt.subplots(figsize = (14,6))

    freqs = open_tune(path_to_tune, duration=duration)
    track, album, artist = get_names(path_to_tune)

    freqs = freqs / float(2 ** 15)

    time = [x/float(sample_rate) for x in range(len(freqs))]

    if feature == 'ZCR':
        feat = librosa.feature.zero_crossing_rate(freqs)[0]
        times = [x*float(HOP)/SAMP_RATE for x in range(len(feat))]

    if feature == 'RMS':
        feat = librosa.feature.rmse(freqs)[0]
        times = [x*float(HOP)/SAMP_RATE for x in range(len(feat))]

    if feature == 'SC':
        feat = librosa.feature.spectral_centroid(freqs, sr = 44100)[0]
        times = [x * float(HOP) / SAMP_RATE for x in range(len(feat))]

    max_feat = max(feat)
    feat_norm = [x/max_feat for x in feat]

    # Matplotlib does not handle non-ASCII characters too well so:
    ascii_title, ascii_album, ascii_artist = names_to_ascii(track, album, artist)

    ax.specgram(freqs, Fs=sample_rate)
    ax.set_title(feature + ": " + ascii_artist + " - " + ascii_album + ":\n" + ascii_title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Amplitde (AU)", fontsize=8)
    ax.set_xlim([0, time[-1]])
    ax.set_ylim([0, 22050])

    ax2 = ax.twinx()
    ax2.plot(times, feat_norm, color='b')
    ax2.tick_params(labelsize=8)
    ax2.set_ylabel(feature + ' (AU)', fontsize=8)
    ax2.set_xlim([0, time[-1]])
    ax2.set_ylim([0,1])


    fig.tight_layout()
    fig.savefig(feature+' - '+track[:-4]+'_wave.png')


###############################################################################
# Bar plots for comparison
def bar_plots(feature_name, bar1, bar2, genres):
    N = len(bar1)
    x_locs = np.arange(N)
    width = 0.35

    fig, ax = plt.subplots(figsize = (12,8))

    rects1 = ax.bar(x_locs, bar1, width, color='r')
    ax.set_ylabel(feature_name + ' mean')
    ax.set_title(feature_name + ' by genre')
    ax.set_xticks(x_locs + width)
    ax.set_xticklabels(list(genres))

    ax2 = ax.twinx()
    rects2 = ax2.bar(x_locs+width, bar2, width, color='y')
    ax2.set_ylabel(feature_name + ' std')



    ax.legend((rects1[0], rects2[0]), ('Mean', 'Std'), loc = 'upper left')

    plt.savefig(feature_name+' by genre.png')


###############################################################################
# Plot the MFCC of a tune
def plot_mfcc(path_to_tune, delta=None, duration=-1):
    fig, ax = plt.subplots(figsize = (14,6))

    freqs = open_tune(path_to_tune, duration=duration)
    track, album, artist = get_names(path_to_tune)

    mfccs = librosa.feature.mfcc(freqs, sr=SAMP_RATE, n_mfcc=13)

    if delta == 1:
        mfcc_cop = mfccs
        mfccs = librosa.feature.delta(mfcc_cop)
    if delta == 2:
        mfcc_cop = mfccs
        mfccs = librosa.feature.delta(mfcc_cop, order=2)

    librosa.display.specshow(mfccs, x_axis='time')

    fig.tight_layout()
    if delta == None:
        fig.savefig(track+'_mfcc.png')
    if delta == 1:
        fig.savefig(track + '_mfccDelta.png')
    if delta == 2:
        fig.savefig(track + '_mfccDeltaDelta.png')

###############################################################################
# Plot the spec and mfc of a tune
def plot_spec_mfccs(path_to_tune, duration=-1):
    freqs = open_tune(path_to_tune, duration=duration)
    track, album, artist = get_names(path_to_tune)

    mfccs = librosa.feature.mfcc(freqs, sr=SAMP_RATE, n_mfcc=13)
    mfccd = librosa.feature.delta(mfccs)
    mfccdd = librosa.feature.delta(mfccs, order=2)

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    librosa.display.specshow(mfccs, sr=SAMP_RATE)
    plt.ylabel('MFCC')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(mfccd, sr=SAMP_RATE)
    plt.ylabel('MFCC-$\Delta$')
    plt.colorbar()

    plt.subplot(3, 1, 3)
    librosa.display.specshow(mfccdd, sr=SAMP_RATE, x_axis='time')
    plt.ylabel('MFCC-$\Delta^2$')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(track[:-4]+'mfccs.png')




###############################################################################
# Plot PCA results
def pca_results(good_data, pca):
    '''
    Inspired from Udacity's Machine Learning Nanodegree, Project 3.
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar')
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)
    plt.legend(loc=6, prop={'size': 1})
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, ncol=1)

    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

    plt.savefig('principal_components.png')


###############################################################################
# Plot confusion matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(['F','Mi','Mo','P','St','Sy']))
    plt.xticks(tick_marks, ['F','Mi','Mo','P','St','Sy'], rotation=45)
    plt.yticks(tick_marks, ['F','Mi','Mo','P','St','Sy'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


###############################################################################
# PLOT PCA-REDUCED DATA COLOR CODED BY GENRE
def channel_results(reduced_data, y, labels):
    '''
    Inspired from Udacity's Machine Learning Nanodegree, Project 3.
    Visualizes the PCA-reduced cluster data in two dimensions using the full dataset
    Data is labeled by "Channel" and cues added for student-selected sample data
    '''

    labeled = pd.concat([reduced_data, y], axis=1)

    # Generate the cluster plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color map
    cmap = cm.get_cmap('viridis')

    # Color the points based on assigned Channel
    labels = labels
    grouped = labeled.groupby('genre')
    for i, channel in grouped:
        channel.plot(ax=ax, kind='scatter', x='Dimension 1', y='Dimension 2', \
                     color=cmap(i*60), label=labels[i - 1], s=30);

    # Set plot title
    ax.set_title("PCA-Reduced Data Labeled by Genre")
    plt.savefig('plot_genres.png')