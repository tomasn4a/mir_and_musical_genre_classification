# -*- coding: utf8 -*-

from plots import *
from helpers import *
from features import *
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns

MAIN_DIR    = '/Users/tomas/Documents/MOOCS/udacity/mlnd/p5'
GENRES_DIR  = '/Users/tomas/Music/Genres'
MUSIC_DIR   = '/Users/tomas/Music/Test'
GENRES_LIST = ['bluegrass', 'classical', 'flamenco', 'hip hop', 'jazz', 'metal']


# #####################################################################################################################
# #####################################################################################################################
#
# MUSIC INFORMATION RETRIEVAL
#
# #####################################################################################################################
# #####################################################################################################################

# ############################################################
# # PLOT WAVEFORMS
# bill = GENRES_DIR+"/Jazz/Bill Evans/Everybody Digs Bill Evans/09 Oleo.wav"
# paco = GENRES_DIR+"/Flamenco/Paco De Lucia/Fuente Y Caudal/01 Entre Dos Aguas (Rumba).wav"
# noam = GENRES_DIR+"/Bluegrass/Noam Pikelny/Noam Pikelny Plays Kenny Baker Plays Bill Monroe/01 Road to Columbus.wav"
# tunes = [bill, paco, noam]
# plot_waves(tunes, n_seconds=20)


# ############################################################
# # PLOT ZCR ON TOP OF WAVEFORM
# paco = GENRES_DIR+"/Flamenco/Paco De Lucia/Fuente Y Caudal/01 Entre Dos Aguas (Rumba).wav"
# sup_wave(paco,'ZCR', duration=20)

# ############################################################
# # PLOT ZCR MEAN AND STD ACROSS GENRES
# bluegrass = GENRES_DIR+"/Bluegrass/Noam Pikelny/Noam Pikelny Plays Kenny Baker Plays Bill Monroe/07 Big Sandy River.wav"
# classical = GENRES_DIR+"/Classical/Herbert von Karajan/Holst - The Planets (DG)/04 Jupiter, the Bringer of Jollity.wav"
# flamenco  = GENRES_DIR+"/Flamenco/Paco de Lucia/Fuente y Caudal/07 Los Pinares (Tangos).wav"
# hiphop    = GENRES_DIR+"/Hip Hop/VA/Hip Hop and Rap/Mike Jones - Drop And Gimme 50 Ft. Hurricane Chris.wav"
# jazz      = GENRES_DIR+"/Jazz/Art Blakey and the Jazz Messengers/Moanin'/02 Moanin'.wav"
# metal     = GENRES_DIR+"/Metal/August Burns Red/Constellations/02 Existence.wav"
#
# genres = ['bluegrass', 'classical', 'flamenco', 'hip hop', 'jazz', 'metal']
# genre_songs = [bluegrass, classical, flamenco, hiphop, jazz, metal]
#
# zcr_means = []
# zcr_std = []
#
# for tune in genre_songs:
#     print "Opening: " + tune
#     data = open_tune(tune)
#
#     print "Extracting ZCR"
#     zeros = librosa.feature.zero_crossing_rate(data)
#     zcr_means.append(np.mean(zeros))
#     zcr_std.append(np.std(zeros))
#
# bar_plots('ZCR', zcr_means, zcr_std, genres)


# ############################################################
# # PLOT RMS ON TOP OF WAVEFORM
# paco = GENRES_DIR+"/Flamenco/Paco De Lucia/Fuente Y Caudal/01 Entre Dos Aguas (Rumba).wav"
# sup_wave(paco,'RMS', duration=20)


# ############################################################
# # PLOT RMS MEAN AND STD ACROSS GENRES
# bluegrass = GENRES_DIR+"/Bluegrass/Noam Pikelny/Noam Pikelny Plays Kenny Baker Plays Bill Monroe/07 Big Sandy River.wav"
# classical = GENRES_DIR+"/Classical/Herbert von Karajan/Holst - The Planets (DG)/04 Jupiter, the Bringer of Jollity.wav"
# flamenco  = GENRES_DIR+"/Flamenco/Paco de Lucia/Fuente y Caudal/07 Los Pinares (Tangos).wav"
# hiphop    = GENRES_DIR+"/Hip Hop/VA/Hip Hop and Rap/Mike Jones - Drop And Gimme 50 Ft. Hurricane Chris.wav"
# jazz      = GENRES_DIR+"/Jazz/Art Blakey and the Jazz Messengers/Moanin'/02 Moanin'.wav"
# metal     = GENRES_DIR+"/Metal/August Burns Red/Constellations/02 Existence.wav"
#
# genres = ['bluegrass', 'classical', 'flamenco', 'hip hop', 'jazz', 'metal']
# genre_songs = [bluegrass, classical, flamenco, hiphop, jazz, metal]
#
# rms_means = []
# rms_std = []
#
# for tune in genre_songs:
#     print "Opening: " + tune
#     data = open_tune(tune)
#
#     print "Extracting RMS"
#     rms = librosa.feature.rmse(data)
#     rms_means.append(np.mean(rms))
#     rms_std.append(np.std(rms))
#
# bar_plots('RMS', rms_means, rms_std, genres)

# ############################################################
# # PLOT SPECTROGRAMS
# bill = GENRES_DIR+"/Jazz/Bill Evans/Everybody Digs Bill Evans/09 Oleo.wav"
# paco = GENRES_DIR+"/Flamenco/Paco De Lucia/Fuente Y Caudal/01 Entre Dos Aguas (Rumba).wav"
# noam = GENRES_DIR+"/Bluegrass/Noam Pikelny/Noam Pikelny Plays Kenny Baker Plays Bill Monroe/01 Road to Columbus.wav"
# tunes = [bill, paco, noam]
# plot_specs(tunes, n_seconds=20)


# ############################################################
# # PLOT WAVE AND SPECTROGRAM
# wagner = GENRES_DIR+"/Classical/Herbert von Karajan/Wagner/03 Die Meistersinger von Nurnberg_ Prelude to Act III.wav"
# plot_full_comp(wagner)


# ############################################################
# # PLOT SC ON TOP OF WAVEFORM
# paco = GENRES_DIR+"/Flamenco/Paco De Lucia/Fuente Y Caudal/01 Entre Dos Aguas (Rumba).wav"
# sup_spec(paco,'SC', duration=20)


# ############################################################
# # PLOT SC MEAN AND STD ACROSS GENRES
# bluegrass = GENRES_DIR+"/Bluegrass/Noam Pikelny/Noam Pikelny Plays Kenny Baker Plays Bill Monroe/07 Big Sandy River.wav"
# classical = GENRES_DIR+"/Classical/Herbert von Karajan/Holst - The Planets (DG)/04 Jupiter, the Bringer of Jollity.wav"
# flamenco  = GENRES_DIR+"/Flamenco/Paco de Lucia/Fuente y Caudal/07 Los Pinares (Tangos).wav"
# hiphop    = GENRES_DIR+"/Hip Hop/VA/Hip Hop and Rap/Mike Jones - Drop And Gimme 50 Ft. Hurricane Chris.wav"
# jazz      = GENRES_DIR+"/Jazz/Art Blakey and the Jazz Messengers/Moanin'/02 Moanin'.wav"
# metal     = GENRES_DIR+"/Metal/August Burns Red/Constellations/02 Existence.wav"
#
# genres = ['bluegrass', 'classical', 'flamenco', 'hip hop', 'jazz', 'metal']
# genre_songs = [bluegrass, classical, flamenco, hiphop, jazz, metal]
#
# sc_means = []
# sc_std = []
#
# for tune in genre_songs:
#     print "Opening: " + tune
#     data = open_tune(tune)
#
#     print "Extracting SC"
#     sc = librosa.feature.spectral_centroid(data, sr=44100)
#     sc_means.append(np.mean(sc))
#     sc_std.append(np.std(sc))
#
# bar_plots('SC', sc_means, sc_std, genres)


# ############################################################
# # PLOT SPECTROGRAMS AND MFCC
# paco = GENRES_DIR+"/Flamenco/Paco De Lucia/Fuente Y Caudal/01 Entre Dos Aguas (Rumba).wav"
# plot_spec_mfccs(paco)


# #####################################################################################################################
# #####################################################################################################################
#
# MACHINE LEARNING
#
# #####################################################################################################################
# #####################################################################################################################

# ############################################################
# # GET FEATURES
# df = populate_df(GENRES_DIR, duration=300)
# df.to_pickle(MAIN_DIR+'/genres_300.pkl')
#
# df = populate_df(GENRES_DIR, duration=-1)
# df.to_pickle(MAIN_DIR+'/genres_full.pkl')


############################################################
# READ STORED FEATURES AND SPLIT INTO TRAIN/TEST
raw_data = pd.read_pickle(MAIN_DIR+'/genres_300.pkl')

# CODIFIY GENRES
raw_data['genre'] = raw_data['genre'].astype('category')
cat_columns = raw_data.select_dtypes(['category']).columns
raw_data[cat_columns] = raw_data[cat_columns].apply(lambda x: x.cat.codes)

# DROP FIRST MFCC AND MFCC_DELTA
cols_drop = ['MFCC0 mean', 'MFCC0 std', 'MFCCD0 mean', 'MFCCD0 std']
mod_data = raw_data.drop(cols_drop, axis=1)

# SPLIT FEATURES AND LABEL
features = list(mod_data.columns[1:])
labels = mod_data.columns[0]

X_all = raw_data[features]
y_all = raw_data[labels]

# STRATIFIED SPLIT INTO TRAINING AND TESTING SETS
sss = StratifiedShuffleSplit(y_all, n_iter=1, test_size=0.25, random_state=4)

for train_index, test_index in sss:
    X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
    y_train, y_test = y_all[train_index], y_all[test_index]

# # NAIVE KNN
# classifier = KNeighborsClassifier(n_neighbors=1)
# classifier.fit(X_train, y_train)
#
# y_preds = classifier.predict(X_test)
#
# print classification_report(y_test, y_preds, target_names=GENRES_LIST)
#
# conf_matrix = confusion_matrix(y_test, y_preds)
# df_cm = pd.DataFrame(conf_matrix, index=GENRES_LIST,columns=GENRES_LIST)
# plt.figure(figsize=(10,7))
# sns.heatmap(df_cm, annot=True)
# plt.savefig('naiveKNN.png')

# # IMPROVING THE KNN MODEL
# parameters = {'n_neighbors':np.arange(1,25,1),
#               'weights':['uniform','distance']}
#
# gs_classifier = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, scoring='f1_weighted')
# gs_classifier.fit(X_train,y_train)
#
# print gs_classifier.best_params_
#
# y_preds = gs_classifier.predict(X_test)
#
# print classification_report(y_test, y_preds, target_names=GENRES_LIST)
#
# conf_matrix = confusion_matrix(y_test, y_preds)
# df_cm = pd.DataFrame(conf_matrix, index=GENRES_LIST,columns=GENRES_LIST)
# plt.figure(figsize=(10,7))
# sns.heatmap(df_cm, annot=True)
# plt.savefig('gsKNN.png')

# # NAIVE BAYES
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
#
# y_preds = classifier.predict(X_test)
#
# print classification_report(y_test, y_preds, target_names=GENRES_LIST)
#
# conf_matrix = confusion_matrix(y_test, y_preds)
# df_cm = pd.DataFrame(conf_matrix, index=GENRES_LIST,columns=GENRES_LIST)
# plt.figure(figsize=(10,7))
# sns.heatmap(df_cm, annot=True)
# plt.savefig('gaussianNB.png')


# #####################################################################################################################
# #####################################################################################################################
#
# VISUALIZATION
#
# #####################################################################################################################
# #####################################################################################################################

############################################################
# PRINCIPAL COMPONENT ANALYSIS AND PLOTTING

# Keep only MFCC
cols_keep = ['genre']
for i in range(12):
    cols_keep.append('MFCC'+str(i+1)+' mean')
for i in range(12):
    cols_keep.append('MFCC' + str(i + 1) + ' std')

mod_data = raw_data[cols_keep]

# Split features and labels
features = list(mod_data.columns[1:])
labels = mod_data.columns[0]

X_all = raw_data[features]
y_all = raw_data[labels]

# Perform PCA
n_components = 2
pca = PCA(n_components=n_components).fit(X_all)
pca_plot = pca_results(X_all, pca)

reduced_data = pca.transform(X_all)
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Plot music library
channel_results(reduced_data,y_all,GENRES_LIST)
