# -*- coding: utf8 -*-

from plots import *
from helpers import *
from features import *
from sklearn.preprocessing import normalize, StandardScaler, RobustScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.decomposition import PCA
import seaborn as sns

MAIN_DIR = '/Users/tomas/Documents/MOOCS/udacity/mlnd/p5'
GENRES_DIR = '/Users/tomas/Music/Genres'
BAYES_DIR = '/Users/tomas/Music/Bayes test'
GENRES_LIST = ['bluegrass', 'classical', 'flamenco', 'hip hop', 'jazz', 'metal']

# #####################################################################################################################
# #####################################################################################################################
#
# # MUSIC INFORMATION RETRIEVAL
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
# bluegrass = GENRES_DIR + "/Bluegrass/Noam Pikelny/Noam Pikelny Plays Kenny Baker Plays Bill Monroe/07 Big Sandy River.wav"
# classical = GENRES_DIR + "/Classical/Herbert von Karajan/Holst - The Planets (DG)/04 Jupiter, the Bringer of Jollity.wav"
# flamenco = GENRES_DIR + "/Flamenco/Paco de Lucia/Fuente y Caudal/07 Los Pinares (Tangos).wav"
# hiphop = GENRES_DIR + "/Hip Hop/VA/Hip Hop and Rap/Mike Jones - Drop And Gimme 50 Ft. Hurricane Chris.wav"
# jazz = GENRES_DIR + "/Jazz/Art Blakey and the Jazz Messengers/Moanin'/02 Moanin'.wav"
# metal = GENRES_DIR + "/Metal/August Burns Red/Constellations/02 Existence.wav"
#
# genres = ['bluegrass', 'classical', 'flamenco', 'hip hop', 'jazz', 'metal']
# genre_songs = [bluegrass, classical, flamenco, hiphop, jazz, metal]
#
# zcrs = []
#
# for tune in genre_songs:
#     print "Opening: " + tune
#     data = open_tune(tune)
#
#     print "Extracting ZCR"
#     zeros = librosa.feature.zero_crossing_rate(data)
#     zcrs.append(zeros[0])
#
# box_plots(zcrs, genres, 'ZCR')


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
# rmss = []
#
# for tune in genre_songs:
#     print "Opening: " + tune
#     data = open_tune(tune)
#
#     print "Extracting RMS"
#     rms = librosa.feature.rmse(data)
#     rmss.append(rms[0])
#
# box_plots(rmss, genres, 'RMS')


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
# scs = []
#
# for tune in genre_songs:
#     print "Opening: " + tune
#     data = open_tune(tune)
#
#     print "Extracting SC"
#     sc = librosa.feature.spectral_centroid(data, sr=44100)
#     scs.append(sc[0])
#
# box_plots(scs, genres, 'SC')


# ############################################################
# # PLOT SPECTROGRAMS AND MFCC
# paco = GENRES_DIR+"/Flamenco/Paco De Lucia/Fuente Y Caudal/01 Entre Dos Aguas (Rumba).wav"
# plot_spec_mfccs(paco)


# #####################################################################################################################
# #####################################################################################################################
#
# # MACHINE LEARNING
#
# #####################################################################################################################
# #####################################################################################################################

# ############################################################
# # GET FEATURES
# df = populate_df(GENRES_DIR, duration=300)
# df.to_pickle(MAIN_DIR+'/genres_300_full.pkl')
#
# df = populate_df(GENRES_DIR, duration=-1)
# df.to_pickle(MAIN_DIR+'/genres_300_full.pkl')


# ###########################################################
# READ STORED FEATURES AND GET MEAN AND STD
raw_data = pd.read_pickle(MAIN_DIR + '/genres_300_full.pkl')

raw_data['ZCR mean'] = [np.mean(x.reshape(1, -1)) for x in raw_data['ZCR']]
raw_data['ZCR std'] = [np.std(x.reshape(1, -1)) for x in raw_data['ZCR']]
del raw_data['ZCR']
raw_data['RMS mean'] = [np.mean(x.reshape(1, -1)) for x in raw_data['RMS']]
raw_data['RMS std'] = [np.std(x.reshape(1, -1)) for x in raw_data['RMS']]
del raw_data['RMS']
raw_data['SC mean'] = [np.mean(x.reshape(1, -1)) for x in raw_data['SC']]
raw_data['SC std'] = [np.std(x.reshape(1, -1)) for x in raw_data['SC']]
del raw_data['SC']
for i in range(13):
    raw_data['MFCC' + str(i) + ' mean'] = [np.mean(x.reshape(1, -1)) for x in raw_data['MFCC' + str(i)]]
    raw_data['MFCC' + str(i) + ' std'] = [np.std(x.reshape(1, -1)) for x in raw_data['MFCC' + str(i)]]
    raw_data['MFCCD' + str(i) + ' mean'] = [np.mean(x.reshape(1, -1)) for x in raw_data['MFCCD' + str(i)]]
    raw_data['MFCCD' + str(i) + ' std'] = [np.std(x.reshape(1, -1)) for x in raw_data['MFCCD' + str(i)]]
    del raw_data['MFCC' + str(i)]
    del raw_data['MFCCD' + str(i)]

# CODIFIY GENRES
raw_data['genre'] = raw_data['genre'].astype('category')
cat_columns = raw_data.select_dtypes(['category']).columns
raw_data[cat_columns] = raw_data[cat_columns].apply(lambda l: l.cat.codes)

# DROP FIRST MFCC AND MFCC_DELTA
cols_drop = ['MFCC0 mean', 'MFCC0 std', 'MFCCD0 mean', 'MFCCD0 std']
mod_data = raw_data.drop(cols_drop, axis=1)

# SPLIT FEATURES AND LABEL
features = list(mod_data.columns[1:])
labels = mod_data.columns[0]
X_all = mod_data[features]
y_all = mod_data[labels]

# # PLOT CORRELATION MATRIX
# plot_correlation_matrix(X_all)
#

# #NAIVE BAYES
#
# # STRATIFIED SPLIT INTO TRAINING AND TESTING SETS
# sss = StratifiedShuffleSplit(y_all, n_iter=7, test_size=0.25, random_state=4)
# precisions = []
# recalls = []
# f1s = []
# accuracies = []
# for train_index, test_index in sss:
#     X_train, X_test = X_all.iloc[train_index], X_all.iloc[test_index]
#     y_train, y_test = y_all[train_index], y_all[test_index]
#
#     classifier = GaussianNB()
#     classifier.fit(X_train, y_train)
#
#     y_preds = classifier.predict(X_test)
#
#     precisions.append(precision_score(y_test, y_preds, average=None))
#     recalls.append(recall_score(y_test, y_preds, average=None))
#     f1s.append(f1_score(y_test, y_preds, average=None))
#     accuracies.append(accuracy_score(y_test, y_preds))
#
# # # GET SCORES
# print np.round(np.mean(precisions, axis=0),2)
# print np.round(np.mean(recalls, axis=0),2)
# print np.round(np.mean(f1s, axis=0),2)
# print np.round(np.mean([np.mean(precisions, axis=0),np.mean(recalls, axis=0),np.mean(f1s, axis=0)], axis=1),3)
# print np.round(np.mean(accuracies),3)

# # PLOT CONFUSION MATRIX FOR LATEST FOLD
# conf_matrix = confusion_matrix(y_test, y_preds)
# df_cm = pd.DataFrame(conf_matrix, index=GENRES_LIST,columns=GENRES_LIST)
# plt.figure(figsize=(10,7))
# sns.heatmap(df_cm, annot=True)
# plt.savefig('gaussianNB.png')

# # CHECK RESULTS ON UNSEEN DATA
# samples = populate_df(BAYES_DIR, duration=-1)
# samples.to_pickle(MAIN_DIR+'/bayes_full.pkl')
#
# samples = pd.read_pickle(MAIN_DIR + '/bayes_full.pkl')
# samples['ZCR mean'] = [np.mean(x.reshape(1, -1)) for x in samples['ZCR']]
# samples['ZCR std'] = [np.std(x.reshape(1, -1)) for x in samples['ZCR']]
# del samples['ZCR']
# samples['RMS mean'] = [np.mean(x.reshape(1, -1)) for x in samples['RMS']]
# samples['RMS std'] = [np.std(x.reshape(1, -1)) for x in samples['RMS']]
# del samples['RMS']
# samples['SC mean'] = [np.mean(x.reshape(1, -1)) for x in samples['SC']]
# samples['SC std'] = [np.std(x.reshape(1, -1)) for x in samples['SC']]
# del samples['SC']
# for i in range(13):
#     samples['MFCC' + str(i) + ' mean'] = [np.mean(x.reshape(1, -1)) for x in samples['MFCC' + str(i)]]
#     samples['MFCC' + str(i) + ' std'] = [np.std(x.reshape(1, -1)) for x in samples['MFCC' + str(i)]]
#     samples['MFCCD' + str(i) + ' mean'] = [np.mean(x.reshape(1, -1)) for x in samples['MFCCD' + str(i)]]
#     samples['MFCCD' + str(i) + ' std'] = [np.std(x.reshape(1, -1)) for x in samples['MFCCD' + str(i)]]
#     del samples['MFCC' + str(i)]
#     del samples['MFCCD' + str(i)]
# new_songs = samples['genre']
# del samples['genre']
# # DROP FIRST MFCC AND MFCC_DELTA
# cols_drop = ['MFCC0 mean', 'MFCC0 std', 'MFCCD0 mean', 'MFCCD0 std']
# samples = samples.drop(cols_drop, axis=1)
#
# print pd.DataFrame(np.round(classifier.predict_proba(samples),3), columns=GENRES_LIST, index=new_songs)


# # K-NN
#
# # STANDARDIZING DATA
# scaler = StandardScaler()
# X_all_s = scaler.fit_transform(X_all)
# X_all_s = pd.DataFrame(X_all_s, columns=X_all.columns)
#
# # STRATIFIED SPLIT INTO TRAINING AND TESTING SETS
# sss = StratifiedShuffleSplit(y_all, n_iter=7, test_size=0.25, random_state=4)
# precisions = []
# recalls = []
# f1s = []
# accuracies = []
# for train_index, test_index in sss:
#     X_train, X_test = X_all_s.iloc[train_index], X_all_s.iloc[test_index]
#     y_train, y_test = y_all[train_index], y_all[test_index]
#
#     classifier = KNeighborsClassifier(n_neighbors=1)
#     classifier.fit(X_train, y_train)
#
#     y_preds = classifier.predict(X_test)
#
#     precisions.append(precision_score(y_test, y_preds, average=None))
#     recalls.append(recall_score(y_test, y_preds, average=None))
#     f1s.append(f1_score(y_test, y_preds, average=None))
#     accuracies.append(accuracy_score(y_test, y_preds))
#
# # GET SCORES
# print np.round(np.mean(precisions, axis=0),2)
# print np.round(np.mean(recalls, axis=0),2)
# print np.round(np.mean(f1s, axis=0),2)
# print np.round(np.mean([np.mean(precisions, axis=0),np.mean(recalls, axis=0),np.mean(f1s, axis=0)], axis=1),3)
# print np.round(np.mean(accuracies),3)

#
# # PLOT CONFUSION MATRIX FOR LATEST FOLD
# conf_matrix = confusion_matrix(y_test, y_preds)
# df_cm = pd.DataFrame(conf_matrix, index=GENRES_LIST,columns=GENRES_LIST)
# plt.figure(figsize=(10,7))
# sns.heatmap(df_cm, annot=True)
# plt.savefig('naiveKNN.png')

# # IMPROVING THE KNN MODEL
#
# parameters = {'n_neighbors':np.arange(1,25,1),
#               'weights':['uniform','distance'],
#               'p':[1,2]}
#
# # STANDARDIZING DATA
# scaler = StandardScaler()
# X_all_s = scaler.fit_transform(X_all)
# X_all_s = pd.DataFrame(X_all_s, columns=X_all.columns)
#
# sss = StratifiedShuffleSplit(y_all, n_iter=1, test_size=0.25, random_state=4)
# for train_index, test_index in sss:
#     X_train_s, X_test_s = X_all_s.iloc[train_index], X_all_s.iloc[test_index]
#     y_train, y_test = y_all[train_index], y_all[test_index]
#
# gs_classifier = GridSearchCV(KNeighborsClassifier(), parameters, cv=7, scoring='f1_weighted')
# gs_classifier.fit(X_train_s,y_train)
#
# print gs_classifier.best_params_
#
# y_preds = gs_classifier.predict(X_test_s)
#
# print classification_report(y_test, y_preds, target_names=GENRES_LIST)
#
# conf_matrix = confusion_matrix(y_test, y_preds)
# df_cm = pd.DataFrame(conf_matrix, index=GENRES_LIST,columns=GENRES_LIST)
# plt.figure(figsize=(10,7))
# sns.heatmap(df_cm, annot=True)
# plt.show()
# # plt.savefig('gsKNN.png')



# #####################################################################################################################
# #####################################################################################################################
#
# # VISUALIZATION
#
# #####################################################################################################################
# #####################################################################################################################

############################################################
# PRINCIPAL COMPONENT ANALYSIS AND PLOTTING

# Keep only MFCC
cols_keep = ['genre']
for i in range(12):
    cols_keep.append('MFCC' + str(i + 1) + ' mean')
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
sns.reset_orig()
genre_results(reduced_data,y_all,GENRES_LIST)




# cols = X_all.columns
#
# scaler = RobustScaler()
# X_all_s = scaler.fit_transform(X_all)
# X_all_s = pd.DataFrame(X_all_s, columns=cols)
#
# # Perform PCA
# n_components = 2
# pca = PCA(n_components=n_components).fit(X_all_s)
# pca_t = PCA(n_components=n_components).fit_transform(X_all_s)
# pca_plot = pca_results(X_all_s, pca)
#
# columns = []
# for i in range(n_components):
#     columns.append('Dimension ' + str(i + 1))
#
# reduced_data = pca.transform(X_all_s)
# reduced_data = pd.DataFrame(reduced_data, columns=columns)
#
# # Plot music library
# genre_results(reduced_data, y_all, GENRES_LIST)
