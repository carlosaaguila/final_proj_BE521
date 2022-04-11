import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scipy
from scipy.stats import pearsonr
from scipy import signal as sig
from scipy.io import loadmat
from scipy.signal import ellip, lfilter, filtfilt, find_peaks, butter, sosfiltfilt, sosfilt

leaderboard_data = loadmat('/Users/carlosaguila/Downloads/drive-download-20220406T170639Z-001/leaderboard_data.mat')
raw_training_data = loadmat('/Users/carlosaguila/Downloads/drive-download-20220406T170639Z-001/raw_training_data.mat')

# glove data for training - per subject
train_dg_s1 = raw_training_data['train_dg'][0][0]
train_dg_s2 = raw_training_data['train_dg'][1][0]
train_dg_s3 = raw_training_data['train_dg'][2][0]

# ecog data for training - per subject
train_ecog_s1 = raw_training_data['train_ecog'][0][0]
train_ecog_s2 = raw_training_data['train_ecog'][1][0]
train_ecog_s3 = raw_training_data['train_ecog'][2][0]

# leaderboard ecog signal per patient
leaderboard_data_s1 = leaderboard_data['leaderboard_ecog'][0][0]
leaderboard_data_s2 = leaderboard_data['leaderboard_ecog'][1][0]
leaderboard_data_s3 = leaderboard_data['leaderboard_ecog'][2][0]


# number of windows in signal given winLen and winDisp
def NumWins(x, fs, winLen, winDisp):
    return (len(x) - winLen * fs + winDisp * fs) // (winDisp * fs)


def filter_data(raw_eeg, fs=1000):
    """
    Write a filter function to clean underlying data.
    Filter type and parameters are up to you. Points will be awarded for reasonable filter type, parameters and application.
    Please note there are many acceptable answers, but make sure you aren't throwing out crucial data or adversly
    distorting the underlying data!

    Input:
      raw_eeg (samples x channels): the raw signal
      fs: the sampling rate (1000 for this dataset)
    Output:
      clean_data (samples x channels): the filtered signal
    """
    raw_eeg_t = raw_eeg.transpose()
    filtered = []

    nyq = fs / 2

    # (b, a) = ellip(4, 0.1, 40, 20/nyq, btype='lowpass')
    sos = butter(8, [0.15, 200], btype='bandpass', output='sos', fs=fs)

    for ch_data in raw_eeg_t:
        # filtered_ch = filtfilt(b, a, ch_data)
        filtered_ch = sosfiltfilt(sos, ch_data)
        filtered.append(filtered_ch)

    filtered = np.array(filtered)

    return filtered.transpose()


# line length
def LL(x):
    return np.sum(np.absolute(np.ediff1d(x)))


# energy
def E(x):
    return np.sum(x ** 2)


# area
def A(x):
    return np.sum(np.absolute(x))


# spectral amp
def spectral_amplitude(x):
    x_fft = np.fft.fft(x)
    return np.mean(x_fft)


# number of crossings (zero) - not in
def ZX(x):
    x_demean = x - np.mean(x)
    num_crossings = 0
    for i in range(1, len(x)):
        fromAbove = False
        fromBelow = False
        if x_demean[i - 1] > 0 and x_demean[i] < 0:
            fromAbove = True
        if x_demean[i - 1] < 0 and x_demean[i] > 0:
            fromBelow = True

        if fromAbove or fromBelow:
            num_crossings += 1
    return num_crossings


def bandpower(x, fs, fmin, fmax):
    f, Pxx = sig.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])


# gets features, load features you want calculated from here
def get_features(filtered_window, fs=1000):
    """
      Write a function that calculates features for a given filtered window.
      Feel free to use features you have seen before in this class, features that
      have been used in the literature, or design your own!

      Input:
        filtered_window (window_samples x channels): the window of the filtered ecog signal
        fs: sampling rate
      Output:
        features (channels x num_features): the features calculated on each channel for the window
    """

    filtered_window_t = filtered_window.transpose()

    features = []

    for ch in filtered_window_t:
        features.append(np.array([LL(ch),  # Line-Length
                                  E(ch),  # Energy
                                  bandpower(ch, fs, 5, 15),
                                  bandpower(ch, fs, 20, 25),
                                  bandpower(ch, fs, 75, 115),
                                  bandpower(ch, fs, 125, 160),
                                  bandpower(ch, fs, 160, 175)
                                  ]))

    features = np.array(features)

    return features


# get_windowed_feats - filters raw ecog signal and finds features
def get_windowed_feats(raw_ecog, fs, window_length, window_overlap):
    """
      Write a function which processes data through the steps of filtering and
      feature calculation and returns features. Points will be awarded for completing
      each step appropriately (note that if one of the functions you call within this script
      returns a bad output, you won't be double penalized). Note that you will need
      to run the filter_data and get_features functions within this function.

      Inputs:
        raw_eeg (samples x channels): the raw signal
        fs: the sampling rate (1000 for this dataset)
        window_length: the window's length
        window_overlap: the window's overlap
      Output:
        all_feats (num_windows x (channels x features)): the features for each channel for each time window
          note that this is a 2D array.
    """

    cleaned_ecog = filter_data(raw_ecog)
    num_wins = NumWins(cleaned_ecog.transpose()[0], fs, window_length, window_overlap)
    all_feats_3d = []
    for winStart in np.arange(0, int(num_wins), 1):
        clip = cleaned_ecog[
               int(winStart * window_overlap * fs):int(winStart * window_overlap * fs + (window_length * fs))]
        all_feats_3d.append(get_features(clip))

    num_channels = len(all_feats_3d[0])
    num_features = len(all_feats_3d[0][0])

    all_feats = np.zeros([len(all_feats_3d), num_features * num_channels])

    for k in range(int(len(all_feats_3d))):
        q = flatten_list = [j for sub in all_feats_3d[k] for j in sub]
        all_feats[k, :] = q

    return np.array(all_feats)


#all_feats_s1 = get_windowed_feats(train_ecog_s1, 1000, 0.1, 0.05)  # output of get_windowed_feats
#all_feats_s2 = get_windowed_feats(train_ecog_s2, 1000, 0.1, 0.05)
#all_feats_s3 = get_windowed_feats(train_ecog_s3, 1000, 0.1, 0.05)

#feats_LB_s1 = get_windowed_feats(leaderboard_data_s1, 1000, 0.1, 0.05)
#feats_LB_s2 = get_windowed_feats(leaderboard_data_s2, 1000, 0.1, 0.05)
#feats_LB_s3 = get_windowed_feats(leaderboard_data_s3, 1000, 0.1, 0.05)


# normalization function
def normalize(TRAIN_DATA, LEADERBOARD_DATA):
    feats1_idx_s1 = np.arange(0, len(TRAIN_DATA.transpose()), 7)
    feats2_idx_s1 = np.arange(1, len(TRAIN_DATA.transpose()), 7)
    feats3_idx_s1 = np.arange(2, len(TRAIN_DATA.transpose()), 7)
    feats4_idx_s1 = np.arange(3, len(TRAIN_DATA.transpose()), 7)
    feats5_idx_s1 = np.arange(4, len(TRAIN_DATA.transpose()), 7)
    feats6_idx_s1 = np.arange(5, len(TRAIN_DATA.transpose()), 7)
    feats7_idx_s1 = np.arange(6, len(TRAIN_DATA.transpose()), 7)
    feat1_s1 = [];
    feat2_s1 = [];
    feat3_s1 = [];
    feat4_s1 = [];
    feat5_s1 = [];
    feat6_s1 = [];
    feat7_s1 = [];

    for i in feats1_idx_s1:
        feat1_s1.append(TRAIN_DATA[:][i])
    for i in feats2_idx_s1:
        feat2_s1.append(TRAIN_DATA[:][i])
    for i in feats3_idx_s1:
        feat3_s1.append(TRAIN_DATA[:][i])
    for i in feats4_idx_s1:
        feat4_s1.append(TRAIN_DATA[:][i])
    for i in feats5_idx_s1:
        feat5_s1.append(TRAIN_DATA[:][i])
    for i in feats6_idx_s1:
        feat6_s1.append(TRAIN_DATA[:][i])
    for i in feats7_idx_s1:
        feat7_s1.append(TRAIN_DATA[:][i])

    feat1_mean_s1 = np.mean(feat1_s1);
    feat1_std_s1 = np.std(feat1_s1)
    feat2_mean_s1 = np.mean(feat2_s1);
    feat2_std_s1 = np.std(feat2_s1)
    feat3_mean_s1 = np.mean(feat3_s1);
    feat3_std_s1 = np.std(feat3_s1)
    feat4_mean_s1 = np.mean(feat4_s1);
    feat4_std_s1 = np.std(feat4_s1)
    feat5_mean_s1 = np.mean(feat5_s1);
    feat5_std_s1 = np.std(feat5_s1)
    feat6_mean_s1 = np.mean(feat6_s1);
    feat6_std_s1 = np.std(feat6_s1)
    feat7_mean_s1 = np.mean(feat7_s1);
    feat7_std_s1 = np.std(feat7_s1)

    all_feats_normalized_s1 = TRAIN_DATA
    for i in feats1_idx_s1:
        all_feats_normalized_s1[:][i] = np.divide((TRAIN_DATA[:][i] - feat1_mean_s1), feat1_std_s1)
    for i in feats2_idx_s1:
        all_feats_normalized_s1[:][i] = np.divide((TRAIN_DATA[:][i] - feat2_mean_s1), feat2_std_s1)
    for i in feats3_idx_s1:
        all_feats_normalized_s1[:][i] = np.divide((TRAIN_DATA[:][i] - feat3_mean_s1), feat3_std_s1)
    for i in feats4_idx_s1:
        all_feats_normalized_s1[:][i] = np.divide((TRAIN_DATA[:][i] - feat4_mean_s1), feat4_std_s1)
    for i in feats5_idx_s1:
        all_feats_normalized_s1[:][i] = np.divide((TRAIN_DATA[:][i] - feat5_mean_s1), feat5_std_s1)
    for i in feats6_idx_s1:
        all_feats_normalized_s1[:][i] = np.divide((TRAIN_DATA[:][i] - feat6_mean_s1), feat6_std_s1)
    for i in feats7_idx_s1:
        all_feats_normalized_s1[:][i] = np.divide((TRAIN_DATA[:][i] - feat7_mean_s1), feat7_std_s1)

    all_feats_normalized_LB_s1 = LEADERBOARD_DATA
    for i in feats1_idx_s1:
        all_feats_normalized_LB_s1[:][i] = np.divide((LEADERBOARD_DATA[:][i] - feat1_mean_s1), feat1_std_s1)
    for i in feats2_idx_s1:
        all_feats_normalized_LB_s1[:][i] = np.divide((LEADERBOARD_DATA[:][i] - feat2_mean_s1), feat2_std_s1)
    for i in feats3_idx_s1:
        all_feats_normalized_LB_s1[:][i] = np.divide((LEADERBOARD_DATA[:][i] - feat3_mean_s1), feat3_std_s1)
    for i in feats4_idx_s1:
        all_feats_normalized_LB_s1[:][i] = np.divide((LEADERBOARD_DATA[:][i] - feat4_mean_s1), feat4_std_s1)
    for i in feats5_idx_s1:
        all_feats_normalized_LB_s1[:][i] = np.divide((LEADERBOARD_DATA[:][i] - feat5_mean_s1), feat5_std_s1)
    for i in feats6_idx_s1:
        all_feats_normalized_LB_s1[:][i] = np.divide((LEADERBOARD_DATA[:][i] - feat6_mean_s1), feat6_std_s1)
    for i in feats7_idx_s1:
        all_feats_normalized_LB_s1[:][i] = np.divide((LEADERBOARD_DATA[:][i] - feat7_mean_s1), feat7_std_s1)

    return all_feats_normalized_s1, all_feats_normalized_LB_s1


#norm_all_feats_s1, norm_feats_LB_s1 = normalize(all_feats_s1, feats_LB_s1)
#norm_all_feats_s2, norm_feats_LB_s2 = normalize(all_feats_s2, feats_LB_s2)
#norm_all_feats_s3, norm_feats_LB_s3 = normalize(all_feats_s3, feats_LB_s3)

train_dg_s1_downsample = train_dg_s1[::50][:-1]
train_dg_s2_downsample = train_dg_s2[::50][:-1]
train_dg_s3_downsample = train_dg_s3[::50][:-1]

#np.save('C:/Users/cagui/PycharmProjects/BE521/all_feats_s1.npy', all_feats_s1)
#np.save('C:/Users/cagui/PycharmProjects/BE521/all_feats_s2.npy', all_feats_s2)
#np.save('C:/Users/cagui/PycharmProjects/BE521/all_feats_s3.npy', all_feats_s3)

#np.save('C:/Users/cagui/PycharmProjects/BE521/feats_LB_s1.npy', feats_LB_s1)
#np.save('C:/Users/cagui/PycharmProjects/BE521/feats_LB_s2.npy', feats_LB_s2)
#np.save('C:/Users/cagui/PycharmProjects/BE521/feats_LB_s3.npy', feats_LB_s3)

#np.save('C:/Users/cagui/PycharmProjects/BE521/norm_all_feats_s1.npy', norm_all_feats_s1)
#np.save('C:/Users/cagui/PycharmProjects/BE521/norm_all_feats_s2.npy', norm_all_feats_s2)
#np.save('C:/Users/cagui/PycharmProjects/BE521/norm_all_feats_s3.npy', norm_all_feats_s3)

#np.save('C:/Users/cagui/PycharmProjects/BE521/norm_feats_LB_s1.npy', norm_feats_LB_s1)
#np.save('C:/Users/cagui/PycharmProjects/BE521/norm_feats_LB_s2.npy', norm_feats_LB_s2)
#np.save('C:/Users/cagui/PycharmProjects/BE521/norm_feats_LB_s3.npy', norm_feats_LB_s3)

# load all npy's
all_feats_s1 = np.load('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/all_feats_s1.npy')
all_feats_s2 = np.load('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/all_feats_s2.npy')
all_feats_s3 = np.load('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/all_feats_s3.npy')

feats_LB_s1 = np.load('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/feats_LB_s1.npy')
feats_LB_s2 = np.load('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/feats_LB_s2.npy')
feats_LB_s3 = np.load('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/feats_LB_s3.npy')

norm_all_feats_s1 = np.load('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/norm_all_feats_s1.npy')
norm_all_feats_s2 = np.load('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/norm_all_feats_s2.npy')
norm_all_feats_s3 = np.load('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/norm_all_feats_s3.npy')

norm_feats_LB_s1 = np.load('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/norm_feats_LB_s1.npy')
norm_feats_LB_s2 = np.load('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/norm_feats_LB_s2.npy')
norm_feats_LB_s3 = np.load('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/norm_feats_LB_s3.npy')


from sklearn.ensemble import RandomForestRegressor

# rfr subj1
rfr_reg_s1 = RandomForestRegressor(n_estimators=1000).fit(norm_all_feats_s1, train_dg_s1_downsample)
LB_pred_s1 = rfr_reg_s1.predict(norm_feats_LB_s1)  # predicting from features from leaderboard data

np.save('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/LB_pred_s1.npy', LB_pred_s1)

model_fname_s1 = '/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/subject1_rfr_1000.model'
pickle.dump(rfr_reg_s1, open(model_fname_s1, 'wb'))

# rfr subj2
rfr_reg_s2 = RandomForestRegressor(n_estimators=1000).fit(norm_all_feats_s2, train_dg_s2_downsample)
LB_pred_s2 = rfr_reg_s2.predict(norm_feats_LB_s2)

np.save('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/LB_pred_s2.npy', LB_pred_s2)

model_fname_s2 = '/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/subject2_rfr_1000.model'
pickle.dump(rfr_reg_s2, open(model_fname_s2, 'wb'))

# rfr subj3
rfr_reg_s3 = RandomForestRegressor(n_estimators=1000).fit(norm_all_feats_s3, train_dg_s3_downsample)
LB_pred_s3 = rfr_reg_s3.predict(norm_feats_LB_s3)

np.save('/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/LB_pred_s3.npy', LB_pred_s3)

model_fname_s3 = '/Users/carlosaguila/PycharmProjects/BE521_Proj/pycharm - be521/subject3_rfr_1000.model'
pickle.dump(rfr_reg_s3, open(model_fname_s3, 'wb'))
