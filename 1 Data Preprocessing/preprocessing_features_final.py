# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 09:02:26 2021

Feature implementation large copied from:
https://github.com/SebastianRestrepoA/EMG-pattern-recognition

@author: StephanLehmler
"""

import numpy as np
import pandas as pd
import math
import pywt
import os

def features_calculation(signal, fs):

    features_names = ['VAR', 'RMS', 'IEMG', 'MAV', 'LOG', 'WL', 'ACC', 'DASDV', 'ZC', 'WAMP', 'MYOP', "FR", "MNP", "TP","MNF", "MDF", "PKF", "WENT"]
    print("Start feature calculation")
    time_features = calc_time_features(signal)
    frequency_features = calc_frequency_features(signal, fs)
    wavelet_features = calc_wavelet_features(signal)
    all_features_df = pd.DataFrame(np.column_stack((time_features, frequency_features, wavelet_features)).T,index=features_names)
    print('Features extraction succesful')

    return all_features_df


def calc_time_features(signal):
    variance = []
    rms = []
    iemg = []
    mav = []
    log_detector = []
    wl = []
    aac = []
    dasdv = []
    zc = []
    wamp = []
    myop = []

    th = np.mean(signal) + 3 * np.std(signal)

    for x in signal:
        frame = x.shape[0]
        variance.append(np.var(x))
        rms.append(np.sqrt(np.mean(x ** 2))) #root mean square
        iemg.append(np.sum(abs(x)))  # Integral
        mav.append(np.sum(np.absolute(x)) / frame)  # Mean Absolute Value
        log_detector.append(np.exp(np.sum(np.log10(np.absolute(x))) / frame))
        wl.append(np.sum(abs(np.diff(x))))  # Wavelength
        aac.append(np.sum(abs(np.diff(x))) / frame)  # Average Amplitude Change
        dasdv.append(
            math.sqrt((1 / (frame - 1)) * np.sum((np.diff(x)) ** 2)))  # Difference absolute standard deviation value
        zc.append(zcruce(x, th))  # Zero-Crossing
        wamp.append(wilson_amplitude(x, th))  # Willison amplitude
        myop.append(myopulse(x, th))  # Myopulse percentage rate

    time_features = np.column_stack((variance, rms, iemg, mav, log_detector, wl, aac, dasdv, zc, wamp, myop))
    return time_features


def wilson_amplitude(signal, th):
    x = abs(np.diff(signal))
    umbral = x >= th
    return np.sum(umbral)


def myopulse(signal, th):
    umbral = signal >= th
    return np.sum(umbral) / len(signal)

def zcruce(X, th):
    th = 0
    cruce = 0
    for cont in range(len(X) - 1):
        can = X[cont] * X[cont + 1]
        can2 = abs(X[cont] - X[cont + 1])
        if can < 0 and can2 > th:
            cruce = cruce + 1
    return cruce

def calc_frequency_features(signal, fs):
    fr = []
    mnp = []
    tot = []
    mnf = []
    mdf = []
    pkf = []

    for x in signal:
        frequency, power = spectrum(x, fs)

        fr.append(frequency_ratio(frequency, power))  # Frequency ratio
        mnp.append(np.sum(power) / len(power))  # Mean power
        tot.append(np.sum(power))  # Total power
        mnf.append(mean_freq(frequency, power))  # Mean frequency
        mdf.append(median_freq(frequency, power))  # Median frequency
        pkf.append(frequency[power.argmax()])  # Peak frequency

    frequency_features = np.column_stack((fr, mnp, tot, mnf, mdf, pkf))

    return frequency_features

def spectrum(signal, fs):
    m = len(signal)
    n = 2 ** (x - 1).bit_length()
    y = np.fft.fft(signal, n)
    yh = y[0:int(n / 2 - 1)]
    fh = (fs / n) * np.arange(0, n / 2 - 1, 1)
    power = np.real(yh * np.conj(yh) / n)

    return fh, power


def frequency_ratio(frequency, power):
    power_low = power[(frequency >= 30) & (frequency <= 250)]
    power_high = power[(frequency > 250) & (frequency <= 500)]
    ULC = np.sum(power_low)
    UHC = np.sum(power_high)

    return ULC / UHC

def shannon(x):
    N = len(x)
    nb = 19
    hist, bin_edges = np.histogram(x, bins=nb)
    counts = hist / N
    nz = np.nonzero(counts)

    return np.sum(counts[nz] * np.log(counts[nz]) / np.log(2))

def mean_freq(frequency, power):
    num = 0
    den = 0
    for i in range(int(len(power) / 2)):
        num += frequency[i] * power[i]
        den += power[i]

    return num / den

def median_freq(frequency, power):
    power_total = np.sum(power) / 2
    temp = 0
    tol = 0.01
    errel = 1
    i = 0

    while abs(errel) > tol:
        temp += power[i]
        errel = (power_total - temp) / power_total
        i += 1
        if errel < 0:
            errel = 0
            i -= 1

    return frequency[i]

def calc_wavelet_features(signal):
    h_wavelet = []

    for x in signal:

        E_a, E = wavelet_energy(x, 'db2', 4)
        E.insert(0, E_a)
        E = np.asarray(E) / 100

        h_wavelet.append(-np.sum(E * np.log2(E)))

    return h_wavelet

def wavelet_energy(x, mother, nivel):
    coeffs = pywt.wavedecn(x, wavelet=mother, level=nivel)
    arr, _ = pywt.coeffs_to_array(coeffs)
    Et = np.sum(arr ** 2)
    cA = coeffs[0]
    Ea = 100 * np.sum(cA ** 2) / Et
    Ed = []

    for k in range(1, len(coeffs)):
        cD = list(coeffs[k].values())
        cD = np.asarray(cD)
        Ed.append(100 * np.sum(cD ** 2) / Et)

    return Ea, Ed


import NinaPro_Utility
import random

def get_all_subjects_normalized_filtered(nb_subjects = 10,db="DB4"):
    #create preprocessed data for multiple subjects at once
    list_of_df = []
    subject_index = {}
    last_end = 0
    for subject in range(1,nb_subjects+1):
        print(subject)
        filename= "S{}_E1_A1.mat".format(str(subject))
        foldername = "../0 Data/"+db+"/data"
        data = NinaPro_Utility.get_data(foldername,filename)
        start = last_end
        end = start + len(data.index)
        last_end = end +1
        subject_index[subject] = (start,end)
        list_of_df.append(data)
    data = pd.concat(list_of_df,ignore_index=True)
    reps = [3]
    data = NinaPro_Utility.normalise(data, reps)
    emg_band = NinaPro_Utility.filter_data(data=data, f=(20,40), butterworth_order=4, btype='bandpass')
    return emg_band,subject_index

def preprocess_emg(emg_band, db, sampleSetting):
    #depending on setting
    reps = [1,2,4,5,6] # does not contain 3 because three is used for normalization
    if db =="DB4":
        nbGestures = 12
    else:
        nbGestures = 17
    #reps and gestures depend on sample setting
    if sampleSetting =="60Perc":
        #all gestures, 2/3th of the samples for training
        gestures = [i for i in range(1,nbGestures+1)]
        test_reps = random.sample(reps, k=1)+[3] #three was used for normalization
        train_reps = [r for r in reps if r not in test_reps]
    elif sampleSetting == "30Perc":
        #all gestures, 1/3th of the samples for training
        gestures = [i for i in range(1,nbGestures+1)]
        test_reps = random.sample(reps, k=3) + [3] #three was used for normalization
        train_reps = [r for r in reps if r not in test_reps]
    elif sampleSetting == "16Perc":
        #all gestures, 1/6th of the samples for training
        gestures = [i for i in range(1,nbGestures+1)]
        test_reps = random.sample(reps, k=4) +[3]#three was used for normalization
        train_reps = [r for r in reps if r not in test_reps]
    elif sampleSetting == "fiveMovements":
        #5 random gestures, 1/6th of the samples for training
        gestures = random.sample(range(1,nbGestures+1), k=5)
        test_reps = random.sample(reps, k=4)+[3]#three was used for normalization
        train_reps = [r for r in reps if r not in test_reps]
    elif sampleSetting == "oneMovement":
        #5 random gestures, 1/6th of the samples for training
        gestures = random.sample(range(1,nbGestures+1), k=1)
        test_reps = random.sample(reps, k=4) +[3]#three was used for normalization
        train_reps = [r for r in reps if r not in test_reps]
    elif sampleSetting == "83Perc":
        #all gestures, 5/6th of the samples for training
        gestures = [i for i in range(1,nbGestures+1)]
        test_reps = [3]#three was used for normalization
        train_reps = [r for r in reps if r not in test_reps]

    #windowing
    win_len = 400
    win_stride = 20
    X_train, y_train, r_train = NinaPro_Utility.windowing(emg_band, train_reps, gestures, win_len, win_stride)
    X_test, y_test, r_test = NinaPro_Utility.windowing(emg_band, test_reps, [i for i in range(1,nbGestures+1)], win_len, win_stride)

    #y_train = NinaPro_Utility.get_categorical(y_train)

    if sampleSetting in ("fiveMovements","oneMovement"):
        label_enc = np.zeros((len(y_train),nbGestures), dtype=int)
        for i in range(len(y_train)):
            label_enc[i,int(y_train[i])-1] = 1
        y_train = label_enc
    else:
        y_train = NinaPro_Utility.get_categorical(y_train)

    y_test = NinaPro_Utility.get_categorical(y_test)

    return X_train, y_train, X_test, y_test



def get_multiple_subjects(nb_subjects=10,retrain_subject=1,sampleSetting="83Perc", db="DB4"):
    #create preprocessed data for multiple subjects at once
    list_of_df = []
    retrain_index_start = 0
    for subject in range(1,nb_subjects+1):
        print(subject)
        filename= "S{}_E1_A1.mat".format(str(subject))
        foldername = "../0 Data/"+db+"/data"
        data = NinaPro_Utility.get_data(foldername,filename)
        if subject < retrain_subject:
            retrain_index_start = retrain_index_start + len(data.index)
        elif subject == retrain_subject:
            retrain_index_end = retrain_index_start + len(data.index)
        list_of_df.append(data)
    data = pd.concat(list_of_df,ignore_index=True)



    #continue as before
    reps = [1,2,3,4,5,6]
    if db =="DB4":
        nbGestures = 12
    else:
        nbGestures = 17
    #reps and gestures depend on sample setting
    if sampleSetting =="60Perc":
        #all gestures, 2/3th of the samples for training
        gestures = [i for i in range(1,nbGestures+1)]
        test_reps = random.sample(reps, k=2)
        train_reps = [r for r in reps if r not in test_reps]
    elif sampleSetting == "30Perc":
        #all gestures, 1/3th of the samples for training
        gestures = [i for i in range(1,nbGestures+1)]
        test_reps = random.sample(reps, k=4)
        train_reps = [r for r in reps if r not in test_reps]
    elif sampleSetting == "16Perc":
        #all gestures, 1/6th of the samples for training
        gestures = [i for i in range(1,nbGestures+1)]
        test_reps = random.sample(reps, k=5)
        train_reps = [r for r in reps if r not in test_reps]
    elif sampleSetting == "fiveMovements":
        #5 random gestures, 1/6th of the samples for training
        gestures = random.sample(range(1,nbGestures+1), k=5)
        test_reps = random.sample(reps, k=5)
        train_reps = [r for r in reps if r not in test_reps]
    elif sampleSetting == "oneMovement":
        #5 random gestures, 1/6th of the samples for training
        gestures = random.sample(range(1,nbGestures+1), k=1)
        test_reps = random.sample(reps, k=5)
        train_reps = [r for r in reps if r not in test_reps]
    elif sampleSetting == "83Perc":
        #all gestures, 5/6th of the samples for training
        gestures = [i for i in range(1,nbGestures+1)]
        test_reps = random.sample(reps, k=1)
        train_reps = [r for r in reps if r not in test_reps]


    data = NinaPro_Utility.normalise(data, reps)

    #emg_low = NinaPro_Utility.filter_data(data=data, f=20, butterworth_order=4, btype='lowpass')
    emg_band = NinaPro_Utility.filter_data(data=data, f=(20,40), butterworth_order=4, btype='bandpass')
    #emg_rectified = NinaPro_Utility.rectify(emg_low)
    #emg_rectified = NinaPro_Utility.rectify(emg_band)

    #before windowing separate retrain subject

    retrain_emg_band = emg_band.iloc[retrain_index_start:retrain_index_end+1,:]

    #np.delete(emg_band,list(range(retrain_index_start,retrain_index_end+1)),axis=0)
    #pre_subject = emg_band.iloc[:retrain_index_start-1,:]
    #post_subject = emg_band.iloc[retrain_index_end+1:,:]
    emg_band = emg_band.iloc[list(range(retrain_index_start)) + list(range(retrain_index_end+1,emg_band.shape[0])) ,:]

    win_len = 400
    win_stride = 20
    X_train, y_train, r_train = NinaPro_Utility.windowing(emg_band, train_reps, gestures, win_len, win_stride)
    X_test, y_test, r_test = NinaPro_Utility.windowing(emg_band, test_reps, [i for i in range(1,nbGestures+1)], win_len, win_stride)

    #y_train = NinaPro_Utility.get_categorical(y_train)

    if sampleSetting in ("fiveMovements","oneMovement"):
        label_enc = np.zeros((len(y_train),nbGestures), dtype=int)
        for i in range(len(y_train)):
            label_enc[i,int(y_train[i])-1] = 1
        y_train = label_enc
    else:
        y_train = NinaPro_Utility.get_categorical(y_train)

    y_test = NinaPro_Utility.get_categorical(y_test)

    #preprocess retrain emg
    X_REtrain, y_REtrain, r_REtrain = NinaPro_Utility.windowing(retrain_emg_band, train_reps, gestures, win_len, win_stride)
    X_REtest, y_REtest, r_REtest = NinaPro_Utility.windowing(retrain_emg_band, test_reps, [i for i in range(1,nbGestures+1)], win_len, win_stride)

    #y_train = NinaPro_Utility.get_categorical(y_train)

    if sampleSetting in ("fiveMovements","oneMovement"):
        label_enc = np.zeros((len(y_REtrain),nbGestures), dtype=int)
        for i in range(len(y_REtrain)):
            label_enc[i,int(y_REtrain[i])-1] = 1
        y_REtrain = label_enc
    else:
        y_REtrain = NinaPro_Utility.get_categorical(y_REtrain)

    y_REtest = NinaPro_Utility.get_categorical(y_REtest)

    return X_train, y_train, X_test, y_test, X_REtrain, y_REtrain, X_REtest, y_REtest


for setting in [(10,"DB4"),(11,"DB3"),(40,"DB2")]:
    nbSubjects = setting[0]
    dbName = setting[1]
    emg,index_dict = get_all_subjects_normalized_filtered(nbSubjects,dbName)

    for s in range(1,nbSubjects+1):
        #extract subject data
        start, end = index_dict[s]
        subject_emg = emg.iloc[start:end+1,:]
        for samplesize in ["83Perc","60Perc","30Perc","16Perc","fiveMovements","oneMovement"]:
            X_train, y_train, X_test, y_test = preprocess_emg(subject_emg, dbName, samplesize)
            fs = 2000
            F_train = np.empty((X_train.shape[0],18,X_train.shape[2]))
            for channel in range(X_train.shape[2]):
                print(channel)
                r = features_calculation(X_train[:,:,channel], str(channel), fs)
                F_train[:,:,channel] = r.values.T

            F_test = np.empty((X_test.shape[0],18,X_test.shape[2]))
            for channel in range(X_test.shape[2]):
                print(channel)
                r = features_calculation(X_test[:,:,channel], str(channel), fs)
                F_test[:,:,channel] = r.values.T
            #save in folder
            foldername = "../0 Data/{0}/{0}_preproc_Features/{1}/{2}/".format(dbName, str(s), samplesize)
            os.makedirs(foldername,exist_ok=True)
            print(foldername)
            np.save(foldername+"_X_train", F_train)
            np.save(foldername+"_X_test", F_test)
            np.save(foldername+"_y_train", y_train)
            np.save(foldername+"_y_test", y_test)
