# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 09:02:26 2021

@author: StephanLehmler
"""

import numpy as np
import pandas as pd
import os
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
    #data = NinaPro_Utility.normalise(data, reps)
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
    



#for setting in [(10,"DB4"),(11,"DB3")]:
for setting in [(40,"DB2")]:
    nbSubjects = setting[0]
    dbName = setting[1]
    emg,index_dict = get_all_subjects_normalized_filtered(nbSubjects,dbName)
    
    for s in range(1,nbSubjects+1):
        #extract subject data
        start, end = index_dict[s]
        subject_emg = emg.iloc[start:end+1,:]
        for samplesize in ["83Perc","60Perc","30Perc","16Perc","fiveMovements","oneMovement"]:
            X_train, y_train, X_test, y_test = preprocess_emg(subject_emg, dbName, samplesize)
            #save in folder 
            foldername = "../0 Data/{0}/{0}_preproc_raw/{1}/{2}/".format(dbName, str(s), samplesize)
            os.makedirs(foldername,exist_ok=True)
            print(foldername)
            np.save(foldername+"_X_train", X_train)
            np.save(foldername+"_X_test", X_test)
            np.save(foldername+"_y_train", y_train)
            np.save(foldername+"_y_test", y_test)
       
