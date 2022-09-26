# -*- coding: utf-8 -*-
"""
Created on Tue May  4 22:05:19 2021

@author: StephanLehmler
"""

import numpy as np
import random
import tensorflow as tf
import optuna

def get_subject(subject=1):
    foldername = "../0 Data/DB2/DB2_preproc_Features/{}/83Perc/".format(str(subject))
    X_train = np.load(foldername+"_X_train.npy")
    X_test = np.load(foldername+"_X_test.npy")
    y_train = np.load(foldername+"_y_train.npy")
    y_test = np.load(foldername+"_y_test.npy")
    
    return X_train, y_train, X_test, y_test


X_train = []
X_test= []
y_train= []
y_test= []
print("load training data")
for i in range(1,11):
    print(i)
    X_tr, y_tr, X_te, y_te = get_subject(i)
    X_train.append(X_tr)
    X_test.append(X_te)
    y_train.append(y_tr)
    y_test.append(y_te)
#flatten
X_train = np.concatenate(X_train)
X_test = np.concatenate(X_test)
y_train = np.concatenate(y_train)
y_test = np.concatenate(y_test)
#shuffle
# random_ids_train = list(range(y_train.shape[0]))
# random.shuffle(random_ids_train)
# X_train= X_train[random_ids_train]
# y_train= y_train[random_ids_train]
    

# 1. Define an objective function to be maximized.
def objective(trial):
    """I want to tune:
        nb of cnn layers
            nb of kernel
            dropout factor
        nb of dense layers
        batchsize
        epochs
        Adam
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
    """    


    nb_dense_layers = trial.suggest_int("nb_dense_layers",1,8)
    dense_dropout_factor = trial.suggest_uniform('dense_dropout_factor',0.0,0.5)
    batchsize = trial.suggest_categorical('batchsize', [16,32,64,128,256,512])
    learning_rate = trial.suggest_loguniform('learning_rate',0.0001, 0.1)
    beta_1 = trial.suggest_loguniform('beta_1',0.1, 2.0)
    beta_2 = trial.suggest_loguniform('beta_2',0.1, 2.0)
    epsilon = trial.suggest_loguniform('epsilon',1e-07, 1.0)
    
    # 2. Suggest values of the hyperparameters using a trial object.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(18,12)))
    
    num_hidden_input = int(trial.suggest_loguniform('num_hidden_input', 36, 512))
    batchnorm = trial.suggest_categorical("batchnorm", [True,False])
    input_dropout_factor = trial.suggest_uniform('input_dropout_factor',0.0,0.5)
    if batchnorm:
            model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(num_hidden_input, activation='relu', kernel_regularizer="l2"))
    model.add(tf.keras.layers.Dropout(input_dropout_factor))

    for i in range(nb_dense_layers):
        num_hidden = int(trial.suggest_loguniform('num_hidden{}'.format(i), 4, 512))
        model.add(tf.keras.layers.Dense(num_hidden, activation='relu', kernel_regularizer="l2"))
        if batchnorm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dense_dropout_factor))
    model.add(tf.keras.layers.Dense(17, kernel_regularizer="l2", activation="softmax"))
    
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1=beta_1, beta_2 = beta_2,epsilon=epsilon)
    
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                      patience=10, 
                                                      min_delta=0.01,
                                                      mode='auto',
                                                      restore_best_weights=True
                                                      )    
                    
    history = model.fit(X_train, y_train ,validation_split=0.1, epochs=150, batch_size=batchsize,callbacks=[early_stopping] )
    #history = model.fit(X_train, y_train ,validation_data=(X_test,y_test), epochs=150, batch_size=batchsize,callbacks=[early_stopping] )
    test_accuracy = model.evaluate(X_test, y_test)[1]
    return test_accuracy
    #return history.history["val_accuracy"][-1]

# 3. Create a study object and optimize the objective function.
import logging
import sys

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "study_10subjects_features_new_shuffle"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(direction= 'maximize',study_name=study_name, storage=storage_name)

#study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

df = study.trials_dataframe()
df.to_csv("study_10subjects_features_shuffle.csv")
