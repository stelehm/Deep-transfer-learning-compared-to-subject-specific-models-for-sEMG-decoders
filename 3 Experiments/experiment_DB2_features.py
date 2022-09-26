

# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:24:27 2021

@author: StephanLehmler
"""

#import NinaPro_Utility
import numpy as np
import random
import tensorflow as tf
import os


def get_model():
    #from optuna:
    #     batchnorm = True
    #     batchsize = 256 
    #     beta_1 = 0.4
    #     beta_2 = 0.1 - 0.13
    #     epsilon = 0.36 - 0.41
    #     learning_rate = 0.0004- 0.0015

    # dense_dropout = 0.08-0.1
    # input_dropout = 0.36-0.41

    # hidden_0 = 180 - 320
    # hidden_1 = 40 - 72
    # hidden_2 = 91 -379
    # hidden_3 = 407 -511
    # hidden_4 = 86 - 145
    # hidden_5 = 65- 182
    # hidden_6 = nan - 364-509
    num_hidden_input = 300
    nb_dense_layers = [200,100,50]
    #[250, 50,200,450, 150,75]
    input_dropout_factor = 0.4
    #dense_dropout_factor = 0.08
    
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(18,12)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(num_hidden_input, activation='relu', kernel_regularizer="l2"))
    
    model.add(tf.keras.layers.Dropout(input_dropout_factor))
    
    for num_hidden in nb_dense_layers:
        model.add(tf.keras.layers.Dense(num_hidden, activation='relu', kernel_regularizer="l2"))
        model.add(tf.keras.layers.BatchNormalization())
        #model.add(tf.keras.layers.Dropout(dense_dropout_factor))
    model.add(tf.keras.layers.Dense(17, kernel_regularizer="l2", activation="softmax"))
    
    return model


def get_subject_preproc(subject,samplesize="83Perc"):
    foldername = "../0 Data/DB2/DB2_preproc_Features/{0}/{1}/".format(str(subject),samplesize)
    X_train = np.load(foldername+"_X_train.npy")
    X_test = np.load(foldername+"_X_test.npy")
    y_train = np.load(foldername+"_y_train.npy")
    y_test = np.load(foldername+"_y_test.npy")
    return X_train, y_train, X_test, y_test


with open("results_DB2_features_small.csv","w") as wf:
     wf.write("subject,train_epochs,retraining_epochs,retraining_samples,retraining_method,train_loss,val_loss,train_accuracy,val_accuracy,test_accuracy,pre_retraining_train_accuracy,pre_retraining_test_accuracy,new_train_loss,new_val_loss,new_train_accuracy,new_val_accuracy,new_test_accuracy,r_train_loss,r_val_loss,r_train_accuracy,r_val_accuracy,r_test_accuracy")
     wf.write("\n")
     
#load 83Perc data
all_83Perc_X_train = []
all_83Perc_y_train= []
all_83Perc_X_test = []
all_83Perc_y_test = []
for i in range(1,41):
    X_tr, y_tr, X_te, y_te = get_subject_preproc(i)
    all_83Perc_X_train.append(X_tr)
    all_83Perc_y_train.append(y_tr)
    all_83Perc_X_test.append(X_te)
    all_83Perc_y_test.append(y_te)

#! forgot last subject    
for subject in range(11,41):
    print(subject)
    #build model
    pretrained_model = get_model()
    learning_rate = 0.001
    beta_1 = 0.4
    beta_2 = 0.1
    epsilon = 0.38
    batchsize = 256
    #opt = tf.keras.optimizers.Adam(learning_rate = learning_rate, beta_1=beta_1, beta_2 = beta_2,epsilon=epsilon)
    opt = tf.keras.optimizers.Adam()

    pretrained_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    
    #remove subject from _83Perc_
    X_train = all_83Perc_X_train[:i] +all_83Perc_X_train[i+1:]
    y_train = all_83Perc_y_train[:i] +all_83Perc_y_train[i+1:]
    X_test = all_83Perc_X_test[:i] +all_83Perc_X_test[i+1:]
    y_test = all_83Perc_y_test[:i] +all_83Perc_y_test[i+1:]
    
    #flatten
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    #shuffle
    random_ids_train = list(range(y_train.shape[0]))
    random.shuffle(random_ids_train)
    X_train= X_train[random_ids_train]
    y_train= y_train[random_ids_train]
    
    # #!
    # from sklearn.neural_network import MLPClassifier

    # clf = MLPClassifier(hidden_layer_sizes=(200,200,200,200,200),random_state=1, max_iter=300)
    # X = np.reshape(X_train, (X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
    # y = np.argwhere(y_train>0)[:,1]
    # clf.fit(X,y)
    # print("MLP-train on new features:" + str(clf.score(X,y)))
    # X_t = np.reshape(X_test, (X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
    # y_t = np.argwhere(y_test>0)[:,1]
    # print("MLP-test on new features:" + str(clf.score(X_t,y_t)))
    # #!
    
    # print(bob)
        
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                      patience=10, 
                                                      min_delta=0.001,
                                                      mode='auto',
                                                      restore_best_weights=True
                                                      )
    #csv_logger = tf.keras.callbacks.CSVLogger("model{}_history_log.csv".format(str(subject)), append=True)
    
    history = pretrained_model.fit(X_train, y_train, epochs=150 ,validation_split=0.1,batch_size=batchsize, callbacks=[early_stopping])
    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]
    train_accuracy = history.history["accuracy"][-1]
    val_accuracy = history.history["val_accuracy"][-1]

    
    nb_train_epochs = len(history.history['loss'])
    
    
    #test_accuracy = mean(test_accuracy)
    test_accuracy = pretrained_model.evaluate(X_test, y_test)[1]
    
    #free memory
    #X_train = []
    #X_test= []
    #y_train= []
    #y_test= []
    
    

    #save model
    foldername = "./models/DB2_features_small/model{}/".format(str(subject))
    os.mkdir(foldername) 
    pretrained_model.save_weights(foldername+"model_{}".format(str(subject)))
    
    #iterate amount of samples
    for retraining_samples in ["83Perc","60Perc", "30Perc","16Perc","fiveMovements","oneMovement"]:
        X_re_train, y_re_train, X_re_test, y_re_test = get_subject_preproc(i,retraining_samples)
        
        #shuffle
        random_ids_retrain = list(range(y_re_train.shape[0]))
        random.shuffle(random_ids_retrain)
        X_re_train= X_re_train[random_ids_retrain]
        y_re_train= y_re_train[random_ids_retrain]
        
        X = X_re_train
        y = y_re_train
        #if I want to use previous data while retraining
        # X = np.concatenate([X_re_train,X_train])
        # y = np.concatenate([y_re_train,y_train])
        random_ids = list(range(y.shape[0]))
        #if i want to shuffle
        random.shuffle(random_ids)
        X = X[random_ids]
        y = y[random_ids]
        
        
        pre_retraining_train_accuracy = pretrained_model.evaluate(X_re_train, y_re_train)[1]
        pre_retraining_test_accuracy = pretrained_model.evaluate(X_re_test, y_re_test)[1]
        
        #model without pretraining
        #get empy model
        new_model = get_model()
        new_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
        csv_logger_new = tf.keras.callbacks.CSVLogger("model{}_{}_newModel_history_log.csv".format(str(subject),retraining_samples), append=True)
        #train
        new_history = new_model.fit(X,y,epochs=150,validation_split=0.1,batch_size=batchsize,callbacks=[csv_logger_new,early_stopping])
        new_epochs = len(new_history.history['loss'])
        
        new_train_loss = new_history.history["loss"][-1]
        new_val_loss = new_history.history["val_loss"][-1]
        new_train_accuracy = new_history.history["accuracy"][-1]
        new_val_accuracy = new_history.history["val_accuracy"][-1]
        new_test_accuracy = new_model.evaluate(X_re_test, y_re_test)[1]
    
        for retraining_method in ["all","first", "last","first_last","new"]:    
            #copy model
            #retrain_model = tf.keras.models.clone_model(pretrained_model)
            retrain_model = get_model()
            retrain_model.load_weights(foldername+"model_{}".format(str(subject)))
            retrain_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
            #set trainability
            if retraining_method == "first":
                for layer in retrain_model.layers[1:]:
                    layer.trainable = False 
            elif retraining_method == "last":
                for layer in retrain_model.layers[:-1]:
                    layer.trainable = False 
            elif retraining_method == "first_last":
                for layer in retrain_model.layers[1:-1]:
                    layer.trainable = False 
            elif retraining_method =="new":
                for layer in retrain_model.layers:
                    layer.trainable = False
                    retrain_model.add(tf.keras.layers.Dense(17, kernel_regularizer="l2", activation="softmax"))
            elif retraining_method == "all":
                print("all")
            #retrain model
            #retrain_history = retrain_model.fit(X_re_train, y_re_train ,validation_split=0.2, epochs=retraining_epochs, batch_size=32)#batchsize)
            
            csv_logger_retrain = tf.keras.callbacks.CSVLogger("model{}_{}_{}_retrain_history_log.csv".format(str(subject),retraining_samples,retraining_method), append=True)
            retrain_history = retrain_model.fit(X,y,epochs=150,validation_split=0.1,batch_size=batchsize,callbacks=[csv_logger_retrain,early_stopping])
           
            retraining_epochs = len(retrain_history.history['loss'])
            
            r_train_loss = retrain_history.history["loss"][-1]
            r_val_loss = retrain_history.history["val_loss"][-1]
            r_train_accuracy = retrain_history.history["accuracy"][-1]
            r_val_accuracy = retrain_history.history["val_accuracy"][-1]
            r_test_accuracy = retrain_model.evaluate(X_re_test, y_re_test)[1]
            
            
            #write results
            variable_list = [subject,nb_train_epochs,retraining_epochs,retraining_samples,retraining_method,
                             train_loss, val_loss, train_accuracy,val_accuracy,test_accuracy,
                             pre_retraining_train_accuracy, pre_retraining_test_accuracy,
                             new_train_loss, new_val_loss, new_train_accuracy, new_val_accuracy, new_test_accuracy,
                             r_train_loss, r_val_loss, r_train_accuracy, r_val_accuracy, r_test_accuracy]
            with open("results_DB2_features_small.csv","a") as wf:
                wf.write(', '.join([str(measure) for measure in variable_list ]))
                wf.write("\n")
