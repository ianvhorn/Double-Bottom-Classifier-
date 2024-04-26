# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:45:15 2024

@author: ianva
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:23:31 2024

@author: ianva
"""
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras #api for tensorflow
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution1D, Dense, MaxPool1D, Dropout, BatchNormalization
from sklearn import metrics



CWD = os.path.dirname(os.getcwd())
SAVE_PATH = os.path.join(CWD, 'DBimages')
TRU_DIR = os.path.join(CWD, r'DB_sections')
TRU_PATH = os.path.join(CWD, r'DB_sections\SPY_TRU.csv')
TRAIN_PATH = os.path.join(CWD, "TRAIN")
DATA = os.path.join(CWD,"RAW_DATA")

def csv_annotations(TRAIN_PATH = TRAIN_PATH):
    
    annotations_df = pd.read_csv(r"C:\Users\ianva\TechnicalAnalysisCNN\TRAIN\Val_set_info.csv", usecols = [1,2,3],skiprows=[0], header = None)
    data_df = pd.read_csv(os.path.join(DATA,"ALL_DATA.csv"),usecols=[2],skiprows=[0])

    x_all = []
    start_all = []
    end_all = []
    #print(data_df.head())

    for index in range(annotations_df.shape[0]):
        start,end = annotations_df.iloc[index,[0,1]] # X is the datavalue y is the truth
        x = data_df.iloc[start:end,0]
        x = (x-x.min())
        x = x/x.max()
        start_all.append(start)
        end_all.append(end)
        x_all.append(x)
        #y_all.append(y)
        
        

    x_arr = np.array(x_all)
    y = np.array(annotations_df.iloc[:,2])

    '''
    yn = []
    count = 0
    while count < y.size:
        if y[count] == 1:
            yn.append([0,1])
        else:
            yn.append([1,0])
        count+=1
    '''
    return x_arr,y,start_all,end_all
    

def evaluate_model(data, labels, model):
    x = data.reshape(len(data), 15, 1)
    y = np.array(labels).squeeze()

    

    #predictions = np.argmax(np.array(model.predict(x)), axis = 2)
   # print(predictions)
    predictions = np.round(model.predict(x))
    acc = np.mean(predictions == y)
    print("ACCURACY")
    print(acc)
    true_pos = []
    true_neg = []
    for i in range(len(y)):
        if y[i] ==1:
            val = y[i] and predictions[i]
            true_pos.append(val)
        if y[i] == 0:
            val = not(y[i] or predictions[i])
            true_neg.append(val)
            
    matrix = np.array([[np.mean(true_neg),1-np.mean(true_neg)],
                      [1-np.mean(true_pos),np.mean(true_pos)]])
                       
    precision = sum(true_pos)/(sum(predictions))
    recall = sum(true_neg)/(predictions.shape[0]-sum(predictions))
    
    print("Precision")
    print (precision)
    
    print("Recall")
    print(recall)
    
    cm_display = metrics.ConfusionMatrixDisplay(matrix, display_labels = ['Backround', 'Double Bottom'])
    
    cm_display.plot()
    plt.show()
    '''
    m = metrics.classification_report(y, predictions, digits = 2)
    cm = metrics.confusion_matrix(y, predictions, normalize = 'true')
    print(cm)
    cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels = ['Noise / Rebound', 'Single Peaks'])
    
    cm_display.plot()
    plt.show()
    '''
    #return(acc, m)


def generate_model(weights = [80,0,20], lr = .001, epochs = 300, loss ='binary_crossentropy' ): #'kullback_leibler_divergence'
    data,labels,start,end = csv_annotations()
    x_test = []
    x_train = []
    x_val = []
    y_val = []
    y_train = []
    y_test = []
    val_start = []
    val_end = []
    
    for i in range(len(data)):
        select = random.choices(list(range(len(weights))), weights = weights)
        
        if select[0] == 0:
            
            x_train.append(np.float_(data[i]))
            y_train.append(np.int_(labels[i]))
            
            
        elif select[0] == 1:
            x_test.append(np.float_(data[i]))
            y_test.append(np.int_(labels[i]))
        elif select[0] == 2:
            x_val.append(np.float_(data[i]))
            y_val.append(np.int_(labels[i]))
            val_start.append(start[i])
            val_end.append(end[i])
            
    val_info = pd.DataFrame({'start' : val_start,
                             'end' : val_end,
                             'class' : y_val})
    pd.DataFrame.to_csv(val_info,r"C:\Users\ianva\TechnicalAnalysisCNN\TRAIN\Val_set_info.csv")
            
    model = Sequential([
         Convolution1D(filters=4, kernel_size=3,  padding = 'same', name='c1d', activation = 'relu'),
         MaxPool1D(2, name = 'mp1'),
         BatchNormalization(),
         Convolution1D(filters=8, kernel_size=3,  padding = 'same', name='c1d2', activation = 'relu'),
         MaxPool1D(2, name = 'mp2'),
         BatchNormalization(),
         Convolution1D(filters=16, kernel_size=3,  padding = 'same', name='c1d3', activation = 'relu'),
         MaxPool1D(2, name = 'mp3'),
         Dropout(0.5),
         BatchNormalization(),
         Dense(1, activation='sigmoid')
    ])
    data_length = len(x_train)
    x_train = np.array(x_train).reshape(data_length, 15, 1)
    y_train = np.array(y_train).reshape(data_length, 1, 1)

    
    model.compile(optimizer = Adam(learning_rate = lr), loss = loss)
    model.fit(x_train, y_train,200, epochs = epochs)
    model.summary()
    print("FINISHED TRAIN")
    path = os.path.join(CWD,"savedModel.keras")
    model.save(path)
    evaluate_model(np.array(x_val), np.array(y_val), model)
    
#generate_model()
allldata=csv_annotations()
data = allldata[0]
labels = allldata[1]

model = keras.models.load_model(r"C:\Users\ianva\TechnicalAnalysisCNN\Models\FINAL\savedModel.keras")#load model
evaluate_model(data, labels, model)