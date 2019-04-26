# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:41:50 2019

@author: ADWANI
"""

import random
import keras
from keras import optimizers
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import os
import cv2 
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import Sequential,load_model
from keras.layers import Lambda, concatenate
from keras import Model
from keras import callbacks
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import model_from_json
import gc
import glob
import cv2
import os
import numpy as np

#======End Library=========

directory='D:\justRandomCodes\Secure Computing Proj\crop_part1\*.jpg'
path='D:\justRandomCodes\Secure Computing Proj\crop_part1\\'

m=[]
f=[]
ml=[]
fl=[]
for filename in os.listdir(path):
    if(filename[3]=='0'):
        x=cv2.imread(path+filename)
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        res = cv2.resize(gray, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
        m.append(res)
        ml.append(0)
    else:
        x=cv2.imread(path+filename)
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        res = cv2.resize(gray, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
        f.append(res)
        fl.append(1)

Images=m+f
labels=ml+fl
ImagesA=np.asarray(Images)
ImagesA=ImagesA.reshape((len(ImagesA),100,100,1))
LabelsA=np.asarray(labels)

x_train,x_test,y_train,y_test=train_test_split(ImagesA, LabelsA, test_size=0.2, random_state=75)
#collection=skimage.io.imread_collection(directory)


#
filename='model_gender.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
callbacks_list = [csv_log]

#====End Variables=====

#=====Model Start=========

input_shape=((100,100,1))
num_classes=2


model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=input_shape))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))


#=====Model End=========
   
 


model.compile(optimizer =Adam(lr = 1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
hist=model.fit(x_train, y_train, epochs = 10, validation_data=(x_test, y_test),callbacks=callbacks_list)


model_json = model.to_json()
with open("model_gender.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_gender.h5")
print("Saved model to disk")

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

   
epochs = 10
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
