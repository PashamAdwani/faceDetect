# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:38:13 2019

@author: ADWANI
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:51:25 2019

@author: ADWANI
"""

import numpy as np
import cv2 
from keras.models import model_from_json
import keras 
import matplotlib as plt
from skimage import color
from skimage import io
import tkinter as tk
from tkinter import *

json_file = open(r'D:\justRandomCodes\SecureComputingProj\model_gender.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("D:\justRandomCodes\SecureComputingProj\model_gender.h5")
print("Loaded model from disk")
face_cascade = cv2.CascadeClassifier(r'D:\justRandomCodes\Eye Blinking\opencv-master\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'\D:\justRandomCodes\Eye Blinking\opencv-master\opencv-master\data\haarcascades\haarcascade_eye.xml')

def CaptureImage():
    # initialize the camera
    cam = cv2.VideoCapture(0)   # 0 -> index of camera
    s, img = cam.read()
    return img

count=0
r = tk.Tk() 
r.title('Women Safety App') 
while True:
    x=CaptureImage()
    faces = face_cascade.detectMultiScale(x, 1.3, 5)
    gray = color.rgb2gray(x)
    if(len(faces)!=0):
        for (x,y,w,h) in faces:
            roi=gray[x:x+w,y:y+h]
            print('Face Detected')
    img=cv2.resize(roi, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
    img=img.reshape((1,100,100,1))
    y=loaded_model.predict(img) 
    if(np.argmax(y, axis=None, out=None)==0):
        print('male')
        gender='male'
        count=count+1
        if(count==10):
            ourMessage ='Male face detected, can not login'
            messageVar = Message(r, text = ourMessage) 
            messageVar.config(bg='lightgreen') 
            messageVar.pack()
            break
    elif(np.argmax(y, axis=None, out=None)==1):
        print('female')
        gender='female'
        count=count+1
        if(count==10):
            ourMessage ='Female face detected, Welcome'
            messageVar = Message(r, text = ourMessage) 
            messageVar.config(bg='lightgreen') 
            messageVar.pack( ) 
            button = tk.Button(r, text='Login', width=25, command=r.destroy) 
            button.pack()
            break

r.mainloop()
#if(gender=='male'):    
#    ourMessage ='Male face detected, can not login'
#    messageVar = Message(r, text = ourMessage) 
#    messageVar.config(bg='lightgreen') 
#    messageVar.pack( ) 
#    r.mainloop()
#elif(gender=='female'):
#    ourMessage ='Female face detected, Welcome'
#    messageVar = Message(r, text = ourMessage) 
#    messageVar.config(bg='lightgreen') 
#    messageVar.pack( ) 
#    button = tk.Button(r, text='Login', width=25, command=r.destroy) 
#    button.pack()     