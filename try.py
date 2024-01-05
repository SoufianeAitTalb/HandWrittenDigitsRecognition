import tkinter as tk
import numpy as np
import cv2

#pip install pillow
from PIL import ImageTk,Image,ImageDraw

from keras.models import Sequential
from keras.layers import Dense,Flatten

from tkinter import messagebox

model=Sequential()

model.add(Flatten(input_shape=(28,28)))

model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.load_weights('FFNN-MNIST.h5')

def event_function(event):