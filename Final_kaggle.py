# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 22:29:16 2019

@author: Hemanath
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.model_selection import train_test_split

data1=pd.read_csv('train.csv')
y=np.array(data1['0'])
xt=np.array(data1)
xt=np.array(data1.drop('0',axis=1))
xt=xt.reshape(28708,48,48,1)
w=xt[595]
w.shape
plt.figure(figsize=(5,5))
plt.imshow(w)
plt.colorbar()
plt.show()

labels = np.array(y)
lb=LabelBinarizer()
labels = lb.fit_transform(labels)
xtrain,xtest,ytrain,ytest = train_test_split(xt,y,test_size=0.2,random_state=42)
xtrain.shape

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(xtrain)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)