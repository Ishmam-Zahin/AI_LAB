from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.models import Sequential, Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

EPOCHS = 1

inputs = Input((1,))
h1 = Dense(8, activation='relu', name='h1')(inputs)
h2 = Dense(16, activation='relu', name='h2')(h1)
h3 = Dense(4, activation='relu', name='h3')(h2)
outputs = Dense(1, name='h4')(h3)

model = Model(inputs, outputs)
model.summary()

model.compile(
    loss='mse',
)

# model.fit(trainX, trainY, validation = (valX, valY), epochs=50)

def myfunc(x):
    y = 7 * x**4 + 5 * x**3 + 2 * x**2 - 7 * x + 10
    return y

n = 1000
x = np.random.randint(0, n, n)
y = myfunc(x)

train_n = n * 0.7