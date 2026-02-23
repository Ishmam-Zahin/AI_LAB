#!/usr/bin/env python
# coding: utf-8

from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

def drawLossGraph(equation_names = [], histories = [], rows = 2, cols = 2, figsize = (16, 9)):
    if len(equation_names) != len(histories):
        raise Exception('size do not match')
    plt.figure(figsize=figsize)
    for index, history in enumerate(histories):
        plt.subplot(rows, cols, index + 1)
        plt.plot(history['loss'][1:], label = 'train loss')
        plt.plot(history['val_loss'][1:], label = 'validation loss', linestyle = '--')
        plt.title(equation_names[index])
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
    plt.suptitle('Training Loss VS Validation Loss')
    plt.tight_layout()
    plt.show()

def drawTestGraph(equation_names = [], models = [], x_test = None, y_tests  = [], rows = 2, cols = 2, figsize = (16, 9)):
    if len(equation_names) != len(models):
        raise Exception('size do not match')
    plt.figure(figsize=figsize)
    for index, model in enumerate(models):
        y_test = y_tests[index]
        y_test_predict = model.predict(x_test)
        plt.subplot(rows, cols, index + 1)
        plt.scatter(x_test, y_test, label = 'actual', marker = 'o')
        plt.scatter(x_test, y_test_predict, label = 'predicted', marker = 'x')
        plt.title(equation_names[index])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
    plt.suptitle('Actual VS Predicted')
    plt.tight_layout()
    plt.show()

def eq1(x):
    y = 5*x + 10
    return y

def eq2(x):
    y = 3*x**2 + 5*x + 10
    return y

def eq3(x):
    y = 4*x**3 + 3*x**2 + 5*x + 10
    return y

equation_names = ['5x + 10', '3x^2 + 5x + 10', '4x^3 + 3x^2 + 5x + 10']
dataset_size = 100000
train_split = int(0.8 * dataset_size)
low = -100
high = 100

x = np.linspace(low, high, dataset_size)
random.shuffle(x)
x_train = x[:train_split]
x_test = x[train_split:]
x_train = np.array(x_train)
x_test = np.array(x_test)
x_mean = x_train.mean()
x_std = x_train.std()
x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std
print(x_train.shape, x_test.shape)

y_train1 = eq1(x_train)
y_test1 = eq1(x_test)
y_mean1 = y_train1.mean()
y_std1 = y_train1.std()
y_train1 = (y_train1 - y_mean1) / y_std1
y_test1 = (y_test1 - y_mean1) / y_std1
print(y_train1.shape, y_test1.shape)

y_train2 = eq2(x_train)
y_test2 = eq2(x_test)
y_mean2 = y_train2.mean()
y_std2 = y_train2.std()
y_train2 = (y_train2 - y_mean2) / y_std2
y_test2 = (y_test2 - y_mean2) / y_std2
print(y_train2.shape, y_test2.shape)

y_train3 = eq3(x_train)
y_test3 = eq3(x_test)
y_mean3 = y_train3.mean()
y_std3 = y_train3.std()
y_train3 = (y_train3 - y_mean3) / y_std3
y_test3 = (y_test3 - y_mean3) / y_std3
print(y_train3.shape, y_test3.shape)

model1 = Sequential([
    layers.Input(shape = (1,)),
    layers.Dense(2, activation='relu'),
    layers.Dense(1, activation=None),
])
model1.summary(show_trainable=True)

model2 = Sequential([
    layers.Input(shape = (1,)),
    layers.Dense(4, activation='relu'),
    layers.Dense(2, activation='relu'),
    layers.Dense(1, activation=None),
])
model2.summary(show_trainable=True)

model3 = Sequential([
    layers.Input(shape = (1,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='relu'),
    layers.Dense(2, activation='relu'),
    layers.Dense(1, activation=None),
])
model3.summary(show_trainable=True)

model1.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='mse',
)

model2.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='mse',
)

model3.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='mse',
)

history1 = model1.fit(x_train, y_train1, epochs=20, batch_size=32, validation_split=.1)
model1.evaluate(x_test, y_test1)

history2 = model2.fit(x_train, y_train2, epochs=20, batch_size=32, validation_split=.1)
model2.evaluate(x_test, y_test2)

history3 = model3.fit(x_train, y_train3, epochs=20, batch_size=32, validation_split=.1)
model3.evaluate(x_test, y_test3)

drawLossGraph(equation_names=equation_names, histories=[history1.history, history2.history, history3.history])
drawTestGraph(equation_names=equation_names, models = [model1, model2, model3], x_test=x_test[:50], y_tests=[y_test1[:50], y_test2[:50], y_test3[:50]])