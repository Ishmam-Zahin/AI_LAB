import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(6, activation='relu', input_shape=(3,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()



# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
# from keras.models import Sequential
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from keras.datasets import mnist
# import cv2
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import numpy as np

# EPOCHS = 1
# IMG_WIDTH = 28
# IMG_HEIGHT = 28

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(x_train.shape[0], IMG_WIDTH, IMG_HEIGHT, 1).astype('float32') / 255
# x_test = x_test.reshape(x_test.shape[0], IMG_WIDTH, IMG_HEIGHT, 1).astype('float32') / 255
# # print(x_train.shape, x_test.shape)


# # print(y_train.shape, y_test.shape)
# # y_train = to_categorical(y_train, num_classes=10)
# # y_test = to_categorical(y_test, num_classes=10)
# # print(y_train.shape, y_test.shape)

# # x_train = cv2.resize(x_train, (IMG_WIDTH, IMG_HEIGHT))
# # x_test = cv2.resize(x_test, (IMG_WIDTH, IMG_HEIGHT))
# # print(x_train[0].shape)

# model = Sequential([
#     Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
#     Conv2D(32, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, kernel_size=(3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax')
# ])

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.fit(x_train, y_train, epochs=EPOCHS)
# model.evaluate(x_test, y_test, verbose=2)
# # print(x_test[5].shape)
# tmp = model.predict(x_test[41:42])
# print(np.argmax(tmp, axis=1))

# plt.imshow(x_test[41].reshape(IMG_WIDTH, IMG_HEIGHT), cmap='gray')
# plt.show()