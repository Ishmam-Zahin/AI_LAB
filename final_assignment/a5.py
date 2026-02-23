






from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random


(x_train_1, y_train_1), (x_test_1, y_test_1) = tf.keras.datasets.fashion_mnist.load_data()
(x_train_2, y_train_2), (x_test_2, y_test_2) = tf.keras.datasets.mnist.load_data()
(x_train_3, y_train_3), (x_test_3, y_test_3) = tf.keras.datasets.cifar10.load_data()

num_classes = 10

x_train_1 = x_train_1[..., np.newaxis]
x_test_1  = x_test_1[..., np.newaxis]
x_train_2 = x_train_2[..., np.newaxis]
x_test_2  = x_test_2[..., np.newaxis]



x_train_1 = x_train_1.astype('float32') / 255.0
x_test_1  = x_test_1.astype('float32')  / 255.0
x_train_2 = x_train_2.astype('float32') / 255.0
x_test_2  = x_test_2.astype('float32')  / 255.0
x_train_3 = x_train_3.astype('float32') / 255.0
x_test_3  = x_test_3.astype('float32')  / 255.0


y_train_1 = tf.keras.utils.to_categorical(y_train_1, num_classes)
y_test_1  = tf.keras.utils.to_categorical(y_test_1, num_classes)
y_train_2 = tf.keras.utils.to_categorical(y_train_2, num_classes)
y_test_2  = tf.keras.utils.to_categorical(y_test_2, num_classes)
y_train_3 = y_train_3.reshape(-1) if y_train_3.ndim > 1 else y_train_3
y_test_3  = y_test_3.reshape(-1)  if y_test_3.ndim > 1 else y_test_3
y_train_3 = tf.keras.utils.to_categorical(y_train_3, num_classes)
y_test_3  = tf.keras.utils.to_categorical(y_test_3, num_classes)


print('fashion mnist | train: ', x_train_1.shape, y_train_1.shape, ' | test: ', x_test_1.shape, y_test_1.shape)
print('digit mnist | train: ', x_train_2.shape, y_train_2.shape, ' | test: ', x_test_2.shape, y_test_2.shape)
print('cipher 10 | train: ', x_train_3.shape, y_train_3.shape, ' | test: ', x_test_3.shape, y_test_3.shape)


fashion_labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
                  'Sandal','Shirt','Sneaker','Bag','Ankle boot']
mnist_labels = [str(i) for i in range(10)]
cifar_labels = ['airplane','automobile','bird','cat','deer',
                'dog','frog','horse','ship','truck']






fig, axes = plt.subplots(6, 5, figsize=(12, 8))
fig.suptitle("Fashion MNIST, MNIST & CIFAR-10 Samples", fontsize=16)
datasets = [
    ("Fashion MNIST - Train", x_train_1, y_train_1, fashion_labels, 'gray'),
    ("Fashion MNIST - Test", x_test_1, y_test_1, fashion_labels, 'gray'),
    ("MNIST - Train", x_train_2, y_train_2, mnist_labels, 'gray'),
    ("MNIST - Test", x_test_2, y_test_2, mnist_labels, 'gray'),
    ("CIFAR-10 - Train", x_train_3, y_train_3, cifar_labels, None),
    ("CIFAR-10 - Test", x_test_3, y_test_3, cifar_labels, None),
]
for row, (title, x_data, y_data, labels, cmap) in enumerate(datasets):
    for col in range(5):
        ax = axes[row, col]
        ax.axis("off")
        img = x_data[col]
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        if y_data.ndim > 1 and getattr(y_data, 'shape', None) and y_data.shape[1] > 1:
            label_idx = np.argmax(y_data[col])
        else:
            label_idx = y_data[col][0] if y_data.ndim > 1 else y_data[col]
        ax.set_title(labels[int(label_idx)], fontsize=8)
    axes[row, 0].set_ylabel(title, fontsize=10, rotation=0, labelpad=50, va='center')
plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.5)
plt.show()






model1 = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
model1.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()






history1 = model1.fit(
    x_train_1, y_train_1,
    validation_data=(x_test_1, y_test_1),
    epochs=10,
    batch_size=32,
    verbose=1
)






model2 = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
model2.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model2.summary()






history2 = model2.fit(
    x_train_2, y_train_2,
    validation_data=(x_test_2, y_test_2),
    epochs=10,
    batch_size=32,
    verbose=1
)






model3 = Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
model3.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model3.summary()






history3 = model3.fit(
    x_train_3, y_train_3,
    validation_data=(x_test_3, y_test_3),
    epochs=10,
    batch_size=32,
    verbose=1
)






def plot_all_histories(histories, titles):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for i, (history, title) in enumerate(zip(histories, titles)):
        axes[i, 0].plot(history.history['accuracy'], label='Train Acc')
        axes[i, 0].plot(history.history['val_accuracy'], label='Val Acc')
        axes[i, 0].set_title(f"{title} Accuracy")
        axes[i, 0].set_xlabel("Epoch")
        axes[i, 0].set_ylabel("Accuracy")
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        axes[i, 1].plot(history.history['loss'], label='Train Loss')
        axes[i, 1].plot(history.history['val_loss'], label='Val Loss')
        axes[i, 1].set_title(f"{title} Loss")
        axes[i, 1].set_xlabel("Epoch")
        axes[i, 1].set_ylabel("Loss")
        axes[i, 1].legend()
        axes[i, 1].grid(True)
    plt.tight_layout()
    plt.show()


plot_all_histories([history1, history2, history3], ["Fashion MNIST", "MNIST", "CIFAR-10"])






def plot_random_predictions(models, x_tests, y_tests, label_lists, titles):
    import random
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for row, (model, x_test, y_test, labels, title) in enumerate(zip(models, x_tests, y_tests, label_lists, titles)):
        idxs = random.sample(range(x_test.shape[0]), 5)
        preds = model.predict(x_test[idxs], verbose=0)
        for col, (i, pred) in enumerate(zip(idxs, preds)):
            ax = axes[row, col]
            actual_idx = np.argmax(y_test[i])
            pred_idx = np.argmax(pred)
            confidence = np.max(pred)
            img = x_test[i]
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"{title}\nA: {labels[actual_idx]}\nP: {labels[pred_idx]}\nC: {confidence:.2f}", fontsize=9)
    plt.tight_layout()
    plt.show()


plot_random_predictions(
    [model1, model2, model3],
    [x_test_1, x_test_2, x_test_3],
    [y_test_1, y_test_2, y_test_3],
    [fashion_labels, mnist_labels, cifar_labels],
    ["Fashion MNIST", "MNIST", "CIFAR-10"]
)







