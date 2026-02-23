





import os
import cv2
import numpy as np
import glob
import random
from sklearn.preprocessing import LabelEncoder





x_train_dirs = glob.glob("/home/zahin/Desktop/AI_LAB/datasets/shoe/train/*/*")
x_test_dirs = glob.glob("/home/zahin/Desktop/AI_LAB/datasets/shoe/test/*/*")
random.shuffle(x_train_dirs)
random.shuffle(x_test_dirs)





y_train = [(x.split('/')[-2]) for x in x_train_dirs]
y_test = [(x.split('/')[-2]) for x in x_test_dirs]
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)





x_train = []
x_test = []
for dir in x_train_dirs:
    img = cv2.imread(dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x_train.append(img)
for dir in x_test_dirs:
    img = cv2.imread(dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x_test.append(img)





x_train = np.array(x_train, dtype='float32') / 255.0
x_test = np.array(x_test, dtype='float32') / 255.0
y_train = np.array(y_train)
y_test = np.array(y_test)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)





import tensorflow as tf
from keras.applications import VGG16
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam





base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')


base_model.trainable = False
base_model.summary(show_trainable=True)





model = Sequential([
    base_model,
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(2048, activation='relu'),
    Dropout(0.4),
    Dense(1024, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])
model.summary(show_trainable=True)





model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)





history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=32
)





import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()





test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")





indices = random.sample(range(len(x_test)), 10)

sample_images = x_test[indices]
sample_labels = y_test[indices]

pred_probs = model.predict(sample_images)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(sample_labels, axis=1) if sample_labels.ndim > 1 else sample_labels

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(sample_images[i])
    plt.axis('off')
    plt.title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}")
plt.tight_layout()
plt.show()

