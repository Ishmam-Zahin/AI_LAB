import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam
import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(filters = 8, kernel_size = (3, 3), activation = 'relu'),
    Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu'),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax'),
])

model.summary(show_trainable = True)

model.compile(
    optimizer = Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train, y_train_cat,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    verbose=2
)

test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# plt.figure(figsize=(16, 9))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Acc')
# plt.plot(history.history['val_accuracy'], label='Val Acc')
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Training vs Validation Accuracy")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training vs Validation Loss")
# plt.legend()

# plt.tight_layout()
# # plt.savefig("fcnn_mnist.png")
# plt.show()
