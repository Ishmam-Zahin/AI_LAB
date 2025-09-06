import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score


data = np.load("my_dataset.npz")
X_train = data['x_train']
X_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']
# print(X_train.shape)
# print(X_test.shape)

X_train = X_train.astype("float32")/255.0
X_test  = X_test.astype("float32")/255.0

model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(10, activation='softmax'),
])

model.compile(
    optimizer = Adam(),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

output = model.fit(
    X_train, y_train,
    epochs = 50,
    batch_size = 64,
    validation_split = 0.1
)

# y_prob = model.predict(X_test)
# y_pred = y_prob.argmax(axis=1)

# acc = accuracy_score(y_test, y_pred)

# print("Test Accuracy:", float(acc))

print("Evaluation on test set:")
model.evaluate(X_test, y_test)

plt.figure()
plt.plot(output.history['loss'], label='train_loss')
plt.plot(output.history['val_loss'], label='val_loss')
plt.title('Loss Curves')
plt.legend()
plt.show()