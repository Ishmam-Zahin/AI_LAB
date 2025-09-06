import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

x = np.linspace(-10, 10, 1000000)
np.random.shuffle(x)
y = 5 * x**2 + 10 * x - 2

x_train = x[:700000]
y_train = y[:700000]
x_eval = x[70000:800000]
y_eval = y[70000:800000]
x_test = x[800000:]
y_test = y[800000:]


model = Sequential([
    Input(shape=(1,)),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1),
])

model.compile(optimizer=Adam(), loss='mse')

model.fit(x_train, y_train, validation_data=(x_eval, y_eval), epochs=10, batch_size=32)
model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

plt.plot(x_test, y_test, label='Original test f(x)', color='blue')
plt.plot(x_test, y_pred, label='Predicted test f(x)', color='red')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Comparison of Original and Predicted f(x)')
# plt.savefig('plot.png')
plt.show()
