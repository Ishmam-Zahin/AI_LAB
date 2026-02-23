from keras.models import Model
from keras.layers import Input, Dense


def main():
    inputs = Input(shape = (3,), name = 'input-layer')
    h1 = Dense(3, activation = 'relu', name = 'hidden 1')(inputs)
    h2 = Dense(4, activation = 'relu', name = 'hidden 2')(h1)
    outputs = Dense(2, activation = 'softmax', name = 'output')(h2)

    model = Model(inputs = inputs, outputs = outputs, name = 'FCFNN')
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(model.summary())




if __name__ == '__main__':
    main()