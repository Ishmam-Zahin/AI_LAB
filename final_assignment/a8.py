





import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam






def build_vgg16(input_shape=(224, 224, 3), num_classes=1000):

    model = models.Sequential([
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'),

        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'),

        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'),

        
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'),

        
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'),
        layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'),

        
        layers.Flatten(name='flatten'),
        layers.Dense(4096, activation='relu', name='fc1'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu', name='fc2'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ])

    return model





model = build_vgg16(input_shape=(224, 224, 3), num_classes=10)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()







