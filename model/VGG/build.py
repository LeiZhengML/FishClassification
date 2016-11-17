from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import yaml
from keras import backend as K

def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)

def build():
    with open('CONFIG.yaml') as f:
        CONFIG = yaml.load(f)
    NUM_CLASSES = len(CONFIG['DATA']['FISH_CLASSES'])

    model = Sequential()
    model.add(Activation(activation=center_normalize,
                         input_shape=(CONFIG['DATA']['ROWS'], CONFIG['DATA']['COLS'], CONFIG['DATA']['CHANNELS'])))

    model.add(Convolution2D(32, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3,))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model
