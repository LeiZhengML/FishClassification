from build import *
from load_data import *
from keras.callbacks import EarlyStopping

X_train,X_test,y_train,y_test = load_data()
vgg = build()
early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

vgg.fit(X_train, y_train, batch_size=64, nb_epoch=100,
          validation_split=0.2, verbose=1, shuffle=True, callbacks=[early_stopping])