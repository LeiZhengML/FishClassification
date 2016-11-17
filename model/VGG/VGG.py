from build import *
from load_data import *
from keras.callbacks import EarlyStopping
import tensorflow
from tensorflow.python.ops import control_flow_ops
tensorflow.python.control_flow_ops = control_flow_ops
from sklearn.metrics import log_loss
from utils import *

with open('CONFIG.yaml') as f:
    CONFIG = yaml.load(f)

X_train,X_val,y_train,y_val = load_train_data()
vgg = build()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

vgg.fit(X_train, y_train, batch_size=64, nb_epoch=100,
        validation_data=(X_val,y_val), verbose=1, shuffle=True, callbacks=[early_stopping])


preds = vgg.predict(X_val, verbose=1)
print("Validation Log Loss: {}".format(log_loss(y_val, preds)))

test = load_test_data()
test_preds = vgg.predict(test, verbose=1)
create_submission(test_preds,CONFIG=CONFIG)

