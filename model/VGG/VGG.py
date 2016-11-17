from build import *
from load_data import *
from keras.callbacks import EarlyStopping
#import tensorflow
#from tensorflow.python.ops import control_flow_ops
#tensorflow.python.control_flow_ops = control_flow_ops
from sklearn.metrics import log_loss
from utils import *

with open('CONFIG.yaml') as f:
    CONFIG = yaml.load(f)

X_all,y_all = load_train_data()
vgg = build()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

vgg.fit(X_all,y_all, batch_size=CONFIG['VGG']['BATCH_SIZE'], nb_epoch=CONFIG['VGG']['NB_EPOCH'],
        validation_split=CONFIG['VGG']['VAL_RATIO'], verbose=1, shuffle=True, callbacks=[early_stopping])
#preds = vgg.predict(X_val, verbose=1)
#print("Validation Log Loss: {}".format(log_loss(y_val, preds)))

test = load_test_data()

test_preds = vgg.predict(test, verbose=1)
create_submission(test_preds,CONFIG=CONFIG)

