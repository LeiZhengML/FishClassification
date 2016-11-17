import os, cv2, random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#%matplotlib inline
from keras.utils import np_utils
import yaml

def get_images(fish,TRAIN_DIR='../../data/train/'):
    """Load files from train folder"""
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir)]
    return images

def read_image(src,ROWS,COLS):
    """Read and resize individual images"""
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
    return im

def load_data(TRAIN_DIR='../../data/train/'):
    with open('CONFIG.yaml') as f:
        CONFIG = yaml.load(f)
    files = []
    y_all = []

    for fish in CONFIG['DATA']['FISH_CLASSES']:
        fish_files = get_images(fish)
        files.extend(fish_files)

        y_fish = np.tile(fish, len(fish_files))
        y_all.extend(y_fish)
        print("{0} photos of {1}".format(len(fish_files), fish))

    y_all = np.array(y_all)

    X_all = np.ndarray((len(files), CONFIG['DATA']['ROWS'], CONFIG['DATA']['COLS'], CONFIG['DATA']['CHANNELS']), dtype=np.uint8)

    for i, im in enumerate(files):
        X_all[i] = read_image(TRAIN_DIR+im,CONFIG['DATA']['ROWS'],CONFIG['DATA']['COLS'])
        if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))

    print(X_all.shape)

    # One Hot Encoding Labels
    y_all = LabelEncoder().fit_transform(y_all)
    y_all = np_utils.to_categorical(y_all)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,
                                                          test_size=0.2, random_state=23,
                                                          stratify=y_all)
    return X_train,X_test,y_train,y_test