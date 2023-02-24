import os
import sys
import pydantic_argparse
from params import TrainParams

import tensorflow as tf
# import numpy as np
# import pandas as pd
# from scipy import stats
# from sklearn import metrics
# import seaborn as sns
# from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LayerNormalization
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras import regularizers as reg

if sys.platform == "darwin":
    Adam = tf.keras.optimizers.legacy.Adam
else:
    Adam = tf.keras.optimizers.Adam


def load_existing_model(params: TrainParams):
    return load_model(params.trained_model_dir + "/" + params.model_name + ".h5") 


def define_model(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding = 'same', kernel_regularizer=reg.l2(l=0.15)))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding = 'same', kernel_regularizer=reg.l2(l=0.15)))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding = 'same', kernel_regularizer=reg.l2(l=0.15)))
    model.add(Dropout(0.4))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(n_outputs, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy', 
        optimizer=Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-9), 
        metrics=['accuracy', 'mean_absolute_error'])

    return model


def create_parser():
    """Create CLI argument parser
    Returns:
        ArgumentParser: Arg parser
    """
    return pydantic_argparse.ArgumentParser(
        model=TrainParams,
        prog="Human Activity Recognition Train Command",
        description="Train HAR model",
    )


if __name__ == "__main__":
    parser = create_parser()
    params = parser.parse_typed_args()

    if fine_tune:
        model = load_existing_model(params)
    else:
        model = define_model(n_timesteps, n_features, n_outputs)

