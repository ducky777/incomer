import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, MaxPooling1D, \
    Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, \
        Dropout, LSTM, LeakyReLU, TimeDistributed, BatchNormalization, Activation, \
            Add, Input, AveragePooling1D, Concatenate

from absl import app
from absl import flags
from tqdm import tqdm

import pickle
import json

FLAGS = flags.FLAGS

flags.DEFINE_string('symbol', None, '(str) Symbol being trained on')
flags.DEFINE_integer('timeframe', None, '(int) Timeframe of data')
flags.DEFINE_integer('lookbacks', 241, '(int) Number of bars to lookback')
flags.DEFINE_integer('totalbars', 10000, '(int) Number of training data')
flags.DEFINE_integer('forwardbars', 12, '(int) Number of bars to look forward')
flags.DEFINE_integer('num_rounds', 2000, '(int) Number of rounds to train a new model')
flags.DEFINE_integer('epochs', 50, '(int) Epochs per round')
flags.DEFINE_integer('batchsize', 256, '(int) Batch size to train')
flags.DEFINE_float('validsplit', 0.01, '(float) Validation split')

# flags.mark_flag_as_required("lookbacks")
# flags.mark_flag_as_required("totalbars")
# flags.mark_flag_as_required("forwardbars")
flags.mark_flag_as_required("symbol")
flags.mark_flag_as_required("timeframe")

MODEL_PATH = ''

def res_block(x, filters, size, stride, downsample=False):
    y = Conv1D(filters, size, (1 if not downsample else 2), padding='same')(x)
    y = Activation('relu')(y)
    y = Conv1D(filters, size, 1, padding='same')(y)

    if downsample:
        x = Conv1D(filters=filters,
                   kernel_size=1,
                   strides=2,
                   padding='same')(x)

    out = Add()([x, y])
    out = Activation('relu')(out)
    return out

def get_model():
    start_model = Input(shape=(lookbacks - 1, 3))
    input_model = Conv1D(kernel_size=24,
                        strides=3,
                        filters=16,
                        padding='same')(start_model)
    input_model = res_block(start_model, 32, 12, 3, downsample=True)
    input_model = res_block(input_model, 64, 6, 3, downsample=True)
    input_model = res_block(input_model, 128, 3, 3, downsample=True)
    input_model = MaxPooling1D()(input_model)
    input_model = Flatten()(input_model)

    # actor layers
    actor = Dense(32)(input_model)
    actor = LeakyReLU()(actor)
    actor = Dense(num_forward_bars*2, activation='sigmoid')(actor)

    model = Model([start_model], [actor])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    return model

def get_eq(op):


if __name__ == '__main__':
    filename = 'data/US5001440.csv'

    data = pd.read_csv(filename, names=['Date', 'Time',
                                                'Open', 'High',
                                                'Low', 'Close',
                                                'Volume'])

    lookbacks = 600
    total_bars = 3000

    hightail = np.array((data.High - data.Open) / data.Open)
    lowtail = np.array((data.Low - data.Open) / data.Open)
    body = np.array((data.Close - data.Open) / data.Open)

    market_data = np.stack((hightail, lowtail, body), axis=1)

    df = pd.DataFrame(market_data)

    for i in range(1, lookbacks):
        df['lb_ht'+str(i)] = df[0].shift(i)
        df['lb_lt'+str(i)] = df[1].shift(i)
        df['lb_body'+str(i)] = df[2].shift(i)

    df = df.iloc[lookbacks:]
    dt = np.array(df)
    dt = dt[-total_bars:]
    x = dt[:, 3:]
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = x.reshape(-1, lookbacks-1, 3)

    num_forward_bars = 5

    op = np.array(data.Open)
    y = []
    for i, _ in enumerate(op[:-num_forward_bars]):
        rn = op[i:i+num_forward_bars]
        highest = np.argmax(rn)
        lowest = np.argmin(rn)
        yt = [0] * num_forward_bars*2
        yt[lowest] = 1
        yt[num_forward_bars + highest] = 1
        y.append(yt)
    y = np.array(y)
    y = y[lookbacks:]
    x = x[num_forward_bars:]
    op = op[lookbacks:]

    signals = y[0].reshape(2, num_forward_bars)
    signals


    k.clear_session()
    model = get_model()
    model.fit(x, y, epochs=100, validation_split=0.2)