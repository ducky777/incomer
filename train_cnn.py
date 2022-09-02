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

def check_model_dir():
    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)

def get_eq(pr, body):
    # pr = [-1 if i == 2 else i for i in pr]
    p_return = []
    for i, signal in enumerate(pr):
        returns = 0
        if signal[1] > 50:
            returns = body[i]
            if pr[i-1][1] < 0.05:
                returns -= 0.0001
        # elif signal[2] > 0.05:
        #     returns = -body[i]
        #     if pr[i-1][2] < 0.05:
        #         returns -= 0.0001
        else:
            returns = 0
    p_return.append(returns)
    return p_return

def get_data():

    lookbacks = FLAGS.lookbacks
    total_bars = FLAGS.totalbars
    forward_sharpe = FLAGS.forwardbars

    filename = 'data/%s%i.csv' %(FLAGS.symbol, FLAGS.timeframe)

    data = pd.read_csv(filename, names=['Date', 'Time',
                                                'Open', 'High',
                                                'Low', 'Close',
                                                'Volume'])

    data_info = {}

    # hightail = np.array(data.High - data.Open) / np.array(data.Open)
    # lowtail = np.array(data.Low - data.Open) / np.array(data.Open)
    # body = np.array(data.Close - data.Open) / np.array(data.Open)

    hightail = np.array(data.High - data.Open)
    lowtail = np.array(data.Low - data.Open)
    body = np.array(data.Close - data.Open)

    # hightail = np.log(data.High) - np.log(data.Open)
    # lowtail = np.log(data.Low) - np.log(data.Open)
    # body = np.log(data.Close) - np.log(data.Open)

    market_data = np.stack((hightail, lowtail, body), axis=1)
    # market_data = market_data[:-1]

    body = np.array(data.Close - data.Open)
    # body = body[lookbacks:]
    body = body[-total_bars:]
    # body = body[1:]

    df = pd.DataFrame(market_data)

    for i in range(1, lookbacks):
        df['lb_ht'+str(i)] = df[0].shift(i)
        df['lb_lt'+str(i)] = df[1].shift(i)
        df['lb_body'+str(i)] = df[2].shift(i)

    df = df.iloc[lookbacks:]
    data_copy = data.iloc[-total_bars:]
    data = np.array(df)
    data = data[-total_bars:]
    y = data[:, 2]
    x = data[:, 3:]
    # body = y[:]

    x_max = np.max(np.abs(x))
    y_max = np.max(np.abs(y))

    x = x/x_max
    y = y/y_max

    x = x.reshape((-1, lookbacks-1, 3))

    data_info['x_max'] = x_max

    valid_idx = int(0.99 * len(x))
    y_sharpe = []

    for i in range(len(body)-forward_sharpe):
        sharpe_ratio = np.mean(body[i:i+forward_sharpe]) / \
            np.std(body[i:i+forward_sharpe])
        actions = [0.] * 3
        if sharpe_ratio > 0:
            actions[1] = sharpe_ratio
            actions[2] = -sharpe_ratio
        elif sharpe_ratio < 0:
            actions[1] = sharpe_ratio
            actions[2] = -sharpe_ratio
        y_sharpe.append(actions)

    y_sharpe = y_sharpe/np.max(np.abs(y_sharpe))
    y_sharpe = (y_sharpe - np.mean(y_sharpe[:valid_idx])) / np.std(y_sharpe[:valid_idx])

    x = x[:-forward_sharpe]
    body = body[:-forward_sharpe]

    x_mean = np.mean(x[:valid_idx])
    x_std = np.std(x[:valid_idx])

    # x = (x - x_mean) / x_std

    data_info['x_mean'] = x_mean
    data_info['x_std'] = x_std

    print('x shape is: %s' % (x.shape,))
    print('y shape is: %s' % (y_sharpe.shape,))

    with open('%s/vars.json' % MODEL_PATH, 'w') as f:
        json.dump(data_info, f)

    return x, y_sharpe, body, data_info

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
    start_model = Input(shape=(FLAGS.lookbacks - 1, 3))
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
    actor = Dense(3, activation='linear')(actor)

    model = Model([start_model], [actor])
    model.compile(optimizer='adam', loss='huber')
    return model

def train(x, y, body):
    train_split = 1. - FLAGS.validsplit
    valid_idx = int(train_split * len(x))

    best_eq = 0

    print("Training start...")

    for i in range(FLAGS.num_rounds):
        k.clear_session()
        model = get_model()
        for epo in tqdm(range(FLAGS.epochs)):
            model.fit(x, y, validation_split = FLAGS.validsplit, shuffle=True,
                    batch_size=FLAGS.batchsize, epochs=1, verbose=0)
            pr = model.predict(x)
            # pr = np.argmax(pr, axis=1)
            eq = get_eq(pr, body)

            total_eq = sum(eq[valid_idx:])
            if total_eq > best_eq:
                savefile_name = '%s/%.5f.h5' % (MODEL_PATH, total_eq)
                model.save(savefile_name)
                best_eq = total_eq
                print('Best return @ time %i with epoch %i: %.5f' %
                      (i, epo, total_eq))
                plt.plot(np.cumsum(eq))
                plt.draw()
                plt.pause(0.001)
                # plt.show()
                # plt.close()
                # plt.plot(np.cumsum(eq[valid_idx:]))
                # plt.show()
                # plt.close()

def main(argv):
    print('non-flag arguments:', argv)

    plt.ion()
    plt.show()

    global MODEL_PATH

    MODEL_PATH = 'models/%s%i_%i' % \
        (FLAGS.symbol, FLAGS.timeframe, FLAGS.lookbacks)

    # main module
    check_model_dir()
    x, y, body, _ = get_data()
    train(x, y, body)

if __name__ == '__main__':
    app.run(main)