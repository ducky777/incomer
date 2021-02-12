#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, MaxPooling1D, \
    Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, \
        Dropout, LSTM, LeakyReLU, TimeDistributed, BatchNormalization, Activation, \
            Add, Input, AveragePooling1D

def get_eq(pr):
    return body * pr[:,0]

def train_model(model, epochs):

    best_eq = 0
    best_model = None
    for i in range(epochs):
        print('\r%s' % i, end='\r')
        model.fit(x, y, epochs=1, validation_split=0.02, batch_size=256, shuffle=True, verbose=0)
        pr = model.predict(x)
        eq = get_eq(pr)
        val_eq = sum(eq[-int(len(x)*0.02):])/np.std(eq[-int(len(x)*0.02):])
        if val_eq > best_eq:
            best_eq = val_eq
            best_model = model
            plt.plot(np.cumsum(eq[-int(len(x)*0.02):]))
            plt.show()
            plt.close()
    return best_model

lookbacks = 241
total_bars = 10000

data = pd.read_csv('EURUSD1440.csv', names=['Date', 'Time',
                                            'Open', 'High',
                                            'Low', 'Close',
                                            'Volume'])

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

body = np.array(data.Close - data.Open)
body = body[lookbacks:]
body = body[-total_bars:]

df = pd.DataFrame(market_data)

for i in range(1, lookbacks):
    df['lb_ht'+str(i)] = df[0].shift(i)
    df['lb_lt'+str(i)] = df[1].shift(i)
    df['lb_body'+str(i)] = df[2].shift(i)

df = df.iloc[lookbacks:]
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

x.shape
#%%
def res_block(x, filters, size, stride, downsample=False):
    y = Conv1D(filters, size, (1 if not downsample else 2), padding='same')(x)
    y = Activation('relu')(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.3)(y)
    y = Conv1D(filters, size, 1, padding='same')(y)

    if downsample:
        x = Conv1D(filters=filters,
                   kernel_size=1,
                   strides=2,
                   padding='same')(x)

    out = Add()([x, y])
    out = Activation('relu')(out)
    out = BatchNormalization()(out)
    return out

k.clear_session()

start_model = Input(shape=(lookbacks - 1, 3))
input_model = BatchNormalization()(start_model)
# input_model = SpatialDropout1D(0.3)(start_model)
input_model = Conv1D(kernel_size=7,
                     strides=1,
                     filters=32,
                     padding='same',
                     dilation_rate=5)(input_model)
input_model = res_block(start_model, 64, 5, 2, downsample=True)
input_model = res_block(input_model, 128, 3, 3, downsample=True)
input_model = res_block(input_model, 256, 2, 3, downsample=True)
input_model = MaxPooling1D()(input_model)
input_model = Flatten()(input_model)
# input_model = BatchNormalization()(input_model)
input_model = Dense(24)(input_model)
input_model = BatchNormalization()(input_model)
# input_model = Dropout(0.3)(input_model)
# input_model = BatchNormalization()(input_model)
# model.add(Activation('relu'))
input_model = LeakyReLU()(input_model)
outputs = Dense(1, activation='linear')(input_model)

# model = Sequential()
# model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
# model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True)))
# model.add(TimeDistributed(Dense(12)))
# model.add(LeakyReLU())
# model.add(Dropout(0.1))
# model.add(Dense(1, activation='linear'))

model = Model([start_model], [outputs])

model.compile(loss='mse', optimizer='adam')
#%%
val_split = 0.3

best_eq = 0
for i in range(1000):
    print('\r%s' % i, end='\r')
    model.fit(x, y, epochs=1, validation_split=val_split, batch_size=128,
              shuffle=True, verbose=0)
    pr = model.predict(x)
    eq = get_eq(pr)
    # eq[-int(len(x)*val_split):] = -eq[-int(len(x)*val_split):]
    val_eq = sum(eq[-int(len(x)*val_split):])/np.std(eq[-int(len(x)*val_split):])
    if val_eq > best_eq:
        best_eq = val_eq
        model.save('./best_EURUSD60.h5')
        plt.plot(np.cumsum(eq[-int(len(x)*val_split):]))
        plt.show()
        plt.close()

print("Done!")
#%%

best_model = tf.keras.models.load_model('./best_EURUSD60.h5')
pr = best_model.predict(x)

eq = get_eq(pr)

plt.plot(np.cumsum(eq))
plt.show()
plt.close()

plt.plot(np.cumsum(eq[-int(len(x)*val_split):]))
plt.show()
plt.close()
#%%
plt.plot(np.cumsum(eq))
plt.show()
plt.close()
#%%
x_max
#%%