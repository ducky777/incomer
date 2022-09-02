#%%
import pandas as pd
import numpy as np
import tensorflow_addons as tfa

from datetime import datetime as dt

import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Dense, Flatten, MaxPooling1D, \
    Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, \
        Dropout, LSTM, LeakyReLU, TimeDistributed, BatchNormalization, Activation, \
            Add, Input, AveragePooling1D, Concatenate

from matplotlib import pyplot as plt

def get_date_ordinal(cycle):
    date_ordinal = np.array([dt.strptime(t.replace(".", "/"), "%Y/%m/%d").\
    toordinal() % cycle for t in np.array(data.Date)])

    date_ordinal = (date_ordinal - np.min(date_ordinal)) \
        / (np.max(date_ordinal) - np.min(date_ordinal))
    return date_ordinal

filename = 'data/US5001440.csv'
valid_idx = 200

data = pd.read_csv(filename, names=['Date', 'Time',
                                    'Open', 'High',
                                    'Low', 'Close',
                                    'Volume'])

lookbacks = 10
total_bars = 3500
num_forward_bars = 30

hightail = np.array((data.High - data.Open) / data.Open)
hightail = (hightail - np.min(hightail)) / (np.max(hightail) - np.min(hightail))

lowtail = np.array((data.Low - data.Open) / data.Open)
lowtail = (lowtail - np.min(lowtail)) / (np.max(lowtail) - np.min(lowtail))

body = np.array((data.Close - data.Open) / data.Open)
body = (body - np.min(body)) / (np.max(body) - np.min(body))

op = np.array(data.Open)
date_ordinal1 = get_date_ordinal(5)
date_ordinal2 = get_date_ordinal(10)
date_ordinal3 = get_date_ordinal(60)
date_ordinal4 = get_date_ordinal(240)
date_ordinal5 = get_date_ordinal(600)

market_data = np.stack((op, hightail, lowtail, body,
                        date_ordinal1, date_ordinal2, date_ordinal3,
                        date_ordinal4, date_ordinal5), axis=1)

df = pd.DataFrame(market_data)

for i in range(1, lookbacks):
    df['lb_ht'+str(i)] = df[1].shift(i)
    df['lb_lt'+str(i)] = df[2].shift(i)
    df['lb_body'+str(i)] = df[3].shift(i)
    df['ord1'+str(i)] = df[4].shift(i)
    df['ord2'+str(i)] = df[5].shift(i)
    df['ord3'+str(i)] = df[6].shift(i)
    df['ord4'+str(i)] = df[7].shift(i)
    df['ord5'+str(i)] = df[8].shift(i)

df = df.iloc[lookbacks:]
dt = np.array(df)
dt = dt[-total_bars:]
x = dt[:, 9:]
# x = (x - np.min(x)) / (np.max(x) - np.min(x))
x = x.reshape(-1, lookbacks-1, 8)
op = dt[:,0]

y = []
for i, _ in enumerate(op[:-num_forward_bars]):
    rn = op[i:i+num_forward_bars]
    yt = (rn - rn.min()) / (rn.max() - rn.min())
    # yt = [0., 0., 0.]
    # if np.argmin(rn) == 0:
    #     yt[0] = 1
    # elif np.argmax(rn) == 0:
    #     yt[1] = 1
    # else:
    #     yt[2] = 1
    y.append(yt)
y = np.array(y)
# x = x[:-num_forward_bars]
op = op[:-num_forward_bars]
#%%
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
    start_model = Input(shape=(lookbacks - 1, 8))
    input_model = Conv1D(kernel_size=24,
                        strides=1,
                        filters=16,
                        padding='same')(start_model)
    input_model = res_block(start_model, 16, 12, 3, downsample=True)
    input_model = res_block(input_model, 32, 6, 5, downsample=True)
    input_model = res_block(input_model, 64, 3, 7, downsample=True)
    input_model = MaxPooling1D()(input_model)
    input_model = Flatten()(input_model)

    # actor layers
    actor = Dense(64, activation=tfa.activations.mish)(input_model)
    # actor = LeakyReLU()(actor)
    actor = Dense(32, activation=tfa.activations.mish)(actor)
    actor = Dense(16, activation=tfa.activations.mish)(actor)
    actor = Dense(32, activation=tfa.activations.mish)(actor)
    actor = Dense(64, activation=tfa.activations.mish)(actor)
    actor = Dense(num_forward_bars, activation='linear')(actor)

    model = Model([start_model], [actor])
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mean_squared_error'])
    return model

def get_eq(signals):
    entries = []
    eq = []
    for i, _ in enumerate(signals):
        if len(entries) > 0:
            eq.append(len(entries)*(np.sum(op[i] - op[i-1] - 0.44)))
        else:
            eq.append(0)

        entry_sig = np.argmin(signals[i])
        exit_sig = np.argmax(signals[i])
        if entry_sig == 0:
            entries.append(op[i])
        if exit_sig == 0:
            entries = []
    return eq

def get_acc_growth(pr, start_amount=1000, eq_pcnt=0.002):
    eq = []
    track_trades = []
    acc_equity = start_amount
    lots = 0
    for i, _ in enumerate(pr):
        if i == 0:
            eq.append(0)
            continue
        pnl = np.sum(op[i] - op[i-1] - 0.44)
        if len(track_trades) > 0:
            for t in track_trades:
                t.append(pnl)

        pnl *= lots
        acc_equity += pnl

        if acc_equity < 100:
            return -1, []

        eq.append(pnl)

        entry_sig = np.argmin(pr[i])
        exit_sig = np.argmax(pr[i])
        if entry_sig == 0:
            lots = int(round((acc_equity * eq_pcnt) - lots))
            lots = max(lots, 0)
            track_trades.append([])
        if exit_sig == 0:
            lots = 0
    return eq, track_trades

def get_acc_equity(pr, start_amount=1000, eq_pcnt=0.002):
    eq = []
    track_trades = []
    acc_equity = start_amount
    lots = 0
    interest = np.log(0.44)
    for i, _ in enumerate(pr):
        if i == 0:
            eq.append(0)
            continue
        pnl = np.log(op[i] - interest) - np.log(op[i-1])
        if len(track_trades) > 0:
            for t in track_trades:
                t.append(pnl)

        pnl *= lots
        acc_equity += pnl

        if acc_equity < 100:
            return -1, []

        eq.append(pnl)

        entry_sig = np.argmin(pr[i])
        exit_sig = np.argmax(pr[i,:10])
        if entry_sig == 0:
            lots = int(round((acc_equity * eq_pcnt)))
            lots = max(lots, 1)
            # lots = 1
            track_trades.append([])
        if exit_sig == 0:
            lots = 0
    return eq, track_trades

def get_eq2(signals):
    entries = []
    eq = []
    for i, _ in enumerate(signals):
        if len(entries) > 0:
            eq.append((np.sum(op[i] - op[i-1])))
        else:
            eq.append(0)
        sig = np.argmax(signals[i])
        if sig == 0 and len(entries) == 0:
            entries.append(op[i])
        if sig == 1:
            entries = []
    return eq

def maxdd(eq):
    return -np.min(np.cumsum(eq) - np.maximum.accumulate(eq))
#%%
k.clear_session()
model = get_model()

best_gain = 0
best_eq = []
best_signals = []

xt = x[:-num_forward_bars]
for _ in range(1000):
    model.fit(xt[:-valid_idx], y[:-valid_idx], epochs=1,
            validation_data=(xt[-valid_idx:], y[-valid_idx:]),
            batch_size=64, shuffle=True, verbose=0)
    signals = model.predict(xt)
    eq, track_trades = get_acc_equity(signals,
                        start_amount=5000,
                        eq_pcnt=0.001)
    if eq == -1:
        continue
    # eq_val = np.mean(eq[-valid_idx:]) / maxdd(eq[-valid_idx:])
    eq_val = np.mean(eq[-valid_idx:]) / np.std(eq[-valid_idx:])
    if eq_val > best_gain:
        model.save("models/US500.h5")
        best_gain = eq_val
        best_eq = eq
        best_signals = signals
        best_trades = np.array(track_trades)
        plt.plot(np.cumsum(eq[-valid_idx:]))
        plt.show()
        plt.close()
#%%
k.clear_session()
model = tf.keras.models.load_model("models/US500.h5")
pr = model(xt)
eq, trades = get_acc_equity(pr, start_amount=1000, eq_pcnt=0.01)

plt.plot(np.cumsum(eq[-200:]))
plt.show()
plt.close()
#%%
model = tf.keras.models.load_model("models/US500.h5")
pr = model(x)
np.argmin(pr[-10:], axis=1)
#%%
pr[-1]
#%%
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

dt.strptime(np.array(data.Date)[1].replace(".", "/"), "%Y/%m/%d").toordinal()

df = data.iloc[lookbacks:]
date_ordinal = np.array([dt.strptime(t.replace(".", "/"), "%Y/%m/%d").\
    toordinal() for t in np.array(df.Date)])

y = np.array(data.Open.shift(-1) - data.Open)
date_ordinal = date_ordinal[1:]
op = np.array(df.Open)[1:]

x = []
for d in range(2, 600):
    dord = date_ordinal % d
    x.append(dord)
x = np.array(x)[1:]
x = np.swapaxes(x, 0, 1)
sim = cosine_similarity(x[-200:], x[:-200])
sim = np.argmax(sim, axis=1)
sig = [np.cumsum(y[i:i+30]) for i in sim]
#%%
def get_acc_equity(pr, op, start_amount=1000, eq_pcnt=0.002):
    eq = []
    track_trades = []
    acc_equity = start_amount
    lots = 0
    for i, _ in enumerate(pr):
        if i == 0:
            eq.append(0)
            continue
        pnl = np.sum(op[i] - op[i-1] - 0.44)
        if len(track_trades) > 0:
            for t in track_trades:
                t.append(pnl)

        pnl *= lots
        acc_equity += pnl

        if acc_equity < 100:
            return -1, []

        eq.append(pnl)

        entry_sig = np.argmin(pr[i])
        exit_sig = np.argmax(pr[i])
        if entry_sig == 0:
            lots = int(round((acc_equity * eq_pcnt) - lots))
            lots = max(lots, 0)
            track_trades.append([])
        if exit_sig == 0:
            lots = 0
    return eq, track_trades

def mse(x_pred):
    err = [x-s for s in x_pred]
    err = np.mean(err, axis=1)
    err = np.square(err)
    err = np.argmin(err, axis=1)
    return err
#%%
# best 161
best_eq = 0

for _ in range(200):
    for d in range(2, 600):
        dord = date_ordinal % d
        x.append(dord)
    x = np.array(x)
    x = np.swapaxes(x, 0, 1)
    sim = cosine_similarity(x[-200:], x[:-200])
    sim = np.argmax(sim, axis=1)
    sig = [np.cumsum(y[i:i+2]) for i in sim]
#%%
# sim = cosine_similarity(x, x[:-200])
sim = mse(x[:-200])
sim = np.argmax(sim, axis=1)
sig = [np.cumsum(y[i:i+2]) for i in sim]
eq, trades = get_acc_equity(sig, op)

plt.plot(np.cumsum(eq))
#%%
err = [x-s for s in x[-20:]]
err = np.mean(err, axis=1)
err = np.square(err)
err = np.argmin(err, axis=1)
err
#%%
x = []

for d in range(20, 600):
    dord = date_ordinal % d
    x.append(dord)
x = np.array(x)
x = np.swapaxes(x, 0, 1)
sim = cosine_similarity(x[-200:], x[:-200])
#%%
np.argmax(sim, axis=1)
#%%
x[:, 2] = 0
#%%
import datetime

datetime.datetime.now()
#%%
int("%.2i%.2i" % (10, 00))
#%%
import numpy as np
import pandas as pd

lookbacks = 5
smas = [5, 10, 20, 50, 100, 200]
moms = [5, 10, 20, 30, 50, 100, 200]

df = pd.read_csv("data/US5001440.csv")

df.drop("Volume", axis=1, inplace=True)
df['Body'] = (df.Close - df.Open) / df.Open

df['Hightail'] = (df.High - df.Open) / df.Open
df['Lowtail'] = (df.Open - df.Low) / df.Open

for lb in range(1, lookbacks + 1):
    df['Body_%i' % lb] = df.Body.shift(lb)
    df['Hightail_%i' % lb] = df.Hightail.shift(lb)
    df['Lowtail_%i' % lb] = df.Lowtail.shift(lb)

for s in smas:
    df['SMA%i' % s] = df.Close.rolling(s).mean() / df.Close

for m in moms:
    df["MOM%i" % m] = df.Close.pct_change(m)

df['Returns'] = df.Close - df.Open
df.Returns = df.Returns.shift(-1)
df.Open = df.Open.shift(-1)
df.High = df.High.shift(-1)
df.Low = df.Low.shift(-1)
df.Close = df.Close.shift(-1)
# df.Hightail = df.Hightail.shift(-1)
# df.Lowtail = df.Lowtail.shift(-1)
# df.Body = df.Body.shift(-1)

features = np.array(df.columns)[5:]

df
#%%
from sklearn.ensemble import RandomForestRegressor
data = np.array(df)[200:-1, 6:]
x = data[:, :-1]
y = data[:, -1]

rf = RandomForestRegressor(100, max_depth=5, random_state=0)
rf.fit(x[:-100], y[:-100])
#%%
predictions = rf.predict(x[-100:])
#%%
from sklearn.metrics import accuracy_score

acc = accuracy_score([0 if i < 0 else 1 for i in y[-100:]], [0 if i < 0 else 1 for i in predictions])
acc
#%%
print(y[-10])
print(predictions[-10])
#%%
impt = rf.feature_importances_
impt_idx = np.argsort(impt)[::-1]
#%%
for i in impt_idx:
    print("%s -- %.5f" % (features[i], impt[i]))
#%%
