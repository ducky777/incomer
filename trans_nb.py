#%%
import pandas as pd
import numpy as np
from datetime import datetime as dt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as k

def get_date_ordinal(cycle):
    date_ordinal = np.array([dt.strptime(t.replace(".", "/"), "%d/%m/%Y").\
    toordinal() % cycle for t in np.array(data.Date)])

    date_ordinal = (date_ordinal - np.min(date_ordinal)) \
        / (np.max(date_ordinal) - np.min(date_ordinal))
    return date_ordinal

filename = 'data/US5001440.csv'
valid_idx = 200

data = pd.read_csv(filename, names=['Date', 'Time',
                                    'Open', 'High',
                                    'Low', 'Close',
                                    'Volume'],
                   header=0).iloc[1:]

lookbacks = 120
total_bars = 5000
num_forward_bars = 10

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
y_cat = []
for i, _ in enumerate(op[:-num_forward_bars]):
    pcnt_gain = []
    rn = op[i:i+num_forward_bars+1]
    for j, p in enumerate(rn[:-1]):
        gain = (rn[j+1] - p) / p
        pcnt_gain.append(gain)
    # yt = (rn - rn.min()) / (rn.max() - rn.min())
    y.append(pcnt_gain)
    cat = 0
    if np.argmin(rn) == 0:
        cat = 1
    elif np.argmax(rn) == 0:
        cat = 2
    y_cat.append(cat)

y_cat = np.expand_dims(y_cat, -1)
y = np.array(y)
op = op[:-num_forward_bars]

ymin, ymax = y.min(), y.max()
y = (y - ymin) / (ymax - ymin)
print(y.shape)
print(y.max(), y.min())
#%%
print(len(y_cat[y_cat == 0]))
print(len(y_cat[y_cat == 1]))
print(len(y_cat[y_cat == 2]))
#%%
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalMaxPooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    return keras.Model(inputs, outputs)

input_shape = x.shape[1:]

k.clear_session()

model = build_model(
    input_shape,
    head_size=128,
    num_heads=8,
    ff_dim=4,
    num_transformer_blocks=16,
    mlp_units=[128],
    mlp_dropout=0.1,
    dropout=0.1,
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"],
)
model.summary()
#%%
from imblearn.over_sampling import SMOTE

xtrain = x[:-num_forward_bars]
xtrain = xtrain[:-valid_idx]

ytrain = y_cat[:-valid_idx]
oversample = SMOTE(sampling_strategy='not majority')
x_over, y_over = oversample.fit_resample(xtrain.reshape(-1, (x.shape[1] * x.shape[2])), ytrain)
x_over = x_over.reshape(-1, x.shape[1], x.shape[2])
#%%
xtrain = np.array([*x_over, *xtrain])
ytrain = np.array([*y_over, *ytrain])
#%%
ytrain = ytrain.reshape(-1, 1).astype(np.float32)
print(xtrain.shape)
print(ytrain.shape)
#%%
def calculate_dd(eq):
    end = np.argmax(np.maximum.accumulate(eq) - eq)
    start = np.argmax(eq[:end])
    return eq[end] - eq[start]

def get_eq(predictions):
    add_balance = 50000
    min_elasped = 20

    # gains = []
    trades = []
    eq = []
    store_trades = []
    entry_price = 0
    entry_elasped = min_elasped
    balance = []
    closed = []

    ops = op[-valid_idx:-num_forward_bars]
    for j, (pr, o)in enumerate(zip(predictions, ops)):
        lots = round(add_balance / o)
        # lots = 1
        if np.argmax(pr)==1 and entry_elasped >= min_elasped:
            print(f"Entered: {o} / {lots} @ {j}")
            trades.append([o, lots])
            entry_elasped = 0
        elif np.argmax(pr)==2 and len(trades) > 0:
            # closed.append(pnl)
            trades = []
            print(f"Closed: {o} / {lots} @ {j}")
            # entry_elasped = min_elasped
        pnl = 0
        for _, l in trades:
            pnl += ((o - ops[j-1]) * l)
        eq.append(pnl)
        store_trades.append(trades)
        entry_elasped += 1

    return np.array(eq)

# xt = x[:-num_forward_bars]
# model = keras.models.load_model("data/best_lb30_f20.h5")

best_eq = 0
for _ in range(999999):
    model.fit(xtrain, ytrain, epochs=1,validation_split=0.9,
            batch_size=32, shuffle=True, verbose=1)
    predictions = model.predict(x[-valid_idx:-num_forward_bars])
    eq = get_eq(predictions)
    try:
        dd = calculate_dd(eq)
    except Exception:
        continue
    if dd == 0:
        dd = 1
    score = np.sum(eq) / abs(dd)
    if score > best_eq:
        model.save("data/best_lb30_f20.h5")
        best_eq = score
        print(best_eq)
#%%
# model.save("data/best.h5")
#%%
len(predictions)
#%%
predictions = model.predict(x[-valid_idx:-num_forward_bars])
#%%
np.argmax(predictions, 1)
#%%
print(len(predictions))
len(op[-valid_idx:-num_forward_bars])
#%%
import matplotlib.pyplot as plt

model = keras.models.load_model("data/best_lb30_f20.h5")

model.summary()
#%%
from keras.models import Model

extractor = Model(model.input, model.layers[-3].output)
extractor.summary()
#%%
predictions = extractor.predict(x[-valid_idx:-num_forward_bars])
#%%
base = extractor.predict(x[:-valid_idx])
#%%
from sklearn.metrics.pairwise import cosine_similarity

cosine = cosine_similarity(predictions, base)
#%%
max_signals = np.argmax(cosine, axis=1)
denorm = y * (ymax - ymin) + ymin

signals = denorm[max_signals]

signals[0]
#%%
import matplotlib.pyplot as plt

add_balance = 50000
min_elasped = 20

# gains = []
trades = []
eq = []
store_trades = []
entry_price = 0
entry_elasped = min_elasped
balance = []
closed = []

ops = op[-valid_idx:-num_forward_bars]
for j, (pr, o)in enumerate(zip(signals, ops)):
    csum = np.cumsum(pr)
    lots = round(add_balance / o)
    # lots = 1
    if np.argmin(csum)==0 and entry_elasped >= min_elasped:
        print(f"Entered: {o} / {lots} @ {j}")
        trades.append([o, lots])
        entry_elasped = 0
    elif np.argmax(csum)==0 and len(trades) > 0:
        # closed.append(pnl)
        trades = []
        print(f"Closed: {o} / {lots} @ {j}")
        # entry_elasped = min_elasped
    pnl = 0
    for _, l in trades:
        pnl += ((o - ops[j-1]) * l)
    eq.append(pnl)
    store_trades.append(trades)
    entry_elasped += 1

eq =  np.array(eq)

plt.plot(np.cumsum(eq))
plt.show()
#%%
dd = calculate_dd(eq)
np.sum(eq) / -dd
#%%
def get_eq(predictions):
    add_balance = 50000
    min_elasped = 20

    # gains = []
    trades = []
    eq = []
    store_trades = []
    entry_price = 0
    entry_elasped = min_elasped
    balance = []
    closed = []

    ops = op[-valid_idx:-num_forward_bars]
    for j, (pr, o)in enumerate(zip(predictions, ops)):
        lots = round(add_balance / o)
        # lots = 1
        if np.argmax(pr)==1 and entry_elasped >= min_elasped:
            print(f"Entered: {o} / {lots} @ {j}")
            trades.append([o, lots])
            entry_elasped = 0
        elif np.argmax(pr)==2 and len(trades) > 0:
            # closed.append(pnl)
            trades = []
            print(f"Closed: {o} / {lots} @ {j}")
            # entry_elasped = min_elasped
        pnl = 0
        for _, l in trades:
            pnl += ((o - ops[j-1]) * l)
        eq.append(pnl)
        store_trades.append(trades)
        entry_elasped += 1

    return np.array(eq)

eq = get_eq(predictions)
plt.plot(np.cumsum(eq))
plt.show()
#%%
def calculate_dd(eq):
    end = np.argmax(np.maximum.accumulate(eq) - eq)
    start = np.argmax(eq[:end])
    return eq[end] - eq[start]

dd = calculate_dd(eq)
dd
# print(eq[start] - eq[end])
# end - start
#%%
np.sum(eq)
#%%