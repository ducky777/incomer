#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

#%%
symbol = "EURUSD1440"
num_data = 800
valid_ratio = 0.3

data = pd.read_csv('%s.csv' % symbol, names=['Date', 'Time',
                                            'Open', 'High',
                                            'Low', 'Close',
                                            'Volume'])

# hightail = np.array(data.High - data.Open)
# lowtail = np.array(data.Low - data.Open)
# body = np.array(data.Close - data.Open)

hightail = np.array(np.log(data.High) - np.log(data.Open))
lowtail = np.array(np.log(data.Low) - np.log(data.Open))
body = np.array(np.log(data.Close) - np.log(data.Open))

# hightail = np.array((data.High - data.Open) / data.Open)
# lowtail = np.array((data.Low - data.Open) / data.Open)
# body = np.array((data.Close - data.Open) / data.Open)

market_data = np.stack((hightail, lowtail, body), axis=1)

valid_idx = int(num_data * (1 - valid_ratio))

def get_pr(lookbacks):
    df = pd.DataFrame(market_data)

    for i in range(1, lookbacks):
        df['lb_ht'+str(i)] = df[0].shift(i)
        df['lb_lt'+str(i)] = df[1].shift(i)
        df['lb_body'+str(i)] = df[2].shift(i)

    df = df.iloc[lookbacks:]
    data = np.array(df)
    data = data[-num_data:]
    y = data[:, 2]
    x = data[:, 3:]

    body = y[:]

    y = y/np.max(np.abs(y))
    x = x/np.max(np.abs(x))

    x = x.reshape((-1, (lookbacks-1) * 3))
    model = KNeighborsRegressor(1)
    model.fit(x[:valid_idx], y[:valid_idx])
    pr = model.predict(x)
    return pr

def get_eq(pr):
    eq = []
    track_position = 0

    for i, result in enumerate(pr):
        if result > 0.15:
            pnl = body[i]
            if track_position != 1:
                pnl -= 0.00005
            pnl = pnl * result
            track_position = 1
            eq.append(pnl)
        elif result < -0.15:
            pnl = -body[i]
            if track_position != -1:
                pnl -= 0.00005
            pnl = pnl * result
            track_position = -1
            eq.append(pnl)
        else:
            eq.append(0)
    return np.array(eq)
#%%
pr = []
for i in tqdm(range(3, 70)):
    result = get_pr(i)
    pr.append(result)
pr = np.array(pr)
#%%
best_eq = 0
best_weights = []
for _ in range(50000):
    weights = np.random.uniform(-1, 1.0001, size=(pr.shape[0], 1))
    weighted_pr = pr * weights
    weighted_pr = np.sum(weighted_pr, axis=0)

    eq = get_eq(weighted_pr)
    # eq[valid_idx:-60] *= -1

    total_eq = np.mean(eq[valid_idx:-60]) / np.std(eq[valid_idx:-60])
    if total_eq > best_eq:
        # plt.plot(np.cumsum(eq))
        # plt.show()
        # plt.close()
        plt.plot(np.cumsum(eq[valid_idx:]))
        plt.show()
        plt.close()
        best_eq = total_eq
        best_weights = weights

print("Done!")
#%%
weighted_pr = pr * best_weights
weighted_pr = np.sum(weighted_pr, axis=0)
eq = get_eq(weighted_pr)
plt.plot(np.cumsum(eq))
plt.show()
plt.close()
plt.plot(np.cumsum(eq[valid_idx:]))
plt.show()
plt.close()
plt.plot(np.cumsum(eq[-60:]))
plt.show()
plt.close()
#%%
best_weights
#%%
np.savetxt("%s_knnweights.npy" % symbol, best_weights)
#%%
eq[valid_idx:-60]
#%%
import numpy as np

weights = np.loadtxt("EURUSD1440_knnweights.npy")
weights
#%%