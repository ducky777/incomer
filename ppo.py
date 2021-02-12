#%%
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
#%%
lookbacks = 61
total_bars = 1000

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

x.shape
#%%
class FXEnv:
    def __init__(self, x, y, op, hi, lo, cl,
                 spread=0.0001, gamma=0.05, max_trades=4,
                 periods_per_episode=20):
        # initialize data
        self.open = np.array(op)
        self.high = np.array(hi)
        self.low = np.array(lo)
        self.close = np.array(cl)
        self.states = tf.convert_to_tensor(x)
        self.y = y
        self.max_trades = max_trades
        self.periods_per_episode = periods_per_episode

        # initialize environment
        self.spread = spread
        self.pnl = 0
        self.current_position = 0
        self.nu_trades_in_ep = 0
        self.observation_space = np.array((x.shape[1], x.shape[2]))
        self.action_space = 3
        self.metadata = "No Meta Data"

    def reset(self):
        start_idx = random.randint(0, len(self.y) - self.periods_per_episode -1)
        self.pnl = 0
        self.current_position = 0
        self.nu_trades_in_ep = 0
        self.current_bar = start_idx
        self.running_bars = 0
        return self.states[self.current_bar]

    def step(self, action):
        if action == 2:
            action = -1
        reward = 0
        done = False
        if action != 0:
            if action != self.current_position:
                reward -= self.spread
                self.nu_trades_in_ep += 1
                if self.nu_trades_in_ep >= self.max_trades:
                    return 0, 0, True, "No Info"
        reward += action * self.y[self.current_bar]
        self.current_position = action
        self.current_bar += 1
        self.running_bars += 1
        if self.running_bars > self.periods_per_episode:
            done = True
            new_state = 0
            reward = 0
        else:
            new_state = self.states[self.current_bar]
        return new_state, reward, done, "No Info"

env = FXEnv(x, body, data_copy.Open, data_copy.High,
            data_copy.Low, data_copy.Close)
state = env.reset()
#%%
model = PPO2(MlpPolicy, env, verbose=1)
# %%
env.states